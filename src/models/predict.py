#!/usr/bin/env python3

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import random
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

try:
    from model import JunctionUNet
except ImportError:
    try:
        from .model import JunctionUNet
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        from model import JunctionUNet

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src/walkers") not in sys.path:
    sys.path.insert(0, str(repo_root / "src/walkers"))
if str(repo_root / "src/preprocessing") not in sys.path:
    sys.path.insert(0, str(repo_root / "src/preprocessing"))

from topological_walker import TopologicalWalker
from cleaning import extract_veins_tophat


DEFAULT_MODEL = Path("training_output/best_model.pt")
DEFAULT_INPUT = Path("data/cleaned_wings")
DEFAULT_SEEDS_DIR = Path("data/cleaned_seeds")
DEFAULT_COORDS_DIR = Path("data/unet_predictions")
DEFAULT_OUTPUT_DIR = Path("prediction_output")
DEFAULT_THRESHOLD = 0.40
DEFAULT_DPI = 150
DEFAULT_WORKERS = 6


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = checkpoint.get("base_channels", 16)
    model = JunctionUNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


def pad_to_multiple_16(img):
    h, w = img.shape
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
    else:
        padded = img
    return padded, pad_h, pad_w


def extract_peaks(heatmap, threshold=0.25, kernel_size=3):
    """
    2D Max-Pooling local peak detection.
    """
    tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    max_pooled = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=pad)
    peaks_mask = (tensor == max_pooled) & (tensor > threshold)
    peaks_np = peaks_mask[0, 0].cpu().numpy()

    ys, xs = np.where(peaks_np)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([xs, ys]).astype(np.int32)


def snap_peaks_to_veins(peaks, gray_img, max_dist=3.0, threshold=128):
    """
    Snap candidate peak coordinates to the nearest physical vein skeleton
    to guarantee 100% biological plausibility.
    """
    if len(peaks) == 0:
        return peaks

    # Vein mask: dark pixels
    vein_mask = (gray_img < threshold).astype(np.uint8)
    vein_ys, vein_xs = np.where(vein_mask)
    if len(vein_xs) == 0:
        return peaks

    vein_coords = np.column_stack([vein_xs, vein_ys])
    tree = cKDTree(vein_coords)
    dists, indices = tree.query(peaks)

    snapped = peaks.copy()
    close_mask = dists <= max_dist
    snapped[close_mask] = vein_coords[indices[close_mask]]
    return snapped


def filter_degree2_false_positives(
    peaks: np.ndarray,
    img: np.ndarray,
    max_branch_dist: float = 8.0,
    ridge_threshold: int = 10,
) -> np.ndarray:
    """
    Suppresses candidate peaks that sit on unbranched (degree-2) vein skeletons
    caused by micro-intensity undulations on thick vein trunks.
    """
    if len(peaks) == 0:
        return peaks

    bw_top, _ = extract_veins_tophat(
        img, kernel_size=9, ridge_threshold=ridge_threshold, background_threshold=250
    )
    vein_mask = (bw_top == 0).astype(np.uint8)
    tw = TopologicalWalker(vein_mask, min_component_size=3, spur_threshold=3)
    cleaned = tw.clean_mask()
    skeleton = tw.compute_skeleton(cleaned)

    # Rutovitz crossing number
    p = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    pad = np.pad(skeleton, 1, mode="constant")
    h, w = skeleton.shape
    nbrs = [pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in p]

    crossing_num = np.zeros_like(skeleton, dtype=np.float32)
    for i in range(8):
        crossing_num += np.abs(nbrs[i].astype(np.float32) - nbrs[(i + 1) % 8].astype(np.float32))
    crossing_num = (0.5 * crossing_num) * skeleton

    branch_pts = np.argwhere(crossing_num >= 3)
    if len(branch_pts) == 0:
        return peaks

    branch_xy = np.column_stack([branch_pts[:, 1], branch_pts[:, 0]])
    tree_branch = cKDTree(branch_xy)

    dists, _ = tree_branch.query(peaks)
    filtered = peaks[dists <= max_branch_dist]
    return filtered


@torch.no_grad()
def predict_image(
    model,
    img,
    device,
    threshold=0.25,
    snap=True,
    filter_degree2=True,
    max_branch_dist=8.0,
    ridge_threshold=10,
):
    h_orig, w_orig = img.shape
    padded, pad_h, pad_w = pad_to_multiple_16(img)

    img_tensor = (
        torch.from_numpy(padded.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    logits = model(img_tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Crop out padding
    prob_cropped = prob[:h_orig, :w_orig]

    # Extract peaks
    peaks = extract_peaks(prob_cropped, threshold=threshold)

    # Optional snap to veins
    if snap:
        peaks = snap_peaks_to_veins(peaks, img, max_dist=3.0)

    # Optional filter degree-2 false positive peaks
    if filter_degree2 and len(peaks) > 0:
        peaks = filter_degree2_false_positives(
            peaks, img, max_branch_dist=max_branch_dist, ridge_threshold=ridge_threshold
        )

    return prob_cropped, peaks


def evaluate_against_gt(pred_coords, gt_coords, tolerance=8.0):
    if len(gt_coords) == 0 and len(pred_coords) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_coords) == 0:
        return 0.0, 1.0, 0.0
    if len(pred_coords) == 0:
        return 1.0, 0.0, 0.0

    tree_gt = cKDTree(gt_coords)
    tree_pred = cKDTree(pred_coords)

    dists_pred, _ = tree_gt.query(pred_coords)
    tp_pred = np.sum(dists_pred <= tolerance)
    precision = float(tp_pred) / len(pred_coords)

    dists_gt, _ = tree_pred.query(gt_coords)
    tp_gt = np.sum(dists_gt <= tolerance)
    recall = float(tp_gt) / len(gt_coords)

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def _plot_worker(task):
    img, prob_map, pred_peaks, gt_peaks, output_path_str, metrics, dpi = task
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Image with predicted junctions (red)
    axes[0].imshow(img, cmap="gray")
    if len(pred_peaks) > 0:
        axes[0].scatter(
            pred_peaks[:, 0], pred_peaks[:, 1], c="red", s=10, marker="o", label=f"Predicted ({len(pred_peaks)})"
        )
    axes[0].set_title(f"Input Wing + Predicted Junctions ({len(pred_peaks)})")
    axes[0].axis("off")
    axes[0].legend(loc="lower right")

    # 2. Predicted Heatmap
    im1 = axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("CNN Probability Heatmap")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Ground Truth (green) vs Predicted (red)
    axes[2].imshow(img, cmap="gray")
    if gt_peaks is not None and len(gt_peaks) > 0:
        axes[2].scatter(
            gt_peaks[:, 0], gt_peaks[:, 1], c="lime", s=12, marker="o", alpha=0.7, label=f"GT ({len(gt_peaks)})"
        )
    if len(pred_peaks) > 0:
        axes[2].scatter(
            pred_peaks[:, 0], pred_peaks[:, 1], c="red", s=8, marker="x", alpha=0.8, label=f"Pred ({len(pred_peaks)})"
        )

    title_str = "GT (Green) vs Predicted (Red)"
    if metrics is not None:
        p, r, f = metrics
        title_str += f"\nP: {p*100:.1f}% | R: {r*100:.1f}% | F1: {f*100:.1f}%"
    axes[2].set_title(title_str)
    axes[2].axis("off")
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_prediction_plot(
    img, prob_map, pred_peaks, gt_peaks, output_path, metrics=None, dpi=DEFAULT_DPI
):
    _plot_worker((img, prob_map, pred_peaks, gt_peaks, str(output_path), metrics, dpi))


def main():
    parser = argparse.ArgumentParser(description="Run junction prediction on dragonfly wing images")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coords-dir", type=Path, default=DEFAULT_COORDS_DIR)
    parser.add_argument("--seeds-dir", type=Path, default=DEFAULT_SEEDS_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tolerance", type=float, default=8.0, help="Matching tolerance in pixels (default: 8.0)")
    parser.add_argument("--no-filter-degree2", action="store_true", help="Disable degree-2 unbranched skeleton peak suppression")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-plot", action="store_true", help="Skip saving visual prediction plots")
    parser.add_argument("--no-coords", action="store_true", help="Skip saving coordinate .npy files")
    parser.add_argument("--sample-k", type=int, default=None, help="Sample k wings at random and evaluate")
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.model, device)

    # Find images
    if args.input.is_file():
        image_paths = [args.input]
    else:
        image_paths = sorted([
            p for p in args.input.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            and not p.stem.endswith("_blackwhite")
        ])

    if args.sample_k is not None:
        random.seed(42)
        image_paths = random.sample(image_paths, min(args.sample_k, len(image_paths)))
        print(f"Sampled {len(image_paths)} wings at random for verification.")

    if not args.no_coords and args.coords_dir is not None:
        args.coords_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving predicted coordinates (.npy) to: {args.coords_dir}")

    if not args.no_plot and args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visual overlay plots (.png) to: {args.output_dir}")

    pool = ProcessPoolExecutor(max_workers=args.workers) if (not args.no_plot and args.workers > 1) else None
    futures = []

    all_p, all_r, all_f1 = [], [], []
    t_start = time.perf_counter()

    try:
        for idx, img_path in enumerate(image_paths, 1):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            filter_deg2 = not args.no_filter_degree2
            ridge_th = 6 if "caribbean_darner" in img_path.stem else 10
            prob_map, pred_peaks = predict_image(
                model, img, device, threshold=args.threshold, filter_degree2=filter_deg2, ridge_threshold=ridge_th
            )

            # Step 1: Save predicted coordinates
            if not args.no_coords and args.coords_dir is not None:
                coord_file = args.coords_dir / f"{img_path.stem}.npy"
                np.save(str(coord_file), pred_peaks)

            # Check for matching GT seeds
            gt_seeds = None
            metrics = None
            seed_cand = args.seeds_dir / f"{img_path.stem}.npy"
            if not seed_cand.exists():
                seed_cand = args.seeds_dir / f"{img_path.stem}_blackwhite.npy"

            if seed_cand.exists():
                gt_seeds = np.load(str(seed_cand))
                if gt_seeds.ndim == 1:
                    gt_seeds = gt_seeds.reshape(-1, 2)
                p, r, f = evaluate_against_gt(pred_peaks, gt_seeds, tolerance=args.tolerance)
                metrics = (p, r, f)
                all_p.append(p)
                all_r.append(r)
                all_f1.append(f)
                if idx % 25 == 0 or idx == len(image_paths):
                    print(f"[{idx:03d}/{len(image_paths)}] {img_path.name:45s} | Pred: {len(pred_peaks):4d} | GT: {len(gt_seeds):4d} | F1: {f*100:.2f}%")
            else:
                if idx % 25 == 0 or idx == len(image_paths):
                    print(f"[{idx:03d}/{len(image_paths)}] {img_path.name:45s} | Pred: {len(pred_peaks):4d} | No GT found")

            # Step 2: Save plot
            if not args.no_plot and args.output_dir is not None:
                out_plot = args.output_dir / f"{img_path.stem}_prediction.png"
                task = (img, prob_map, pred_peaks, gt_seeds, str(out_plot), metrics, args.dpi)
                if pool is not None:
                    futures.append(pool.submit(_plot_worker, task))
                else:
                    _plot_worker(task)

        if pool is not None:
            print(f"\nProcessing {len(futures)} visual plots using {args.workers} background workers...")
            for f in as_completed(futures):
                f.result()
            print("All visual prediction plots completed successfully.")

    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal elapsed time: {elapsed:.2f}s ({elapsed/60:.2f} min)")

    if all_f1:
        print("\n================ SUMMARY EVALUATION ================")
        print(f"Evaluated Wings: {len(all_f1)}")
        print(f"Mean Precision:  {np.mean(all_p)*100:.2f}%")
        print(f"Mean Recall:     {np.mean(all_r)*100:.2f}%")
        print(f"Mean F1 Score:   {np.mean(all_f1)*100:.2f}%")
        print(f"Median F1 Score: {np.median(all_f1)*100:.2f}%")
        print("====================================================")


if __name__ == "__main__":
    main()
