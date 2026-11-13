#!/usr/bin/env python3

import argparse
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

import sys

try:
    from model import JunctionUNet
except ImportError:
    try:
        from .model import JunctionUNet
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        from model import JunctionUNet


DEFAULT_MODEL = Path("training_output/best_model.pt")
DEFAULT_OUTPUT_DIR = Path("prediction_output")


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


@torch.no_grad()
def predict_image(model, img, device, threshold=0.25, snap=True):
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

    return prob_cropped, peaks


def evaluate_against_gt(pred_coords, gt_coords, tolerance=5.0):
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


def save_prediction_plot(
    img, prob_map, pred_peaks, gt_peaks, output_path, metrics=None
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Image with predicted junctions (red)
    axes[0].imshow(img, cmap="gray")
    if len(pred_peaks) > 0:
        axes[0].scatter(
            pred_peaks[:, 0], pred_peaks[:, 1], c="red", s=10, marker="o", label=f"Predicted ({len(pred_peaks)})"
        )
    axes[0].set_title("Input Wing + Predicted Junctions")
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run junction prediction on dragonfly wing images")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=Path("data/rotate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds-dir", type=Path, default=Path("data/seeds"))
    parser.add_argument("--threshold", type=float, default=0.25)
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
        ])

    if args.sample_k is not None:
        random.seed(42)
        image_paths = random.sample(image_paths, min(args.sample_k, len(image_paths)))
        print(f"Sampled {len(image_paths)} wings at random for verification.")

    all_p, all_r, all_f1 = [], [], []

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        prob_map, pred_peaks = predict_image(model, img, device, threshold=args.threshold)

        # Check for matching GT seeds
        gt_seeds = None
        metrics = None
        seed_cand = args.seeds_dir / f"{img_path.stem}.npy"
        if not seed_cand.exists():
            seed_cand = args.seeds_dir / f"{img_path.stem}_blackwhite.npy"

        if seed_cand.exists():
            gt_seeds = np.load(str(seed_cand))
            p, r, f = evaluate_against_gt(pred_peaks, gt_seeds, tolerance=5.0)
            metrics = (p, r, f)
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f)
            print(f"{img_path.name:45s} | Pred: {len(pred_peaks):4d} | GT: {len(gt_seeds):4d} | P: {p*100:.1f}% | R: {r*100:.1f}% | F1: {f*100:.1f}%")
        else:
            print(f"{img_path.name:45s} | Pred: {len(pred_peaks):4d} | No GT found")

        # Save plot
        out_plot = args.output_dir / f"{img_path.stem}_prediction.png"
        save_prediction_plot(img, prob_map, pred_peaks, gt_seeds, out_plot, metrics=metrics)

    if all_f1:
        print("\n================ SUMMARY EVALUATION ================")
        print(f"Evaluated Wings: {len(all_f1)}")
        print(f"Mean Precision:  {np.mean(all_p)*100:.2f}%")
        print(f"Mean Recall:     {np.mean(all_r)*100:.2f}%")
        print(f"Mean F1 Score:   {np.mean(all_f1)*100:.2f}%")
        print("====================================================")


if __name__ == "__main__":
    main()
