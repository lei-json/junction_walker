#!/usr/bin/env python3

from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import torch

try:
    from model import JunctionUNet
    from predict import predict_image, evaluate_against_gt
except ImportError:
    try:
        from .model import JunctionUNet
        from .predict import predict_image, evaluate_against_gt
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        from model import JunctionUNet
        from predict import predict_image, evaluate_against_gt


def verify_k_random_wings(
    k=10,
    seeds_dir="data/seeds",
    image_dir="data/rotate",
    bw_dir="data/blackwhite_rotate",
    checkpoint_path="training_output/best_model.pt",
    output_dir="data/verification_k_wings",
    threshold=0.40,
    tolerance=5.0,
    seed=100,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir = Path(seeds_dir)
    image_dir = Path(image_dir)
    bw_dir = Path(bw_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path} on {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = checkpoint.get("base_channels", 16)
    model = JunctionUNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Discover paired wings
    seed_files = sorted(seeds_dir.glob("*.npy"))
    valid_pairs = []
    for sf in seed_files:
        stem = sf.stem.replace("_blackwhite", "")
        for d in [image_dir, bw_dir]:
            found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = d / f"{stem}{ext}"
                candidate_bw = d / f"{stem}_blackwhite{ext}"
                if candidate.exists():
                    valid_pairs.append((candidate, sf))
                    found = True
                    break
                elif candidate_bw.exists():
                    valid_pairs.append((candidate_bw, sf))
                    found = True
                    break
            if found:
                break

    print(f"Found {len(valid_pairs)} total paired wings available for sampling.")
    random.seed(seed)
    sampled_pairs = random.sample(valid_pairs, k)
    print(f"\n================ SAMPLING k={k} RANDOM WINGS ================\n")

    results = []

    for i, (img_path, seed_path) in enumerate(sampled_pairs, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt_seeds = np.load(str(seed_path))
        if gt_seeds.ndim == 1:
            gt_seeds = gt_seeds.reshape(-1, 2)

        prob_map, pred_peaks = predict_image(model, img, device, threshold=threshold, snap=True)
        precision, recall, f1 = evaluate_against_gt(pred_peaks, gt_seeds, tolerance=tolerance)

        passed = (f1 >= 0.90) and (recall >= 0.90) and (precision >= 0.85)
        status_str = "PASSED (>=90%)" if passed else "FAILED"

        print(
            f"[{i:02d}/{k:02d}] {img_path.name:45s} | "
            f"Pred: {len(pred_peaks):4d} | GT: {len(gt_seeds):4d} | "
            f"Precision: {precision*100:.2f}% | "
            f"Recall: {recall*100:.2f}% | "
            f"F1: {f1*100:.2f}% | {status_str}"
        )

        results.append({
            "index": i,
            "filename": img_path.name,
            "img_path": str(img_path),
            "num_pred": len(pred_peaks),
            "num_gt": len(gt_seeds),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "passed": passed,
        })

        # Generate detailed 4-panel verification image
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        # 1. Full image with predicted junctions (red circles)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img, cmap="gray")
        if len(pred_peaks) > 0:
            ax1.scatter(
                pred_peaks[:, 0], pred_peaks[:, 1],
                c="red", s=14, marker="o", edgecolors="black", linewidths=0.5,
                label=f"Predicted Junctions ({len(pred_peaks)})"
            )
        ax1.set_title(
            f"Wing #{i}: {img_path.name}\n"
            f"Predicted Vein Junctions (Red Circles) | Accuracy F1: {f1*100:.2f}%",
            fontsize=12, fontweight="bold"
        )
        ax1.axis("off")
        ax1.legend(loc="lower right")

        # 2. Probability Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(prob_map, cmap="inferno", vmin=0, vmax=1)
        ax2.set_title("CNN Keypoint Probability Heatmap", fontsize=12, fontweight="bold")
        ax2.axis("off")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Ground truth (green) vs Predicted (red) Overlay
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(img, cmap="gray")
        if len(gt_seeds) > 0:
            ax3.scatter(
                gt_seeds[:, 0], gt_seeds[:, 1],
                c="lime", s=16, marker="o", alpha=0.7, label=f"Ground Truth ({len(gt_seeds)})"
            )
        if len(pred_peaks) > 0:
            ax3.scatter(
                pred_peaks[:, 0], pred_peaks[:, 1],
                c="red", s=10, marker="x", alpha=0.9, label=f"Predicted ({len(pred_peaks)})"
            )
        ax3.set_title(
            f"Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1: {f1*100:.2f}%\n"
            f"Tolerance = {tolerance} px",
            fontsize=12, fontweight="bold"
        )
        ax3.axis("off")
        ax3.legend(loc="lower right")

        # 4. Zoomed-in Detail (Middle Section of Wing)
        ax4 = fig.add_subplot(gs[1, 1])
        h, w = img.shape
        # Center crop around 1/4 to 3/4
        cy, cx = h // 2, w // 2
        crop_h = min(h, 150)
        crop_w = min(w, 250)
        y0, y1 = max(0, cy - crop_h // 2), min(h, cy + crop_h // 2)
        x0, x1 = max(0, cx - crop_w // 2), min(w, cx + crop_w // 2)

        crop_img = img[y0:y1, x0:x1]
        ax4.imshow(crop_img, cmap="gray")

        # Crop predicted & GT points
        pred_in_crop = (
            (pred_peaks[:, 0] >= x0) & (pred_peaks[:, 0] < x1) &
            (pred_peaks[:, 1] >= y0) & (pred_peaks[:, 1] < y1)
        )
        gt_in_crop = (
            (gt_seeds[:, 0] >= x0) & (gt_seeds[:, 0] < x1) &
            (gt_seeds[:, 1] >= y0) & (gt_seeds[:, 1] < y1)
        )
        if np.any(gt_in_crop):
            ax4.scatter(
                gt_seeds[gt_in_crop, 0] - x0, gt_seeds[gt_in_crop, 1] - y0,
                c="lime", s=40, marker="o", edgecolors="black", linewidths=1.0, label="GT"
            )
        if np.any(pred_in_crop):
            ax4.scatter(
                pred_peaks[pred_in_crop, 0] - x0, pred_peaks[pred_in_crop, 1] - y0,
                c="red", s=25, marker="x", linewidths=2.0, label="Predicted"
            )
        ax4.set_title("High-Resolution Zoomed Venation Detail", fontsize=12, fontweight="bold")
        ax4.axis("off")
        ax4.legend(loc="lower right")

        plt.tight_layout()
        out_png = output_dir / f"wing_{i:02d}_{img_path.stem}_verification.png"
        fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Summary
    all_f1 = [r["f1"] for r in results]
    all_p = [r["precision"] for r in results]
    all_r = [r["recall"] for r in results]
    all_passed = all(r["passed"] for r in results)

    print("\n================ FINAL VERIFICATION SUMMARY ================")
    print(f"Total Sampled Wings (k): {k}")
    print(f"All Wings Passed (>=90% Accuracy): {all_passed}")
    print(f"Mean Precision: {np.mean(all_p)*100:.2f}%")
    print(f"Mean Recall:    {np.mean(all_r)*100:.2f}%")
    print(f"Mean F1 Score:  {np.mean(all_f1)*100:.2f}%")
    print(f"Min F1 Score:   {np.min(all_f1)*100:.2f}%")
    print(f"Max F1 Score:   {np.max(all_f1)*100:.2f}%")
    print(f"Verification plots saved to: {output_dir.resolve()}")
    print("============================================================\n")

    return results, all_passed


if __name__ == "__main__":
    verify_k_random_wings(k=10)
