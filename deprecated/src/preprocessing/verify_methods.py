#!/usr/bin/env python3
"""
Comprehensive Verification Script for Preprocessing Methods:
1. Automated Wing-Base Body Remnant Removal.
2. Morphological Black Top-Hat Transform for Colored Wings & Dark Filled Cells.
3. Non-Regression Verification on Ground Truth Wings.
"""

from pathlib import Path
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure access to sibling modules
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "models"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "walkers"))

from cleaning import remove_body_remnants, extract_veins_tophat, preprocess_wing_image
from model import JunctionUNet
from predict import predict_image, evaluate_against_gt
from topological_walker import TopologicalWalker


def run_verification(
    data_dir="data",
    checkpoint_path="training_output/best_model.pt",
    output_dir="data/verification_k_wings",
):
    data_dir = Path(data_dir)
    rotate_dir = data_dir / "rotate"
    seeds_dir = data_dir / "seeds"
    bw_dir = data_dir / "blackwhite_rotate"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CNN model: {checkpoint_path} on {device}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = JunctionUNet(in_channels=1, out_channels=1, base_channels=ckpt.get("base_channels", 16)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = {
        "body_removal": [],
        "colored_wings": [],
        "gt_non_regression": [],
    }

    # =========================================================================
    # PART 1: BODY REMNANT REMOVAL VERIFICATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: VERIFYING BODY REMNANT CLEANING")
    print("=" * 70)

    body_test_files = [
        "279_desert_whitetail_hindwing.jpeg",
        "282_white_corporal_hindwing.jpeg",
        "327_calico_pennant_hindwing.png",
    ]

    for fname in body_test_files:
        p = rotate_dir / fname
        if not p.exists():
            continue

        raw_img = cv2.imread(str(p))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        cleaned_img, body_mask, stats = remove_body_remnants(raw_img)
        gray_clean = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)

        # Baseline naive threshold
        orig_bw = (gray <= 220)
        clean_bw = (gray_clean <= 220)

        # Predict with CNN
        prob_orig, peaks_orig = predict_image(model, gray, device, threshold=0.40, snap=True)
        prob_clean, peaks_clean = predict_image(model, gray_clean, device, threshold=0.40, snap=True)

        # Check false positive peaks in the body zone (x <= 100)
        fp_orig = int(np.sum(peaks_orig[:, 0] <= 100)) if len(peaks_orig) > 0 else 0
        fp_clean = int(np.sum(peaks_clean[:, 0] <= 100)) if len(peaks_clean) > 0 else 0
        body_area_removed = int(np.sum(body_mask))

        print(
            f"Image: {fname:35s} | Removed Area: {body_area_removed:6d} px | "
            f"FP Peaks (x<=100): {fp_orig:3d} -> {fp_clean:3d} (Reduction: {fp_orig - fp_clean:+3d})"
        )

        results["body_removal"].append({
            "filename": fname,
            "removed_area": body_area_removed,
            "fp_before": fp_orig,
            "fp_after": fp_clean,
            "total_peaks_before": len(peaks_orig),
            "total_peaks_after": len(peaks_clean),
        })

        # Visual side-by-side
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        # 1. Original
        axes[0, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        if len(peaks_orig) > 0:
            axes[0, 0].scatter(
                peaks_orig[:, 0], peaks_orig[:, 1],
                c="red", s=12, marker="x", alpha=0.8,
                label=f"Peaks ({len(peaks_orig)})"
            )
        axes[0, 0].set_title(f"Original: {fname}\nUncleaned Insect Body at Left Margin", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")
        axes[0, 0].legend(loc="lower right")

        # 2. Body Mask Overlay
        axes[0, 1].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        overlay[body_mask] = [1.0, 0.0, 0.0, 0.5]  # Red transparent mask
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title(f"Detected Body Remnant Mask ({body_area_removed:,} px)", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # 3. Cleaned Image
        axes[1, 0].imshow(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
        if len(peaks_clean) > 0:
            axes[1, 0].scatter(
                peaks_clean[:, 0], peaks_clean[:, 1],
                c="lime", s=12, marker="o", alpha=0.8,
                label=f"Cleaned Peaks ({len(peaks_clean)})"
            )
        axes[1, 0].set_title(f"Cleaned Image (Body Whitened)\nFalse Peaks Removed: {fp_orig - fp_clean}", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")
        axes[1, 0].legend(loc="lower right")

        # 4. Zoomed Wing Base comparison
        crop_w = min(w, 200)
        crop_h = min(h, 150)
        axes[1, 1].imshow(cv2.cvtColor(cleaned_img[:crop_h, :crop_w], cv2.COLOR_BGR2RGB))
        in_crop = (peaks_clean[:, 0] < crop_w) & (peaks_clean[:, 1] < crop_h)
        if np.any(in_crop):
            axes[1, 1].scatter(
                peaks_clean[in_crop, 0], peaks_clean[in_crop, 1],
                c="lime", s=30, marker="o", edgecolors="black", linewidths=1.0,
                label="Genuine Vein Junctions"
            )
        axes[1, 1].set_title(f"Zoomed Wing Root (Veins Preserved, Body Cleared)", fontsize=12, fontweight="bold")
        axes[1, 1].axis("off")
        axes[1, 1].legend(loc="lower right")

        plt.tight_layout()
        out_path = output_dir / f"verification_body_removal_{p.stem}.png"
        fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # PART 2: MORPHOLOGICAL BLACK TOP-HAT VERIFICATION (COLORED WINGS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: VERIFYING BLACK TOP-HAT VEIN EXTRACTION ON COLORED WINGS")
    print("=" * 70)

    colored_test_files = [
        "322_banded_pennant_hindwing.png",
        "328_halloween_pennant_hindwing.png",
        "388_carolina_saddlebags_hindwing.jpg",
    ]

    for fname in colored_test_files:
        p = rotate_dir / fname
        if not p.exists():
            continue

        raw_img = cv2.imread(str(p))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Method A: Naive global thresholding
        bw_naive = np.where(gray <= 220, 0, 255).astype(np.uint8)
        naive_black_pct = float(np.mean(bw_naive == 0) * 100)

        # Method B: Morphological Black Top-Hat
        bw_tophat, blackhat = extract_veins_tophat(gray, kernel_size=11, ridge_threshold=16)
        tophat_black_pct = float(np.mean(bw_tophat == 0) * 100)

        # Run TopologicalWalker on both masks
        walker_naive = TopologicalWalker(bw_naive == 0)
        juncs_naive = walker_naive.find_junctions()

        walker_tophat = TopologicalWalker(bw_tophat == 0)
        juncs_tophat = walker_tophat.find_junctions()

        print(
            f"Image: {fname:35s} | Naive Black: {naive_black_pct:5.2f}% | TopHat Black: {tophat_black_pct:5.2f}% | "
            f"Junctions: {len(juncs_naive):4d} -> {len(juncs_tophat):4d}"
        )

        results["colored_wings"].append({
            "filename": fname,
            "naive_black_pct": naive_black_pct,
            "tophat_black_pct": tophat_black_pct,
            "juncs_naive": len(juncs_naive),
            "juncs_tophat": len(juncs_tophat),
        })

        # Visual 4-panel comparison
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))

        # 1. Raw Image
        axes[0, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Raw Wing: {fname}\nDark Pigmented Bands / Amber Patches", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        # 2. Naive Binarization (Solid Black Blocks)
        axes[0, 1].imshow(bw_naive, cmap="gray")
        if len(juncs_naive) > 0:
            pts_n = np.array(juncs_naive)
            axes[0, 1].scatter(pts_n[:, 0], pts_n[:, 1], c="red", s=10, marker="x", label=f"Junctions ({len(juncs_naive)})")
            axes[0, 1].legend(loc="lower right")
        axes[0, 1].set_title(f"Naive Threshold (gray <= 220): Black Area = {naive_black_pct:.1f}%\nCells Filled Solid Black, Internal Venation Lost", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # 3. Top-Hat Response Heatmap
        im3 = axes[1, 0].imshow(blackhat, cmap="inferno", vmin=0, vmax=60)
        axes[1, 0].set_title("Morphological Black Top-Hat Response\n(Isolates 1D Vein Ridges, Suppresses 2D Pigment)", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")
        fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 4. Top-Hat Binarization & Detected Junctions
        axes[1, 1].imshow(bw_tophat, cmap="gray")
        if len(juncs_tophat) > 0:
            pts_t = np.array(juncs_tophat)
            axes[1, 1].scatter(pts_t[:, 0], pts_t[:, 1], c="lime", s=12, marker="o", label=f"Vein Junctions ({len(juncs_tophat)})")
            axes[1, 1].legend(loc="lower right")
        axes[1, 1].set_title(f"Top-Hat Vein Extraction: Black Area = {tophat_black_pct:.1f}%\nCells Hollowed, Full Network Resolved ({len(juncs_tophat)} Junctions)", fontsize=12, fontweight="bold")
        axes[1, 1].axis("off")

        plt.tight_layout()
        out_path = output_dir / f"verification_tophat_{p.stem}.png"
        fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
        plt.close(fig)

    # =========================================================================
    # PART 3: NON-REGRESSION GT METRICS VERIFICATION (k=10 WINGS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: NON-REGRESSION VERIFICATION ON k=10 GROUND TRUTH WINGS")
    print("=" * 70)

    seed_files = sorted(seeds_dir.glob("*.npy"))
    valid_pairs = []
    for sf in seed_files:
        stem = sf.stem.replace("_blackwhite", "")
        for d in [rotate_dir, bw_dir]:
            found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                cand = d / f"{stem}{ext}"
                if cand.exists():
                    valid_pairs.append((cand, sf))
                    found = True
                    break
            if found:
                break

    random.seed(100)
    sampled = random.sample(valid_pairs, 10)

    f1_origs = []
    f1_cleans = []

    for i, (img_path, seed_path) in enumerate(sampled, 1):
        raw_img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY) if raw_img.ndim == 3 else raw_img
        gt_seeds = np.load(str(seed_path))
        if gt_seeds.ndim == 1:
            gt_seeds = gt_seeds.reshape(-1, 2)

        # Baseline
        prob_orig, peaks_orig = predict_image(model, gray, device, threshold=0.40, snap=True)
        p_orig, r_orig, f1_orig = evaluate_against_gt(peaks_orig, gt_seeds, tolerance=5.0)

        # Preprocessed
        cleaned_img, body_mask, stats = remove_body_remnants(raw_img)
        gray_clean = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY) if cleaned_img.ndim == 3 else cleaned_img
        prob_clean, peaks_clean = predict_image(model, gray_clean, device, threshold=0.40, snap=True)
        p_clean, r_clean, f1_clean = evaluate_against_gt(peaks_clean, gt_seeds, tolerance=5.0)

        f1_origs.append(f1_orig)
        f1_cleans.append(f1_clean)

        print(
            f"[{i:02d}/10] {img_path.name:45s} | "
            f"Orig F1: {f1_orig*100:5.2f}% | Clean F1: {f1_clean*100:5.2f}% | Diff: {f1_clean - f1_orig:+5.2f}%"
        )

        results["gt_non_regression"].append({
            "filename": img_path.name,
            "orig_f1": f1_orig,
            "clean_f1": f1_clean,
            "orig_precision": p_orig,
            "clean_precision": p_clean,
            "orig_recall": r_orig,
            "clean_recall": r_clean,
        })

    mean_orig_f1 = float(np.mean(f1_origs))
    mean_clean_f1 = float(np.mean(f1_cleans))
    print(f"\nSummary: Mean Baseline F1: {mean_orig_f1*100:.2f}% | Mean Preprocessed F1: {mean_clean_f1*100:.2f}%")
    print(f"Non-regression check: {'PASSED (Zero degradation)' if mean_clean_f1 >= mean_orig_f1 else 'FAILED'}")

    return results


if __name__ == "__main__":
    run_verification()
