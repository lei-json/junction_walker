#!/usr/bin/env python3
"""
Generate comprehensive evaluation plots and figures showing JunctionUNet
performance against Topological Walker ground truth across all cleaned wings.

Produces:
1. distribution_f1_accuracy.png: Histogram & KDE of F1 accuracy across all wings.
2. scatter_unet_vs_topological_counts.png: Scatter plot of UNet vs TopoGT junction counts with R^2 fit.
3. precision_vs_recall_scatter.png: Precision vs Recall distribution across the dataset.
4. comprehensive_performance_dashboard.png: 4-panel publication-ready dashboard.
5. visual_prediction_gallery.png: Visual ground truth vs predicted junction overlays on representative wings.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
import torch
import cv2

import sys
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src/models"))
sys.path.insert(0, str(repo_root / "src/walkers"))

from model import JunctionUNet
from predict import predict_image


DEFAULT_METRICS_CSV = Path("training_output/cleaned_wings_evaluation.csv")
DEFAULT_OUTPUT_DIR = Path("training_output/plots")


def generate_plots(metrics_csv=DEFAULT_METRICS_CSV, output_dir=DEFAULT_OUTPUT_DIR):
    repo_root = Path(__file__).resolve().parents[2]
    metrics_csv = repo_root / metrics_csv
    output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv)
    print(f"Loaded metrics for {len(df)} wings from {metrics_csv}")

    # Set styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 300

    # =========================================================================
    # FIGURE 1: ACCURACY / F1 SCORE DISTRIBUTION (HISTOGRAM)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    f1_pct = df["f1_accuracy"] * 100
    mean_f1 = f1_pct.mean()
    median_f1 = f1_pct.median()
    pct_passing = (f1_pct >= 85.0).mean() * 100

    n, bins, patches = ax.hist(
        f1_pct, bins=35, color="#1f77b4", edgecolor="black", alpha=0.85, label="Dragonfly Wings"
    )

    # Highlight passing vs failing bins
    for b_left, patch in zip(bins[:-1], patches):
        if b_left < 85.0:
            patch.set_facecolor("#d62728")  # Red for < 85%
            patch.set_alpha(0.65)
        else:
            patch.set_facecolor("#2ca02c")  # Green for >= 85%
            patch.set_alpha(0.85)

    ax.axvline(85.0, color="#d62728", linestyle="--", linewidth=2.2, label="Accuracy Target (85%)")
    ax.axvline(mean_f1, color="#1f77b4", linestyle="-", linewidth=2.5, label=f"Mean F1: {mean_f1:.2f}%")
    ax.axvline(median_f1, color="#ff7f0e", linestyle=":", linewidth=2.5, label=f"Median F1: {median_f1:.2f}%")

    ax.set_title("Distribution of Model Accuracy (F1 Score) Across Cleaned Wings\n(JunctionUNet vs. Topological Walker Ground Truth)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Agreement Accuracy (F1 Score %)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Dragonfly Wings", fontsize=12, fontweight="bold")
    ax.set_xlim(55, 101)

    # Stats text box
    stats_text = (
        f"Total Wings: {len(df)}\n"
        f"Passing (>=85%): {pct_passing:.1f}%\n"
        f"Mean Accuracy: {mean_f1:.2f}%\n"
        f"Median Accuracy: {median_f1:.2f}%\n"
        f"Std Dev: {f1_pct.std():.2f}%"
    )
    ax.text(
        0.05, 0.85, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#999999", alpha=0.9)
    )

    ax.legend(loc="upper right", frameon=True, fontsize=11)
    fig.tight_layout()
    p1 = output_dir / "distribution_f1_accuracy.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # =========================================================================
    # FIGURE 2: SCATTER PLOT - UNET VS TOPOLOGICAL WALKER JUNCTION COUNTS
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    x = df["topo_gt_count"]
    y = df["unet_count"]

    # Linear regression
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    r2 = r_val ** 2

    scatter = ax.scatter(
        x, y, c=df["f1_accuracy"] * 100, cmap="viridis",
        s=36, alpha=0.85, edgecolors="none", label="Cleaned Wings"
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Agreement F1 Accuracy (%)", fontsize=11, fontweight="bold")

    # Identity 1:1 line
    max_val = max(x.max(), y.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.8, label="Ideal 1:1 Agreement")
    ax.plot(
        np.array([0, max_val]), slope * np.array([0, max_val]) + intercept,
        color="#e41a1c", linewidth=2.0, label=f"Linear Fit ($R^2 = {r2:.4f}$, slope={slope:.3f})"
    )

    ax.set_title(f"Correlation Between UNet and Topological Walker Junction Counts\n({len(df)} Cleaned Wings)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Topological Walker Ground Truth Junction Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("UNet Predicted Junction Count", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(loc="upper left", frameon=True, fontsize=11)

    fig.tight_layout()
    p2 = output_dir / "scatter_unet_vs_topological_counts.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    # =========================================================================
    # FIGURE 3: PRECISION VS RECALL SCATTER
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    prec = df["precision"] * 100
    rec = df["recall"] * 100

    sc = ax.scatter(
        rec, prec, c=df["f1_accuracy"] * 100, cmap="plasma",
        s=40, alpha=0.85, edgecolors="black", linewidths=0.3
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1 Score (%)", fontsize=11, fontweight="bold")

    ax.axhline(85.0, color="#666666", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.axvline(85.0, color="#666666", linestyle=":", linewidth=1.5, alpha=0.7)

    # Highlight passing quadrant
    ax.axvspan(85, 102, ymin=(85 - 50) / (102 - 50), ymax=1.0, color="#2ca02c", alpha=0.1, label="Target Zone (P >= 85%, R >= 85%)")

    ax.set_title("Precision vs. Recall Across All Cleaned Dragonfly Wings", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (% Genuine Junctions Detected)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision (% Predicted Peaks Valid)", fontsize=12, fontweight="bold")
    ax.set_xlim(50, 102)
    ax.set_ylim(50, 102)
    ax.legend(loc="lower left", frameon=True, fontsize=11)

    fig.tight_layout()
    p3 = output_dir / "precision_vs_recall_scatter.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3}")

    # =========================================================================
    # FIGURE 4: COMPREHENSIVE 4-PANEL PUBLICATION DASHBOARD
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel A: Accuracy Histogram
    axes[0, 0].hist(f1_pct, bins=30, color="#2b5c8f", edgecolor="black", alpha=0.8)
    axes[0, 0].axvline(85.0, color="#d62728", linestyle="--", linewidth=2.0, label="85% Target")
    axes[0, 0].axvline(mean_f1, color="#e6550d", linestyle="-", linewidth=2.2, label=f"Mean: {mean_f1:.1f}%")
    axes[0, 0].set_title("(A) Accuracy (F1) Distribution", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("F1 Score (%)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Wing Count", fontsize=11, fontweight="bold")
    axes[0, 0].legend(loc="upper left")

    # Panel B: Scatter Counts with 1:1 line
    axes[0, 1].scatter(x, y, c="#2ca02c", s=25, alpha=0.75, edgecolors="none")
    axes[0, 1].plot([0, max_val], [0, max_val], "k--", linewidth=1.5, label="1:1 Perfect Agreement")
    axes[0, 1].set_title(f"(B) UNet vs TopoGT Junction Counts ($R^2 = {r2:.4f}$)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Topological Walker Ground Truth Count", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("UNet Predicted Count", fontsize=11, fontweight="bold")
    axes[0, 1].legend(loc="upper left")

    # Panel C: Precision vs Recall
    axes[1, 0].scatter(rec, prec, c="#756bb1", s=30, alpha=0.8, edgecolors="none")
    axes[1, 0].axhline(85.0, color="#e41a1c", linestyle=":", linewidth=1.5)
    axes[1, 0].axvline(85.0, color="#e41a1c", linestyle=":", linewidth=1.5)
    axes[1, 0].set_title("(C) Precision vs. Recall", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Recall (%)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Precision (%)", fontsize=11, fontweight="bold")

    # Panel D: Inference Runtime vs Image Pixels (Area)
    pixel_k = (df["width"] * df["height"]) / 1000.0
    axes[1, 1].scatter(pixel_k, df["unet_time_sec"], c="#31a354", s=30, alpha=0.8, edgecolors="none")
    axes[1, 1].set_title("(D) UNet CPU Inference Runtime vs. Wing Resolution", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Image Resolution (Thousands of Pixels)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Inference Runtime (Seconds)", fontsize=11, fontweight="bold")

    plt.suptitle("Dragonfly Wing Venation Benchmark: JunctionUNet vs. Topological Walker Ground Truth", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout()
    p4 = output_dir / "comprehensive_performance_dashboard.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p4}")

    return {
        "mean_f1": mean_f1,
        "median_f1": median_f1,
        "pct_passing": pct_passing,
        "r2": r2,
    }


def generate_visual_showcase(output_dir=DEFAULT_OUTPUT_DIR):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / output_dir

    device = torch.device("cpu")
    ckpt_path = repo_root / "training_output/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = JunctionUNet(1, 1, ckpt.get("base_channels", 16)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_wings = [
        "100_stillwater_clubtail_forewing.png",
        "147_cherokee_clubtail_forewing.png",
        "234_texas_emerald_forewing.png",
        "44_blue_faced_darner_hindwing.png",
    ]

    fig, axes = plt.subplots(len(sample_wings), 2, figsize=(18, 4 * len(sample_wings)))

    for idx, fname in enumerate(sample_wings):
        img_p = repo_root / "data/cleaned_wings" / fname
        seed_p = repo_root / "data/cleaned_seeds" / fname.replace(".png", ".npy")

        if not img_p.exists() or not seed_p.exists():
            continue

        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        gt = np.load(str(seed_p))
        prob_map, peaks = predict_image(model, img, device, threshold=0.40, snap=True, filter_degree2=True)

        # Panel 1: Full wing overlay
        axes[idx, 0].imshow(img, cmap="gray")
        axes[idx, 0].scatter(gt[:, 0], gt[:, 1], c="red", s=12, marker="x", alpha=0.7, label=f"TopoGT ({len(gt)})")
        axes[idx, 0].scatter(peaks[:, 0], peaks[:, 1], c="lime", s=8, marker="o", alpha=0.7, label=f"UNet Pred ({len(peaks)})")
        axes[idx, 0].set_title(f"{fname} (Full Wing: {img.shape[1]}x{img.shape[0]})", fontsize=12, fontweight="bold")
        axes[idx, 0].axis("off")
        axes[idx, 0].legend(loc="lower right", frameon=True)

        # Panel 2: Zoomed central venation network
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        hw, hh = min(150, img.shape[1] // 4), min(100, img.shape[0] // 4)
        crop = img[cy - hh:cy + hh, cx - hw:cx + hw]

        gt_in = (gt[:, 0] >= cx - hw) & (gt[:, 0] < cx + hw) & (gt[:, 1] >= cy - hh) & (gt[:, 1] < cy + hh)
        p_in = (peaks[:, 0] >= cx - hw) & (peaks[:, 0] < cx + hw) & (peaks[:, 1] >= cy - hh) & (peaks[:, 1] < cy + hh)

        axes[idx, 1].imshow(crop, cmap="gray")
        axes[idx, 1].scatter(gt[gt_in, 0] - (cx - hw), gt[gt_in, 1] - (cy - hh), c="red", s=36, marker="x", linewidths=1.5, label="Topo Walker GT")
        axes[idx, 1].scatter(peaks[p_in, 0] - (cx - hw), peaks[p_in, 1] - (cy - hh), c="lime", s=28, marker="o", facecolors="none", edgecolors="lime", linewidths=1.5, label="UNet Predicted")
        axes[idx, 1].set_title(f"Zoomed Venation Details: {fname}", fontsize=12, fontweight="bold")
        axes[idx, 1].axis("off")
        axes[idx, 1].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    gallery_p = output_dir / "visual_prediction_gallery.png"
    fig.savefig(gallery_p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {gallery_p}")


if __name__ == "__main__":
    generate_plots()
    generate_visual_showcase()
