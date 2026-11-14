#!/usr/bin/env python3
"""
Publication Figure Generator for Unified Multi-Dataset Junction Benchmark.

Generates:
1. 5-Dataset Visual Comparison Panel:
   Abstract Geometric Voronoi, Realistic Synthetic Dragonfly, Real PNAS Dragonfly,
   DRIVE Retinal Vasculature, HALVS Leaf Venation.
   Shows raw input, ground-truth junctions, and UNet detections with matched localization.
2. Comparative Performance Charts:
   F1-score across all 5 benchmark dataset tiers and dual noise robustness curves.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch

import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src/walkers"))
sys.path.insert(0, str(repo_root / "src/models"))
sys.path.insert(0, str(repo_root / "src/preprocessing"))
sys.path.insert(0, str(repo_root / "src/benchmark"))

from model import JunctionUNet
from synthetic_ground_truth_benchmark import generate_synthetic_wing_venation
from dragonfly_synthetic_generator import generate_realistic_dragonfly_wing
from external_datasets_loader import (
    load_pnas_dragonfly_benchmark,
    load_halvs_leaf_benchmark,
    load_drive_retina_benchmark,
)
from unified_junction_benchmark import predict_unet_tiled


def plot_visual_4panel_comparison(
    checkpoint_path: str = "training_output/best_model.pt",
    output_path: str = "training_output/empirical/plots/unified_benchmark_4panel_visual.png",
):
    out_file = repo_root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_p = repo_root / checkpoint_path
    ckpt = torch.load(ckpt_p, map_location=device)
    base_ch = ckpt.get("base_channels", 16)
    unet = JunctionUNet(1, 1, base_ch).to(device)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    # 1. Abstract Geometric Voronoi
    v_full, v_gt_full = generate_synthetic_wing_venation(
        width=800, height=450, n_seeds=130, noise_sigma=2.0, seed=42
    )
    v_crop = v_full[100:300, 250:550]
    v_valid = (v_gt_full[:, 0] >= 250) & (v_gt_full[:, 0] < 550) & (v_gt_full[:, 1] >= 100) & (v_gt_full[:, 1] < 300)
    v_gt = v_gt_full[v_valid] - np.array([250, 100], dtype=np.float32)

    # 2. Realistic Synthetic Dragonfly
    synth_img, synth_gt, _ = generate_realistic_dragonfly_wing(
        width=800, height=360, wing_type="hindwing", noise_sigma=2.0, seed=123
    )
    s_crop = synth_img[100:300, 300:600]
    s_valid = (synth_gt[:, 0] >= 300) & (synth_gt[:, 0] < 600) & (synth_gt[:, 1] >= 100) & (synth_gt[:, 1] < 300)
    s_gt = synth_gt[s_valid] - np.array([300, 100], dtype=np.float32)

    # 3. PNAS Real Dragonfly
    p_crop, p_gt, _ = load_pnas_dragonfly_benchmark(crop_box=(300, 500, 2200, 2500))

    # 4. DRIVE Retina
    d_full, d_gt, _ = load_drive_retina_benchmark(index=1)
    d_crop = d_full[180:380, 180:480]
    d_valid = (d_gt[:, 0] >= 180) & (d_gt[:, 0] < 480) & (d_gt[:, 1] >= 180) & (d_gt[:, 1] < 380)
    d_gt_crop = d_gt[d_valid] - np.array([180, 180], dtype=np.float32)

    # 5. HALVS Leaf
    l_crop, l_gt, _ = load_halvs_leaf_benchmark(crop_box=(700, 900, 500, 800))

    datasets = [
        ("Abstract Geometric Voronoi", v_crop, v_gt),
        ("Realistic Synthetic Dragonfly", s_crop, s_gt),
        ("Real Dragonfly (PNAS 2018)", p_crop, p_gt),
        ("Retinal Vasculature (DRIVE)", d_crop, d_gt_crop),
        ("Leaf Venation (HALVS)", l_crop, l_gt),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(14, 18.5))
    tol = 5.0

    for row_idx, (title, img, gt) in enumerate(datasets):
        preds = predict_unet_tiled(unet, img, device, threshold=0.35)

        # Matched evaluation
        tp_pred = np.zeros(len(preds), dtype=bool)
        matched_gt = np.zeros(len(gt), dtype=bool)

        if len(preds) > 0 and len(gt) > 0:
            tree_gt = cKDTree(gt)
            dists, idxs = tree_gt.query(preds)
            tp_pred = dists <= tol
            for i, (d, idx) in enumerate(zip(dists, idxs)):
                if d <= tol:
                    matched_gt[idx] = True

        p = float(np.sum(tp_pred)) / max(1, len(preds))
        r = float(np.sum(matched_gt)) / max(1, len(gt))
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

        # Column 0: Raw Grayscale Image
        axes[row_idx, 0].imshow(img, cmap="gray")
        axes[row_idx, 0].set_title(f"{title}\n[Input Specimen]", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Column 1: Ground Truth Overlay
        axes[row_idx, 1].imshow(img, cmap="gray")
        if len(gt) > 0:
            axes[row_idx, 1].scatter(gt[:, 0], gt[:, 1], c="#00E676", s=28, edgecolors="black", linewidths=0.7, label="Ground Truth")
        axes[row_idx, 1].set_title(f"Ground Truth Junctions\n(N = {len(gt)})", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")
        if row_idx == 0:
            axes[row_idx, 1].legend(loc="lower right", fontsize=8.5, framealpha=0.85)

        # Column 2: UNet Prediction Overlay
        axes[row_idx, 2].imshow(img, cmap="gray")
        if len(preds) > 0:
            tp_pts = preds[tp_pred]
            fp_pts = preds[~tp_pred]
            if len(tp_pts) > 0:
                axes[row_idx, 2].scatter(tp_pts[:, 0], tp_pts[:, 1], c="#0055FF", s=30, marker="o", edgecolors="black", linewidths=0.8, label="True Positive (<=5px)")
            if len(fp_pts) > 0:
                axes[row_idx, 2].scatter(fp_pts[:, 0], fp_pts[:, 1], c="#FF1744", s=26, marker="x", linewidths=1.3, label="False Positive")

        axes[row_idx, 2].set_title(f"JunctionUNet Detection\nF1@5px = {f1*100:.1f}% | P = {p*100:.1f}%, R = {r*100:.1f}%", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")
        if row_idx == 0:
            axes[row_idx, 2].legend(loc="lower right", fontsize=8.5, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    # Also save with 5panel name
    p5_out = out_file.parent / "unified_benchmark_5panel_visual.png"
    plt.savefig(p5_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visual comparison panel to: {out_file} and {p5_out}")


def plot_summary_charts(
    summary_csv: str = "training_output/empirical/unified_benchmark_summary.csv",
    output_path: str = "training_output/empirical/plots/unified_benchmark_performance_comparison.png",
):
    sum_p = repo_root / summary_csv
    if not sum_p.exists():
        print(f"Summary CSV not found at {sum_p}, skipping plot.")
        return

    df = pd.read_csv(sum_p)
    out_file = repo_root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.0, 1.0])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # 1. Bar Chart: F1@5px across distinct datasets
    target_conditions = [
        ("Abstract Voronoi\n(Clean σ=2)", "Abstract Geometric Voronoi", "sigma_2"),
        ("Abstract Voronoi\n(Noisy σ=20)", "Abstract Geometric Voronoi", "sigma_20"),
        ("Realistic Synth\n(Clean σ=2)", "Realistic Synthetic Dragonfly", "sigma_2"),
        ("Realistic Synth\n(Noisy σ=20)", "Realistic Synthetic Dragonfly", "sigma_20"),
        ("Real Dragonfly\n(PNAS 2018)", "PNAS Dragonfly Wing", "real_specimen"),
        ("Retinal Vessels\n(DRIVE)", "DRIVE Retinal Vasculature", "clinical_fundus"),
        ("Leaf Venation\n(HALVS)", "HALVS Leaf Venation", "real_cleared_leaf"),
    ]

    methods_order = [
        "JunctionUNet (Raw)",
        "TopologicalWalker",
        "Classical Skeleton",
        "Shi-Tomasi",
        "Harris Corner",
        "FAST",
        "SIFT",
    ]

    bar_data = []
    labels = []
    for label, ds, cond in target_conditions:
        sub = df[(df["dataset"] == ds) & (df["condition"] == cond)]
        if len(sub) > 0:
            labels.append(label)
            m_scores = {}
            for _, row in sub.iterrows():
                m_scores[row["method"]] = row["f1_5px"] * 100.0
            bar_data.append(m_scores)

    x = np.arange(len(labels))
    width = 0.11

    colors = ["#2962FF", "#00C853", "#FF6D00", "#AA00FF", "#00B0FF", "#D50000", "#78909C"]

    for i, m_name in enumerate(methods_order):
        vals = [b.get(m_name, 0.0) for b in bar_data]
        offset = (i - len(methods_order) / 2) * width + width / 2
        rects = ax1.bar(x + offset, vals, width, label=m_name, color=colors[i % len(colors)], alpha=0.9, edgecolor="black", linewidth=0.5)

    ax1.set_ylabel("F1-Score at 5px Tolerance (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Cross-Dataset Junction F1-Score Comparison", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # 2. Noise Robustness on Abstract Geometric Voronoi
    v_sub = df[df["dataset"] == "Abstract Geometric Voronoi"]
    noise_sigmas = [2.0, 10.0, 20.0]

    for i, m_name in enumerate(methods_order):
        m_rows = v_sub[v_sub["method"] == m_name]
        f1_vals = []
        for c in ["sigma_2", "sigma_10", "sigma_20"]:
            val = m_rows[m_rows["condition"] == c]["f1_5px"].values
            f1_vals.append(val[0] * 100.0 if len(val) > 0 else 0.0)

        marker = "o" if "UNet" in m_name else ("s" if "Topological" in m_name else "^")
        lw = 2.5 if "UNet" in m_name or "Topological" in m_name else 1.5
        ax2.plot(noise_sigmas, f1_vals, label=m_name, marker=marker, linewidth=lw, color=colors[i % len(colors)], markersize=5.5)

    ax2.set_xlabel("Noise Std Dev (σ)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("F1 @ 5px (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Abstract Geometric Voronoi Noise Response", fontsize=12, fontweight="bold")
    ax2.set_xticks(noise_sigmas)
    ax2.set_xticklabels(["σ = 2", "σ = 10", "σ = 20"], fontsize=9.5)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Noise Robustness on Realistic Synthetic Dragonfly Wing
    synth_sub = df[df["dataset"] == "Realistic Synthetic Dragonfly"]

    for i, m_name in enumerate(methods_order):
        m_rows = synth_sub[synth_sub["method"] == m_name]
        f1_vals = []
        for c in ["sigma_2", "sigma_10", "sigma_20"]:
            val = m_rows[m_rows["condition"] == c]["f1_5px"].values
            f1_vals.append(val[0] * 100.0 if len(val) > 0 else 0.0)

        marker = "o" if "UNet" in m_name else ("s" if "Topological" in m_name else "^")
        lw = 2.5 if "UNet" in m_name or "Topological" in m_name else 1.5
        ax3.plot(noise_sigmas, f1_vals, label=m_name, marker=marker, linewidth=lw, color=colors[i % len(colors)], markersize=5.5)

    ax3.set_xlabel("Noise Std Dev (σ)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("F1 @ 5px (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Realistic Dragonfly Synthetic Noise Response", fontsize=12, fontweight="bold")
    ax3.set_xticks(noise_sigmas)
    ax3.set_xticklabels(["σ = 2", "σ = 10", "σ = 20"], fontsize=9.5)
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved performance comparison charts to: {out_file}")


if __name__ == "__main__":
    plot_visual_4panel_comparison()
    plot_summary_charts()
