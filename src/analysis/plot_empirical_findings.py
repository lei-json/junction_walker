#!/usr/bin/env python3
"""
Publication Figure Generator for Empirical Mitigation and Hypothesis Testing.

Generates:
1. baseline_comparison_dashboard.png
2. tolerance_response_curves.png
3. synthetic_ground_truth_validation.png
4. biological_hypotheses_validation.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
output_plots_dir = repo_root / "training_output/empirical/plots"
output_plots_dir.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})


def plot_baselines_dashboard(csv_path: Path):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)

    # Average metrics per method
    summary = df.groupby("method").agg({
        "f1_3px": "mean",
        "f1_5px": "mean",
        "f1_8px": "mean",
        "p_8px": "mean",
        "r_8px": "mean",
        "mean_loc_error_px": "mean",
        "latency_ms": "mean",
    }).loc[[
        "Harris Corner",
        "Shi-Tomasi",
        "FAST",
        "SIFT",
        "Classical Skeleton",
        "TopologicalWalker",
        "JunctionUNet (Raw)",
    ]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: F1 Comparison across tolerances (3px, 5px, 8px)
    methods = summary.index.tolist()
    x = np.arange(len(methods))
    width = 0.25

    ax = axes[0, 0]
    ax.bar(x - width, summary["f1_3px"] * 100, width, label="τ = 3.0 px", color="#4575b4")
    ax.bar(x, summary["f1_5px"] * 100, width, label="τ = 5.0 px", color="#74add1")
    ax.bar(x + width, summary["f1_8px"] * 100, width, label="τ = 8.0 px", color="#fdae61")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("(A) Keypoint Detection Accuracy Across Matching Tolerances")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")

    # Panel B: Precision vs Recall at 8px
    ax = axes[0, 1]
    colors = ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4", "#313695"]
    for i, m in enumerate(methods):
        p = summary.loc[m, "p_8px"] * 100
        r = summary.loc[m, "r_8px"] * 100
        ax.scatter(r, p, s=180, color=colors[i], label=m, edgecolors="black", zorder=4)
        ax.annotate(m, (r + 1.2, p - 1.0), fontsize=9)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("(B) Precision vs. Recall (τ = 8.0 px)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    # Panel C: Mean Localization Error (px)
    ax = axes[1, 0]
    bars = ax.bar(methods, summary["mean_loc_error_px"], color="#2ca02c", alpha=0.85, edgecolor="black")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="1.0 px Sub-Pixel Threshold")
    ax.set_ylabel("Mean Euclidean Error (pixels)")
    ax.set_title("(C) Spatial Localization Accuracy on True Positive Junctions")
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.legend(loc="upper right")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f} px", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel D: Inference Latency (Log Scale)
    ax = axes[1, 1]
    latencies = summary["latency_ms"]
    ax.bar(methods, latencies, color="#9467bd", alpha=0.85, edgecolor="black")
    ax.set_ylabel("Latency per Wing (ms, Log Scale)")
    ax.set_yscale("log")
    ax.set_title("(D) Computational Efficiency & Throughput")
    ax.set_xticklabels(methods, rotation=25, ha="right")
    for i, lat in enumerate(latencies):
        ax.annotate(f"{lat:.1f} ms", xy=(i, lat), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Benchmark: Classical & Deep Keypoint Baselines on Held-Out Test Wings (N=69)", y=0.99)
    plt.tight_layout()
    out_file = output_plots_dir / "baseline_comparison_dashboard.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_tolerance_curves(csv_path: Path):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)

    tolerances = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: F1 vs Tolerance
    ax = axes[0]
    for model_name, color in [("TopologicalWalker", "#1f77b4"), ("JunctionUNet (Raw)", "#ff7f0e")]:
        sub = df[df["model"] == model_name]
        f1_means = [sub[f"f1_tau_{t:.1f}"].mean() * 100 for t in tolerances]
        p_means = [sub[f"p_tau_{t:.1f}"].mean() * 100 for t in tolerances]
        r_means = [sub[f"r_tau_{t:.1f}"].mean() * 100 for t in tolerances]

        ax.plot(tolerances, f1_means, "o-", label=f"{model_name} (F1)", color=color, linewidth=2.5)
        ax.plot(tolerances, p_means, "--", label=f"{model_name} (Precision)", color=color, alpha=0.5)
        ax.plot(tolerances, r_means, ":", label=f"{model_name} (Recall)", color=color, alpha=0.5)

    ax.axvline(8.0, color="red", linestyle=":", linewidth=1.5, label="Legacy 8 px Threshold")
    ax.axvline(3.0, color="green", linestyle=":", linewidth=1.5, label="Strict 3 px Threshold")
    ax.set_xlabel("Matching Tolerance Radius τ (pixels)")
    ax.set_ylabel("Metric Score (%)")
    ax.set_title("(A) Continuous Metric Response vs. Matching Radius")
    ax.set_xlim(1.0, 15.0)
    ax.set_ylim(40, 102)
    ax.legend(loc="lower right", fontsize=8)

    # Panel B: Localization Error Cumulative Distribution
    ax = axes[1]
    for model_name, color in [("TopologicalWalker", "#1f77b4"), ("JunctionUNet (Raw)", "#ff7f0e")]:
        sub = df[df["model"] == model_name]
        p1 = sub["pct_within_1px"].mean()
        p2 = sub["pct_within_2px"].mean()
        p3 = sub["pct_within_3px"].mean()
        p5 = sub["f1_tau_5.0"].mean() * 100

        radii = [1.0, 2.0, 3.0, 5.0, 8.0]
        cum_pct = [p1, p2, p3, p5, sub["f1_tau_8.0"].mean() * 100]

        ax.plot(radii, cum_pct, "s-", label=f"{model_name} (Median Err: {sub['loc_error_median'].mean():.2f} px)", color=color, linewidth=2.5)

    ax.set_xlabel("Error Distance from Ground Truth (pixels)")
    ax.set_ylabel("Cumulative Matched Landmarks (%)")
    ax.set_title("(B) Sub-Vein Spatial Precision Distribution")
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")

    plt.suptitle("Tolerance Sensitivity & Sub-Vein Localization Precision Analysis", y=0.98)
    plt.tight_layout()
    out_file = output_plots_dir / "tolerance_response_curves.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_biological_hypotheses(json_path: Path, features_csv: Path):
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return
    with open(json_path) as f:
        data = json.load(f)

    df = pd.read_csv(features_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Lewis's Law Linear Regression
    ax = axes[0, 0]
    lewis = data["lewis_law"]
    v_tab = pd.DataFrame(lewis["valence_table"])
    ax.scatter(v_tab["sides"], v_tab["norm_mean_area"], s=120, color="#1f77b4", edgecolor="black", zorder=4, label="Empirical Cell Areas")

    # Fitted line
    x_fit = np.linspace(3, 8, 100)
    y_fit = lewis["slope"] * (x_fit - 6) + lewis["intercept"]
    ax.plot(x_fit, y_fit, "r--", linewidth=2.0, label=f"Fit: Ā_n / Ā = {lewis['slope']:.2f}(n-6) + {lewis['intercept']:.2f} (R² = {lewis['r_squared']:.3f})")

    ax.set_xlabel("Cell Polygonal Sides (n)")
    ax.set_ylabel("Normalized Mean Cell Area (Ā_n / Ā)")
    ax.set_title(f"(A) Lewis's Law: Area Scaling vs. Side Count (p = {lewis['p_val']:.2e})")
    ax.set_xticks(range(3, 9))
    ax.legend(loc="upper left")

    # Panel B: Shear Truss Valency (Degree 3 vs Degree 4)
    ax = axes[0, 1]
    deg_means = [data["shear_truss_valency"]["mean_deg3_pct"], data["shear_truss_valency"]["mean_deg4_pct"]]
    deg_stds = [data["shear_truss_valency"]["std_deg3_pct"], data["shear_truss_valency"]["std_deg4_pct"]]
    bars = ax.bar(["Degree-3 (Trivalent Y-Nodes)", "Degree-4 (Quadrivalent X-Nodes)"], deg_means, yerr=deg_stds, capsize=6,
                  color=["#2ca02c", "#d62728"], alpha=0.85, edgecolor="black")
    ax.set_ylabel("Proportion of Total Junctions (%)")
    ax.set_title(f"(B) Shear Truss Hypothesis: Node Valency Avoidance (t = {data['shear_truss_valency']['t_stat']:.1f}, p < 1e-15)")
    ax.set_ylim(0, 105)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h + 3),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Panel C: Forewing vs Hindwing Compactness Dimorphism (Paired)
    ax = axes[1, 0]
    fore_comp = df[df["wing_type"] == "forewing"]["wing_compactness_mec"]
    hind_comp = df[df["wing_type"] == "hindwing"]["wing_compactness_mec"]
    bp = ax.boxplot([fore_comp.dropna(), hind_comp.dropna()], labels=["Forewing", "Hindwing"],
                    patch_artist=True, widths=0.5)
    colors = ["#3288bd", "#d53e4f"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    dimorph = data["aerodynamic_dimorphism"]["wing_compactness_mec"]
    ax.set_ylabel("Wing Compactness (A_hull / A_mec)")
    ax.set_title(f"(C) Aerodynamic Dimorphism: Wing Compactness (t = {dimorph['t_stat']:.1f}, d = {dimorph['cohens_d']:.2f})")

    # Panel D: Phylogenetic Family Differences in Aspect Ratio
    ax = axes[1, 1]
    def parse_family(s):
        s = str(s).lower()
        if any(k in s for k in ['clubtail', 'snaketail', 'sanddragon', 'forceptail', 'spinylegs', 'dragonhunter']): return 'Gomphidae'
        elif any(k in s for k in ['darner']): return 'Aeshnidae'
        elif any(k in s for k in ['skimmer', 'glider', 'pennant', 'whiteface', 'dragonlet']): return 'Libellulidae'
        elif any(k in s for k in ['emerald', 'baskettail']): return 'Corduliidae'
        return 'Other'

    df["fam"] = df["species"].apply(parse_family)
    fams = ["Gomphidae", "Aeshnidae", "Libellulidae", "Corduliidae"]
    fam_data = [df[df["fam"] == f]["aspect_ratio"].dropna() for f in fams]

    bp2 = ax.boxplot(fam_data, labels=fams, patch_artist=True, widths=0.5)
    fam_colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
    for patch, c in zip(bp2["boxes"], fam_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    phylo = data["phylogenetic_morphometrics"]["aspect_ratio"]
    ax.set_ylabel("Wing Aspect Ratio (W / H)")
    ax.set_title(f"(D) Family-Level Morphometrics: Aspect Ratio (ANOVA F = {phylo['f_stat']:.1f}, η² = {phylo['eta_squared']:.2f})")

    plt.suptitle("Empirical Testing of Biological Hypotheses in Dragonfly Venation", y=0.99)
    plt.tight_layout()
    out_file = output_plots_dir / "biological_hypotheses_validation.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_synthetic_benchmark(csv_path: Path):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    methods = ["TopologicalWalker", "JunctionUNet (Raw)", "Classical Skeleton", "Shi-Tomasi", "FAST"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    noise_levels = sorted(df["noise_sigma"].unique())

    for m, c in zip(methods, colors):
        sub = df[df["method"] == m]
        means = [sub[sub["noise_sigma"] == n]["f1_5px"].mean() * 100 for n in noise_levels]
        ax.plot(noise_levels, means, "o-", label=m, color=c, linewidth=2.5)

    ax.set_xlabel("Additive Sensor Noise Standard Deviation (σ)")
    ax.set_ylabel("F1 Score at τ = 5.0 px (%)")
    ax.set_title("Independent Analytical Benchmark: Robustness to Noise against Exact Mathematical Ground Truth")
    ax.set_xticks(noise_levels)
    ax.set_xticklabels([f"Clean (σ={n})" if n==2 else f"Noisy (σ={n})" for n in noise_levels])
    ax.set_ylim(20, 102)
    ax.legend(loc="lower left")

    plt.tight_layout()
    out_file = output_plots_dir / "synthetic_ground_truth_validation.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_pigmented_stress_test(csv_path: Path):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Pigment Coverage vs False Positive Rate
    ax = axes[0]
    ax.scatter(df["dark_fraction_pct"], df["tw_pigment_fp_rate_pct"], color="#1f77b4", alpha=0.75, s=60, label="TopologicalWalker")
    ax.scatter(df["dark_fraction_pct"], df["unet_pigment_fp_rate_pct"], color="#ff7f0e", alpha=0.75, s=60, marker="^", label="JunctionUNet (Raw)")

    # Trend line for UNet
    m, b = np.polyfit(df["dark_fraction_pct"], df["unet_pigment_fp_rate_pct"], 1)
    x_line = np.linspace(0, df["dark_fraction_pct"].max(), 50)
    ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, label=f"Linear Trend (Slope = +{m:.2f} %/%)")

    ax.set_xlabel("Wing Membrane Pigment Coverage (%)")
    ax.set_ylabel("False Positive Rate in Pigment (%)")
    ax.set_title("(A) False Landmark Triggering vs. Membrane Pigmentation")
    ax.set_ylim(-2, 55)
    ax.legend(loc="upper left")

    # Panel B: False Positive Distribution Comparison
    ax = axes[1]
    bp = ax.boxplot([df["tw_pigment_fp_rate_pct"], df["unet_pigment_fp_rate_pct"]],
                    labels=["TopologicalWalker", "JunctionUNet"], patch_artist=True, widths=0.45)
    bp["boxes"][0].set_facecolor("#1f77b4")
    bp["boxes"][1].set_facecolor("#ff7f0e")
    for patch in bp["boxes"]:
        patch.set_alpha(0.7)

    ax.set_ylabel("False Positive Rate in Pigment Patches (%)")
    ax.set_title("(B) Pigment Artifact Rate Across Excluded Specimens (N=66)")
    ax.set_ylim(-2, 55)

    plt.suptitle("Empirical Stress Test on 66 Excluded Pigmented / Mottled Wings", y=0.98)
    plt.tight_layout()
    out_file = output_plots_dir / "pigmented_wings_stress_test.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_baselines_dashboard(repo_root / "training_output/baselines_benchmark.csv")
    plot_tolerance_curves(repo_root / "training_output/empirical/tolerance_sweep_results.csv")
    plot_biological_hypotheses(
        repo_root / "training_output/empirical/biological_hypothesis_test_results.json",
        repo_root / "data/wing_features.csv",
    )
    plot_synthetic_benchmark(repo_root / "training_output/empirical/synthetic_ground_truth_benchmark.csv")
    plot_pigmented_stress_test(repo_root / "training_output/empirical/pigmented_wings_stress_test.csv")
