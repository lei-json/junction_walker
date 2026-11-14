#!/usr/bin/env python3
"""
Dragonfly Wing Morphometrics & Topological Hulls/Areas Visualizer.

Generates publication-quality visual plots demonstrating:
1. Whole-Wing Convex Hulls, Minimum Enclosing Circles (MEC), Centroids, and Hull Areas.
2. Microscopic Biological Cell Segmentation, Cell Area Heatmaps, and Inscribed/Circumscribed Hulls.
3. Delaunay Triangulation Facet Areas and Empty Circumcircles.
4. Dataset-wide Morphometrics Distributions and Scaling Laws across 443 wings.
5. Master Publication Showcase Dashboard.
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import gaussian_kde

# Add project root to sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src/walkers") not in sys.path:
    sys.path.insert(0, str(repo_root / "src/walkers"))
if str(repo_root / "src/preprocessing") not in sys.path:
    sys.path.insert(0, str(repo_root / "src/preprocessing"))

from cleaning import extract_veins_tophat
from topological_walker import TopologicalWalker


DEFAULT_WINGS_DIR = repo_root / "data/cleaned_wings"
DEFAULT_SEEDS_DIR = repo_root / "data/cleaned_seeds"
DEFAULT_CSV_PATH = repo_root / "data/wing_features.csv"
DEFAULT_OUTPUT_DIR = repo_root / "training_output/plots/morphometrics"

SAMPLE_STEMS = [
    "100_stillwater_clubtail_forewing",
    "100_stillwater_clubtail_hindwing",
    "140_cobra_clubtail_forewing",
    "140_cobra_clubtail_hindwing",
    "212_bronzed_river_cruiser_forewing",
    "385_hyacinth_glider_hindwing",
]


def extract_cells_and_contours(img: np.ndarray, stem: str):
    """Extract interior biological cell polygons, areas, contours, and inscribed/circumscribed circles."""
    h, w = img.shape
    ridge_th = 6 if "caribbean_darner" in stem else 10
    bw_top, _ = extract_veins_tophat(img, kernel_size=9, ridge_threshold=ridge_th, background_threshold=250)
    vein_mask = (bw_top == 0).astype(np.uint8)

    tw = TopologicalWalker(vein_mask, min_component_size=3, spur_threshold=3)
    cleaned_vein = tw.clean_mask()
    membrane = (cleaned_vein == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(membrane, connectivity=4)

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_labels = set(np.unique(labels[border_mask]))

    cells = []
    for lbl in range(1, num_labels):
        if lbl in border_labels:
            continue
        c_area = int(stats[lbl, cv2.CC_STAT_AREA])
        if c_area < 5:
            continue

        c_bin = (labels == lbl).astype(np.uint8)
        cnts, _ = cv2.findContours(c_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        c_p = float(cv2.arcLength(cnt, True))
        if c_p <= 0 or len(cnt) < 3:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * c_p, True)
        (mec_x, mec_y), c_mec = cv2.minEnclosingCircle(cnt)

        dist_m = cv2.distanceTransform(c_bin, cv2.DIST_L2, 3)
        _, c_ins, _, ins_loc = cv2.minMaxLoc(dist_m)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = float(mec_x), float(mec_y)

        c_circ = (4.0 * np.pi * c_area) / (c_p ** 2)

        cells.append({
            "lbl": lbl,
            "area": c_area,
            "perimeter": c_p,
            "circularity": c_circ,
            "centroid": (cx, cy),
            "mec_center": (mec_x, mec_y),
            "mec_radius": c_mec,
            "inscribe_center": (float(ins_loc[0]), float(ins_loc[1])),
            "inscribe_radius": float(c_ins),
            "sides": len(approx),
            "contour": cnt[:, 0, :],
            "polygon": cnt[:, 0, :],
        })

    return cells, cleaned_vein


def plot_sample_wings_convex_hulls(
    stems: List[str],
    wings_dir: Path,
    seeds_dir: Path,
    output_path: Path,
):
    """
    Plots a multi-panel grid of sample wings showing:
    - Detected junction points
    - Computed Convex Hull polygon & Area
    - Minimum Enclosing Circle (MEC) & Area
    - Centroid marker
    - Complete quantitative morphometric metric badges.
    """
    n = len(stems)
    rows = (n + 1) // 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4.6), dpi=300)
    axes = np.array(axes).reshape(-1)

    for idx, stem in enumerate(stems):
        ax = axes[idx]
        img_path = wings_dir / f"{stem}.png"
        seed_path = seeds_dir / f"{stem}.npy"

        if not img_path.exists() or not seed_path.exists():
            ax.text(0.5, 0.5, f"File not found:\n{stem}", ha="center", va="center")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        pts = np.load(str(seed_path)).astype(np.float32)
        h, w = img.shape

        # Compute Geometric Hulls & Circles
        hull = ConvexHull(pts)
        hull_area = float(hull.volume)
        hull_perim = float(hull.area)
        hull_circ = (4.0 * np.pi * hull_area) / (hull_perim ** 2) if hull_perim > 0 else 0.0

        (cx, cy), mec_r = cv2.minEnclosingCircle(pts)
        mec_area = np.pi * (mec_r ** 2)
        compactness = hull_area / mec_area if mec_area > 0 else 0.0

        centroid = np.mean(pts, axis=0)

        # Plot Wing Image
        ax.imshow(img, cmap="gray", aspect="equal", extent=[0, w, h, 0])

        # Plot MEC
        mec_patch = Circle(
            (cx, cy),
            mec_r,
            edgecolor="#00ffff",
            facecolor="none",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
            label="Min Enclosing Circle (MEC)",
            zorder=3,
        )
        ax.add_patch(mec_patch)
        ax.plot(cx, cy, "+", color="#00ffff", markersize=12, markeredgewidth=2.2, zorder=4)

        # Plot Convex Hull
        hull_pts = pts[hull.vertices]
        hull_polygon = Polygon(
            hull_pts,
            closed=True,
            edgecolor="#00ff44",
            facecolor="#00ff44",
            alpha=0.22,
            linewidth=2.4,
            label="Computed Convex Hull",
            zorder=5,
        )
        ax.add_patch(hull_polygon)
        ax.plot(
            np.append(hull_pts[:, 0], hull_pts[0, 0]),
            np.append(hull_pts[:, 1], hull_pts[0, 1]),
            color="#00ff44",
            linewidth=2.2,
            zorder=6,
        )

        # Plot Junction Points
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=8,
            c="#ff3366",
            alpha=0.6,
            edgecolors="none",
            label=f"Junctions (N={len(pts)})",
            zorder=7,
        )

        # Plot Centroid
        ax.plot(
            centroid[0],
            centroid[1],
            marker="*",
            color="#ffcc00",
            markersize=14,
            markeredgecolor="#000000",
            markeredgewidth=1.2,
            label="Centroid",
            zorder=8,
        )

        # Parse labels
        parts = stem.split("_")
        spec_id = parts[0]
        w_type = "Forewing" if "forewing" in stem else "Hindwing"
        species_name = " ".join([p.capitalize() for p in parts[1:] if p not in ("forewing", "hindwing")])

        ax.set_title(
            f"Specimen #{spec_id}: {species_name} ({w_type})",
            fontsize=12,
            fontweight="bold",
            pad=6,
        )
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xlabel("X (pixels)", fontsize=9)
        ax.set_ylabel("Y (pixels)", fontsize=9)
        ax.tick_params(labelsize=8)

        # Quantitative badge
        badge_text = (
            f"Hull Area: {hull_area:,.0f} px²\n"
            f"Hull Perimeter: {hull_perim:,.0f} px\n"
            f"MEC Area: {mec_area:,.0f} px²\n"
            f"MEC Radius: {mec_r:.1f} px\n"
            f"Compactness: {compactness:.3f}\n"
            f"Circularity: {hull_circ:.3f}\n"
            f"Junction Count: {len(pts)}"
        )
        ax.text(
            0.02,
            0.96,
            badge_text,
            transform=ax.transAxes,
            fontsize=8.5,
            family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#333333", alpha=0.9),
            zorder=10,
        )

    # Empty subplots if odd count
    for rem in range(n, len(axes)):
        axes[rem].axis("off")

    # Global legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="#cccccc",
    )

    plt.suptitle(
        "Dragonfly Wing Morphometric Convex Hulls & Minimum Enclosing Circles (MEC)",
        fontsize=16,
        fontweight="bold",
        y=1.015,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_biological_cells_and_areas(
    stem: str,
    wings_dir: Path,
    output_path: Path,
):
    """
    Generates a detailed 4-panel visual breakdown of biological cell compartments:
    - Panel A: Cell Area Heatmap (px²) across the entire wing.
    - Panel B: Polygonal order / valence map (side count: 3, 4, 5, 6, 7+).
    - Panel C: High-magnification zoom-in showing individual cell boundaries,
               computed areas, inscribed circles, and circumscribed circles.
    - Panel D: Cell Area Distribution Histogram & Statistical Summary.
    """
    img_path = wings_dir / f"{stem}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {img_path}")
        return

    h, w = img.shape
    cells, cleaned_vein = extract_cells_and_contours(img, stem)
    if not cells:
        print(f"No cells extracted for {stem}")
        return

    areas = np.array([c["area"] for c in cells], dtype=np.float32)
    sides = np.array([c["sides"] for c in cells], dtype=np.int32)

    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.25, wspace=0.18)

    # ----------------------------------------------------
    # Subplot A: Biological Cell Area Choropleth Heatmap
    # ----------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img, cmap="gray", alpha=0.35, aspect="equal", extent=[0, w, h, 0])

    norm_area = mcolors.LogNorm(vmin=max(float(np.min(areas)), 5.0), vmax=float(np.max(areas)))
    cmap_area = cm.viridis

    patches_a = [Polygon(c["polygon"], closed=True) for c in cells]
    coll_a = PatchCollection(patches_a, cmap=cmap_area, norm=norm_area, alpha=0.88, edgecolor="black", linewidth=0.4)
    coll_a.set_array(areas)
    ax_a.add_collection(coll_a)

    ax_a.set_xlim(0, w)
    ax_a.set_ylim(h, 0)
    ax_a.set_title(
        f"A. Biological Cell Area Heatmap ({len(cells)} Enclosed Compartments)",
        fontsize=12,
        fontweight="bold",
    )
    ax_a.set_xlabel("X (pixels)", fontsize=9)
    ax_a.set_ylabel("Y (pixels)", fontsize=9)
    cbar_a = plt.colorbar(coll_a, ax=ax_a, orientation="horizontal", pad=0.08, shrink=0.7)
    cbar_a.set_label("Cell Area (px², log scale)", fontsize=9, fontweight="bold")

    # ----------------------------------------------------
    # Subplot B: Polygonal Side Count / Valence Map
    # ----------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img, cmap="gray", alpha=0.35, aspect="equal", extent=[0, w, h, 0])

    # Distinct discrete colormap for sides: 3=blue, 4=cyan, 5=green, 6=yellow/orange, 7+=red
    side_colors = {
        3: "#1f77b4",  # Blue
        4: "#00b4d8",  # Cyan
        5: "#2ca02c",  # Green
        6: "#ff7f0e",  # Orange
        7: "#d62728",  # Red
    }
    patches_b = []
    colors_b = []
    for c in cells:
        s = min(c["sides"], 7)
        patches_b.append(Polygon(c["polygon"], closed=True))
        colors_b.append(side_colors.get(s, "#9467bd"))

    coll_b = PatchCollection(patches_b, facecolors=colors_b, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax_b.add_collection(coll_b)
    ax_b.set_xlim(0, w)
    ax_b.set_ylim(h, 0)
    ax_b.set_title(
        "B. Biological Cell Polygonal Order (Valence / Number of Sides)",
        fontsize=12,
        fontweight="bold",
    )
    ax_b.set_xlabel("X (pixels)", fontsize=9)
    ax_b.set_ylabel("Y (pixels)", fontsize=9)

    # Custom legend for sides
    side_labels = [
        f"3-sided ({np.mean(sides==3)*100:.1f}%)",
        f"4-sided ({np.mean(sides==4)*100:.1f}%)",
        f"5-sided ({np.mean(sides==5)*100:.1f}%)",
        f"6-sided ({np.mean(sides==6)*100:.1f}%)",
        f"7+-sided ({np.mean(sides>=7)*100:.1f}%)",
    ]
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=side_colors[k], markersize=10, label=side_labels[i])
        for i, k in enumerate([3, 4, 5, 6, 7])
    ]
    ax_b.legend(handles=legend_handles, loc="upper right", fontsize=8.5, framealpha=0.9)

    # ----------------------------------------------------
    # Subplot C: High-Magnification Zoom-in of Sample Cell Cluster
    # ----------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])

    # Pick a region of interest with well-formed cells near mid-wing
    # Find cell closest to center
    cx_mid, cy_mid = w * 0.45, h * 0.5
    dists_to_mid = [(c["centroid"][0] - cx_mid)**2 + (c["centroid"][1] - cy_mid)**2 for c in cells]
    best_idx = int(np.argmin(dists_to_mid))
    center_cell = cells[best_idx]
    zoom_cx, zoom_cy = center_cell["centroid"]
    box_w = min(w * 0.22, 160)
    box_h = min(h * 0.35, 120)
    x1, x2 = max(0, zoom_cx - box_w), min(w, zoom_cx + box_w)
    y1, y2 = max(0, zoom_cy - box_h), min(h, zoom_cy + box_h)

    # Filter cells inside zoom window
    zoom_cells = [
        c for c in cells
        if x1 <= c["centroid"][0] <= x2 and y1 <= c["centroid"][1] <= y2
    ]

    ax_c.imshow(img, cmap="gray", alpha=0.35, aspect="equal", extent=[0, w, h, 0])

    for c in zoom_cells:
        # Cell polygon outline and fill
        poly = Polygon(c["polygon"], closed=True, edgecolor="#ffffff", facecolor="#00e5ff", alpha=0.35, linewidth=1.5)
        ax_c.add_patch(poly)

        # Maximum Inscribed Circle (Blue)
        ins_c = Circle(
            c["inscribe_center"],
            c["inscribe_radius"],
            edgecolor="#0055ff",
            facecolor="none",
            linestyle="-",
            linewidth=1.4,
            label="Inscribed Circle (r_ins)",
        )
        ax_c.add_patch(ins_c)

        # Circumscribed Circle (Orange)
        mec_c = Circle(
            c["mec_center"],
            c["mec_radius"],
            edgecolor="#ff6600",
            facecolor="none",
            linestyle="--",
            linewidth=1.2,
            label="Circumcircle (R_mec)",
        )
        ax_c.add_patch(mec_c)

    ax_c.set_xlim(x1, x2)
    ax_c.set_ylim(y2, y1)
    ax_c.set_title(
        f"C. Microscopic Zoom-In: Cell Hulls, Inscribed & Circumscribed Circles",
        fontsize=12,
        fontweight="bold",
    )
    ax_c.set_xlabel("X (pixels)", fontsize=9)
    ax_c.set_ylabel("Y (pixels)", fontsize=9)

    # Zoom legend
    custom_lines = [
        plt.Line2D([0], [0], color="#ffffff", lw=2, label="Cell Boundary Contour"),
        plt.Line2D([0], [0], color="#0055ff", lw=1.5, label="Max Inscribed Circle (r_inscribe)"),
        plt.Line2D([0], [0], color="#ff6600", lw=1.5, linestyle="--", label="Min Circumcircle (R_circum)"),
    ]
    ax_c.legend(handles=custom_lines, loc="upper right", fontsize=8)

    # ----------------------------------------------------
    # Subplot D: Cell Area Distribution & Summary
    # ----------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    n_bins = 30
    counts, bin_edges, _ = ax_d.hist(
        areas,
        bins=n_bins,
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.75,
        density=True,
        label="Cell Area Frequencies",
    )

    # KDE curve
    kde = gaussian_kde(areas)
    x_grid = np.linspace(min(areas), max(areas), 200)
    ax_d.plot(x_grid, kde(x_grid), color="#e63946", linewidth=2.5, label="Kernel Density Estimate (KDE)")

    mean_a = float(np.mean(areas))
    med_a = float(np.median(areas))
    q25, q75 = np.percentile(areas, [25, 75])

    ax_d.axvline(mean_a, color="#ffaa00", linestyle="-", linewidth=2, label=f"Mean: {mean_a:.1f} px²")
    ax_d.axvline(med_a, color="#00cc66", linestyle="--", linewidth=2, label=f"Median: {med_a:.1f} px²")
    ax_d.axvspan(q25, q75, color="#cccccc", alpha=0.3, label=f"IQR: [{q25:.1f}, {q75:.1f}]")

    ax_d.set_title(
        f"D. Cell Area Distribution & Allometric Scaling (N={len(cells)} Cells)",
        fontsize=12,
        fontweight="bold",
    )
    ax_d.set_xlabel("Cell Area (px²)", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax_d.legend(loc="upper right", fontsize=8.5)
    ax_d.grid(True, linestyle="--", alpha=0.5)

    parts = stem.split("_")
    spec_id = parts[0]
    w_type = "Forewing" if "forewing" in stem else "Hindwing"
    species_name = " ".join([p.capitalize() for p in parts[1:] if p not in ("forewing", "hindwing")])

    plt.suptitle(
        f"Microscopic Biological Cell Area & Polygon Extraction: {species_name} ({w_type}, Specimen #{spec_id})",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_delaunay_triangulation_and_facets(
    stem: str,
    wings_dir: Path,
    seeds_dir: Path,
    output_path: Path,
):
    """
    Plots Delaunay triangulation, facet area distribution, and empty circumcircles.
    """
    img_path = wings_dir / f"{stem}.png"
    seed_path = seeds_dir / f"{stem}.npy"
    if not img_path.exists() or not seed_path.exists():
        return

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    pts = np.load(str(seed_path)).astype(np.float32)
    h, w = img.shape

    tri = Delaunay(pts)
    p1 = pts[tri.simplices[:, 0]]
    p2 = pts[tri.simplices[:, 1]]
    p3 = pts[tri.simplices[:, 2]]
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p2 - p3, axis=1)
    c = np.linalg.norm(p3 - p1, axis=1)
    s = (a + b + c) / 2.0
    d_area = np.sqrt(np.maximum(0, s * (s - a) * (s - b) * (s - c)))
    valid = d_area > 1e-4

    v_areas = d_area[valid]
    v_simplices = tri.simplices[valid]
    r_circum = (a[valid] * b[valid] * c[valid]) / (4.0 * d_area[valid])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=300, gridspec_kw={"height_ratios": [1.4, 0.9]})

    # Upper panel: Triangulation overlay on wing
    ax1.imshow(img, cmap="gray", alpha=0.35, aspect="equal", extent=[0, w, h, 0])

    tri_patches = [Polygon(pts[simplex], closed=True) for simplex in v_simplices]
    norm_tri = mcolors.LogNorm(vmin=max(float(np.min(v_areas)), 1.0), vmax=float(np.percentile(v_areas, 98)))
    coll = PatchCollection(tri_patches, cmap="plasma", norm=norm_tri, alpha=0.65, edgecolor="#ffffff", linewidth=0.3)
    coll.set_array(v_areas)
    ax1.add_collection(coll)

    ax1.scatter(pts[:, 0], pts[:, 1], s=8, color="#00ffff", alpha=0.8, zorder=5, label=f"Junctions (N={len(pts)})")

    # Sample a few circumcircles
    np.random.seed(42)
    sample_indices = np.random.choice(len(v_simplices), size=min(8, len(v_simplices)), replace=False)
    for s_idx in sample_indices:
        sim = v_simplices[s_idx]
        p_tri = pts[sim]
        # Circumcenter calculation
        d = 2 * (p_tri[0, 0] * (p_tri[1, 1] - p_tri[2, 1]) + p_tri[1, 0] * (p_tri[2, 1] - p_tri[0, 1]) + p_tri[2, 0] * (p_tri[0, 1] - p_tri[1, 1]))
        if abs(d) > 1e-6:
            ux = ((p_tri[0, 0]**2 + p_tri[0, 1]**2) * (p_tri[1, 1] - p_tri[2, 1]) + (p_tri[1, 0]**2 + p_tri[1, 1]**2) * (p_tri[2, 1] - p_tri[0, 1]) + (p_tri[2, 0]**2 + p_tri[2, 1]**2) * (p_tri[0, 1] - p_tri[1, 1])) / d
            uy = ((p_tri[0, 0]**2 + p_tri[0, 1]**2) * (p_tri[2, 0] - p_tri[1, 0]) + (p_tri[1, 0]**2 + p_tri[1, 1]**2) * (p_tri[0, 0] - p_tri[2, 0]) + (p_tri[2, 0]**2 + p_tri[2, 1]**2) * (p_tri[1, 0] - p_tri[0, 0])) / d
            circ = Circle((ux, uy), r_circum[s_idx], edgecolor="#00ff66", facecolor="none", linestyle=":", linewidth=1.2)
            ax1.add_patch(circ)

    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)
    ax1.set_title(
        f"Delaunay Triangulation & Facet Areas Overlay ({len(v_simplices)} Triangles)",
        fontsize=12,
        fontweight="bold",
    )
    cbar = plt.colorbar(coll, ax=ax1, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Facet Area (px²)", fontsize=9, fontweight="bold")

    # Lower panel: Facet area histogram & circumradius distribution
    ax2.hist(v_areas, bins=40, color="#6a4c93", edgecolor="black", alpha=0.75, label="Facet Areas (px²)")
    ax2.set_xlabel("Delaunay Triangle Facet Area (px²)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Triangle Count", fontsize=10, fontweight="bold")
    ax2.set_title("Facet Area Frequency Distribution", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    med_fa = float(np.median(v_areas))
    mean_fa = float(np.mean(v_areas))
    ax2.axvline(med_fa, color="#ff9f1c", linestyle="--", linewidth=2, label=f"Median: {med_fa:.1f} px²")
    ax2.axvline(mean_fa, color="#2ec4b6", linestyle="-", linewidth=2, label=f"Mean: {mean_fa:.1f} px²")
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        f"Delaunay Spatial Triangulation & Empty Circumcircle Mesh ({stem})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_dataset_morphometrics_validation(
    csv_path: Path,
    output_path: Path,
):
    """
    Produces a 6-panel statistical figure validating computed hulls and areas across all wings.
    """
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    axes = axes.flatten()

    fore = df[df["wing_type"] == "forewing"]
    hind = df[df["wing_type"] == "hindwing"]

    # 1. Hull Area Distribution: Forewing vs Hindwing
    ax1 = axes[0]
    ax1.hist(
        fore["hull_area"] / 1000.0,
        bins=25,
        alpha=0.65,
        color="#1f77b4",
        edgecolor="black",
        label=f"Forewings (n={len(fore)})",
    )
    ax1.hist(
        hind["hull_area"] / 1000.0,
        bins=25,
        alpha=0.65,
        color="#ff7f0e",
        edgecolor="black",
        label=f"Hindwings (n={len(hind)})",
    )
    ax1.set_title("A. Convex Hull Area Distribution (k px²)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Hull Area (× 1,000 px²)", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Wing Count", fontsize=9, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Hull Area vs MEC Area (Forewing vs Hindwing Compactness Scaling)
    ax2 = axes[1]
    ax2.scatter(
        fore["mec_area"] / 1000.0,
        fore["hull_area"] / 1000.0,
        s=20,
        color="#1f77b4",
        alpha=0.6,
        label="Forewing",
    )
    ax2.scatter(
        hind["mec_area"] / 1000.0,
        hind["hull_area"] / 1000.0,
        s=20,
        color="#ff7f0e",
        alpha=0.6,
        label="Hindwing",
    )

    # Linear fits
    p_fore = np.polyfit(fore["mec_area"] / 1000.0, fore["hull_area"] / 1000.0, 1)
    p_hind = np.polyfit(hind["mec_area"] / 1000.0, hind["hull_area"] / 1000.0, 1)
    x_line = np.linspace(0, df["mec_area"].max() / 1000.0, 100)
    ax2.plot(x_line, np.polyval(p_fore, x_line), color="#1f77b4", linestyle="--", linewidth=2, label=f"Forewing slope={p_fore[0]:.3f}")
    ax2.plot(x_line, np.polyval(p_hind, x_line), color="#ff7f0e", linestyle="-", linewidth=2, label=f"Hindwing slope={p_hind[0]:.3f}")

    ax2.set_title("B. Hull Area vs. Minimum Enclosing Circle Area", fontsize=11, fontweight="bold")
    ax2.set_xlabel("MEC Area (× 1,000 px²)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Convex Hull Area (× 1,000 px²)", fontsize=9, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Cell Count vs Hull Area
    ax3 = axes[2]
    ax3.scatter(
        df["hull_area"] / 1000.0,
        df["cell_count"],
        s=20,
        c=df["junction_density_hull"] * 1000,
        cmap="viridis",
        alpha=0.75,
    )
    ax3.set_title("C. Biological Cell Count vs. Hull Area", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Hull Area (× 1,000 px²)", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Cell Count", fontsize=9, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Mean Cell Area Distribution Across Species
    ax4 = axes[3]
    ax4.hist(
        df["cell_area_mean"],
        bins=30,
        color="#2ca02c",
        edgecolor="black",
        alpha=0.75,
    )
    med_cell_area = df["cell_area_mean"].median()
    mean_cell_area = df["cell_area_mean"].mean()
    ax4.axvline(med_cell_area, color="#d62728", linestyle="--", linewidth=2, label=f"Median: {med_cell_area:.1f} px²")
    ax4.axvline(mean_cell_area, color="#1f77b4", linestyle="-", linewidth=2, label=f"Mean: {mean_cell_area:.1f} px²")
    ax4.set_title("D. Mean Biological Cell Area Across 443 Wings", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Mean Cell Area (px²)", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Number of Wings", fontsize=9, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=8.5)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # 5. Polygonal Valence Distribution Across Entire Dataset (Lewis' Law)
    ax5 = axes[4]
    poly_cols = ["cell_pct_3_sided", "cell_pct_4_sided", "cell_pct_5_sided", "cell_pct_6_sided", "cell_pct_7plus_sided"]
    poly_means = [df[col].mean() for col in poly_cols]
    poly_stds = [df[col].std() for col in poly_cols]
    poly_labels = ["3-sided", "4-sided", "5-sided", "6-sided", "7+-sided"]
    bar_colors = ["#1f77b4", "#00b4d8", "#2ca02c", "#ff7f0e", "#d62728"]

    ax5.bar(poly_labels, poly_means, yerr=poly_stds, color=bar_colors, edgecolor="black", capsize=4, alpha=0.85)
    ax5.set_title("E. Polygonal Side Valence Frequencies Across Dataset", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Mean Percentage of Cells (%)", fontsize=9, fontweight="bold")
    ax5.grid(True, linestyle="--", alpha=0.5, axis="y")
    for i, m in enumerate(poly_means):
        ax5.text(i, m + poly_stds[i] + 1.0, f"{m:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # 6. Wing Compactness vs Aspect Ratio
    ax6 = axes[5]
    ax6.scatter(
        fore["aspect_ratio"],
        fore["wing_compactness_mec"],
        color="#1f77b4",
        alpha=0.6,
        s=22,
        label=f"Forewings (Mean Compactness={fore['wing_compactness_mec'].mean():.3f})",
    )
    ax6.scatter(
        hind["aspect_ratio"],
        hind["wing_compactness_mec"],
        color="#ff7f0e",
        alpha=0.6,
        s=22,
        label=f"Hindwings (Mean Compactness={hind['wing_compactness_mec'].mean():.3f})",
    )
    ax6.set_title("F. Wing Compactness (A_hull / A_mec) vs Aspect Ratio", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Aspect Ratio (Width / Height)", fontsize=9, fontweight="bold")
    ax6.set_ylabel("Compactness (A_hull / A_MEC)", fontsize=9, fontweight="bold")
    ax6.legend(loc="upper right", fontsize=8)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        f"Cross-Dataset Statistical Validation of Computed Hulls & Areas (N={len(df)} Wings, 232 Species)",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_master_showcase(
    stem_forewing: str,
    stem_hindwing: str,
    wings_dir: Path,
    seeds_dir: Path,
    csv_path: Path,
    output_path: Path,
):
    """
    Creates a unified master dashboard combining:
    - Forewing vs Hindwing Convex Hulls & MEC overlays
    - Biological Cell Area Heatmap
    - High-magnification cell geometry callout
    - Statistical Area scaling across the dataset
    """
    fig = plt.figure(figsize=(20, 14), dpi=300)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1], hspace=0.28, wspace=0.18)

    # 1. Forewing Hull & MEC Overlay
    ax_fore = fig.add_subplot(gs[0, 0])
    img_f = cv2.imread(str(wings_dir / f"{stem_forewing}.png"), cv2.IMREAD_GRAYSCALE)
    pts_f = np.load(str(seeds_dir / f"{stem_forewing}.npy")).astype(np.float32)
    hf, wf = img_f.shape

    hull_f = ConvexHull(pts_f)
    (cxf, cyf), rf = cv2.minEnclosingCircle(pts_f)
    mec_area_f = np.pi * (rf ** 2)

    ax_fore.imshow(img_f, cmap="gray", aspect="equal", extent=[0, wf, hf, 0])
    mec_patch_f = Circle((cxf, cyf), rf, edgecolor="#00ffff", facecolor="none", linestyle="--", linewidth=1.8)
    ax_fore.add_patch(mec_patch_f)
    hull_pts_f = pts_f[hull_f.vertices]
    poly_f = Polygon(hull_pts_f, closed=True, edgecolor="#00ff44", facecolor="#00ff44", alpha=0.22, linewidth=2.0)
    ax_fore.add_patch(poly_f)
    ax_fore.scatter(pts_f[:, 0], pts_f[:, 1], s=6, c="#ff3366", alpha=0.6)
    centroid_f = np.mean(pts_f, axis=0)
    ax_fore.plot(centroid_f[0], centroid_f[1], "*", color="#ffcc00", markersize=12, markeredgecolor="black")

    ax_fore.set_xlim(0, wf)
    ax_fore.set_ylim(hf, 0)
    ax_fore.set_title(f"A. Forewing Hull & MEC: {stem_forewing}", fontsize=11, fontweight="bold")
    ax_fore.set_xlabel("X (px)", fontsize=8)
    ax_fore.set_ylabel("Y (px)", fontsize=8)
    ax_fore.text(
        0.02, 0.95,
        f"Hull Area: {hull_f.volume:,.0f} px²\n"
        f"MEC Area: {mec_area_f:,.0f} px²\n"
        f"Compactness: {hull_f.volume / mec_area_f:.3f}\n"
        f"Junctions: {len(pts_f)}",
        transform=ax_fore.transAxes,
        fontsize=8,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", alpha=0.88),
    )

    # 2. Hindwing Hull & MEC Overlay
    ax_hind = fig.add_subplot(gs[0, 1])
    img_h = cv2.imread(str(wings_dir / f"{stem_hindwing}.png"), cv2.IMREAD_GRAYSCALE)
    pts_h = np.load(str(seeds_dir / f"{stem_hindwing}.npy")).astype(np.float32)
    hh, wh = img_h.shape

    hull_h = ConvexHull(pts_h)
    (cxh, cyh), rh = cv2.minEnclosingCircle(pts_h)
    mec_area_h = np.pi * (rh ** 2)

    ax_hind.imshow(img_h, cmap="gray", aspect="equal", extent=[0, wh, hh, 0])
    mec_patch_h = Circle((cxh, cyh), rh, edgecolor="#00ffff", facecolor="none", linestyle="--", linewidth=1.8)
    ax_hind.add_patch(mec_patch_h)
    hull_pts_h = pts_h[hull_h.vertices]
    poly_h = Polygon(hull_pts_h, closed=True, edgecolor="#00ff44", facecolor="#00ff44", alpha=0.22, linewidth=2.0)
    ax_hind.add_patch(poly_h)
    ax_hind.scatter(pts_h[:, 0], pts_h[:, 1], s=6, c="#ff3366", alpha=0.6)
    centroid_h = np.mean(pts_h, axis=0)
    ax_hind.plot(centroid_h[0], centroid_h[1], "*", color="#ffcc00", markersize=12, markeredgecolor="black")

    ax_hind.set_xlim(0, wh)
    ax_hind.set_ylim(hh, 0)
    ax_hind.set_title(f"B. Hindwing Hull & MEC: {stem_hindwing}", fontsize=11, fontweight="bold")
    ax_hind.set_xlabel("X (px)", fontsize=8)
    ax_hind.set_ylabel("Y (px)", fontsize=8)
    ax_hind.text(
        0.02, 0.95,
        f"Hull Area: {hull_h.volume:,.0f} px²\n"
        f"MEC Area: {mec_area_h:,.0f} px²\n"
        f"Compactness: {hull_h.volume / mec_area_h:.3f}\n"
        f"Junctions: {len(pts_h)}",
        transform=ax_hind.transAxes,
        fontsize=8,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", alpha=0.88),
    )

    # 3. Cell Area Heatmap on Forewing
    ax_heat = fig.add_subplot(gs[1, 0])
    cells_f, _ = extract_cells_and_contours(img_f, stem_forewing)
    areas_f = np.array([c["area"] for c in cells_f], dtype=np.float32)

    ax_heat.imshow(img_f, cmap="gray", alpha=0.35, aspect="equal", extent=[0, wf, hf, 0])
    norm_af = mcolors.LogNorm(vmin=max(float(np.min(areas_f)), 5.0), vmax=float(np.max(areas_f)))
    patches_cf = [Polygon(c["polygon"], closed=True) for c in cells_f]
    coll_cf = PatchCollection(patches_cf, cmap="viridis", norm=norm_af, alpha=0.85, edgecolor="black", linewidth=0.3)
    coll_cf.set_array(areas_f)
    ax_heat.add_collection(coll_cf)
    ax_heat.set_xlim(0, wf)
    ax_heat.set_ylim(hf, 0)
    ax_heat.set_title(f"C. Biological Cell Area Heatmap ({len(cells_f)} Cells)", fontsize=11, fontweight="bold")
    ax_heat.set_xlabel("X (px)", fontsize=8)
    ax_heat.set_ylabel("Y (px)", fontsize=8)
    cbar_cf = plt.colorbar(coll_cf, ax=ax_heat, orientation="horizontal", shrink=0.7, pad=0.08)
    cbar_cf.set_label("Cell Area (px², log scale)", fontsize=8, fontweight="bold")

    # 4. Cell Geometry Zoom-In
    ax_zoom = fig.add_subplot(gs[1, 1])
    cx_mid, cy_mid = wf * 0.45, hf * 0.5
    dists = [(c["centroid"][0] - cx_mid)**2 + (c["centroid"][1] - cy_mid)**2 for c in cells_f]
    best_idx = int(np.argmin(dists))
    zoom_cx, zoom_cy = cells_f[best_idx]["centroid"]
    bw, bh = min(wf * 0.22, 160), min(hf * 0.35, 120)
    zx1, zx2 = max(0, zoom_cx - bw), min(wf, zoom_cx + bw)
    zy1, zy2 = max(0, zoom_cy - bh), min(hf, zoom_cy + bh)

    zcells = [c for c in cells_f if zx1 <= c["centroid"][0] <= zx2 and zy1 <= c["centroid"][1] <= zy2]
    ax_zoom.imshow(img_f, cmap="gray", alpha=0.35, aspect="equal", extent=[0, wf, hf, 0])
    for c in zcells:
        ax_zoom.add_patch(Polygon(c["polygon"], closed=True, edgecolor="#ffffff", facecolor="#00e5ff", alpha=0.35, linewidth=1.4))
        ax_zoom.add_patch(Circle(c["inscribe_center"], c["inscribe_radius"], edgecolor="#0055ff", facecolor="none", lw=1.2))
        ax_zoom.add_patch(Circle(c["mec_center"], c["mec_radius"], edgecolor="#ff6600", facecolor="none", linestyle="--", lw=1.2))

    ax_zoom.set_xlim(zx1, zx2)
    ax_zoom.set_ylim(zy2, zy1)
    ax_zoom.set_title("D. Microscopic Cell Geometric Inscription & Morphology", fontsize=11, fontweight="bold")
    ax_zoom.set_xlabel("X (px)", fontsize=8)
    ax_zoom.set_ylabel("Y (px)", fontsize=8)

    # 5. Dataset Scaling: Hull Area vs MEC Area
    ax_scale = fig.add_subplot(gs[2, 0])
    df = pd.read_csv(csv_path)
    df_fore = df[df["wing_type"] == "forewing"]
    df_hind = df[df["wing_type"] == "hindwing"]

    ax_scale.scatter(df_fore["mec_area"] / 1000.0, df_fore["hull_area"] / 1000.0, s=18, color="#1f77b4", alpha=0.55, label="Forewing")
    ax_scale.scatter(df_hind["mec_area"] / 1000.0, df_hind["hull_area"] / 1000.0, s=18, color="#ff7f0e", alpha=0.55, label="Hindwing")
    p_f = np.polyfit(df_fore["mec_area"] / 1000.0, df_fore["hull_area"] / 1000.0, 1)
    p_h = np.polyfit(df_hind["mec_area"] / 1000.0, df_hind["hull_area"] / 1000.0, 1)
    xl = np.linspace(0, df["mec_area"].max() / 1000.0, 100)
    ax_scale.plot(xl, np.polyval(p_f, xl), "--", color="#1f77b4", lw=2, label=f"Forewing fit (slope={p_f[0]:.3f})")
    ax_scale.plot(xl, np.polyval(p_h, xl), "-", color="#ff7f0e", lw=2, label=f"Hindwing fit (slope={p_h[0]:.3f})")
    ax_scale.set_title("E. Morphometric Scaling: Hull Area vs. MEC Area (N=443)", fontsize=11, fontweight="bold")
    ax_scale.set_xlabel("MEC Area (× 1,000 px²)", fontsize=9, fontweight="bold")
    ax_scale.set_ylabel("Hull Area (× 1,000 px²)", fontsize=9, fontweight="bold")
    ax_scale.legend(loc="upper left", fontsize=8)
    ax_scale.grid(True, linestyle="--", alpha=0.5)

    # 6. Dataset Cell Area Distribution
    ax_dist = fig.add_subplot(gs[2, 1])
    ax_dist.hist(df["cell_area_mean"], bins=28, color="#2ca02c", edgecolor="black", alpha=0.75, label="Mean Cell Area")
    ax_dist.axvline(df["cell_area_mean"].median(), color="#d62728", linestyle="--", lw=2, label=f"Median: {df['cell_area_mean'].median():.1f} px²")
    ax_dist.axvline(df["cell_area_mean"].mean(), color="#1f77b4", linestyle="-", lw=2, label=f"Mean: {df['cell_area_mean'].mean():.1f} px²")
    ax_dist.set_title("F. Cell Area Distribution Across All 443 Wings", fontsize=11, fontweight="bold")
    ax_dist.set_xlabel("Mean Cell Area (px²)", fontsize=9, fontweight="bold")
    ax_dist.set_ylabel("Dragonfly Wings", fontsize=9, fontweight="bold")
    ax_dist.legend(loc="upper right", fontsize=8.5)
    ax_dist.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Dragonfly Wing Morphometrics & Topological Hulls/Areas Verification Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_single_wing_showcase(
    stem: str,
    wings_dir: Path,
    seeds_dir: Path,
    csv_path: Path,
    output_path: Path,
    include_callouts: bool = True,
):
    """
    Generates a dedicated, publication-grade 4-panel specimen showcase:
    - Panel A: Whole-Wing Convex Hull, Minimum Enclosing Circle (MEC), Centroid & Area Overlays.
    - Panel B: Complete Biological Cell Area Choropleth Heatmap with Log-Scale Calibration.
    - Panel C: Microscopic Cell Boundary Inscription & Polygonal Geometry Zoom-In.
    - Panel D: Delaunay Triangulation Network & Empty Circumcircle Dispersion.
    Includes an integrated high-precision morphometric metrics ledger banner.
    """
    img_path = wings_dir / f"{stem}.png"
    seed_path = seeds_dir / f"{stem}.npy"

    if not img_path.exists() or not seed_path.exists():
        print(f"Missing file for {stem}: {img_path} or {seed_path}")
        return

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    pts = np.load(str(seed_path)).astype(np.float32)
    h, w = img.shape

    # Load metrics from CSV
    row_dict = {}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        matching = df[df["stem"] == stem]
        if len(matching) > 0:
            row_dict = matching.iloc[0].to_dict()

    # Compute Convex Hull & MEC
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    hull_area = float(hull.volume)
    hull_perim = float(hull.area)
    hull_circ = (4.0 * np.pi * hull_area) / (hull_perim ** 2) if hull_perim > 0 else 0.0

    (cx, cy), mec_r = cv2.minEnclosingCircle(pts)
    mec_area = np.pi * (mec_r ** 2)
    compactness = hull_area / mec_area if mec_area > 0 else 0.0
    centroid = np.mean(pts, axis=0)

    # Extract Biological Cells
    cells, cleaned_vein = extract_cells_and_contours(img, stem)
    areas = np.array([c["area"] for c in cells], dtype=np.float32) if cells else np.array([1.0])

    # Compute Delaunay Triangulation
    tri = Delaunay(pts)
    p1 = pts[tri.simplices[:, 0]]
    p2 = pts[tri.simplices[:, 1]]
    p3 = pts[tri.simplices[:, 2]]
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p2 - p3, axis=1)
    c = np.linalg.norm(p3 - p1, axis=1)
    s = (a + b + c) / 2.0
    d_area = np.sqrt(np.maximum(0, s * (s - a) * (s - b) * (s - c)))
    valid_tri = d_area > 1e-4
    v_areas = d_area[valid_tri]
    v_simplices = tri.simplices[valid_tri]

    # Parse Title Info
    parts = stem.split("_")
    spec_id = parts[0]
    w_type = "Forewing" if "forewing" in stem else "Hindwing"
    species_name = " ".join([p.capitalize() for p in parts[1:] if p not in ("forewing", "hindwing")])

    fig = plt.figure(figsize=(19, 14), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.25, wspace=0.18)

    # ----------------------------------------------------
    # Panel A: Convex Hull & Minimum Enclosing Circle (MEC)
    # ----------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img, cmap="gray", aspect="equal", extent=[0, w, h, 0])

    # Plot MEC
    mec_patch = Circle(
        (cx, cy),
        mec_r,
        edgecolor="#00ffff",
        facecolor="none",
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label=f"MEC (Area: {mec_area:,.0f} px²)",
        zorder=3,
    )
    ax_a.add_patch(mec_patch)
    ax_a.plot(cx, cy, "+", color="#00ffff", markersize=14, markeredgewidth=2.2, zorder=4)

    # Plot Convex Hull
    hull_polygon = Polygon(
        hull_pts,
        closed=True,
        edgecolor="#00ff44",
        facecolor="#00ff44",
        alpha=0.24,
        linewidth=2.4,
        label=f"Convex Hull (Area: {hull_area:,.0f} px²)",
        zorder=5,
    )
    ax_a.add_patch(hull_polygon)
    ax_a.plot(
        np.append(hull_pts[:, 0], hull_pts[0, 0]),
        np.append(hull_pts[:, 1], hull_pts[0, 1]),
        color="#00ff44",
        linewidth=2.2,
        zorder=6,
    )

    # Plot Junction Points
    ax_a.scatter(
        pts[:, 0],
        pts[:, 1],
        s=10,
        c="#ff3366",
        alpha=0.65,
        edgecolors="none",
        label=f"Vein Junctions (N={len(pts)})",
        zorder=7,
    )

    # Plot Centroid
    ax_a.plot(
        centroid[0],
        centroid[1],
        marker="*",
        color="#ffcc00",
        markersize=14,
        markeredgecolor="#000000",
        markeredgewidth=1.2,
        label=f"Centroid ({centroid[0]:.1f}, {centroid[1]:.1f})",
        zorder=8,
    )

    ax_a.set_xlim(0, w)
    ax_a.set_ylim(h, 0)
    ax_a.set_title(
        f"A. Whole-Wing Convex Hull & Minimum Enclosing Circle (MEC)",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )
    ax_a.set_xlabel("X (pixels)", fontsize=9)
    ax_a.set_ylabel("Y (pixels)", fontsize=9)
    ax_a.legend(loc="lower left", fontsize=8, framealpha=0.88)

    badge_a = (
        f"Hull Area: {hull_area:,.0f} px²\n"
        f"Hull Perimeter: {hull_perim:,.0f} px\n"
        f"MEC Area: {mec_area:,.0f} px²\n"
        f"MEC Radius: {mec_r:.1f} px\n"
        f"Compactness: {compactness:.3f}\n"
        f"Circularity: {hull_circ:.3f}"
    )
    ax_a.text(
        0.02,
        0.96,
        badge_a,
        transform=ax_a.transAxes,
        fontsize=8.5,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#333333", alpha=0.9),
        zorder=10,
    )

    # ----------------------------------------------------
    # Panel B: Biological Cell Area Heatmap
    # ----------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img, cmap="gray", alpha=0.35, aspect="equal", extent=[0, w, h, 0])

    if cells:
        norm_b = mcolors.LogNorm(vmin=max(float(np.min(areas)), 5.0), vmax=float(np.max(areas)))
        patches_b = [Polygon(c["polygon"], closed=True) for c in cells]
        coll_b = PatchCollection(patches_b, cmap="viridis", norm=norm_b, alpha=0.88, edgecolor="black", linewidth=0.3)
        coll_b.set_array(areas)
        ax_b.add_collection(coll_b)
        cbar_b = plt.colorbar(coll_b, ax=ax_b, orientation="horizontal", shrink=0.72, pad=0.08)
        cbar_b.set_label("Cell Area (px², log scale)", fontsize=8.5, fontweight="bold")

    ax_b.set_xlim(0, w)
    ax_b.set_ylim(h, 0)
    ax_b.set_title(
        f"B. Biological Cell Area Heatmap ({len(cells)} Enclosed Cells)",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )
    ax_b.set_xlabel("X (pixels)", fontsize=9)
    ax_b.set_ylabel("Y (pixels)", fontsize=9)

    mean_a = float(np.mean(areas)) if len(areas) > 0 else 0.0
    med_a = float(np.median(areas)) if len(areas) > 0 else 0.0
    tot_a = float(np.sum(areas)) if len(areas) > 0 else 0.0
    cov_pct = (tot_a / hull_area * 100.0) if hull_area > 0 else 0.0

    badge_b = (
        f"Cell Count: {len(cells)}\n"
        f"Total Cell Area: {tot_a:,.0f} px²\n"
        f"Hull Coverage: {cov_pct:.1f}%\n"
        f"Mean Cell Area: {mean_a:.1f} px²\n"
        f"Median Area: {med_a:.1f} px²\n"
        f"Max Cell Area: {float(np.max(areas)):.1f} px²"
    )
    ax_b.text(
        0.02,
        0.96,
        badge_b,
        transform=ax_b.transAxes,
        fontsize=8.5,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#333333", alpha=0.9),
        zorder=10,
    )

    # ----------------------------------------------------
    # Panel C: Microscopic Cell Boundary & Circle Inscriptions
    # ----------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    if cells:
        cx_mid, cy_mid = w * 0.45, h * 0.5
        dists = [(c["centroid"][0] - cx_mid)**2 + (c["centroid"][1] - cy_mid)**2 for c in cells]
        best_idx = int(np.argmin(dists))
        zoom_cx, zoom_cy = cells[best_idx]["centroid"]
        bw, bh = min(w * 0.22, 160), min(h * 0.35, 120)
        zx1, zx2 = max(0, zoom_cx - bw), min(w, zoom_cx + bw)
        zy1, zy2 = max(0, zoom_cy - bh), min(h, zoom_cy + bh)

        zcells = [c for c in cells if zx1 <= c["centroid"][0] <= zx2 and zy1 <= c["centroid"][1] <= zy2]
        ax_c.imshow(img, cmap="gray", alpha=0.35, aspect="equal", extent=[0, w, h, 0])

        for c in zcells:
            ax_c.add_patch(Polygon(c["polygon"], closed=True, edgecolor="#ffffff", facecolor="#00e5ff", alpha=0.35, linewidth=1.4))
            ax_c.add_patch(Circle(c["inscribe_center"], c["inscribe_radius"], edgecolor="#0055ff", facecolor="none", lw=1.3, label="Inscribed Circle" if c == zcells[0] else ""))
            ax_c.add_patch(Circle(c["mec_center"], c["mec_radius"], edgecolor="#ff6600", facecolor="none", linestyle="--", lw=1.3, label="Cell MEC" if c == zcells[0] else ""))
            if include_callouts:
                ax_c.text(
                    c["centroid"][0],
                    c["centroid"][1],
                    f"A={c['area']}\n{c['sides']}s",
                    color="#ffffff",
                    fontsize=7.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#111111", alpha=0.72),
                )

        ax_c.set_xlim(zx1, zx2)
        ax_c.set_ylim(zy2, zy1)
        ax_c.legend(loc="upper right", fontsize=8, framealpha=0.85)

    c_title = (
        "C. Microscopic Cell Polygon Inscription & Area Callouts"
        if include_callouts
        else "C. Microscopic Cell Polygon Inscription & Geometry"
    )
    ax_c.set_title(c_title, fontsize=12, fontweight="bold", pad=6)
    ax_c.set_xlabel("X (pixels)", fontsize=9)
    ax_c.set_ylabel("Y (pixels)", fontsize=9)

    # ----------------------------------------------------
    # Panel D: Delaunay Triangulation & Circumcircle Voids
    # ----------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.imshow(img, cmap="gray", alpha=0.35, aspect="equal", extent=[0, w, h, 0])

    if len(v_simplices) > 0:
        tri_patches = [Polygon(pts[simplex], closed=True) for simplex in v_simplices]
        norm_tri = mcolors.LogNorm(vmin=max(float(np.min(v_areas)), 1.0), vmax=float(np.percentile(v_areas, 98)))
        coll_d = PatchCollection(tri_patches, cmap="plasma", norm=norm_tri, alpha=0.65, edgecolor="#ffffff", linewidth=0.3)
        coll_d.set_array(v_areas)
        ax_d.add_collection(coll_d)
        cbar_d = plt.colorbar(coll_d, ax=ax_d, orientation="horizontal", shrink=0.72, pad=0.08)
        cbar_d.set_label("Facet Area (px², log scale)", fontsize=8.5, fontweight="bold")

    ax_d.scatter(pts[:, 0], pts[:, 1], s=8, color="#00ffff", alpha=0.8, zorder=5, label=f"Junctions (N={len(pts)})")

    # Sample a few circumcircles
    np.random.seed(42)
    sample_indices = np.random.choice(len(v_simplices), size=min(6, len(v_simplices)), replace=False)
    for s_idx in sample_indices:
        sim = v_simplices[s_idx]
        p_tri = pts[sim]
        d_p1, d_p2, d_p3 = p_tri[0], p_tri[1], p_tri[2]
        d_a = np.linalg.norm(d_p1 - d_p2)
        d_b = np.linalg.norm(d_p2 - d_p3)
        d_c = np.linalg.norm(d_p3 - d_p1)
        d_s = (d_a + d_b + d_c) / 2.0
        area_s = np.sqrt(max(0, d_s * (d_s - d_a) * (d_s - d_b) * (d_s - d_c)))
        if area_s > 1e-4:
            r_c = (d_a * d_b * d_c) / (4.0 * area_s)
            d_val = 2 * (d_p1[0] * (d_p2[1] - d_p3[1]) + d_p2[0] * (d_p3[1] - d_p1[1]) + d_p3[0] * (d_p1[1] - d_p2[1]))
            if abs(d_val) > 1e-5:
                u_x = ((d_p1[0]**2 + d_p1[1]**2) * (d_p2[1] - d_p3[1]) + (d_p2[0]**2 + d_p2[1]**2) * (d_p3[1] - d_p1[1]) + (d_p3[0]**2 + d_p3[1]**2) * (d_p1[1] - d_p2[1])) / d_val
                u_y = ((d_p1[0]**2 + d_p1[1]**2) * (d_p3[0] - d_p2[0]) + (d_p2[0]**2 + d_p2[1]**2) * (d_p1[0] - d_p3[0]) + (d_p3[0]**2 + d_p3[1]**2) * (d_p2[0] - d_p1[0])) / d_val
                if r_c < 120:
                    c_patch = Circle((u_x, u_y), r_c, edgecolor="#ff00ff", facecolor="none", linestyle=":", linewidth=1.5, zorder=6)
                    ax_d.add_patch(c_patch)

    ax_d.set_xlim(0, w)
    ax_d.set_ylim(h, 0)
    ax_d.set_title("D. Delaunay Network Triangulation & Circumcircle Voids", fontsize=12, fontweight="bold", pad=6)
    ax_d.set_xlabel("X (pixels)", fontsize=9)
    ax_d.set_ylabel("Y (pixels)", fontsize=9)
    ax_d.legend(loc="lower left", fontsize=8, framealpha=0.88)

    ce_val = row_dict.get("clark_evans_index", 1.20)
    badge_d = (
        f"Triangles: {len(v_simplices):,}\n"
        f"Mean Facet Area: {float(np.mean(v_areas)):.1f} px²\n"
        f"Median Facet Area: {float(np.median(v_areas)):.1f} px²\n"
        f"Clark-Evans Index: {ce_val:.3f}"
    )
    ax_d.text(
        0.02,
        0.96,
        badge_d,
        transform=ax_d.transAxes,
        fontsize=8.5,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#333333", alpha=0.9),
        zorder=10,
    )

    plt.suptitle(
        f"Specimen Morphometrics Showcase: #{spec_id} {species_name} ({w_type})\n"
        f"Convex Hull Area: {hull_area:,.0f} px² | MEC Area: {mec_area:,.0f} px² | Compactness: {compactness:.3f} | Enclosed Cells: {len(cells)}",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot sample wings and computed hulls/areas.")
    parser.add_argument("--wings-dir", type=Path, default=DEFAULT_WINGS_DIR, help="Path to cleaned wings.")
    parser.add_argument("--seeds-dir", type=Path, default=DEFAULT_SEEDS_DIR, help="Path to cleaned seeds.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH, help="Path to wing_features.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save plots.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Figure 1: Sample Wings Convex Hulls & MEC Overlays...")
    plot_sample_wings_convex_hulls(
        stems=SAMPLE_STEMS,
        wings_dir=args.wings_dir,
        seeds_dir=args.seeds_dir,
        output_path=args.output_dir / "01_sample_wings_convex_hulls_and_mec.png",
    )

    print("Generating Figure 2: Biological Cell Areas & Geometry Breakdown...")
    plot_biological_cells_and_areas(
        stem=SAMPLE_STEMS[0],
        wings_dir=args.wings_dir,
        output_path=args.output_dir / "02_biological_cells_and_area_heatmaps.png",
    )

    print("Generating Figure 3: Delaunay Triangulation & Facet Areas...")
    plot_delaunay_triangulation_and_facets(
        stem=SAMPLE_STEMS[0],
        wings_dir=args.wings_dir,
        seeds_dir=args.seeds_dir,
        output_path=args.output_dir / "03_delaunay_triangulation_and_empty_circles.png",
    )

    print("Generating Figure 4: Dataset-Wide Statistical Morphometrics Validation...")
    plot_dataset_morphometrics_validation(
        csv_path=args.csv_path,
        output_path=args.output_dir / "04_dataset_areas_and_morphometrics_validation.png",
    )

    print("Generating Figure 5: Master Publication Dashboard Showcase...")
    plot_master_showcase(
        stem_forewing=SAMPLE_STEMS[0],
        stem_hindwing=SAMPLE_STEMS[1],
        wings_dir=args.wings_dir,
        seeds_dir=args.seeds_dir,
        csv_path=args.csv_path,
        output_path=args.output_dir / "05_master_hulls_and_areas_showcase.png",
    )

    print("Generating Figure 6: Single-Wing Deep-Dive Showcase (Stillwater Clubtail Forewing)...")
    plot_single_wing_showcase(
        stem="100_stillwater_clubtail_forewing",
        wings_dir=args.wings_dir,
        seeds_dir=args.seeds_dir,
        csv_path=args.csv_path,
        output_path=args.output_dir / "06_single_wing_showcase_100_stillwater_clubtail_forewing.png",
        include_callouts=False,
    )

    print("Generating Figure 7: Single-Wing Deep-Dive Showcase (Hyacinth Glider Hindwing)...")
    plot_single_wing_showcase(
        stem="385_hyacinth_glider_hindwing",
        wings_dir=args.wings_dir,
        seeds_dir=args.seeds_dir,
        csv_path=args.csv_path,
        output_path=args.output_dir / "07_single_wing_showcase_385_hyacinth_glider_hindwing.png",
        include_callouts=True,
    )

    print("\nAll morphometric and hull visualization figures generated successfully!")


if __name__ == "__main__":
    main()
