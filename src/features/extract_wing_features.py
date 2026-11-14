#!/usr/bin/env python3
"""
Dragonfly Wing Morphometric & Topological Feature Extraction Pipeline.

Extracts a comprehensive tabular feature matrix across dragonfly wings:
1. Global Wing Morphometrics & Whole-Wing Minimum Enclosing Circle (MEC).
2. Spatial Point Pattern Statistics (Nearest Neighbor Distances, Clark-Evans Index).
3. Delaunay Triangulation Empty Circumcircles & Facet Areas.
4. Topological Vein Skeleton & Crossing Number Junction Branching (Degree 3 vs 4).
5. Biological Cell Polygon Extraction via Vein Mask Connected Components:
   - Cell area distributions (mean, std, median, IQR, log-variance, skewness)
   - Cell perimeters and isoperimetric quotient / circularity (4*pi*A / P^2)
   - Cell circumcircle radius (R_circum) and inscribed circle radius (r_inscribe)
   - Cell elongation ratio (R_circum / r_inscribe) and packing efficiency
   - Cell polygonal valence / side distributions (% 3, 4, 5, 6, 7+ sided cells).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.stats import skew

# Add repository modules to sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root / "src/walkers") not in sys.path:
    sys.path.insert(0, str(repo_root / "src/walkers"))
if str(repo_root / "src/preprocessing") not in sys.path:
    sys.path.insert(0, str(repo_root / "src/preprocessing"))

from cleaning import extract_veins_tophat
from topological_walker import TopologicalWalker


DEFAULT_WINGS_DIR = repo_root / "data/cleaned_wings"
DEFAULT_SEEDS_DIR = repo_root / "data/cleaned_seeds"
DEFAULT_UNET_DIR = repo_root / "data/unet_predictions"
DEFAULT_MANIFEST = repo_root / "data/cleaned_wings_manifest.csv"
DEFAULT_OUTPUT_CSV = repo_root / "data/wing_features.csv"


def safe_skew(arr: np.ndarray) -> float:
    """Computes skewness safely, returning 0.0 for degenerate or uniform distributions."""
    if len(arr) < 3:
        return 0.0
    s = np.std(arr)
    if s < 1e-6:
        return 0.0
    val = float(skew(arr))
    return 0.0 if np.isnan(val) or np.isinf(val) else val


def safe_log_var(arr: np.ndarray) -> float:
    """Computes the variance of log-transformed values safely."""
    pos = arr[arr > 0]
    if len(pos) < 2:
        return 0.0
    val = float(np.var(np.log(pos)))
    return 0.0 if np.isnan(val) or np.isinf(val) else val


def featurize_single_wing(args_tuple: Tuple[str, Path, Path, Path]) -> Optional[Dict[str, Any]]:
    """
    Featurizes a single dragonfly wing from its image and junction seed file.
    """
    stem, img_path, seed_path, unet_path = args_tuple

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    if not seed_path.exists():
        return None

    junctions = np.load(str(seed_path))
    if junctions.ndim == 1:
        junctions = junctions.reshape(-1, 2)
    N = len(junctions)

    if unet_path.exists():
        unet_juncs = np.load(str(unet_path))
        if unet_juncs.ndim == 1:
            unet_juncs = unet_juncs.reshape(-1, 2)
        N_unet = len(unet_juncs)
    else:
        N_unet = 0

    h, w = img.shape

    # Metadata & Taxonomy Parsing
    parts = stem.split("_")
    specimen_id = parts[0] if parts[0].isdigit() else ""
    wing_type = "forewing" if "forewing" in stem else ("hindwing" if "hindwing" in stem else "unknown")
    species = "_".join([p for p in parts[1:] if p not in ("forewing", "hindwing", "skillet")])

    feats: Dict[str, Any] = {
        "stem": stem,
        "filename": f"{stem}.png",
        "specimen_id": specimen_id,
        "species": species,
        "wing_type": wing_type,
        "img_width": w,
        "img_height": h,
        "aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
        "img_total_pixels": h * w,
        "num_junctions_topo": N,
        "num_junctions_unet": N_unet,
        "junction_count_diff": N_unet - N,
        "junction_count_ratio": round(N_unet / N, 4) if N > 0 else 0.0,
    }

    if N < 4:
        # Cannot compute morphometric geometry on fewer than 4 points
        return feats

    pts = junctions.astype(np.float32)

    # ========================================================
    # 1. Whole-Wing Minimum Enclosing Circle (MEC) & Convex Hull
    # ========================================================
    (cx, cy), mec_radius = cv2.minEnclosingCircle(pts)
    mec_area = np.pi * (mec_radius ** 2)
    feats["mec_center_x"] = round(float(cx), 2)
    feats["mec_center_y"] = round(float(cy), 2)
    feats["mec_radius"] = round(float(mec_radius), 2)
    feats["mec_area"] = round(float(mec_area), 2)

    hull = ConvexHull(pts)
    hull_area = float(hull.volume)  # In 2D, scipy hull.volume = 2D area
    hull_perim = float(hull.area)   # In 2D, scipy hull.area = 2D perimeter
    feats["hull_area"] = round(hull_area, 2)
    feats["hull_perimeter"] = round(hull_perim, 2)
    feats["hull_circularity"] = (
        round((4.0 * np.pi * hull_area) / (hull_perim ** 2), 4) if hull_perim > 0 else 0.0
    )
    feats["wing_compactness_mec"] = round(hull_area / mec_area, 4) if mec_area > 0 else 0.0

    centroid = np.mean(pts, axis=0)
    feats["centroid_x"] = round(float(centroid[0]), 2)
    feats["centroid_y"] = round(float(centroid[1]), 2)
    feats["centroid_dist_to_mec"] = round(float(np.linalg.norm(centroid - [cx, cy])), 2)
    feats["junction_density_hull"] = round(N / hull_area, 6) if hull_area > 0 else 0.0
    feats["junction_density_img"] = round(N / (h * w), 6)

    # ========================================================
    # 2. Longitudinal & Transverse Spatial Gradients
    # ========================================================
    xs = pts[:, 0]
    ys = pts[:, 1]
    feats["junction_x_min"] = round(float(np.min(xs)), 1)
    feats["junction_x_max"] = round(float(np.max(xs)), 1)
    feats["junction_x_mean"] = round(float(np.mean(xs)), 2)
    feats["junction_x_std"] = round(float(np.std(xs)), 2)
    feats["junction_x_skewness"] = round(safe_skew(xs), 4)
    feats["junction_span_x"] = round(float(np.max(xs) - np.min(xs)), 1)

    feats["junction_y_min"] = round(float(np.min(ys)), 1)
    feats["junction_y_max"] = round(float(np.max(ys)), 1)
    feats["junction_y_mean"] = round(float(np.mean(ys)), 2)
    feats["junction_y_std"] = round(float(np.std(ys)), 2)
    feats["junction_y_skewness"] = round(safe_skew(ys), 4)
    feats["junction_span_y"] = round(float(np.max(ys) - np.min(ys)), 1)

    x_mid = 0.5 * (np.min(xs) + np.max(xs))
    y_mid = 0.5 * (np.min(ys) + np.max(ys))
    basal_n = np.sum(xs <= x_mid)
    apical_n = np.sum(xs > x_mid)
    leading_n = np.sum(ys <= y_mid)
    trailing_n = np.sum(ys > y_mid)
    feats["basal_fraction"] = round(float(basal_n / N), 4)
    feats["apical_fraction"] = round(float(apical_n / N), 4)
    feats["leading_fraction"] = round(float(leading_n / N), 4)
    feats["trailing_fraction"] = round(float(trailing_n / N), 4)
    feats["basal_to_apical_ratio"] = round(float(basal_n / max(apical_n, 1)), 4)
    feats["leading_to_trailing_ratio"] = round(float(leading_n / max(trailing_n, 1)), 4)

    # ========================================================
    # 3. Spatial Point Pattern Statistics (Nearest Neighbor & Clark-Evans)
    # ========================================================
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=min(4, N))
    nnd = dists[:, 1]
    feats["nnd_mean"] = round(float(np.mean(nnd)), 3)
    feats["nnd_std"] = round(float(np.std(nnd)), 3)
    feats["nnd_median"] = round(float(np.median(nnd)), 3)
    feats["nnd_min"] = round(float(np.min(nnd)), 3)
    feats["nnd_max"] = round(float(np.max(nnd)), 3)
    feats["nnd_q25"] = round(float(np.percentile(nnd, 25)), 3)
    feats["nnd_q75"] = round(float(np.percentile(nnd, 75)), 3)
    feats["nnd2_mean"] = round(float(np.mean(dists[:, 2])), 3) if dists.shape[1] > 2 else feats["nnd_mean"]
    feats["nnd3_mean"] = round(float(np.mean(dists[:, 3])), 3) if dists.shape[1] > 3 else feats["nnd_mean"]

    density = N / hull_area if hull_area > 0 else 0.0
    expected_nnd = 0.5 / np.sqrt(density) if density > 0 else 1.0
    feats["clark_evans_index"] = round(float(feats["nnd_mean"] / expected_nnd), 4) if expected_nnd > 0 else 1.0

    # ========================================================
    # 4. Delaunay Triangulation Empty Circumcircles & Facets
    # ========================================================
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
    r_circum = (a[valid] * b[valid] * c[valid]) / (4.0 * d_area[valid])
    v_areas = d_area[valid]

    feats["delaunay_triangles_count"] = int(np.sum(valid))
    feats["delaunay_circumradius_mean"] = round(float(np.mean(r_circum)), 3)
    feats["delaunay_circumradius_std"] = round(float(np.std(r_circum)), 3)
    feats["delaunay_circumradius_median"] = round(float(np.median(r_circum)), 3)
    feats["delaunay_circumradius_q25"] = round(float(np.percentile(r_circum, 25)), 3)
    feats["delaunay_circumradius_q75"] = round(float(np.percentile(r_circum, 75)), 3)
    feats["delaunay_facet_area_mean"] = round(float(np.mean(v_areas)), 3)
    feats["delaunay_facet_area_std"] = round(float(np.std(v_areas)), 3)
    feats["delaunay_facet_area_median"] = round(float(np.median(v_areas)), 3)
    feats["delaunay_facet_area_skewness"] = round(safe_skew(v_areas), 4)

    # ========================================================
    # 5. Topological Vein Skeleton & Crossing Number Branch Degrees
    # ========================================================
    ridge_th = 10
    bw_top, _ = extract_veins_tophat(
        img, kernel_size=9, ridge_threshold=ridge_th, background_threshold=250
    )
    vein_mask = (bw_top == 0).astype(np.uint8)

    tw = TopologicalWalker(vein_mask, min_component_size=3, spur_threshold=3)
    cleaned_vein = tw.clean_mask()
    skeleton = tw.compute_skeleton(cleaned_vein)

    # 8-neighbor Rutovitz crossing number calculation
    p_shifts = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    pad = np.pad(skeleton, 1, mode="constant")
    nbrs = [pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in p_shifts]
    crossing_num = np.zeros_like(skeleton, dtype=np.float32)
    for i in range(8):
        crossing_num += np.abs(nbrs[i].astype(np.float32) - nbrs[(i + 1) % 8].astype(np.float32))
    crossing_num = (0.5 * crossing_num) * skeleton

    vein_px = int(np.sum(skeleton > 0))
    deg3 = int(np.sum(crossing_num == 3))
    deg4 = int(np.sum(crossing_num >= 4))
    tot_branch = max(deg3 + deg4, 1)

    feats["vein_pixel_count"] = vein_px
    feats["vein_length_density"] = round(vein_px / hull_area, 4) if hull_area > 0 else 0.0
    feats["junction_degree3_count"] = deg3
    feats["junction_degree4_count"] = deg4
    feats["junction_degree3_ratio"] = round(deg3 / tot_branch, 4)
    feats["junction_degree4_ratio"] = round(deg4 / tot_branch, 4)

    # ========================================================
    # 6. Biological Cell Polygon Extraction via Vein Mask Connected Components
    # ========================================================
    # Membrane cells are the non-vein interior compartments enclosed by veins
    membrane = (cleaned_vein == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(membrane, connectivity=4)

    # Identify background components touching image boundaries
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_labels = set(np.unique(labels[border_mask]))

    c_areas = []
    c_perims = []
    c_circs = []
    c_mec_r = []
    c_ins_r = []
    c_r_ratios = []
    c_mec_effs = []
    c_sides = []

    for lbl in range(1, num_labels):
        if lbl in border_labels:
            continue
        c_area = int(stats[lbl, cv2.CC_STAT_AREA])
        if c_area < 5:  # Filter out sub-vein discretization noise
            continue

        c_bin = (labels == lbl).astype(np.uint8)
        cnts, _ = cv2.findContours(c_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        c_p = float(cv2.arcLength(cnt, True))
        if c_p <= 0:
            continue

        # Douglas-Peucker polygon approximation for polygonal order (valence)
        approx = cv2.approxPolyDP(cnt, 0.04 * c_p, True)
        _, c_mec = cv2.minEnclosingCircle(cnt)

        # Distance transform inside cell for maximum inscribed circle
        dist_m = cv2.distanceTransform(c_bin, cv2.DIST_L2, 3)
        _, c_ins, _, _ = cv2.minMaxLoc(dist_m)

        # Isoperimetric quotient Q = 4 * pi * Area / Perimeter^2
        c_circ = (4.0 * np.pi * c_area) / (c_p ** 2)
        # Area efficiency inside circumscribing circle
        c_eff = c_area / (np.pi * (c_mec ** 2)) if c_mec > 0 else 0.0
        # Ratio of circumscribed circle radius to inscribed circle radius (elongation)
        c_ratio = c_mec / c_ins if c_ins > 0 else 1.0

        c_areas.append(c_area)
        c_perims.append(c_p)
        c_circs.append(c_circ)
        c_mec_r.append(c_mec)
        c_ins_r.append(c_ins)
        c_r_ratios.append(c_ratio)
        c_mec_effs.append(c_eff)
        c_sides.append(len(approx))

    c_areas_arr = np.array(c_areas, dtype=np.float32)
    c_perims_arr = np.array(c_perims, dtype=np.float32)
    c_circs_arr = np.array(c_circs, dtype=np.float32)
    c_mec_r_arr = np.array(c_mec_r, dtype=np.float32)
    c_ins_r_arr = np.array(c_ins_r, dtype=np.float32)
    c_r_ratios_arr = np.array(c_r_ratios, dtype=np.float32)
    c_mec_effs_arr = np.array(c_mec_effs, dtype=np.float32)
    c_sides_arr = np.array(c_sides, dtype=np.int32)

    n_cells = len(c_areas_arr)
    feats["cell_count"] = n_cells
    feats["cell_total_area"] = round(float(np.sum(c_areas_arr)), 1)
    feats["cell_area_coverage_pct"] = (
        round(float(np.sum(c_areas_arr) / hull_area * 100), 2) if hull_area > 0 else 0.0
    )

    if n_cells > 0:
        feats["cell_area_mean"] = round(float(np.mean(c_areas_arr)), 2)
        feats["cell_area_std"] = round(float(np.std(c_areas_arr)), 2)
        feats["cell_area_median"] = round(float(np.median(c_areas_arr)), 2)
        q75, q25 = np.percentile(c_areas_arr, [75, 25])
        feats["cell_area_iqr"] = round(float(q75 - q25), 2)
        feats["cell_area_min"] = round(float(np.min(c_areas_arr)), 1)
        feats["cell_area_max"] = round(float(np.max(c_areas_arr)), 1)
        feats["cell_area_p10"] = round(float(np.percentile(c_areas_arr, 10)), 2)
        feats["cell_area_p90"] = round(float(np.percentile(c_areas_arr, 90)), 2)
        feats["cell_area_skewness"] = round(safe_skew(c_areas_arr), 4)
        feats["cell_area_log_var"] = round(safe_log_var(c_areas_arr), 4)

        feats["cell_perim_mean"] = round(float(np.mean(c_perims_arr)), 2)
        feats["cell_perim_std"] = round(float(np.std(c_perims_arr)), 2)
        feats["cell_perim_median"] = round(float(np.median(c_perims_arr)), 2)

        feats["cell_circularity_mean"] = round(float(np.mean(c_circs_arr)), 4)
        feats["cell_circularity_std"] = round(float(np.std(c_circs_arr)), 4)
        feats["cell_circularity_median"] = round(float(np.median(c_circs_arr)), 4)

        feats["cell_mec_radius_mean"] = round(float(np.mean(c_mec_r_arr)), 2)
        feats["cell_mec_radius_std"] = round(float(np.std(c_mec_r_arr)), 2)
        feats["cell_inscribe_radius_mean"] = round(float(np.mean(c_ins_r_arr)), 2)
        feats["cell_inscribe_radius_std"] = round(float(np.std(c_ins_r_arr)), 2)
        feats["cell_radius_ratio_mean"] = round(float(np.mean(c_r_ratios_arr)), 3)
        feats["cell_radius_ratio_median"] = round(float(np.median(c_r_ratios_arr)), 3)
        feats["cell_mec_efficiency_mean"] = round(float(np.mean(c_mec_effs_arr)), 4)

        feats["cell_pct_3_sided"] = round(float(np.mean(c_sides_arr == 3) * 100), 2)
        feats["cell_pct_4_sided"] = round(float(np.mean(c_sides_arr == 4) * 100), 2)
        feats["cell_pct_5_sided"] = round(float(np.mean(c_sides_arr == 5) * 100), 2)
        feats["cell_pct_6_sided"] = round(float(np.mean(c_sides_arr == 6) * 100), 2)
        feats["cell_pct_7plus_sided"] = round(float(np.mean(c_sides_arr >= 7) * 100), 2)
        feats["cell_sides_mean"] = round(float(np.mean(c_sides_arr)), 2)
        feats["cell_sides_std"] = round(float(np.std(c_sides_arr)), 2)
        feats["cell_sides_mode"] = int(np.bincount(c_sides_arr).argmax())
    else:
        for k in [
            "cell_area_mean", "cell_area_std", "cell_area_median", "cell_area_iqr",
            "cell_area_min", "cell_area_max", "cell_area_p10", "cell_area_p90",
            "cell_area_skewness", "cell_area_log_var", "cell_perim_mean",
            "cell_perim_std", "cell_perim_median", "cell_circularity_mean",
            "cell_circularity_std", "cell_circularity_median", "cell_mec_radius_mean",
            "cell_mec_radius_std", "cell_inscribe_radius_mean", "cell_inscribe_radius_std",
            "cell_radius_ratio_mean", "cell_radius_ratio_median", "cell_mec_efficiency_mean",
            "cell_pct_3_sided", "cell_pct_4_sided", "cell_pct_5_sided", "cell_pct_6_sided",
            "cell_pct_7plus_sided", "cell_sides_mean", "cell_sides_std", "cell_sides_mode"
        ]:
            feats[k] = 0.0

    return feats


def run_extraction_pipeline(
    wings_dir: Path = DEFAULT_WINGS_DIR,
    seeds_dir: Path = DEFAULT_SEEDS_DIR,
    unet_dir: Path = DEFAULT_UNET_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    workers: int = 8,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Executes the full feature extraction pipeline across all cleaned dragonfly wings.
    """
    t_start = time.perf_counter()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    # Target only cleaned wings included in dataset
    inc = df_manifest[df_manifest["status"] == "included"].copy()
    stems = list(inc["stem"])

    if limit is not None and limit > 0:
        stems = stems[:limit]

    print(f"============================================================")
    print(f"DRAGONFLY WING FEATURE EXTRACTION PIPELINE")
    print(f"============================================================")
    print(f"Wings Directory    : {wings_dir}")
    print(f"Seeds Directory    : {seeds_dir}")
    print(f"UNet Directory     : {unet_dir}")
    print(f"Total Wings Target : {len(stems)}")
    print(f"Worker Processes   : {workers}")
    print(f"Output CSV         : {output_csv}")
    print(f"============================================================\n")

    tasks = [
        (
            stem,
            wings_dir / f"{stem}.png",
            seeds_dir / f"{stem}.npy",
            unet_dir / f"{stem}.npy",
        )
        for stem in stems
    ]

    records = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, res in enumerate(executor.map(featurize_single_wing, tasks), start=1):
                if res is not None:
                    records.append(res)
                if idx % 50 == 0 or idx == len(tasks):
                    elapsed = time.perf_counter() - t_start
                    rate = idx / elapsed if elapsed > 0 else 0
                    print(f"[{idx:3d}/{len(tasks):3d}] Processed wings... ({rate:.1f} wings/sec)")
    else:
        for idx, task in enumerate(tasks, start=1):
            res = featurize_single_wing(task)
            if res is not None:
                records.append(res)
            if idx % 50 == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t_start
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"[{idx:3d}/{len(tasks):3d}] Processed wings... ({rate:.1f} wings/sec)")

    df_feats = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_csv(output_csv, index=False)

    total_time = time.perf_counter() - t_start
    print(f"\n============================================================")
    print(f"SUCCESSFULLY EXTRACTED WING FEATURES")
    print(f"Total Wings Saved  : {len(df_feats)}")
    print(f"Features Per Wing  : {df_feats.shape[1]}")
    print(f"Total Elapsed Time : {total_time:.2f} seconds ({len(df_feats)/total_time:.1f} wings/sec)")
    print(f"Saved Dataset To   : {output_csv}")
    print(f"============================================================\n")

    return df_feats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract comprehensive morphometric, topological, and cell polygon features for dragonfly wings."
    )
    parser.add_argument("--wings-dir", type=Path, default=DEFAULT_WINGS_DIR, help="Directory containing cleaned wing images (.png)")
    parser.add_argument("--seeds-dir", type=Path, default=DEFAULT_SEEDS_DIR, help="Directory containing Topological Walker seeds (.npy)")
    parser.add_argument("--unet-dir", type=Path, default=DEFAULT_UNET_DIR, help="Directory containing UNet predictions (.npy)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to cleaned wings manifest CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output destination CSV path")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of wings to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction_pipeline(
        wings_dir=args.wings_dir,
        seeds_dir=args.seeds_dir,
        unet_dir=args.unet_dir,
        manifest_path=args.manifest,
        output_csv=args.output,
        workers=args.workers,
        limit=args.limit,
    )
