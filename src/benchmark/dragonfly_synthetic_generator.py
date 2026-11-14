#!/usr/bin/env python3
"""
Procedural "Dragonfly-Grade" Synthetic Wing Generator with Exact Analytical Ground Truth.

Implements biologically faithful procedural wing synthesis:
1. Primary Longitudinal Veins (Bézier Splines):
   Costa (leading edge), Subcosta (terminating at nodus), Radius (R1/Rs to apex),
   Media, Cubitus, and Anal veins, with realistic width tapering (11px -> 3px).
2. Anisotropic Secondary Reticulation:
   Horizontally elongated Voronoi crossveins (Sx/Sy = 2.0) matching real odonate cell aspect ratios.
3. Key Anatomical Landmarks:
   - Nodus: Mid-wing leading edge inflection point.
   - Pterostigma: Dense, pigmented cuticular cell near the apex.
   - Wing Silhouettes: Forewing (slender) and Hindwing (expanded anal base).
4. Optical Image Formation:
   - Cylindrical chitin profile with Gaussian diffraction roll-off.
   - Cuticular membrane background with subtle illumination gradients.
   - Additive Gaussian sensor noise sweeps (sigma = 2.0, 10.0, 20.0).
5. Exact Analytical Ground Truth:
   Mathematical junction coordinates computed a priori from:
   - Interior trivalent Voronoi vertices
   - Analytical intersection of crossvein segments with primary splines
   - Spline bifurcation nodes
   All known to exact sub-pixel coordinates without any algorithmic labeling bias.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import Polygon, LineString, Point


def generate_realistic_dragonfly_wing(
    width: int = 1200,
    height: int = 480,
    wing_type: str = "hindwing",
    n_seeds: int = 340,
    anisotropy: float = 2.0,
    noise_sigma: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Synthesizes a realistic dragonfly wing image with exact analytical ground truth junctions.

    Returns:
        img: Grayscale uint8 image (H, W)
        gt_junctions: Exact analytical coordinates (N, 2) float32
        metadata: Dictionary containing anatomical landmark info and generation params
    """
    np.random.seed(seed)
    W, H = width, height

    # 1. Aerodynamic Wing Perimeter
    if wing_type.lower() == "hindwing":
        # Broader base and anal lobe
        ctrl_pts = np.array([
            [0.05 * W, 0.45 * H],
            [0.12 * W, 0.22 * H],
            [0.30 * W, 0.12 * H],
            [0.55 * W, 0.10 * H],  # Nodus
            [0.75 * W, 0.11 * H],
            [0.85 * W, 0.12 * H],  # Pterostigma start
            [0.93 * W, 0.14 * H],  # Pterostigma end
            [0.97 * W, 0.22 * H],  # Apex
            [0.92 * W, 0.45 * H],
            [0.75 * W, 0.68 * H],
            [0.50 * W, 0.82 * H],
            [0.25 * W, 0.86 * H],  # Expanded anal margin
            [0.10 * W, 0.75 * H],
            [0.05 * W, 0.45 * H],
        ], dtype=np.float32)
    else:
        # Forewing: Narrower, elongated chord
        ctrl_pts = np.array([
            [0.05 * W, 0.45 * H],
            [0.15 * W, 0.25 * H],
            [0.35 * W, 0.18 * H],
            [0.55 * W, 0.16 * H],  # Nodus
            [0.75 * W, 0.17 * H],
            [0.85 * W, 0.18 * H],  # Pterostigma start
            [0.93 * W, 0.20 * H],  # Pterostigma end
            [0.97 * W, 0.27 * H],  # Apex
            [0.90 * W, 0.45 * H],
            [0.75 * W, 0.58 * H],
            [0.50 * W, 0.68 * H],
            [0.25 * W, 0.70 * H],
            [0.10 * W, 0.62 * H],
            [0.05 * W, 0.45 * H],
        ], dtype=np.float32)

    tck, _ = splprep([ctrl_pts[:, 0], ctrl_pts[:, 1]], s=0, per=True)
    u_fine = np.linspace(0, 1, 400)
    x_fine, y_fine = splev(u_fine, tck)
    boundary_poly = Polygon(np.column_stack([x_fine, y_fine]))

    # 2. Primary Longitudinal Veins (Bézier Trunks with Continuous Decaying Caliber)
    nodus_x = 0.55 * W
    nodus_y = 0.10 * H if wing_type.lower() == "hindwing" else 0.16 * H
    apex_x = 0.97 * W
    apex_y = 0.22 * H if wing_type.lower() == "hindwing" else 0.27 * H

    spline_specs = [
        # Costa (Leading Edge)
        ("Costa", [[0.05 * W, 0.22 * W, nodus_x, 0.82 * W, apex_x],
                   [0.45 * H, 0.17 * H, nodus_y, 0.12 * H, apex_y]], 11.0, 4.0),
        # Subcosta (terminates cleanly at nodus)
        ("Subcosta", [[0.05 * W, 0.22 * W, 0.40 * W, nodus_x],
                      [0.46 * H, 0.22 * H, 0.16 * H, nodus_y]], 8.5, 4.0),
        # Radius R1 (runs to apex just below Costa)
        ("Radius_R1", [[0.05 * W, 0.25 * W, nodus_x, 0.80 * W, 0.95 * W],
                       [0.48 * H, 0.28 * H, 0.24 * H, 0.22 * H, 0.25 * H]], 9.5, 3.5),
        # Media (M)
        ("Media", [[0.10 * W, 0.30 * W, 0.60 * W, 0.88 * W],
                   [0.50 * H, 0.42 * H, 0.45 * H, 0.50 * H]], 7.5, 3.0),
        # Cubitus (Cu)
        ("Cubitus", [[0.08 * W, 0.25 * W, 0.50 * W, 0.70 * W],
                     [0.55 * H, 0.58 * H, 0.68 * H, 0.75 * H]], 7.0, 2.5),
        # Anal Vein (A)
        ("Anal", [[0.06 * W, 0.15 * W, 0.30 * W, 0.45 * W],
                  [0.52 * H, 0.62 * H, 0.78 * H, 0.83 * H]], 6.0, 2.5),
    ]

    spline_data = []
    for name, pts_def, w_base, w_tip in spline_specs:
        k_degree = min(3, len(pts_def[0]) - 1)
        tck_s, _ = splprep(pts_def, k=k_degree, s=0)
        u_s = np.linspace(0, 1, 300)
        xs, ys = splev(u_s, tck_s)
        spline_data.append((name, LineString(np.column_stack([xs, ys])), xs, ys, w_base, w_tip))

    # 3. Pterostigma Anatomical Landmark
    pt_y_top = 0.118 * H if wing_type.lower() == "hindwing" else 0.178 * H
    pt_poly = Polygon([
        [0.82 * W, pt_y_top],
        [0.91 * W, pt_y_top + 0.018 * H],
        [0.90 * W, pt_y_top + 0.075 * H],
        [0.81 * W, pt_y_top + 0.055 * H],
    ])

    # 4. Anisotropic Secondary Voronoi Crossveins
    minx, miny, maxx, maxy = boundary_poly.bounds
    seeds = []
    while len(seeds) < n_seeds:
        candidate_pts = np.random.uniform([minx, miny], [maxx, maxy], size=(n_seeds * 2, 2))
        for p in candidate_pts:
            if boundary_poly.contains(Point(p[0], p[1])):
                seeds.append(p)
                if len(seeds) == n_seeds:
                    break
    seeds = np.array(seeds)

    # Anisotropic scaling
    seeds_scaled = seeds.copy()
    seeds_scaled[:, 0] /= anisotropy
    vor = Voronoi(seeds_scaled)
    vertices = vor.vertices.copy()
    vertices[:, 0] *= anisotropy

    # 5. Exact Analytical Junction Extraction
    junction_candidates = []

    # A. Interior Voronoi trivalent nodes
    for v in vertices:
        pt = Point(v[0], v[1])
        if boundary_poly.contains(pt) and not pt_poly.contains(pt):
            junction_candidates.append(v)

    # B. Geometric intersection between Voronoi crossvein ridges and primary Bézier splines
    for p1_idx, p2_idx in vor.ridge_vertices:
        if p1_idx >= 0 and p2_idx >= 0:
            v1, v2 = vertices[p1_idx], vertices[p2_idx]
            ridge_seg = LineString([v1, v2])
            for _, s_line, _, _, _, _ in spline_data:
                inter = ridge_seg.intersection(s_line)
                if not inter.is_empty:
                    if inter.geom_type == "Point":
                        if boundary_poly.contains(inter) and not pt_poly.contains(inter):
                            junction_candidates.append([inter.x, inter.y])
                    elif inter.geom_type == "MultiPoint":
                        for pt in inter.geoms:
                            if boundary_poly.contains(pt) and not pt_poly.contains(pt):
                                junction_candidates.append([pt.x, pt.y])

    # C. Spline landmark bifurcations / contact nodes
    junction_candidates.append([nodus_x, nodus_y])  # Subcosta-Costa meeting at nodus
    junction_candidates.append([0.10 * W, 0.50 * H])  # Arculus Media origin
    junction_candidates.append([0.08 * W, 0.55 * H])  # Cubitus origin

    junction_candidates = np.array(junction_candidates, dtype=np.float32)

    # D. Cluster sub-pixel duplicate intersections (< 2.5 px tolerance)
    tree = cKDTree(junction_candidates)
    clusters = tree.query_ball_tree(tree, r=2.5)
    visited = set()
    analytical_gt_junctions = []
    for idx, nbrs in enumerate(clusters):
        if idx not in visited:
            pts = junction_candidates[nbrs]
            analytical_gt_junctions.append(np.mean(pts, axis=0))
            visited.update(nbrs)
    analytical_gt_junctions = np.array(analytical_gt_junctions, dtype=np.float32)

    # 6. Optical Image Rendering
    wing_mask = np.zeros((H, W), dtype=np.uint8)
    b_pts = np.column_stack([x_fine, y_fine]).astype(np.int32)
    cv2.fillPoly(wing_mask, [b_pts], 255)

    vein_canvas = np.zeros((H, W), dtype=np.uint8)

    # Render secondary crossvein ridges
    for p1_idx, p2_idx in vor.ridge_vertices:
        if p1_idx >= 0 and p2_idx >= 0:
            v1, v2 = vertices[p1_idx], vertices[p2_idx]
            pt1 = (int(round(v1[0])), int(round(v1[1])))
            pt2 = (int(round(v2[0])), int(round(v2[1])))
            if boundary_poly.contains(Point(v1[0], v1[1])) or boundary_poly.contains(Point(v2[0], v2[1])):
                cv2.line(vein_canvas, pt1, pt2, 225, thickness=2, lineType=cv2.LINE_AA)

    # Render primary splines with continuous decaying width
    for _, _, xs, ys, w_base, w_tip in spline_data:
        n_pts = len(xs)
        for i in range(n_pts - 1):
            t = i / float(n_pts - 1)
            w = int(round(w_base * (1.0 - t) + w_tip * t))
            pt1 = (int(round(xs[i])), int(round(ys[i])))
            pt2 = (int(round(xs[i + 1])), int(round(ys[i + 1])))
            cv2.line(vein_canvas, pt1, pt2, 255, thickness=max(1, w), lineType=cv2.LINE_AA)

    # Render pterostigma cuticular patch
    pt_coords = np.array(pt_poly.exterior.coords, dtype=np.int32)
    cv2.fillPoly(vein_canvas, [pt_coords], 245)

    # Clip all veins to wing interior
    vein_canvas = cv2.bitwise_and(vein_canvas, vein_canvas, mask=wing_mask)

    # Gaussian diffraction blur for cylindrical cross-section
    vein_blurred = cv2.GaussianBlur(vein_canvas.astype(np.float32), (5, 5), sigmaX=1.15)
    norm_veins = vein_blurred / 255.0

    # Cuticular membrane shading: inside wing is subtle gradient, outside is microscope slide background
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    membrane = np.where(wing_mask > 0, 238.0 + 4.0 * np.sin(x_coords * 0.01), 253.0)
    vein_darkness = 28.0

    synthetic_img = membrane * (1.0 - norm_veins) + vein_darkness * norm_veins

    # Sensor noise
    if noise_sigma > 0:
        sensor_noise = np.random.normal(0, noise_sigma, (H, W))
        synthetic_img = synthetic_img + sensor_noise

    synthetic_img = np.clip(synthetic_img, 0, 255).astype(np.uint8)

    metadata = {
        "width": W,
        "height": H,
        "wing_type": wing_type,
        "n_seeds": n_seeds,
        "anisotropy": anisotropy,
        "noise_sigma": noise_sigma,
        "seed": seed,
        "junction_count": len(analytical_gt_junctions),
        "nodus": [nodus_x, nodus_y],
        "apex": [apex_x, apex_y],
    }

    return synthetic_img, analytical_gt_junctions, metadata


if __name__ == "__main__":
    out_dir = Path("data/realistic_synthetic_wings")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating representative Realistic Synthetic Dragonfly Wings...")

    conditions = [
        ("clean_sigma_2", 2.0, "hindwing"),
        ("moderate_sigma_10", 10.0, "hindwing"),
        ("noisy_sigma_20", 20.0, "hindwing"),
        ("forewing_clean", 2.0, "forewing"),
    ]

    for name, noise, w_type in conditions:
        img, gt, meta = generate_realistic_dragonfly_wing(
            width=1200, height=480, wing_type=w_type, noise_sigma=noise, seed=42
        )
        img_p = out_dir / f"synthetic_{name}.png"
        cv2.imwrite(str(img_p), img)

        gt_df = pd.DataFrame(gt, columns=["x", "y"])
        gt_df.to_csv(out_dir / f"synthetic_{name}_gt_junctions.csv", index=False)

        # Overlay
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for j in gt:
            cv2.circle(overlay, (int(round(j[0])), int(round(j[1]))), 3, (0, 0, 255), -1)
        cv2.imwrite(str(out_dir / f"synthetic_{name}_overlay.png"), overlay)

        print(f"[{name}] {w_type.capitalize()} with {len(gt)} junctions saved to {img_p}")
