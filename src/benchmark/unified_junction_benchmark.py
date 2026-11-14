#!/usr/bin/env python3
"""
Comprehensive Unified Benchmark Suite for JunctionUNet and Baseline Detectors.

Unified multi-tier evaluation hierarchy:
1. Tier 1: Abstract Geometric Planar Graph (Basic Voronoi Venation):
   - Uniform-width planar graph lattice with analytical mathematical ground truth
   - Noise sweeps: Clean (sigma=2), Moderate (sigma=10), Heavy Noise (sigma=20)
2. Tier 2: Realistic Dragonfly-Grade Synthetic Benchmark:
   - Parametric wing silhouettes with decaying primary splines (11px -> 3.5px)
   - Anisotropic Voronoi crossveins (Sx/Sy = 2.0), nodus, and pterostigma
   - Exact analytical closed-form intersections and Voronoi graph nodes
   - Noise sweeps: Clean (sigma=2), Moderate (sigma=10), Heavy Noise (sigma=20)
3. Tier 3: External Real Dragonfly Venation Benchmark (Hoffmann et al., PNAS 2018):
   - Real photographic dragonfly wings with 9,586 biological cell junctions
   - Multi-region evaluation (Center, Apex, Nodus, Anal Base)
4. Tier 4: External Real Retinal Vasculature Benchmark (DRIVE):
   - Clinical fundus photographs with dual-observer expert manual ground truth
5. Tier 5: External Real Leaf Venation Benchmark (HALVS):
   - Real plant leaf specimens with hierarchical vascular ground truth

Evaluates all 7 detectors:
- Harris Corner
- Shi-Tomasi
- FAST
- SIFT
- Classical Skeleton
- TopologicalWalker
- JunctionUNet (Raw)
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src/walkers"))
sys.path.insert(0, str(repo_root / "src/models"))
sys.path.insert(0, str(repo_root / "src/preprocessing"))
sys.path.insert(0, str(repo_root / "src/benchmark"))

from model import JunctionUNet
from baselines import (
    detect_harris,
    detect_shi_tomasi,
    detect_fast,
    detect_sift,
    detect_classical_skeleton,
    detect_topological_walker,
    detect_unet_raw,
    evaluate_predictions_multi_tolerance,
)
from synthetic_ground_truth_benchmark import generate_synthetic_wing_venation
from dragonfly_synthetic_generator import generate_realistic_dragonfly_wing
from external_datasets_loader import (
    load_pnas_dragonfly_benchmark,
    load_halvs_leaf_benchmark,
    load_drive_retina_benchmark,
)


def predict_unet_tiled(
    model: torch.nn.Module,
    img: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
    threshold: float = 0.35,
) -> np.ndarray:
    """Tiled UNet inference for large images, guaranteeing constant memory and seamless boundaries."""
    h, w = img.shape
    if h <= tile_size and w <= tile_size:
        return detect_unet_raw(model, img, device, threshold=threshold)

    stride = tile_size - overlap
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    y_starts = list(range(0, max(1, h - tile_size + stride), stride))
    x_starts = list(range(0, max(1, w - tile_size + stride), stride))

    for ys in y_starts:
        ye = min(h, ys + tile_size)
        ys_act = max(0, ye - tile_size)
        for xs in x_starts:
            xe = min(w, xs + tile_size)
            xs_act = max(0, xe - tile_size)

            patch = img[ys_act:ye, xs_act:xe]
            ph, pw = patch.shape
            pad_h = (16 - (ph % 16)) % 16
            pad_w = (16 - (pw % 16)) % 16
            if pad_h > 0 or pad_w > 0:
                padded = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="reflect")
            else:
                padded = patch

            inp = (
                torch.from_numpy(padded.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                out = torch.sigmoid(model(inp))[0, 0, :ph, :pw].cpu().numpy()

            prob_map[ys_act:ye, xs_act:xe] += out
            count_map[ys_act:ye, xs_act:xe] += 1.0

    prob_map = prob_map / np.maximum(count_map, 1.0)

    # Peak extraction
    t = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0)
    mp = F.max_pool2d(t, kernel_size=3, stride=1, padding=1)
    peaks = (t == mp) & (t > threshold)
    ys_p, xs_p = np.where(peaks[0, 0].numpy())
    if len(xs_p) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([xs_p, ys_p]).astype(np.int32)


def run_benchmark_on_sample(
    img: np.ndarray,
    gt_pts: np.ndarray,
    methods: Dict[str, Any],
    dataset_name: str,
    sample_id: str,
    condition: str = "standard",
) -> List[Dict[str, Any]]:
    """Runs all methods on a single image and evaluates against gt_pts."""
    records = []
    for name, fn in methods.items():
        t0 = time.perf_counter()
        try:
            preds = fn(img)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            met = evaluate_predictions_multi_tolerance(preds, gt_pts, [3.0, 5.0, 8.0])
        except Exception as e:
            print(f"Error in {name} on {dataset_name} ({sample_id}): {e}")
            preds = np.empty((0, 2), dtype=np.int32)
            latency_ms = np.nan
            met = {
                "p_3": 0.0, "r_3": 0.0, "f1_3": 0.0,
                "p_5": 0.0, "r_5": 0.0, "f1_5": 0.0,
                "p_8": 0.0, "r_8": 0.0, "f1_8": 0.0,
                "mle": np.nan,
            }

        rec = {
            "dataset": dataset_name,
            "sample_id": sample_id,
            "condition": condition,
            "method": name,
            "image_shape": f"{img.shape[0]}x{img.shape[1]}",
            "gt_count": len(gt_pts),
            "pred_count": len(preds),
            "latency_ms": latency_ms,
            "p_3px": met["p_3"],
            "r_3px": met["r_3"],
            "f1_3px": met["f1_3"],
            "p_5px": met["p_5"],
            "r_5px": met["r_5"],
            "f1_5px": met["f1_5"],
            "p_8px": met["p_8"],
            "r_8px": met["r_8"],
            "f1_8px": met["f1_8"],
            "mean_loc_error_px": met["mle"],
        }
        records.append(rec)
    return records


def run_unified_benchmark(
    checkpoint_path: str = "training_output/best_model.pt",
    output_dir: str = "training_output/empirical",
    n_synthetic_trials: int = 5,
):
    out_p = repo_root / output_dir
    out_p.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Unified Junction Benchmark on device: {device}")

    # Load UNet
    ckpt_p = repo_root / checkpoint_path
    ckpt = torch.load(ckpt_p, map_location=device)
    base_ch = ckpt.get("base_channels", 16)
    unet = JunctionUNet(1, 1, base_ch).to(device)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    methods = {
        "Harris Corner": detect_harris,
        "Shi-Tomasi": detect_shi_tomasi,
        "FAST": detect_fast,
        "SIFT": detect_sift,
        "Classical Skeleton": detect_classical_skeleton,
        "TopologicalWalker": detect_topological_walker,
        "JunctionUNet (Raw)": lambda im: predict_unet_tiled(unet, im, device, threshold=0.35),
    }

    all_records = []

    # =========================================================================
    # 1. ABSTRACT GEOMETRIC PLANAR GRAPH (Basic Voronoi Venation)
    # =========================================================================
    print("\n--- Benchmark 1: Abstract Geometric Planar Graph (Basic Voronoi Venation) ---")
    noise_levels = [2.0, 10.0, 20.0]
    for noise in noise_levels:
        cond_label = f"sigma_{int(noise)}"
        for trial in range(n_synthetic_trials):
            seed = 1000 + int(noise * 10) + trial
            img_v, gt_v = generate_synthetic_wing_venation(
                width=800,
                height=450,
                n_seeds=130,
                vein_thickness=3,
                noise_sigma=noise,
                seed=seed,
            )
            recs = run_benchmark_on_sample(
                img_v, gt_v, methods,
                dataset_name="Abstract Geometric Voronoi",
                sample_id=f"voronoi_trial{trial}",
                condition=cond_label,
            )
            all_records.extend(recs)
            print(f"  [Voronoi {cond_label} trial {trial+1}/{n_synthetic_trials}] GT={len(gt_v)} evaluated.")

    # =========================================================================
    # 2. REALISTIC DRAGONFLY-GRADE SYNTHETIC BENCHMARK
    # =========================================================================
    print("\n--- Benchmark 2: Realistic Dragonfly-Grade Synthetic Wing Benchmark ---")
    for noise in noise_levels:
        cond_label = f"sigma_{int(noise)}"
        for trial in range(n_synthetic_trials):
            seed = 500 + int(noise * 10) + trial
            w_type = "hindwing" if trial % 2 == 0 else "forewing"
            img, gt, meta = generate_realistic_dragonfly_wing(
                width=1000,
                height=400,
                wing_type=w_type,
                n_seeds=280,
                noise_sigma=noise,
                seed=seed,
            )
            recs = run_benchmark_on_sample(
                img, gt, methods,
                dataset_name="Realistic Synthetic Dragonfly",
                sample_id=f"{w_type}_trial{trial}",
                condition=cond_label,
            )
            all_records.extend(recs)
            print(f"  [Synthetic {cond_label} trial {trial+1}/{n_synthetic_trials}] {w_type} GT={len(gt)} evaluated.")

    # =========================================================================
    # 3. REAL DRAGONFLY WING BENCHMARK (Hoffmann et al., PNAS 2018)
    # =========================================================================
    print("\n--- Benchmark 3: Real Dragonfly Wing Venation (Hoffmann et al., PNAS) ---")
    pnas_regions = [
        ("Center Wing", (400, 800, 1000, 1600)),
        ("Wing Apex", (300, 700, 2000, 2600)),
        ("Nodus Region", (200, 600, 1400, 2000)),
        ("Anal Base", (500, 900, 400, 1000)),
    ]
    for reg_name, box in pnas_regions:
        img_p, gt_p, meta_p = load_pnas_dragonfly_benchmark(crop_box=box)
        recs = run_benchmark_on_sample(
            img_p, gt_p, methods,
            dataset_name="PNAS Dragonfly Wing",
            sample_id=reg_name,
            condition="real_specimen",
        )
        all_records.extend(recs)
        print(f"  [PNAS {reg_name}] GT={len(gt_p)} evaluated.")

    # =========================================================================
    # 4. REAL RETINAL VASCULATURE BENCHMARK (DRIVE)
    # =========================================================================
    print("\n--- Benchmark 4: Real Retinal Vasculature (DRIVE Clinical Fundus) ---")
    for i in range(1, 6):
        img_d, gt_d, meta_d = load_drive_retina_benchmark(index=i)
        recs = run_benchmark_on_sample(
            img_d, gt_d, methods,
            dataset_name="DRIVE Retinal Vasculature",
            sample_id=f"fundus_{i:02d}",
            condition="clinical_fundus",
        )
        all_records.extend(recs)
        print(f"  [DRIVE Fundus {i:02d}] GT={len(gt_d)} evaluated.")

    # =========================================================================
    # 5. REAL LEAF VENATION BENCHMARK (HALVS)
    # =========================================================================
    print("\n--- Benchmark 5: Real Leaf Venation (HALVS Plant Science) ---")
    leaf_regions = [
        ("Midrib & Secondary Veins", (600, 1200, 400, 1000)),
        ("Laminar Crossvein Network", (1400, 2000, 500, 1100)),
    ]
    for reg_name, box in leaf_regions:
        img_l, gt_l, meta_l = load_halvs_leaf_benchmark(crop_box=box)
        recs = run_benchmark_on_sample(
            img_l, gt_l, methods,
            dataset_name="HALVS Leaf Venation",
            sample_id=reg_name,
            condition="real_cleared_leaf",
        )
        all_records.extend(recs)
        print(f"  [HALVS {reg_name}] GT={len(gt_l)} evaluated.")

    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    csv_path = out_p / "unified_junction_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed per-sample benchmark results written to: {csv_path}")

    # Also sync synthetic_ground_truth_benchmark.csv for the Voronoi subset
    voronoi_sub = df[df["dataset"] == "Abstract Geometric Voronoi"].copy()
    if len(voronoi_sub) > 0:
        voronoi_csv_path = out_p / "synthetic_ground_truth_benchmark.csv"
        voronoi_sub["trial"] = voronoi_sub["sample_id"].apply(lambda s: int(s.replace("voronoi_trial", "")))
        voronoi_sub["noise_sigma"] = voronoi_sub["condition"].apply(lambda c: float(c.replace("sigma_", "")))
        voronoi_cols = [
            "trial", "noise_sigma", "method", "gt_count", "pred_count", "latency_ms",
            "p_3px", "r_3px", "f1_3px", "p_5px", "r_5px", "f1_5px", "p_8px", "r_8px", "f1_8px", "mean_loc_error_px"
        ]
        voronoi_sub[voronoi_cols].to_csv(voronoi_csv_path, index=False)
        print(f"Synchronized analytical Voronoi benchmark CSV: {voronoi_csv_path}")

    # Summary table per dataset & condition & method
    summary = df.groupby(["dataset", "condition", "method"]).agg({
        "p_3px": "mean",
        "r_3px": "mean",
        "f1_3px": "mean",
        "p_5px": "mean",
        "r_5px": "mean",
        "f1_5px": "mean",
        "p_8px": "mean",
        "r_8px": "mean",
        "f1_8px": "mean",
        "mean_loc_error_px": "mean",
        "latency_ms": "mean",
    }).reset_index()

    summary_csv = out_p / "unified_benchmark_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 105)
    print("                      UNIFIED JUNCTION DETECTION BENCHMARK SUMMARY (ALL DATASETS)")
    print("=" * 105)
    disp = summary.copy()
    disp["P@5px"] = disp["p_5px"].apply(lambda x: f"{x*100:.1f}%")
    disp["R@5px"] = disp["r_5px"].apply(lambda x: f"{x*100:.1f}%")
    disp["F1@5px"] = disp["f1_5px"].apply(lambda x: f"{x*100:.1f}%")
    disp["F1@8px"] = disp["f1_8px"].apply(lambda x: f"{x*100:.1f}%")
    disp["MLE"] = disp["mean_loc_error_px"].apply(lambda x: f"{x:.2f} px")
    disp["Latency"] = disp["latency_ms"].apply(lambda x: f"{x:.1f} ms")
    cols = ["dataset", "condition", "method", "P@5px", "R@5px", "F1@5px", "F1@8px", "MLE", "Latency"]
    print(disp[cols].to_string(index=False))

    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-trials", type=int, default=3, help="Trials per synthetic noise level")
    args = parser.parse_args()
    run_unified_benchmark(n_synthetic_trials=args.synthetic_trials)
