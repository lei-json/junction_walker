#!/usr/bin/env python3
"""
Tolerance Sensitivity & Sub-Vein Localization Precision Sweep.

Addresses Weakness 3:
Sweeps tolerance radius tau from 1.0 px to 15.0 px.
Computes the exact continuous Euclidean localization error distribution
(mean, median, IQR, 90th percentile) to determine whether detections are
accurate at sub-vein scale or relying on loose tolerance slack.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src/walkers"))
sys.path.insert(0, str(repo_root / "src/models"))
sys.path.insert(0, str(repo_root / "src/preprocessing"))
sys.path.insert(0, str(repo_root / "src/benchmark"))

from model import JunctionUNet
from baselines import detect_unet_raw, detect_topological_walker


TOLERANCES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]


def evaluate_tolerance_curve(pred_coords: np.ndarray, gt_coords: np.ndarray) -> Dict[str, Any]:
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return {f"f1_tau_{t:.1f}": 0.0 for t in TOLERANCES}

    tree_gt = cKDTree(gt_coords)
    tree_pred = cKDTree(pred_coords)

    dists_pred, _ = tree_gt.query(pred_coords)
    dists_gt, _ = tree_pred.query(gt_coords)

    res = {}
    for t in TOLERANCES:
        p = float(np.sum(dists_pred <= t)) / len(pred_coords)
        r = float(np.sum(dists_gt <= t)) / len(gt_coords)
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        res[f"p_tau_{t:.1f}"] = p
        res[f"r_tau_{t:.1f}"] = r
        res[f"f1_tau_{t:.1f}"] = f1

    # Exact error distribution for matched true positives within 8 px
    matched = dists_pred[dists_pred <= 8.0]
    if len(matched) > 0:
        res["loc_error_mean"] = float(np.mean(matched))
        res["loc_error_std"] = float(np.std(matched))
        res["loc_error_median"] = float(np.median(matched))
        res["loc_error_q25"] = float(np.percentile(matched, 25))
        res["loc_error_q75"] = float(np.percentile(matched, 75))
        res["loc_error_p90"] = float(np.percentile(matched, 90))
        res["pct_within_1px"] = float(np.mean(dists_pred <= 1.0) * 100)
        res["pct_within_2px"] = float(np.mean(dists_pred <= 2.0) * 100)
        res["pct_within_3px"] = float(np.mean(dists_pred <= 3.0) * 100)
    else:
        res["loc_error_mean"] = np.nan
        res["loc_error_std"] = np.nan
        res["loc_error_median"] = np.nan
        res["loc_error_q25"] = np.nan
        res["loc_error_q75"] = np.nan
        res["loc_error_p90"] = np.nan
        res["pct_within_1px"] = 0.0
        res["pct_within_2px"] = 0.0
        res["pct_within_3px"] = 0.0

    return res


def run_tolerance_sweep(
    split_file: str = "data/splits/specimen_split.json",
    wings_dir: str = "data/cleaned_wings",
    seeds_dir: str = "data/cleaned_seeds",
    checkpoint_path: str = "training_output/best_model.pt",
    output_dir: str = "training_output/empirical",
    limit: int = None,
):
    out_dir = repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(repo_root / split_file) as f:
        split_data = json.load(f)

    test_stems = split_data["test"]
    if limit is not None:
        test_stems = test_stems[:limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(repo_root / checkpoint_path, map_location=device)
    base_ch = ckpt.get("base_channels", 16)
    unet = JunctionUNet(1, 1, base_ch).to(device)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    wings_p = repo_root / wings_dir
    seeds_p = repo_root / seeds_dir

    records = []
    print(f"Executing Tolerance Sweep across {len(test_stems)} Held-Out Test Wings...")

    for i, stem in enumerate(test_stems, 1):
        candidates = list(wings_p.glob(f"{stem}.*"))
        if not candidates:
            continue
        img = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
        gt_file = seeds_p / f"{stem}.npy"
        if not gt_file.exists() or img is None:
            continue
        gt = np.load(str(gt_file))

        # Evaluate Raw JunctionUNet
        raw_unet_preds = detect_unet_raw(unet, img, device, threshold=0.35)
        unet_met = evaluate_tolerance_curve(raw_unet_preds, gt)
        unet_met["stem"] = stem
        unet_met["model"] = "JunctionUNet (Raw)"
        unet_met["num_preds"] = len(raw_unet_preds)
        unet_met["num_gt"] = len(gt)
        records.append(unet_met)

        # Evaluate TopologicalWalker
        tw_preds = detect_topological_walker(img)
        tw_met = evaluate_tolerance_curve(tw_preds, gt)
        tw_met["stem"] = stem
        tw_met["model"] = "TopologicalWalker"
        tw_met["num_preds"] = len(tw_preds)
        tw_met["num_gt"] = len(gt)
        records.append(tw_met)

        if i % 10 == 0 or i == len(test_stems):
            print(f"[{i:02d}/{len(test_stems)}] {stem:35s} | UNet Med Err: {unet_met['loc_error_median']:.2f}px | TW Med Err: {tw_met['loc_error_median']:.2f}px")

    df = pd.DataFrame(records)
    csv_path = out_dir / "tolerance_sweep_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("               TOLERANCE SWEEP & LOCALIZATION ERROR SUMMARY")
    print("=" * 80)
    for model_name, grp in df.groupby("model"):
        print(f"\n--- Model: {model_name} ---")
        print(f"  Mean Localization Error:    {grp['loc_error_mean'].mean():.2f} ± {grp['loc_error_std'].mean():.2f} px")
        print(f"  Median Localization Error:  {grp['loc_error_median'].mean():.2f} px")
        print(f"  90th Percentile Error:      {grp['loc_error_p90'].mean():.2f} px")
        print(f"  Detections within 1.0 px:   {grp['pct_within_1px'].mean():.1f}%")
        print(f"  Detections within 2.0 px:   {grp['pct_within_2px'].mean():.1f}%")
        print(f"  Detections within 3.0 px:   {grp['pct_within_3px'].mean():.1f}%")
        print("  F1 Score by Tolerance:")
        for t in [1.0, 2.0, 3.0, 4.0, 5.0, 8.0]:
            print(f"    tau = {t:4.1f} px: F1 = {grp[f'f1_tau_{t:.1f}'].mean()*100:.2f}% (P={grp[f'p_tau_{t:.1f}'].mean()*100:.1f}%, R={grp[f'r_tau_{t:.1f}'].mean()*100:.1f}%)")

    print(f"\nFull tolerance results saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_tolerance_sweep(limit=args.limit)
