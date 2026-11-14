#!/usr/bin/env python3
"""
Empirical Stress Test on 66 Excluded Pigmented / Mottled Dragonfly Wings.

Addresses Weakness 3:
Tests TopologicalWalker and JunctionUNet on the 66 wings excluded from the curated
dataset due to heavy pigment patches, amber banding, or dark wing spots.
Quantifies failure modes, false-positive density inside pigmented vs clear regions,
and establishes the precise operational limits of both algorithms.
"""

import argparse
from pathlib import Path
import time
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import torch

import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src/walkers"))
sys.path.insert(0, str(repo_root / "src/models"))
sys.path.insert(0, str(repo_root / "src/preprocessing"))
sys.path.insert(0, str(repo_root / "src/benchmark"))

from model import JunctionUNet
from baselines import detect_unet_raw, detect_topological_walker
from cleaning import remove_body_remnants


def run_pigmented_stress_test(
    manifest_csv: str = "data/cleaned_wings_manifest.csv",
    raw_dir: str = "data/raw",
    checkpoint_path: str = "training_output/best_model.pt",
    output_dir: str = "training_output/empirical",
    limit: int = None,
):
    out_dir = repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(repo_root / manifest_csv)
    excl_df = manifest_df[manifest_df["status"] == "excluded_colored_spots"]
    raw_p = repo_root / raw_dir

    if limit is not None:
        excl_df = excl_df.iloc[:limit]

    print(f"Testing on {len(excl_df)} Excluded Pigmented / Mottled Wing Specimens...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(repo_root / checkpoint_path, map_location=device)
    base_ch = ckpt.get("base_channels", 16)
    unet = JunctionUNet(1, 1, base_ch).to(device)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    records = []

    for i, (_, row) in enumerate(excl_df.iterrows(), 1):
        fn = row["filename"]
        img_path = raw_p / fn
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        # Clean wing-base remnants
        cleaned_img, _, _ = remove_body_remnants(img)

        # Detect junctions using Topological Walker
        t0 = time.perf_counter()
        tw_preds = detect_topological_walker(cleaned_img)
        tw_time = time.perf_counter() - t0

        # Detect junctions using Raw JunctionUNet
        t0 = time.perf_counter()
        unet_preds = detect_unet_raw(unet, cleaned_img, device, threshold=0.35)
        unet_time = time.perf_counter() - t0

        # Measure dark pigmentation fraction (< 100 intensity)
        dark_mask = (cleaned_img < 110).astype(np.uint8)
        dark_fraction = float(np.mean(dark_mask) * 100)

        # How many predicted junctions land in solid dark pigment blocks?
        # A solid dark block has low local variance or large connected dark component (> 500 px)
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        pigment_block_mask = np.zeros_like(dark_mask)
        for label_idx in range(1, n_cc):
            if stats[label_idx, cv2.CC_STAT_AREA] > 600:  # Non-vein solid pigment patch
                pigment_block_mask[labels == label_idx] = 1

        # Fraction of detections that fall inside solid pigment patches (false positives)
        tw_fp_pigment = 0
        if len(tw_preds) > 0:
            for x, y in tw_preds:
                if 0 <= y < h and 0 <= x < w:
                    if pigment_block_mask[y, x]:
                        tw_fp_pigment += 1

        unet_fp_pigment = 0
        if len(unet_preds) > 0:
            for x, y in unet_preds:
                if 0 <= y < h and 0 <= x < w:
                    if pigment_block_mask[y, x]:
                        unet_fp_pigment += 1

        tw_fp_rate = (tw_fp_pigment / len(tw_preds) * 100) if len(tw_preds) > 0 else 0.0
        unet_fp_rate = (unet_fp_pigment / len(unet_preds) * 100) if len(unet_preds) > 0 else 0.0

        rec = {
            "filename": fn,
            "width": w,
            "height": h,
            "dark_fraction_pct": dark_fraction,
            "tw_junction_count": len(tw_preds),
            "tw_fp_in_pigment": tw_fp_pigment,
            "tw_pigment_fp_rate_pct": tw_fp_rate,
            "tw_latency_sec": tw_time,
            "unet_junction_count": len(unet_preds),
            "unet_fp_in_pigment": unet_fp_pigment,
            "unet_pigment_fp_rate_pct": unet_fp_rate,
            "unet_latency_sec": unet_time,
        }
        records.append(rec)

        if i % 10 == 0 or i == len(excl_df):
            print(f"[{i:02d}/{len(excl_df)}] {fn:38s} | Dark%: {dark_fraction:4.1f}% | TW Pts: {len(tw_preds):4d} (FP in Pigment: {tw_fp_rate:4.1f}%) | UNet Pts: {len(unet_preds):4d} (FP in Pigment: {unet_fp_rate:4.1f}%)")

    df = pd.DataFrame(records)
    csv_path = out_dir / "pigmented_wings_stress_test.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("             PIGMENTED WINGS STRESS TEST FINDINGS")
    print("=" * 80)
    print(f"Total Pigmented Specimens Evaluated:  {len(df)}")
    print(f"Mean Pigmentation Coverage:           {df['dark_fraction_pct'].mean():.2f}%")
    print(f"TopologicalWalker Mean Junctions:     {df['tw_junction_count'].mean():.1f}")
    print(f"TopologicalWalker Pigment False Pos:  {df['tw_pigment_fp_rate_pct'].mean():.2f}%")
    print(f"JunctionUNet Mean Junctions:          {df['unet_junction_count'].mean():.1f}")
    print(f"JunctionUNet Pigment False Pos:       {df['unet_pigment_fp_rate_pct'].mean():.2f}%")
    print(f"\nDetailed stress test results saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_pigmented_stress_test(limit=args.limit)
