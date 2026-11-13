#!/usr/bin/env python3

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch

import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src/walkers"))
sys.path.insert(0, str(repo_root / "src/models"))

try:
    from model import JunctionUNet
    from predict import predict_image, evaluate_against_gt
except ImportError:
    from .model import JunctionUNet
    from .predict import predict_image, evaluate_against_gt

from topological_walker import TopologicalWalker


def evaluate_all(
    rot_dir="data/rotate",
    bw_dir="data/blackwhite_rotate",
    seed_dir="data/seeds",
    checkpoint_path="training_output/best_model.pt",
    output_csv="training_output/generalized_metrics.csv",
    tolerance=5.0,
    limit=None,
):
    repo_root = Path(__file__).resolve().parents[2]
    rot_dir = repo_root / rot_dir
    bw_dir = repo_root / bw_dir
    seed_dir = repo_root / seed_dir
    checkpoint_path = repo_root / checkpoint_path
    output_csv = repo_root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {checkpoint_path} on {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = checkpoint.get("base_channels", 16)
    model = JunctionUNet(1, 1, base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_wings = sorted(list(rot_dir.glob("*.*")))
    if limit is not None:
        all_wings = all_wings[:limit]

    print(f"Evaluating {len(all_wings)} wings...")

    records = []
    t_start = time.perf_counter()

    for i, img_path in enumerate(all_wings, 1):
        stem = img_path.stem
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        area_100k = (h * w) / 100000.0

        # Look for GT seed
        sf1 = seed_dir / f"{stem}.npy"
        sf2 = seed_dir / f"{stem}_blackwhite.npy"
        gt_seeds = None
        if sf1.exists():
            gt_seeds = np.load(str(sf1))
        elif sf2.exists():
            gt_seeds = np.load(str(sf2))

        if gt_seeds is not None and gt_seeds.ndim == 1:
            gt_seeds = gt_seeds.reshape(-1, 2)

        # UNet inference
        t0 = time.perf_counter()
        prob_map, unet_peaks = predict_image(model, img, device, threshold=0.40, snap=True)
        unet_time = time.perf_counter() - t0

        # Topological Walker inference (on BW if available, else img < 220)
        bw_cands = list(bw_dir.glob(f"{stem}*.*"))
        if bw_cands:
            bw_img = cv2.imread(str(bw_cands[0]), cv2.IMREAD_GRAYSCALE)
            mask = (bw_img < 128).astype(np.uint8)
        else:
            mask = (img < 220).astype(np.uint8)

        t0 = time.perf_counter()
        tw = TopologicalWalker(mask, min_component_size=3, spur_threshold=3)
        topo_peaks = tw.find_junctions()
        topo_time = time.perf_counter() - t0

        # Agreement
        agr_p, agr_r, agr_f1 = evaluate_against_gt(unet_peaks, topo_peaks, tolerance=tolerance)

        rec = {
            "index": i,
            "filename": img_path.name,
            "height": h,
            "width": w,
            "unet_time_sec": unet_time,
            "topo_time_sec": topo_time,
            "unet_count": len(unet_peaks),
            "topo_count": len(topo_peaks),
            "unet_density": len(unet_peaks) / area_100k,
            "topo_density": len(topo_peaks) / area_100k,
            "agreement_precision": agr_p,
            "agreement_recall": agr_r,
            "agreement_f1": agr_f1,
            "has_gt": gt_seeds is not None,
            "gt_count": len(gt_seeds) if gt_seeds is not None else np.nan,
            "unet_gt_precision": np.nan,
            "unet_gt_recall": np.nan,
            "unet_gt_f1": np.nan,
            "topo_gt_precision": np.nan,
            "topo_gt_recall": np.nan,
            "topo_gt_f1": np.nan,
        }

        if gt_seeds is not None:
            up, ur, uf = evaluate_against_gt(unet_peaks, gt_seeds, tolerance=tolerance)
            tp, tr, tf = evaluate_against_gt(topo_peaks, gt_seeds, tolerance=tolerance)
            rec["unet_gt_precision"] = up
            rec["unet_gt_recall"] = ur
            rec["unet_gt_f1"] = uf
            rec["topo_gt_precision"] = tp
            rec["topo_gt_recall"] = tr
            rec["topo_gt_f1"] = tf

        records.append(rec)
        if i % 25 == 0 or i == len(all_wings):
            print(f"[{i:03d}/{len(all_wings)}] {img_path.name:40s} | UNet: {len(unet_peaks):4d} | Topo: {len(topo_peaks):4d} | Agr F1: {agr_f1*100:.1f}%")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    elapsed = time.perf_counter() - t_start

    print("\n================ COMPLETED GENERALIZED EVALUATION ================")
    print(f"Total Wings Evaluated: {len(df)}")
    print(f"Total Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Mean UNet runtime / wing: {df['unet_time_sec'].mean():.3f}s")
    print(f"Mean Topo runtime / wing: {df['topo_time_sec'].mean():.3f}s")
    print(f"Mean Spatial Agreement F1: {df['agreement_f1'].mean()*100:.2f}%")

    gt_df = df[df["has_gt"]]
    if len(gt_df) > 0:
        print(f"\nGround Truth Subsample (N={len(gt_df)} wings):")
        print(f"  UNet Mean F1:  {gt_df['unet_gt_f1'].mean()*100:.2f}% (P: {gt_df['unet_gt_precision'].mean()*100:.2f}%, R: {gt_df['unet_gt_recall'].mean()*100:.2f}%)")
        print(f"  Topo Mean F1:  {gt_df['topo_gt_f1'].mean()*100:.2f}% (P: {gt_df['topo_gt_precision'].mean()*100:.2f}%, R: {gt_df['topo_gt_recall'].mean()*100:.2f}%)")

    print(f"Saved full metrics table to: {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of wings to evaluate")
    args = parser.parse_args()
    evaluate_all(limit=args.limit)
