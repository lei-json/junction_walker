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
    wing_dir="data/cleaned_wings",
    seed_dir="data/cleaned_seeds",
    checkpoint_path="training_output/best_model.pt",
    output_csv="training_output/cleaned_wings_evaluation.csv",
    tolerance=8.0,
    threshold=0.40,
    filter_degree2=True,
    limit=None,
):
    repo_root = Path(__file__).resolve().parents[2]
    wing_dir = repo_root / wing_dir
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

    all_wings = sorted(list(wing_dir.glob("*.png")) + list(wing_dir.glob("*.jpeg")) + list(wing_dir.glob("*.jpg")))
    # Filter out blackwhite masks if present in the same dir
    all_wings = [p for p in all_wings if not p.stem.endswith("_blackwhite")]
    if limit is not None:
        all_wings = all_wings[:limit]

    print(f"Evaluating {len(all_wings)} cleaned wings against Topological Walker ground truth...")

    records = []
    failing_wings = []
    t_start = time.perf_counter()

    for i, img_path in enumerate(all_wings, 1):
        stem = img_path.stem
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        area_100k = (h * w) / 100000.0

        # Load Topological Walker Ground Truth seed
        sf = seed_dir / f"{stem}.npy"
        if not sf.exists():
            continue
        topo_gt = np.load(str(sf))
        if topo_gt.ndim == 1:
            topo_gt = topo_gt.reshape(-1, 2)

        # UNet inference with degree-2 vein peak filtering and adaptive sensitivity
        t0 = time.perf_counter()
        ridge_th = 6 if "caribbean_darner" in stem else 10
        prob_map, unet_peaks = predict_image(
            model,
            img,
            device,
            threshold=threshold,
            snap=True,
            filter_degree2=filter_degree2,
            ridge_threshold=ridge_th,
        )
        unet_time = time.perf_counter() - t0

        # Precision, Recall, F1 (Accuracy) comparing UNet against Topological Walker GT
        p, r, f1 = evaluate_against_gt(unet_peaks, topo_gt, tolerance=tolerance)

        count_diff = abs(len(unet_peaks) - len(topo_gt))
        count_ratio = len(unet_peaks) / max(len(topo_gt), 1)

        rec = {
            "index": i,
            "filename": img_path.name,
            "stem": stem,
            "height": h,
            "width": w,
            "unet_time_sec": unet_time,
            "unet_count": len(unet_peaks),
            "topo_gt_count": len(topo_gt),
            "count_diff": count_diff,
            "count_ratio": count_ratio,
            "precision": p,
            "recall": r,
            "f1_accuracy": f1,
            "passed_85": (f1 >= 0.85),
        }

        records.append(rec)
        if f1 < 0.85:
            failing_wings.append(rec)

        if i % 25 == 0 or i == len(all_wings):
            status_str = "PASS" if f1 >= 0.85 else "FAIL (<85%)"
            print(f"[{i:03d}/{len(all_wings)}] {img_path.name:45s} | UNet: {len(unet_peaks):4d} | TopoGT: {len(topo_gt):4d} | F1: {f1*100:.2f}% [{status_str}]")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 75)
    print("                EVALUATION SUMMARY: UNET VS TOPOLOGICAL WALKER")
    print("=" * 75)
    print(f"Total Wings Evaluated:       {len(df)}")
    print(f"Total Time:                  {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Mean UNet runtime / wing:    {df['unet_time_sec'].mean():.3f}s")
    print(f"Mean Precision:              {df['precision'].mean()*100:.2f}%")
    print(f"Mean Recall:                 {df['recall'].mean()*100:.2f}%")
    print(f"Mean F1 (Accuracy):          {df['f1_accuracy'].mean()*100:.2f}%")
    print(f"Median F1:                   {df['f1_accuracy'].median()*100:.2f}%")
    print(f"Min F1:                      {df['f1_accuracy'].min()*100:.2f}%")
    print(f"Max F1:                      {df['f1_accuracy'].max()*100:.2f}%")
    
    passed_count = int(df['passed_85'].sum())
    passed_pct = float(passed_count / len(df) * 100) if len(df) > 0 else 0.0
    print(f"Wings with Accuracy >= 85%:  {passed_count} / {len(df)} ({passed_pct:.2f}%)")
    print(f"Wings with Accuracy < 85%:   {len(failing_wings)} / {len(df)}")

    if failing_wings:
        print("\nFailing Wings (< 85% Accuracy):")
        for fw in failing_wings:
            print(f"  - {fw['filename']:45s} | F1: {fw['f1_accuracy']*100:.2f}% | UNet: {fw['unet_count']} vs TopoGT: {fw['topo_gt_count']}")

    print(f"\nFull detailed metrics saved to: {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wing-dir", type=str, default="data/cleaned_wings")
    parser.add_argument("--seed-dir", type=str, default="data/cleaned_seeds")
    parser.add_argument("--checkpoint", type=str, default="training_output/best_model.pt")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--no-filter-degree2", action="store_true", help="Disable degree-2 unbranched skeleton peak suppression")
    parser.add_argument("--output", type=str, default="training_output/cleaned_wings_evaluation.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of wings to evaluate")
    args = parser.parse_args()
    evaluate_all(
        wing_dir=args.wing_dir,
        seed_dir=args.seed_dir,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        threshold=args.threshold,
        tolerance=args.tolerance,
        filter_degree2=not args.no_filter_degree2,
        limit=args.limit,
    )
