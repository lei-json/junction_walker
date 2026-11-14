#!/usr/bin/env python3
"""
Evaluate JunctionUNet Generalization Across Strict Held-Out Splits.

Addresses Weakness 2 (Unclear Train/Test Split):
Evaluates the model separately on:
1. Train Partition (In-Distribution seen specimens, N=308)
2. Validation Partition (N=66)
3. Held-Out Test Partition (Specimen-Grouped unseen dragonflies, N=69)
4. Out-of-Distribution Taxonomic Test Partition (Completely unseen genera, N=88)

Measures Precision, Recall, F1, Count Error, and the Generalization Gap.
"""

import json
from pathlib import Path
import time
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch

import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src/models"))
sys.path.insert(0, str(repo_root / "src/walkers"))

from model import JunctionUNet
from predict import predict_image, evaluate_against_gt


def evaluate_split(
    model: torch.nn.Module,
    device: torch.device,
    stems: List[str],
    wings_dir: Path,
    seeds_dir: Path,
    tolerance: float = 8.0,
    threshold: float = 0.40,
    filter_degree2: bool = True,
) -> Dict[str, Any]:
    records = []

    for stem in stems:
        candidates = list(wings_dir.glob(f"{stem}.*"))
        if not candidates:
            continue
        img = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
        seed_f = seeds_dir / f"{stem}.npy"
        if img is None or not seed_f.exists():
            continue
        gt = np.load(str(seed_f))

        _, unet_peaks = predict_image(
            model,
            img,
            device,
            threshold=threshold,
            snap=True,
            filter_degree2=filter_degree2,
            ridge_threshold=10,
        )

        p, r, f1 = evaluate_against_gt(unet_peaks, gt, tolerance=tolerance)
        count_err = abs(len(unet_peaks) - len(gt))

        records.append({
            "stem": stem,
            "gt_count": len(gt),
            "pred_count": len(unet_peaks),
            "count_error": count_err,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    df = pd.DataFrame(records)
    return {
        "count": len(df),
        "mean_precision": float(df["precision"].mean() * 100),
        "mean_recall": float(df["recall"].mean() * 100),
        "mean_f1": float(df["f1"].mean() * 100),
        "median_f1": float(df["f1"].median() * 100),
        "pass_85_pct": float(np.mean(df["f1"] >= 0.85) * 100),
        "mean_count_error": float(df["count_error"].mean()),
    }


def run_split_evaluation(
    checkpoint_path: str = "training_output/best_model.pt",
    output_json: str = "training_output/empirical/split_generalization_results.json",
):
    wings_dir = repo_root / "data/cleaned_wings"
    seeds_dir = repo_root / "data/cleaned_seeds"
    ckpt_path = repo_root / checkpoint_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    base_ch = ckpt.get("base_channels", 16)
    model = JunctionUNet(1, 1, base_ch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load splits
    with open(repo_root / "data/splits/specimen_split.json") as f:
        spec_split = json.load(f)
    with open(repo_root / "data/splits/taxonomic_split.json") as f:
        taxa_split = json.load(f)

    print("Evaluating Model Across Partitions...")

    train_res = evaluate_split(model, device, spec_split["train"], wings_dir, seeds_dir)
    print(f" Train Set (N={train_res['count']}): Mean F1 = {train_res['mean_f1']:.2f}% | Pass (>=85%) = {train_res['pass_85_pct']:.1f}%")

    val_res = evaluate_split(model, device, spec_split["val"], wings_dir, seeds_dir)
    print(f" Val Set (N={val_res['count']}): Mean F1 = {val_res['mean_f1']:.2f}% | Pass (>=85%) = {val_res['pass_85_pct']:.1f}%")

    test_res = evaluate_split(model, device, spec_split["test"], wings_dir, seeds_dir)
    print(f" Held-Out Test Set (N={test_res['count']}): Mean F1 = {test_res['mean_f1']:.2f}% | Pass (>=85%) = {test_res['pass_85_pct']:.1f}%")

    taxa_test_res = evaluate_split(model, device, taxa_split["test"], wings_dir, seeds_dir)
    print(f" Unseen Taxa Test (N={taxa_test_res['count']}): Mean F1 = {taxa_test_res['mean_f1']:.2f}% | Pass (>=85%) = {taxa_test_res['pass_85_pct']:.1f}%")

    gen_gap_specimen = train_res["mean_f1"] - test_res["mean_f1"]
    gen_gap_taxa = train_res["mean_f1"] - taxa_test_res["mean_f1"]

    summary = {
        "train": train_res,
        "val": val_res,
        "held_out_test_specimen": test_res,
        "unseen_taxa_test": taxa_test_res,
        "generalization_gap_specimen_pct": round(gen_gap_specimen, 2),
        "generalization_gap_taxa_pct": round(gen_gap_taxa, 2),
    }

    out_file = repo_root / output_json
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("             GENERALIZATION & LEAKAGE-FREE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Train Set Mean F1 (In-Distribution):       {train_res['mean_f1']:.2f}%")
    print(f"Held-Out Test Set Mean F1 (Unseen Specs):   {test_res['mean_f1']:.2f}%")
    print(f"Unseen Taxa Test Mean F1 (Zero-Shot Gen):  {taxa_test_res['mean_f1']:.2f}%")
    print(f"Generalization Gap (Train - Held-Out):      {gen_gap_specimen:+.2f}%")
    print(f"Generalization Gap (Train - Unseen Taxa):   {gen_gap_taxa:+.2f}%")
    print(f"Results archived to: {out_file}")


if __name__ == "__main__":
    run_split_evaluation()
