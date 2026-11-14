#!/usr/bin/env python3
"""
Benchmark Classical and Deep Keypoint Detectors on Dragonfly Wing Venation.

Compares:
1. Harris Corner Detector
2. Shi-Tomasi (Good Features to Track)
3. FAST Corner Detector
4. SIFT Keypoint Detector
5. Classical Morphological Skeleton Crossing Number Detector
6. TopologicalWalker (Proposed Graph Walker)
7. JunctionUNet (Raw - Direct Heatmap Peaks without Topological Post-Filtering)
8. JunctionUNet (Hybrid - Heatmap + Topological Branch Verification)
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import torch
import torch.nn.functional as F

import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src/walkers"))
sys.path.insert(0, str(repo_root / "src/models"))
sys.path.insert(0, str(repo_root / "src/preprocessing"))

from model import JunctionUNet
from topological_walker import TopologicalWalker
from cleaning import extract_veins_tophat


# ============================================================
# EVALUATION METRIC HELPER
# ============================================================

def evaluate_predictions_multi_tolerance(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    tolerances: List[float] = [3.0, 5.0, 8.0],
) -> Dict[str, float]:
    """
    Computes precision, recall, f1 at multiple tolerances and mean localization error (MLE).
    """
    metrics = {}
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        for tol in tolerances:
            metrics[f"p_{tol:.0f}"] = 1.0
            metrics[f"r_{tol:.0f}"] = 1.0
            metrics[f"f1_{tol:.0f}"] = 1.0
        metrics["mle"] = 0.0
        return metrics

    if len(pred_pts) == 0:
        for tol in tolerances:
            metrics[f"p_{tol:.0f}"] = 0.0
            metrics[f"r_{tol:.0f}"] = 0.0
            metrics[f"f1_{tol:.0f}"] = 0.0
        metrics["mle"] = np.nan
        return metrics

    if len(gt_pts) == 0:
        for tol in tolerances:
            metrics[f"p_{tol:.0f}"] = 0.0
            metrics[f"r_{tol:.0f}"] = 1.0
            metrics[f"f1_{tol:.0f}"] = 0.0
        metrics["mle"] = np.nan
        return metrics

    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    dists_pred, _ = tree_gt.query(pred_pts)
    dists_gt, _ = tree_pred.query(gt_pts)

    for tol in tolerances:
        tp_pred = np.sum(dists_pred <= tol)
        p = float(tp_pred) / len(pred_pts)

        tp_gt = np.sum(dists_gt <= tol)
        r = float(tp_gt) / len(gt_pts)

        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

        metrics[f"p_{tol:.0f}"] = p
        metrics[f"r_{tol:.0f}"] = r
        metrics[f"f1_{tol:.0f}"] = f1

    # Mean localization error for matches within maximum tolerance
    max_tol = max(tolerances)
    matched_dists = dists_pred[dists_pred <= max_tol]
    metrics["mle"] = float(np.mean(matched_dists)) if len(matched_dists) > 0 else np.nan

    return metrics


# ============================================================
# BASELINE DETECTORS
# ============================================================

def detect_harris(img: np.ndarray, max_corners: int = 1500) -> np.ndarray:
    """Harris Corner Detector."""
    gray = img.copy()
    dst = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    th = 0.01 * dst.max()
    ys, xs = np.where(dst > th)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    # Sort by strength and take top max_corners
    strengths = dst[ys, xs]
    idx = np.argsort(-strengths)[:max_corners]
    return np.column_stack([xs[idx], ys[idx]]).astype(np.int32)


def detect_shi_tomasi(img: np.ndarray, max_corners: int = 1500) -> np.ndarray:
    """Shi-Tomasi / Good Features to Track."""
    corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=3,
    )
    if corners is None or len(corners) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return corners.reshape(-1, 2).astype(np.int32)


def detect_fast(img: np.ndarray, threshold: int = 20) -> np.ndarray:
    """FAST Corner Detector."""
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
    kps = fast.detect(img, None)
    if not kps:
        return np.empty((0, 2), dtype=np.int32)
    pts = np.array([kp.pt for kp in kps], dtype=np.int32)
    return pts


def detect_sift(img: np.ndarray, max_features: int = 1500) -> np.ndarray:
    """SIFT Keypoint Detector."""
    sift = cv2.SIFT_create(nfeatures=max_features)
    kps = sift.detect(img, None)
    if not kps:
        return np.empty((0, 2), dtype=np.int32)
    pts = np.array([kp.pt for kp in kps], dtype=np.int32)
    return pts


def detect_classical_skeleton(img: np.ndarray) -> np.ndarray:
    """
    Standard Classical Skeleton Crossing-Number Detector.
    Uses Otsu thresholding + standard Zhang-Suen / Medial skeletonization + crossing number >= 3.
    (Without TopologicalWalker's custom top-hat or morphological spur pruning).
    """
    # Otsu thresholding on grayscale
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    skel = skeletonize(binary > 0).astype(np.uint8)
    
    # Rutovitz Crossing Number on 8-neighborhood
    p = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    pad = np.pad(skel, 1, mode="constant")
    h, w = skel.shape
    nbrs = [pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in p]

    crossing_num = np.zeros_like(skel, dtype=np.float32)
    for i in range(8):
        crossing_num += np.abs(nbrs[i].astype(np.float32) - nbrs[(i + 1) % 8].astype(np.float32))
    crossing_num = (0.5 * crossing_num) * skel

    branch_pts = np.argwhere(crossing_num >= 3)
    if len(branch_pts) == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    # Simple cluster contraction via connected components
    cand = (crossing_num >= 3).astype(np.uint8)
    n_cc, labels, _, centroids = cv2.connectedComponentsWithStats(cand, connectivity=8)
    junctions = []
    for c in range(1, n_cc):
        cx, cy = centroids[c]
        junctions.append((int(round(cx)), int(round(cy))))
    return np.array(junctions, dtype=np.int32)


def detect_topological_walker(img: np.ndarray) -> np.ndarray:
    """Proposed Topological Walker with Top-Hat Filtering."""
    bw_top, _ = extract_veins_tophat(
        img, kernel_size=11, ridge_threshold=10, background_threshold=240
    )
    vein_mask = (bw_top == 0).astype(np.uint8)
    walker = TopologicalWalker(vein_mask, min_component_size=5, spur_threshold=3)
    return walker.find_junctions()


def detect_unet_raw(
    model: torch.nn.Module,
    img: np.ndarray,
    device: torch.device,
    threshold: float = 0.35,
) -> np.ndarray:
    """Pure Raw JunctionUNet: Heatmap local peak extraction only, zero skeleton filtering."""
    h_orig, w_orig = img.shape
    pad_h = (16 - (h_orig % 16)) % 16
    pad_w = (16 - (w_orig % 16)) % 16
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")

    img_t = (
        torch.from_numpy(padded.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(img_t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob_c = prob[:h_orig, :w_orig]

    # Max pool peak extraction
    tensor = torch.from_numpy(prob_c).unsqueeze(0).unsqueeze(0)
    max_pooled = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    peaks_mask = (tensor == max_pooled) & (tensor > threshold)
    peaks_np = peaks_mask[0, 0].cpu().numpy()

    ys, xs = np.where(peaks_np)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack([xs, ys]).astype(np.int32)


# ============================================================
# RUN BENCHMARK
# ============================================================

def run_benchmark(
    split_file: str = "data/splits/specimen_split.json",
    wings_dir: str = "data/cleaned_wings",
    seeds_dir: str = "data/cleaned_seeds",
    checkpoint_path: str = "training_output/best_model.pt",
    output_csv: str = "training_output/baselines_benchmark.csv",
    sample_limit: int = None,
):
    split_path = repo_root / split_file
    wings_p = repo_root / wings_dir
    seeds_p = repo_root / seeds_dir
    ckpt_p = repo_root / checkpoint_path
    out_p = repo_root / output_csv
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(split_path, "r") as f:
        split_data = json.load(f)

    test_stems = split_data["test"]
    if sample_limit is not None:
        test_stems = test_stems[:sample_limit]

    print(f"Running Baselines Benchmark on {len(test_stems)} Strictly Held-Out Test Wings...")

    # Load UNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        "JunctionUNet (Raw)": lambda im: detect_unet_raw(unet, im, device),
    }

    results = []

    for idx, stem in enumerate(test_stems, 1):
        # Find wing file
        img_candidates = list(wings_p.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        img_path = img_candidates[0]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        seed_file = seeds_p / f"{stem}.npy"
        if not seed_file.exists():
            continue
        gt = np.load(str(seed_file))

        print(f"[{idx:02d}/{len(test_stems)}] Benchmarking specimen: {stem:40s} | GT Junctions: {len(gt):4d}")

        for name, fn in methods.items():
            t0 = time.perf_counter()
            preds = fn(img)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            met = evaluate_predictions_multi_tolerance(preds, gt, tolerances=[3.0, 5.0, 8.0])

            rec = {
                "stem": stem,
                "method": name,
                "gt_count": len(gt),
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
            results.append(rec)

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_p, index=False)

    print("\n" + "=" * 90)
    print("                    BASELINES BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    summary = res_df.groupby("method").agg({
        "p_3px": lambda x: f"{x.mean()*100:.1f}%",
        "r_3px": lambda x: f"{x.mean()*100:.1f}%",
        "f1_3px": lambda x: f"{x.mean()*100:.1f}%",
        "p_5px": lambda x: f"{x.mean()*100:.1f}%",
        "r_5px": lambda x: f"{x.mean()*100:.1f}%",
        "f1_5px": lambda x: f"{x.mean()*100:.1f}%",
        "p_8px": lambda x: f"{x.mean()*100:.1f}%",
        "r_8px": lambda x: f"{x.mean()*100:.1f}%",
        "f1_8px": lambda x: f"{x.mean()*100:.1f}%",
        "mean_loc_error_px": lambda x: f"{x.mean():.2f} px",
        "latency_ms": lambda x: f"{x.mean():.1f} ms",
    }).reset_index()

    print(summary.to_string(index=False))
    print(f"\nDetailed per-wing baseline metrics saved to: {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test wings")
    args = parser.parse_args()
    run_benchmark(sample_limit=args.limit)
