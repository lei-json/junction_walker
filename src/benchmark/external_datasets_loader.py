#!/usr/bin/env python3
"""
External Curvilinear / Vein Dataset Loaders for Benchmarking Junction Detection.

Provides standard interfaces returning:
- Grayscale uint8 image (H, W)
- Ground-truth junction coordinates (N, 2) float32
- Dataset metadata

Datasets covered:
1. Real Dragonfly Wing Venation (Hoffmann et al., PNAS 2018):
   High-resolution specimen photograph (1088 x 3001) with 9,586 biological cell junctions
   derived from Fast Marching Method (FMM) cell segmentations.
2. Real Leaf Venation (HALVS cvpr/plant benchmark):
   Cleared leaf photograph (Sweet Cherry, 3225 x 1530) with expert-annotated hierarchical
   vascular network skeleton junctions.
3. Real Retinal Vasculature (DRIVE benchmark):
   Clinical fundus photographs (584 x 565) with dual-observer expert manual vessel ground truth
   bifurcations and crossings.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

repo_root = Path(__file__).resolve().parents[2]


def extract_junctions_from_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extracts topological junctions (bifurcations and crossovers) from a binary vessel mask.
    Uses Zhang-Suen / medial axis skeletonization + Rutovitz crossing number >= 3.
    """
    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    p = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    pad = np.pad(skel, 1, mode="constant")
    h, w = skel.shape
    nbrs = [pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in p]

    crossing_num = np.zeros_like(skel, dtype=np.float32)
    for i in range(8):
        crossing_num += np.abs(nbrs[i].astype(np.float32) - nbrs[(i + 1) % 8].astype(np.float32))
    crossing_num = (0.5 * crossing_num) * skel

    cand = (crossing_num >= 3).astype(np.uint8)
    n_cc, _, _, centroids = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n_cc <= 1:
        return np.empty((0, 2), dtype=np.float32)

    return centroids[1:].astype(np.float32)


def load_pnas_dragonfly_benchmark(
    crop_box: Tuple[int, int, int, int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads real dragonfly wing photograph and cell segmentation ground truth from Hoffmann et al. PNAS.
    crop_box: Optional (y_start, y_end, x_start, x_end) to evaluate specific regions or full wing.
    """
    img_path = repo_root / "data/external/pnas_sample/PNAS_Paper_Code/Segment_Wing_Image/Arigomphus_villosipes_LHW.png"
    seg_path = repo_root / "data/external/pnas_sample/PNAS_Paper_Code/Segment_Wing_Image/Output_Files/Arigomphus_villosipes_LHW_FMM_seg.csv"

    if not img_path.exists() or not seg_path.exists():
        raise FileNotFoundError(f"PNAS dataset files not found at {img_path} or {seg_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    seg = pd.read_csv(str(seg_path), header=None).values

    # Extract 2x2 corner junctions where 3 or more distinct cells meet
    tl = seg[:-1, :-1]
    tr = seg[:-1, 1:]
    bl = seg[1:, :-1]
    br = seg[1:, 1:]

    stack = np.stack([tl, tr, bl, br], axis=0)
    sorted_stack = np.sort(stack, axis=0)
    diff = sorted_stack[1:] != sorted_stack[:-1]
    distinct_count = np.sum(diff, axis=0) + 1
    has_zero = np.any(stack == 0, axis=0)
    distinct_nonzero = distinct_count - has_zero.astype(int)

    junction_mask = (distinct_nonzero >= 3).astype(np.uint8)
    n_cc, _, _, centroids = cv2.connectedComponentsWithStats(junction_mask, connectivity=8)
    gt_all = centroids[1:].astype(np.float32)

    if crop_box is not None:
        y0, y1, x0, x1 = crop_box
        img_out = img[y0:y1, x0:x1]
        valid = (gt_all[:, 0] >= x0) & (gt_all[:, 0] < x1) & (gt_all[:, 1] >= y0) & (gt_all[:, 1] < y1)
        gt_out = gt_all[valid] - np.array([x0, y0], dtype=np.float32)
    else:
        img_out = img
        gt_out = gt_all

    meta = {
        "dataset": "PNAS Dragonfly (Hoffmann et al.)",
        "specimen": "Arigomphus villosipes (Dragonfly Hindwing)",
        "full_shape": img.shape,
        "crop_box": crop_box,
        "gt_count": len(gt_out),
    }

    return img_out, gt_out, meta


def load_halvs_leaf_benchmark(
    crop_box: Tuple[int, int, int, int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads real leaf venation image and hierarchical vein segmentation from HALVS.
    """
    img_path = repo_root / "data/external/HALVS/HALVS-Hierarchical-Vein-Segment/U2pl/HALVS/JPEGImages/Sweet Cherry_img.png"
    mask_path = repo_root / "data/external/HALVS/HALVS-Hierarchical-Vein-Segment/U2pl/HALVS/SegmentationClass/Sweet Cherry_mask.png"

    if not img_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"HALVS leaf files not found at {img_path} or {mask_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if crop_box is not None:
        y0, y1, x0, x1 = crop_box
        img_crop = img[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]
        # Invert contrast for standard light-background dark-vein pipeline
        img_out = 255 - img_crop
        gt_out = extract_junctions_from_skeleton(mask_crop > 0)
    else:
        img_out = 255 - img
        gt_out = extract_junctions_from_skeleton(mask > 0)

    meta = {
        "dataset": "HALVS Leaf Venation (Plant Science)",
        "species": "Prunus avium (Sweet Cherry)",
        "full_shape": img.shape,
        "crop_box": crop_box,
        "gt_count": len(gt_out),
    }

    return img_out, gt_out, meta


def load_drive_retina_benchmark(
    index: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads clinical retinal fundus photograph and expert manual vessel annotation from DRIVE.
    index: 1 to 5
    """
    idx_str = f"{index:02d}"
    img_path = repo_root / f"data/external/DRIVE/images/{idx_str}.tif"
    mask_path = repo_root / f"data/external/DRIVE/labels/{idx_str}_manual1.png"

    if not img_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"DRIVE image {idx_str} not found at {img_path}")

    # Retinal fundus images are typically green-channel contrast optimized for vasculature
    bgr = cv2.imread(str(img_path))
    green_ch = bgr[:, :, 1]
    # Retinal vessels are dark on orange fundus; green channel provides highest vascular contrast
    img_out = green_ch

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt_out = extract_junctions_from_skeleton(mask > 127)

    meta = {
        "dataset": "DRIVE Retinal Vasculature (Biomedical)",
        "sample_id": idx_str,
        "full_shape": img_out.shape,
        "gt_count": len(gt_out),
    }

    return img_out, gt_out, meta
