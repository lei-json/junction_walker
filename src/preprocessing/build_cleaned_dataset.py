#!/usr/bin/env python3
"""
Build Cleaned Wings Dataset & Generate Topological Walker Ground Truth.

Extracts all dragonfly wings without large colored spots,
cleans wing-base insect body remnants, computes 1-pixel-wide
topological vein skeletons and junction branch points via TopologicalWalker,
and saves the dataset paired with ground-truth junction coordinates.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import time
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "walkers"))

from cleaning import remove_body_remnants, extract_veins_tophat
from topological_walker import TopologicalWalker


DEFAULT_INPUT_DIR = Path("data/rotate")
DEFAULT_OUTPUT_WINGS_DIR = Path("data/cleaned_wings")
DEFAULT_OUTPUT_SEEDS_DIR = Path("data/cleaned_seeds")
DEFAULT_MANIFEST_PATH = Path("data/cleaned_wings_manifest.csv")

# Species known for heavy opaque colored patches/spots or solid membranes
COLORED_SPOT_KEYWORDS = [
    "amberwing",
    "pennant",
    "saddlebags",
    "whitetail",
    "corporal",
    "setwing",
    "filigree",
    "pseudoleon",
]
MAX_DARK_PCT_THRESHOLD = 22.0
THRESHOLD_VEIN = 220


def is_wing_without_large_colored_spots(filename: str, img: np.ndarray) -> bool:
    """
    Check if a wing qualifies for the cleaned dataset (without large colored spots).
    """
    lower_name = filename.lower()
    if any(k in lower_name for k in COLORED_SPOT_KEYWORDS):
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    dark_pct = float(np.mean(gray < THRESHOLD_VEIN) * 100)
    if dark_pct > MAX_DARK_PCT_THRESHOLD:
        return False

    return True


def process_single_wing(args_tuple):
    img_path_str, out_wings_dir_str, out_seeds_dir_str = args_tuple
    img_path = Path(img_path_str)
    out_wings_dir = Path(out_wings_dir_str)
    out_seeds_dir = Path(out_seeds_dir_str)

    t0 = time.perf_counter()
    raw_img = cv2.imread(str(img_path))
    if raw_img is None:
        return None

    filename = img_path.name
    stem = img_path.stem

    # Filter out wings with large colored spots
    if not is_wing_without_large_colored_spots(filename, raw_img):
        return {
            "filename": filename,
            "status": "excluded_colored_spots",
            "width": raw_img.shape[1],
            "height": raw_img.shape[0],
            "num_junctions": 0,
            "runtime_sec": 0.0,
        }

    # Step 1: Clean wing-base body remnants
    cleaned_img, body_mask, body_info = remove_body_remnants(raw_img)
    gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY) if cleaned_img.ndim == 3 else cleaned_img

    # Step 2: Extract veins via Morphological Black Top-Hat
    # Preserves fine 1D vein ridges even when faint or anti-aliased (<= 250)
    # Uses adaptive ridge threshold (threshold=6) for wings with faint crossvenation
    ridge_threshold = 6 if "caribbean_darner" in stem else 10
    bw_top, _ = extract_veins_tophat(
        gray,
        kernel_size=9,
        ridge_threshold=ridge_threshold,
        background_threshold=250,
    )
    vein_mask = (bw_top == 0).astype(np.uint8)

    # Step 3: Run TopologicalWalker
    tw = TopologicalWalker(vein_mask, min_component_size=3, spur_threshold=3)
    junctions = tw.find_junctions()

    # Step 4: Save cleaned grayscale image & ground truth junction seeds
    out_img_path = out_wings_dir / f"{stem}.png"
    cv2.imwrite(str(out_img_path), gray)

    out_seed_path = out_seeds_dir / f"{stem}.npy"
    np.save(str(out_seed_path), junctions)

    runtime_sec = time.perf_counter() - t0

    return {
        "filename": filename,
        "stem": stem,
        "status": "included",
        "width": raw_img.shape[1],
        "height": raw_img.shape[0],
        "bodies_cleaned": len(body_info),
        "dark_pct": float(np.mean(vein_mask) * 100),
        "num_junctions": len(junctions),
        "runtime_sec": runtime_sec,
    }


def build_cleaned_dataset(
    input_dir=DEFAULT_INPUT_DIR,
    out_wings_dir=DEFAULT_OUTPUT_WINGS_DIR,
    out_seeds_dir=DEFAULT_OUTPUT_SEEDS_DIR,
    manifest_path=DEFAULT_MANIFEST_PATH,
    max_workers=6,
    limit=None,
):
    repo_root = Path(__file__).resolve().parent.parent.parent
    input_dir = repo_root / input_dir
    out_wings_dir = repo_root / out_wings_dir
    out_seeds_dir = repo_root / out_seeds_dir
    manifest_path = repo_root / manifest_path

    out_wings_dir.mkdir(parents=True, exist_ok=True)
    out_seeds_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".webp"}
    ])

    if limit is not None:
        all_files = all_files[:limit]

    print(f"Discovered {len(all_files)} total raw wing images.")
    print(f"Building cleaned dataset with {max_workers} worker processes...")

    tasks = [
        (str(p), str(out_wings_dir), str(out_seeds_dir))
        for p in all_files
    ]

    t_start = time.perf_counter()
    results = []
    included_count = 0
    excluded_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process_single_wing, tasks):
            if res is None:
                continue
            results.append(res)
            if res["status"] == "included":
                included_count += 1
                if included_count % 50 == 0:
                    print(f"  [Progress] Processed {included_count} cleaned wings ({res['filename']}: {res['num_junctions']} junctions)")
            else:
                excluded_count += 1

    total_time = time.perf_counter() - t_start

    # Save manifest
    fieldnames = [
        "filename", "stem", "status", "width", "height",
        "bodies_cleaned", "dark_pct", "num_junctions", "runtime_sec"
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print("\n================ COMPLETED CLEANED DATASET GENERATION ================")
    print(f"Total Wings Processed: {len(results)}")
    print(f"Cleaned Wings Included: {included_count}")
    print(f"Wings with Large Spots Excluded: {excluded_count}")
    print(f"Total Wall Time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"Manifest written to: {manifest_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    build_cleaned_dataset(max_workers=args.workers, limit=args.limit)
