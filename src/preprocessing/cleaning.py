#!/usr/bin/env python3
"""
Preprocessing and Cleaning Tools for Dragonfly Wing Venation Analysis.

Includes:
1. Automated Wing-Base / Body Remnant Removal:
   Detects and whitens dense, non-vein body structures (thorax, head, legs)
   remaining at the wing base (left root margin) from specimen preparation.

2. Morphological Black Top-Hat Vein Isolation:
   Isolates narrow 1D vein ridges (<= 7 px) while rejecting broad 2D regions
   such as pigmented wing bands, colored infill, and pterostigmata.
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np


# ============================================================
# CONSTANTS & DEFAULTS
# ============================================================

DEFAULT_ROOT_MARGIN_PCT = 0.15
DEFAULT_DARK_THRESHOLD = 130
DEFAULT_MIN_BODY_AREA = 800
DEFAULT_MAX_VEIN_WIDTH = 15
DEFAULT_MAX_BX = 5
DEFAULT_DILATION_KERNEL_SIZE = 7

DEFAULT_TOPHAT_KERNEL_SIZE = 11
DEFAULT_RIDGE_THRESHOLD = 16
DEFAULT_BACKGROUND_THRESHOLD = 235

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# METHOD 1: BODY REMNANT REMOVAL
# ============================================================

def remove_body_remnants(
    image: np.ndarray,
    root_margin_pct: float = DEFAULT_ROOT_MARGIN_PCT,
    dark_threshold: int = DEFAULT_DARK_THRESHOLD,
    min_body_area: int = DEFAULT_MIN_BODY_AREA,
    max_vein_width: int = DEFAULT_MAX_VEIN_WIDTH,
    max_bx: int = DEFAULT_MAX_BX,
    dilation_kernel_size: int = DEFAULT_DILATION_KERNEL_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Detects and whitens non-vein body components attached to the wing base (left margin).

    Dragonfly wings in horizontal alignment have their root attachment at the left edge.
    Non-wing body remnants (thorax sclerites, legs) are thick, large-area 2D dark structures,
    unlike thin 1D veins.

    Args:
        image: BGR or grayscale image (H, W, 3) or (H, W).
        root_margin_pct: Fraction of image width defining root region (left margin).
        dark_threshold: Intensity cutoff for body pixels (gray < dark_threshold).
        min_body_area: Minimum connected pixel area to classify as a body chunk.
        max_vein_width: Maximum expected vein thickness (px). Veins thinner than this are preserved.
        max_bx: Maximum starting x-coordinate for a body chunk (must touch or be adjacent to x=0).
        dilation_kernel_size: Dilation kernel size to cover anti-aliased gray borders.

    Returns:
        cleaned_image: Cleaned copy of image with body chunks set to pure white (255).
        body_mask: Boolean mask (H, W) where True indicates removed body pixels.
        removed_info: List of dicts with statistics on each removed component.
    """
    is_color = (image.ndim == 3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image.copy()
    h, w = gray.shape

    margin_w = max(int(w * root_margin_pct), 20)
    left_strip = gray[:, :margin_w]

    # Threshold dark structures at the wing root
    dark = (left_strip < dark_threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)

    cleaned = image.copy()
    body_mask = np.zeros((h, w), dtype=bool)
    removed_info = []

    kernel_dilate = (
        np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        if dilation_kernel_size > 0
        else None
    )

    for i in range(1, n_labels):
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # A body component:
        # 1. Starts at or adjacent to left edge (bx <= max_bx)
        # 2. Is thick in both dimensions (bw > max_vein_width and bh > max_vein_width)
        # 3. Has significant area (area >= min_body_area)
        if bx <= max_bx and bw > max_vein_width and bh > max_vein_width and area >= min_body_area:
            comp_mask = (labels == i).astype(np.uint8)
            if kernel_dilate is not None:
                comp_mask = cv2.dilate(comp_mask, kernel_dilate)

            body_mask[:, :margin_w] |= (comp_mask > 0)
            removed_info.append({
                "label": i,
                "area": area,
                "bbox": (bx, by, bw, bh),
            })

    if is_color:
        cleaned[body_mask] = (255, 255, 255)
    else:
        cleaned[body_mask] = 255

    return cleaned, body_mask, removed_info


# ============================================================
# METHOD 2: MORPHOLOGICAL BLACK TOP-HAT TRANSFORM
# ============================================================

def extract_veins_tophat(
    gray_image: np.ndarray,
    kernel_size: int = DEFAULT_TOPHAT_KERNEL_SIZE,
    ridge_threshold: int = DEFAULT_RIDGE_THRESHOLD,
    background_threshold: int = DEFAULT_BACKGROUND_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolates narrow 1D vein ridges while ignoring 2D colored patches and filled cells.

    Black Top-Hat = Closing(I) - I.
    Closing bridges across dark features narrower than kernel_size (1D veins),
    replacing them with the local cell membrane intensity.
    Subtracting the original isolates veins.
    Broad dark patches (pigmented bands, pterostigmata) are larger than the kernel,
    so their interior is unchanged by closing and produces near-zero top-hat response.

    Args:
        gray_image: Grayscale image (H, W).
        kernel_size: Structuring element diameter (px). Should exceed maximum vein width.
        ridge_threshold: Minimum intensity contrast for vein ridges.
        background_threshold: Intensity cutoff to avoid false detections in pure white background.

    Returns:
        bw_veins: Binary image (veins = 0 / black, cells & background = 255 / white).
        blackhat: Raw float/uint8 black top-hat response.
    """
    if gray_image.ndim == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

    # Detect veins: high ridge response and inside the wing area (darker than white border)
    vein_mask = (blackhat >= ridge_threshold) & (gray_image < background_threshold)

    # Convert to project convention: veins = 0 (black), background = 255 (white)
    bw_veins = np.where(vein_mask, 0, 255).astype(np.uint8)

    return bw_veins, blackhat


# ============================================================
# COMPREHENSIVE PIPELINE PREPROCESSOR
# ============================================================

def preprocess_wing_image(
    image: np.ndarray,
    clean_body: bool = True,
    method: str = "tophat",
    threshold: int = 220,
    tophat_kernel_size: int = DEFAULT_TOPHAT_KERNEL_SIZE,
    ridge_threshold: int = DEFAULT_RIDGE_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocesses a wing image by optionally removing body remnants and binarizing.

    Args:
        image: Color or grayscale wing image.
        clean_body: Whether to apply automated wing-base body remnant removal.
        method: Binarization method ('tophat', 'threshold', or 'hybrid').
        threshold: Darkness threshold for 'threshold' method.
        tophat_kernel_size: Kernel diameter for top-hat.
        ridge_threshold: Ridge threshold for top-hat.

    Returns:
        cleaned_image: Cleaned color/gray image (with body remnants whitened if requested).
        bw_image: Binarized image (veins=0, background=255).
        metadata: Execution statistics and details.
    """
    metadata: Dict[str, Any] = {"method": method}

    # Step 1: Body remnant cleaning
    if clean_body:
        cleaned_image, body_mask, removed_info = remove_body_remnants(image)
        metadata["bodies_removed"] = len(removed_info)
        metadata["body_details"] = removed_info
    else:
        cleaned_image = image.copy()
        metadata["bodies_removed"] = 0
        metadata["body_details"] = []

    gray = (
        cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
        if cleaned_image.ndim == 3
        else cleaned_image
    )

    # Step 2: Binarization
    if method == "tophat":
        bw_image, _ = extract_veins_tophat(
            gray,
            kernel_size=tophat_kernel_size,
            ridge_threshold=ridge_threshold,
        )
    elif method == "threshold":
        bw_image = np.where(gray <= threshold, 0, 255).astype(np.uint8)
    elif method == "hybrid":
        bw_top, _ = extract_veins_tophat(
            gray,
            kernel_size=tophat_kernel_size,
            ridge_threshold=ridge_threshold,
        )
        # Combine: thin ridges from top-hat AND high-contrast veins, but avoiding large solid blocks
        bw_thresh = (gray <= threshold)
        # Keep threshold only where vein is somewhat dark or near top-hat ridge
        bw_image = np.where((bw_top == 0) | (gray <= 100), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")

    metadata["black_pixel_pct"] = float(np.mean(bw_image == 0) * 100)
    return cleaned_image, bw_image, metadata


# ============================================================
# CLI TOOL
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dragonfly wing cleaning and vein extraction tools."
    )
    parser.add_argument(
        "input_directory",
        type=Path,
        help="Directory with input wing images.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Directory for binarized output images (defaults to blackwhite_rotate).",
    )
    parser.add_argument(
        "--clean-bodies",
        action="store_true",
        default=True,
        help="Remove body remnants at left wing base margin.",
    )
    parser.add_argument(
        "--method",
        choices=["tophat", "threshold", "hybrid"],
        default="tophat",
        help="Binarization method (default: tophat).",
    )
    parser.add_argument(
        "--tophat-kernel",
        type=int,
        default=DEFAULT_TOPHAT_KERNEL_SIZE,
        help="Top-hat kernel diameter.",
    )
    parser.add_argument(
        "--ridge-threshold",
        type=int,
        default=DEFAULT_RIDGE_THRESHOLD,
        help="Vein ridge threshold for top-hat.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_directory.resolve()
    out_dir = args.output_directory or (input_dir.parent / "blackwhite_rotate")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"Processing {len(images)} images from {input_dir} -> {out_dir}")
    print(f"Clean bodies: {args.clean_bodies} | Method: {args.method}")

    bodies_count = 0
    for img_p in images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        cleaned, bw, meta = preprocess_wing_image(
            img,
            clean_body=args.clean_bodies,
            method=args.method,
            tophat_kernel_size=args.tophat_kernel,
            ridge_threshold=args.ridge_threshold,
        )
        if meta["bodies_removed"] > 0:
            bodies_count += 1
            print(f"  [BODY CLEANED] {img_p.name} ({meta['bodies_removed']} chunk(s) removed)")

        out_name = f"{img_p.stem}_blackwhite{img_p.suffix}"
        cv2.imwrite(str(out_dir / out_name), bw)

    print(f"Finished processing {len(images)} images. Total wings with bodies cleaned: {bodies_count}.")
