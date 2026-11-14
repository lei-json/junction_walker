#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from .cleaning import preprocess_wing_image, remove_body_remnants, extract_veins_tophat
except ImportError:
    try:
        from cleaning import preprocess_wing_image, remove_body_remnants, extract_veins_tophat
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        from cleaning import preprocess_wing_image, remove_body_remnants, extract_veins_tophat


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_THRESHOLD = 220
DEFAULT_TOPHAT_KERNEL = 11
DEFAULT_RIDGE_THRESHOLD = 16


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

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
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Convert every image in a directory to black/white "
            "using darkness thresholding or morphological Black Top-Hat filtering."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "image_directory",
        type=Path,
        help="Directory containing input images.",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=(
            "Pixels with grayscale intensity <= threshold "
            "become black; pixels above become white."
        ),
    )

    parser.add_argument(
        "--method",
        choices=["threshold", "tophat", "hybrid"],
        default="tophat",
        help=(
            "Binarization method: 'tophat' isolates 1D veins and clears "
            "colored bands/pterostigmata; 'threshold' uses static global threshold."
        ),
    )

    parser.add_argument(
        "--clean-bodies",
        action="store_true",
        default=True,
        help="Detect and whiten insect body remnants at left wing base.",
    )

    parser.add_argument(
        "--no-clean-bodies",
        action="store_false",
        dest="clean_bodies",
        help="Disable automated body remnant cleaning.",
    )

    parser.add_argument(
        "--tophat-kernel",
        type=int,
        default=DEFAULT_TOPHAT_KERNEL,
        help="Kernel size for black top-hat transform.",
    )

    parser.add_argument(
        "--ridge-threshold",
        type=int,
        default=DEFAULT_RIDGE_THRESHOLD,
        help="Ridge contrast threshold for top-hat vein detection.",
    )

    return parser.parse_args()


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(directory):

    images = []

    for path in sorted(
        directory.iterdir()
    ):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            images.append(path)

    return images


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    input_path,
    output_dir,
    threshold=DEFAULT_THRESHOLD,
    method="threshold",
    clean_bodies=False,
    tophat_kernel_size=DEFAULT_TOPHAT_KERNEL,
    ridge_threshold=DEFAULT_RIDGE_THRESHOLD,
):

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    output_path = (
        output_dir
        / f"{input_path.stem}_blackwhite"
        f"{input_path.suffix}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:

        raise RuntimeError(
            f"Could not read {input_path}"
        )

    # --------------------------------------------------------
    # Preprocess & Binarize
    # --------------------------------------------------------

    cleaned_image, output, meta = preprocess_wing_image(
        image=image,
        clean_body=clean_bodies,
        method=method,
        threshold=threshold,
        tophat_kernel_size=tophat_kernel_size,
        ridge_threshold=ridge_threshold,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if not cv2.imwrite(
        str(output_path),
        output,
    ):

        raise RuntimeError(
            f"Could not save {output_path}"
        )

    body_note = f" (body cleaned: {meta['bodies_removed']} chunk)" if meta.get("bodies_removed", 0) > 0 else ""
    print(
        f"{input_path.name} -> "
        f"{output_path.name}{body_note}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Validate directory
    # --------------------------------------------------------

    if not args.image_directory.is_dir():

        print(
            f"Error: not a directory: "
            f"{args.image_directory}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Validate threshold
    # --------------------------------------------------------

    if not 0 <= args.threshold <= 255:

        print(
            "Error: threshold must be between 0 and 255"
        )

        raise SystemExit(1)

    image_directory = (
        args.image_directory.resolve()
    )

    # --------------------------------------------------------
    # Output directory
    #
    # If input is:
    #
    #     data/images
    #
    # output becomes:
    #
    #     data/blackwhite
    # --------------------------------------------------------

    output_dir = (
        image_directory.parent
        / "blackwhite_rotate"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = find_images(
        image_directory
    )

    if not images:

        print(
            f"No supported images found in "
            f"{image_directory}"
        )

        return

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    print(
        f"Input directory:  {image_directory}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"Method:           {args.method}"
    )

    print(
        f"Clean bodies:     {args.clean_bodies}"
    )

    print(
        f"Images:           {len(images)}"
    )

    print()

    for input_path in images:

        try:

            process_image(
                input_path=input_path,
                output_dir=output_dir,
                threshold=args.threshold,
                method=args.method,
                clean_bodies=args.clean_bodies,
                tophat_kernel_size=args.tophat_kernel,
                ridge_threshold=args.ridge_threshold,
            )

        except RuntimeError as error:

            print(
                f"Error: {error}"
            )

    print()
    print(
        f"Finished. Processed {len(images)} images."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
