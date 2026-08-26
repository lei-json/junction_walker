#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_MIN_COMPONENT_SIZE = 3


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
            "Skeletonize black vein structures in every image "
            "in a directory."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "image_directory",
        type=Path,
        help="Directory containing thresholded images.",
    )

    parser.add_argument(
        "--min-component-size",
        type=int,
        default=DEFAULT_MIN_COMPONENT_SIZE,
        help=(
            "Remove connected black components smaller than "
            "this number of pixels before skeletonization."
        ),
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
# REMOVE SMALL COMPONENTS
# ============================================================

def remove_small_components(
    mask,
    min_component_size,
):
    """
    Remove small connected black components.

    Input:
        mask = binary image where black/vein pixels are 1.

    Output:
        cleaned binary mask.
    """

    (
        num_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(
        mask
    )

    for label in range(
        1,
        num_labels,
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= min_component_size:

            cleaned[
                labels == label
            ] = 1

    return cleaned


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    input_path,
    output_dir,
    min_component_size,
):

    output_path = (
        output_dir
        / f"{input_path.stem}_skeleton"
        f"{input_path.suffix}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise RuntimeError(
            f"Could not read {input_path}"
        )

    # --------------------------------------------------------
    # Convert black pixels to foreground.
    #
    # Thresholded input:
    #
    #   black = vein
    #   white = background
    #
    # skeletonize() expects:
    #
    #   True = foreground
    # --------------------------------------------------------

    mask = (
        image < 128
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Remove tiny artifacts
    # --------------------------------------------------------

    if min_component_size > 1:

        mask = remove_small_components(
            mask=mask,
            min_component_size=min_component_size,
        )

    # --------------------------------------------------------
    # Skeletonize
    # --------------------------------------------------------

    skeleton = skeletonize(
        mask > 0
    )

    # --------------------------------------------------------
    # Convert back to image format.
    #
    # Skeleton veins = black
    # Background = white
    # --------------------------------------------------------

    output = np.where(
        skeleton,
        0,
        255,
    ).astype(
        np.uint8
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

    print(
        f"{input_path.name} -> "
        f"{output_path.name}"
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
    # Validate component size
    # --------------------------------------------------------

    if args.min_component_size < 1:

        print(
            "Error: --min-component-size must be >= 1"
        )

        raise SystemExit(1)

    image_directory = (
        args.image_directory.resolve()
    )

    # --------------------------------------------------------
    # Output directory
    #
    # Example:
    #
    #     data/threshold
    #
    # becomes:
    #
    #     data/skeleton
    # --------------------------------------------------------

    output_dir = (
        image_directory.parent
        / "skeleton"
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
    # Print settings
    # --------------------------------------------------------

    print(
        f"Input directory:          "
        f"{image_directory}"
    )

    print(
        f"Output directory:         "
        f"{output_dir}"
    )

    print(
        f"Minimum component size:   "
        f"{args.min_component_size}"
    )

    print(
        f"Images:                   "
        f"{len(images)}"
    )

    print()

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    successful = 0

    for input_path in images:

        try:

            process_image(
                input_path=input_path,
                output_dir=output_dir,
                min_component_size=(
                    args.min_component_size
                ),
            )

            successful += 1

        except RuntimeError as error:

            print(
                f"Error: {error}"
            )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()

    print(
        f"Finished. "
        f"Processed {successful}/"
        f"{len(images)} images."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
