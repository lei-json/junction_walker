#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_THRESHOLD = 220


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
            "using a darkness threshold."
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
    threshold,
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
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------
    # Black / white threshold
    #
    # <= threshold -> black
    # >  threshold -> white
    # --------------------------------------------------------

    output = np.where(
        gray <= threshold,
        0,
        255,
    ).astype(np.uint8)

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
        f"Threshold:        {args.threshold}"
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
