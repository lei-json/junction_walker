#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import cv2


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_INPUT_DIR = Path("../../data/")
DEFAULT_SAVE_DIR_NAME = "saved_wings"


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
# FIND IMAGES
# ============================================================

def find_images(directory):
    """
    Find supported images directly inside the directory.

    The saved_wings directory is not searched because only
    files directly inside the input directory are considered.
    """

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):
    """
    Load an image for viewing.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image: {path}"
        )

    return image


# ============================================================
# DRAW HELP
# ============================================================

def draw_help(
    image,
    index,
    total,
    path,
    already_saved,
):
    """
    Add navigation/status information below the image.
    """

    height, width = image.shape[:2]

    bar_height = 105

    canvas = cv2.copyMakeBorder(
        image,
        0,
        bar_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30),
    )

    text1 = (
        f"{index + 1}/{total}  {path.name}"
    )

    if already_saved:
        status = "SAVED"
    else:
        status = "NOT SAVED"

    text2 = (
        f"Status: {status}"
    )

    text3 = (
        "S: save wing | N: next | P: previous | ESC: quit"
    )

    cv2.putText(
        canvas,
        text1,
        (10, height + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text2,
        (10, height + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text3,
        (10, height + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# VIEWER
# ============================================================

class ImageViewer:

    def __init__(
        self,
        images,
        save_dir,
    ):

        self.images = images

        self.save_dir = save_dir

        self.index = 0

        self.window_name = (
            "Save Wings"
        )

    # --------------------------------------------------------
    # Current image path
    # --------------------------------------------------------

    @property
    def current_path(self):

        return self.images[
            self.index
        ]

    # --------------------------------------------------------
    # Current save path
    # --------------------------------------------------------

    @property
    def current_save_path(self):

        return self.save_dir / self.current_path.name

    # --------------------------------------------------------
    # Check whether current image is saved
    # --------------------------------------------------------

    def is_saved(self):

        return self.current_save_path.exists()

    # --------------------------------------------------------
    # Save current image
    # --------------------------------------------------------

    def save_current(self):

        source = self.current_path
        destination = self.current_save_path

        # ----------------------------------------------------
        # Already saved
        # ----------------------------------------------------

        if destination.exists():

            print(
                f"Already saved: {source.name}"
            )

            return

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            source,
            destination,
        )

        print(
            f"Saved: {source.name}"
        )

        print(
            f"  -> {destination}"
        )

        # ----------------------------------------------------
        # Automatically advance
        # ----------------------------------------------------

        if self.index < len(self.images) - 1:

            self.index += 1

            print(
                f"[{self.index + 1}/{len(self.images)}] "
                f"{self.current_path.name}"
            )

        else:

            print(
                "Already at last image."
            )

    # --------------------------------------------------------
    # Next
    # --------------------------------------------------------

    def next_image(self):

        if self.index >= len(self.images) - 1:

            print(
                "Already at last image."
            )

            return

        self.index += 1

        print(
            f"[{self.index + 1}/{len(self.images)}] "
            f"{self.current_path.name}"
        )

    # --------------------------------------------------------
    # Previous
    # --------------------------------------------------------

    def previous_image(self):

        if self.index <= 0:

            print(
                "Already at first image."
            )

            return

        self.index -= 1

        print(
            f"[{self.index + 1}/{len(self.images)}] "
            f"{self.current_path.name}"
        )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(self):

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        while True:

            # ------------------------------------------------
            # Load current image
            # ------------------------------------------------

            try:

                image = load_image(
                    self.current_path
                )

            except RuntimeError as error:

                print(
                    f"Error: {error}"
                )

                self.index += 1

                if self.index >= len(self.images):

                    break

                continue

            # ------------------------------------------------
            # Add navigation information
            # ------------------------------------------------

            display = draw_help(
                image,
                self.index,
                len(self.images),
                self.current_path,
                self.is_saved(),
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                self.window_name,
                display,
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(30) & 0xFF

            # ------------------------------------------------
            # ESC = quit
            # ------------------------------------------------

            if key == 27:

                break

            # ------------------------------------------------
            # S = save
            # ------------------------------------------------

            elif key in (
                ord("s"),
                ord("S"),
            ):

                self.save_current()

            # ------------------------------------------------
            # N = next
            # ------------------------------------------------

            elif key in (
                ord("n"),
                ord("N"),
            ):

                self.next_image()

            # ------------------------------------------------
            # P = previous
            # ------------------------------------------------

            elif key in (
                ord("p"),
                ord("P"),
            ):

                self.previous_image()

        cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "View wing images and save selected wings "
            "to a saved_wings subdirectory."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing wing images."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_dir = (
        args.directory.resolve()
    )

    save_dir = (
        input_dir / DEFAULT_SAVE_DIR_NAME
    )

    # --------------------------------------------------------
    # Validate input directory
    # --------------------------------------------------------

    if not input_dir.is_dir():

        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

    # --------------------------------------------------------
    # Create save directory
    # --------------------------------------------------------

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = find_images(
        input_dir
    )

    if not images:

        raise FileNotFoundError(
            f"No supported images found in: "
            f"{input_dir}"
        )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print(
        f"Found {len(images):,} images."
    )

    print(
        f"Input directory: {input_dir}"
    )

    print(
        f"Save directory: {save_dir}"
    )

    print()

    print(
        "Controls:"
    )

    print(
        "  S           - save current wing and go next"
    )

    print(
        "  N           - next image"
    )

    print(
        "  P           - previous image"
    )

    print(
        "  ESC         - quit"
    )

    print()

    # --------------------------------------------------------
    # Start viewer
    # --------------------------------------------------------

    viewer = ImageViewer(
        images=images,
        save_dir=save_dir,
    )

    viewer.run()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
