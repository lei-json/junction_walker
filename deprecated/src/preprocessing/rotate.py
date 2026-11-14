#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_INPUT_DIR = Path("../../data/raw")


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
    Load an image.
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
# ROTATION
# ============================================================

def rotate_image(image, angle):
    """
    Rotate an image while preserving the entire image.
    """

    height, width = image.shape[:2]

    center = (
        width / 2.0,
        height / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_width = int(
        height * sin +
        width * cos
    )

    new_height = int(
        height * cos +
        width * sin
    )

    matrix[0, 2] += (
        new_width / 2.0
        - center[0]
    )

    matrix[1, 2] += (
        new_height / 2.0
        - center[1]
    )

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ============================================================
# DRAW HELP
# ============================================================

def draw_help(
    image,
    index,
    total,
    path,
    angle,
    polygon_points,
    polygon_mode,
):
    """
    Draw polygon and help information.
    """

    height, width = image.shape[:2]

    bar_height = 115

    canvas = cv2.copyMakeBorder(
        image,
        0,
        bar_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30),
    )

    # --------------------------------------------------------
    # Draw polygon
    # --------------------------------------------------------

    if polygon_points:

        points = np.asarray(
            polygon_points,
            dtype=np.int32,
        )

        if len(points) >= 2:

            # Automatically close polygon.
            cv2.polylines(
                canvas,
                [points],
                True,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Draw vertices
        # ----------------------------------------------------

        for x, y in polygon_points:

            cv2.circle(
                canvas,
                (x, y),
                5,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    text1 = (
        f"{index + 1}/{total}  "
        f"{path.name}  "
        f"Rotation: {angle:+.1f}°"
    )

    if polygon_mode:

        text2 = (
            "POLYGON: Left click=add | "
            "Backspace=undo | ESC=cancel"
        )

        text3 = (
            "ENTER: whiten polygon | "
            "changes save automatically"
        )

    else:

        text2 = (
            "Left/Right: rotate 1° | "
            "A/D: rotate 5° | "
            "R: reset"
        )

        text3 = (
            "C: polygon | "
            "N: next | P: previous | "
            "ESC: quit"
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
        (10, height + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        text3,
        (10, height + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# WHITEN POLYGON
# ============================================================

def whiten_polygon(image, points):
    """
    Make everything inside the polygon white.
    """

    if len(points) < 3:

        return image.copy()

    points_array = np.asarray(
        points,
        dtype=np.int32,
    )

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [points_array],
        255,
    )

    result = image.copy()

    result[mask == 255] = 255

    return result


# ============================================================
# VIEWER
# ============================================================

class ImageViewer:

    def __init__(
        self,
        images,
        output_dir,
    ):

        self.images = images

        self.output_dir = output_dir

        self.index = 0

        self.angle = 0.0

        self.current_image = None

        self.polygon_points = []

        self.polygon_mode = False

        self.window_name = (
            "Rotate / Crop Image Viewer"
        )

    # --------------------------------------------------------
    # Current path
    # --------------------------------------------------------

    @property
    def current_path(self):

        return self.images[
            self.index
        ]

    # --------------------------------------------------------
    # Load current image
    # --------------------------------------------------------

    def load_current_image(self):

        self.current_image = load_image(
            self.current_path
        )

        self.angle = 0.0

        self.polygon_points = []

        self.polygon_mode = False

    # --------------------------------------------------------
    # Save current image
    # --------------------------------------------------------

    def save_current(self):

        if self.current_image is None:
            return False

        output_path = (
            self.output_dir
            / self.current_path.name
        )

        success = cv2.imwrite(
            str(output_path),
            self.current_image,
        )

        if not success:

            print(
                f"Failed to save: {output_path}"
            )

            return False

        print(
            f"Saved: {output_path}"
        )

        return True

    # --------------------------------------------------------
    # Apply rotation
    # --------------------------------------------------------

    def change_rotation(self, delta):

        if self.polygon_mode:
            return

        self.angle += delta

        # --------------------------------------------
        # Rebuild rotation from the original source.
        #
        # Polygon modifications are already contained
        # in current_image, so rotation needs to be
        # performed relative to the current image.
        # --------------------------------------------

        self.current_image = rotate_image(
            self.current_image,
            delta,
        )

        # --------------------------------------------
        # Save immediately.
        # --------------------------------------------

        self.save_current()

    # --------------------------------------------------------
    # Reset rotation
    # --------------------------------------------------------

    def reset_rotation(self):

        if self.polygon_mode:
            return

        # Reload original image.
        self.current_image = load_image(
            self.current_path
        )

        self.angle = 0.0

        self.save_current()

        print(
            "Rotation reset."
        )

    # --------------------------------------------------------
    # Next image
    # --------------------------------------------------------

    def next_image(self):

        if self.index >= len(
            self.images
        ) - 1:

            print(
                "Already at last image."
            )

            return

        self.index += 1

        self.load_current_image()

        print(
            f"[{self.index + 1}/{len(self.images)}] "
            f"{self.current_path.name}"
        )

    # --------------------------------------------------------
    # Previous image
    # --------------------------------------------------------

    def previous_image(self):

        if self.index <= 0:

            print(
                "Already at first image."
            )

            return

        self.index -= 1

        self.load_current_image()

        print(
            f"[{self.index + 1}/{len(self.images)}] "
            f"{self.current_path.name}"
        )

    # --------------------------------------------------------
    # Mouse callback
    # --------------------------------------------------------

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ):

        if not self.polygon_mode:
            return

        image = param

        if image is not None:

            if y >= image.shape[0]:
                return

        # ----------------------------------------------------
        # Left click = add point
        # ----------------------------------------------------

        if event == cv2.EVENT_LBUTTONDOWN:

            self.polygon_points.append(
                (x, y)
            )

            print(
                f"Point {len(self.polygon_points)}: "
                f"({x}, {y})"
            )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(self):

        self.load_current_image()

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        while True:

            # ------------------------------------------------
            # Current display
            # ------------------------------------------------

            display_image = (
                self.current_image.copy()
            )

            # ------------------------------------------------
            # Mouse callback
            # ------------------------------------------------

            cv2.setMouseCallback(
                self.window_name,
                self.mouse_callback,
                display_image,
            )

            # ------------------------------------------------
            # Draw help / polygon
            # ------------------------------------------------

            display = draw_help(
                display_image,
                self.index,
                len(self.images),
                self.current_path,
                self.angle,
                self.polygon_points,
                self.polygon_mode,
            )

            # ------------------------------------------------
            # Show
            # ------------------------------------------------

            cv2.imshow(
                self.window_name,
                display,
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key_raw = cv2.waitKeyEx(30)

            if key_raw == -1:
                continue

            key = key_raw & 0xFF

            # =================================================
            # ESC
            # =================================================

            if key == 27:

                if self.polygon_mode:

                    self.polygon_mode = False
                    self.polygon_points = []

                    print(
                        "Polygon cancelled."
                    )

                else:

                    break

            # =================================================
            # N
            # =================================================

            elif key in (
                ord("n"),
                ord("N"),
            ):

                if not self.polygon_mode:

                    self.next_image()

            # =================================================
            # P
            # =================================================

            elif key in (
                ord("p"),
                ord("P"),
            ):

                if not self.polygon_mode:

                    self.previous_image()

            # =================================================
            # C = polygon mode
            # =================================================

            elif key in (
                ord("c"),
                ord("C"),
            ):

                if not self.polygon_mode:

                    self.polygon_mode = True
                    self.polygon_points = []

                    print()
                    print(
                        "Polygon mode enabled."
                    )
                    print(
                        "Left click to add points."
                    )
                    print(
                        "ENTER to whiten polygon."
                    )

            # =================================================
            # A = rotate counterclockwise 5°
            # =================================================

            elif key in (
                ord("a"),
                ord("A"),
            ):

                self.change_rotation(
                    -5.0
                )

            # =================================================
            # D = rotate clockwise 5°
            # =================================================

            elif key in (
                ord("d"),
                ord("D"),
            ):

                self.change_rotation(
                    5.0
                )

            # =================================================
            # R = reset rotation
            # =================================================

            elif key in (
                ord("r"),
                ord("R"),
            ):

                self.reset_rotation()

            # =================================================
            # LEFT ARROW
            # =================================================

            elif key_raw in (
                81,
                2424832,
            ):

                self.change_rotation(
                    -1.0
                )

            # =================================================
            # RIGHT ARROW
            # =================================================

            elif key_raw in (
                83,
                2555904,
            ):

                self.change_rotation(
                    1.0
                )

            # =================================================
            # BACKSPACE
            # =================================================

            elif key in (
                8,
                127,
            ):

                if self.polygon_mode:

                    if self.polygon_points:

                        removed = (
                            self.polygon_points.pop()
                        )

                        print(
                            f"Removed point: "
                            f"{removed}"
                        )

            # =================================================
            # ENTER = WHITEN POLYGON
            # =================================================

            elif key in (
                10,
                13,
            ):

                if self.polygon_mode:

                    if len(
                        self.polygon_points
                    ) < 3:

                        print(
                            "Polygon needs at least "
                            "3 points."
                        )

                        continue

                    # ----------------------------------------
                    # Whiten selected area
                    # ----------------------------------------

                    self.current_image = (
                        whiten_polygon(
                            self.current_image,
                            self.polygon_points,
                        )
                    )

                    # ----------------------------------------
                    # Clear polygon overlay so the user
                    # sees the actual resulting image.
                    # ----------------------------------------

                    self.polygon_points = []

                    self.polygon_mode = False

                    # ----------------------------------------
                    # Automatically save.
                    # ----------------------------------------

                    self.save_current()

                    print(
                        "Polygon whitened and saved."
                    )

            # =================================================
            # S = manual save
            # =================================================

            elif key in (
                ord("s"),
                ord("S"),
            ):

                if not self.polygon_mode:

                    self.save_current()

        cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Interactively rotate images and "
            "remove polygon-selected regions."
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
            "Directory containing input images."
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

    # --------------------------------------------------------
    # Validate input directory
    # --------------------------------------------------------

    if not input_dir.is_dir():

        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

    # --------------------------------------------------------
    # Output directory
    #
    # data/raw -> data/rotate
    # --------------------------------------------------------

    output_dir = (
        input_dir.parent / "rotate"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    all_images = find_images(
        input_dir
    )

    if not all_images:

        raise FileNotFoundError(
            f"No supported images found in: "
            f"{input_dir}"
        )

    # --------------------------------------------------------
    # Skip images already in rotate/
    # --------------------------------------------------------

    images = [
        path
        for path in all_images
        if not (
            output_dir / path.name
        ).exists()
    ]

    skipped = (
        len(all_images) - len(images)
    )

    # --------------------------------------------------------
    # Nothing left
    # --------------------------------------------------------

    if not images:

        print(
            f"All {len(all_images):,} images "
            "have already been processed."
        )

        print(
            f"Output directory: {output_dir}"
        )

        return

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print(
        f"Found {len(all_images):,} total images."
    )

    print(
        f"Already processed: {skipped:,}"
    )

    print(
        f"Remaining: {len(images):,}"
    )

    print(
        f"Input directory: {input_dir}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print()

    print("Controls:")
    print(
        "  Left / Right       - rotate 1°"
    )
    print(
        "  A / D              - rotate 5°"
    )
    print(
        "  R                  - reset rotation"
    )
    print(
        "  C                  - polygon mode"
    )
    print(
        "  N                  - next image"
    )
    print(
        "  P                  - previous image"
    )
    print(
        "  S                  - manually save"
    )
    print(
        "  ESC                - quit / cancel polygon"
    )

    print()

    print("Polygon:")
    print(
        "  Left click         - add point"
    )
    print(
        "  Backspace          - remove last point"
    )
    print(
        "  Enter              - whiten + automatically save"
    )

    print()

    # --------------------------------------------------------
    # Start viewer
    # --------------------------------------------------------

    viewer = ImageViewer(
        images=images,
        output_dir=output_dir,
    )

    viewer.run()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
