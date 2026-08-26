#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pipeline.random_stage import run_random_stage
from walkers.intersection.intersection import Intersection


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

NUM_SIMULATIONS = 100

DEFAULT_EDGE_THRESHOLD = 80


# ------------------------------------------------------------
# Random walkers
# ------------------------------------------------------------

DEFAULT_VERTICAL_WALKERS = 10000
DEFAULT_HORIZONTAL_WALKERS = 10000

DEFAULT_VERTICAL_CONSUME_SIDE = 5
DEFAULT_HORIZONTAL_CONSUME_SIDE = 5

DEFAULT_VERTICAL_MAX_DRIFT = 10
DEFAULT_HORIZONTAL_MAX_DRIFT = 10

DEFAULT_VERTICAL_MIN_STEPS = 10
DEFAULT_HORIZONTAL_MIN_STEPS = 10


# ------------------------------------------------------------
# Intersection point selection
# ------------------------------------------------------------

DEFAULT_INTERSECTION_COUNT_THRESHOLD = 3
DEFAULT_INTERSECTION_SUPPRESSION = 7.0


# ------------------------------------------------------------
# Randomness
# ------------------------------------------------------------

DEFAULT_RANDOM_STATE = 100


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
# SHUFFLE IMAGES
# ============================================================

def shuffle_images(
    images,
    random_state,
):
    """
    Shuffle the image processing order reproducibly.
    """

    rng = np.random.default_rng(
        random_state
    )

    images = list(images)

    rng.shuffle(images)

    return images


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image


# ============================================================
# EDGE MASK
# ============================================================

def make_edge_mask(
    gray,
    threshold,
):
    """
    Convert a grayscale image into a binary edge mask.

    Pixels darker than the threshold are treated as edges.
    """

    return (
        gray < threshold
    ).astype(np.uint8)


# ============================================================
# CLEANUP EDGE MASK
# ============================================================

def cleanup_edge_mask(mask):
    """
    Remove connected components smaller than 3 pixels.
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

    cleaned = np.zeros_like(mask)

    for label in range(
        1,
        num_labels,
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= 3:

            cleaned[
                labels == label
            ] = 1

    return cleaned


# ============================================================
# RANDOM WALKER DEBUG
# ============================================================

def save_random_combined_overlay(
    gray,
    edge_mask,
    paths,
    consumed_mask,
    output,
    color,
):
    """
    Save one combined random-walker debug image.

    Layers:

        edge mask      -> orange
        consumed mask  -> cyan
        walker paths   -> supplied color

    The consumed footprint is drawn first so the actual
    walker paths remain visible on top.
    """

    image = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    # --------------------------------------------------------
    # Original edge mask
    # --------------------------------------------------------

    image[
        edge_mask > 0
    ] = (
        255,
        180,
        0,
    )

    # --------------------------------------------------------
    # Consumed footprint
    #
    # CYAN
    # --------------------------------------------------------

    image[
        consumed_mask > 0
    ] = (
        255,
        255,
        0,
    )

    # --------------------------------------------------------
    # Walker paths
    # --------------------------------------------------------

    for path in paths:

        if not path:
            continue

        for i in range(
            1,
            len(path),
        ):

            x1, y1 = path[
                i - 1
            ]

            x2, y2 = path[
                i
            ]

            cv2.line(
                image,
                (
                    int(x1),
                    int(y1),
                ),
                (
                    int(x2),
                    int(y2),
                ),
                color,
                1,
                cv2.LINE_8,
            )

        # Draw actual path pixels on top of the
        # consumed footprint.

        for x, y in path:

            x = int(x)
            y = int(y)

            if (
                0 <= x < image.shape[1]
                and 0 <= y < image.shape[0]
            ):

                image[
                    y,
                    x
                ] = color

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output),
        image,
    ):

        raise RuntimeError(
            f"Failed to save debug output: {output}"
        )


# ============================================================
# SAVE RANDOM DEBUG OUTPUTS
# ============================================================

def save_random_debug_outputs(
    gray,
    edge_mask,
    horizontal_left_paths,
    horizontal_right_paths,
    vertical_up_paths,
    vertical_down_paths,
    horizontal_consumed_mask,
    vertical_consumed_mask,
    horizontal_output,
    vertical_output,
):
    """
    Save exactly two random-walker debug images:

        debug/horizontal/<stem>.png
        debug/vertical/<stem>.png

    Horizontal:
        left + right paths
        consumed footprint

    Vertical:
        up + down paths
        consumed footprint
    """

    # --------------------------------------------------------
    # Combine directions ONLY for visualization.
    # --------------------------------------------------------

    horizontal_paths = (
        horizontal_left_paths
        + horizontal_right_paths
    )

    vertical_paths = (
        vertical_up_paths
        + vertical_down_paths
    )

    # --------------------------------------------------------
    # Horizontal
    #
    # GREEN
    # --------------------------------------------------------

    save_random_combined_overlay(
        gray=gray,
        edge_mask=edge_mask,
        paths=horizontal_paths,
        consumed_mask=horizontal_consumed_mask,
        output=horizontal_output,
        color=(
            0,
            255,
            0,
        ),
    )

    # --------------------------------------------------------
    # Vertical
    #
    # MAGENTA
    # --------------------------------------------------------

    save_random_combined_overlay(
        gray=gray,
        edge_mask=edge_mask,
        paths=vertical_paths,
        consumed_mask=vertical_consumed_mask,
        output=vertical_output,
        color=(
            255,
            0,
            255,
        ),
    )


# ============================================================
# INTERSECTION POINT GENERATION
# ============================================================

def generate_intersection_points(
    intersection_counts,
    min_count,
    suppression_distance,
):
    """
    Generate representative intersection points from the
    accumulated random-walker intersection counts.

    Candidates are sorted by intersection frequency.
    Stronger candidates are selected first, and nearby
    candidates are suppressed.

    Returns:

        numpy array with shape (N, 2)

    Coordinates are stored as:

        (x, y)
    """

    heatmap = intersection_counts.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Find candidate pixels
    # --------------------------------------------------------

    candidates = np.argwhere(
        heatmap >= min_count
    )

    # --------------------------------------------------------
    # Sort strongest intersections first
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda p: heatmap[
            p[0],
            p[1],
        ],
        reverse=True,
    )

    intersection_points = []

    # --------------------------------------------------------
    # Non-maximum suppression
    # --------------------------------------------------------

    for y, x in candidates:

        keep = True

        for px, py in intersection_points:

            dx = x - px
            dy = y - py

            distance = np.sqrt(
                dx ** 2
                + dy ** 2
            )

            if distance < suppression_distance:

                keep = False
                break

        if keep:

            intersection_points.append(
                (
                    int(x),
                    int(y),
                )
            )

    # --------------------------------------------------------
    # No points
    # --------------------------------------------------------

    if not intersection_points:

        return np.empty(
            (0, 2),
            dtype=np.int32,
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return np.asarray(
        intersection_points,
        dtype=np.int32,
    )


# ============================================================
# SAVE INTERSECTION OVERLAY
# ============================================================

def save_intersection_overlay(
    gray,
    intersection_points,
    output,
):
    """
    Save the original grayscale image with detected
    intersection points drawn on top.

    Intersection points are shown as red circles.
    """

    image = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    for x, y in intersection_points:

        cv2.circle(
            image,
            (
                int(x),
                int(y),
            ),
            2,
            (
                0,
                0,
                255,
            ),
            -1,
            cv2.LINE_AA,
        )

    output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output),
        image,
    )

    if not success:

        raise RuntimeError(
            f"Failed to save intersection overlay: "
            f"{output}"
        )


# ============================================================
# CREATE SESSION DIRECTORY
# ============================================================

def create_session_directory(
    data_directory,
):
    """
    Create the next available session directory.

    Examples:

        data/session_01
        data/session_02
        data/session_03
        ...

    The first unused session number is selected.
    """

    session_number = 1

    while True:

        session_dir = (
            data_directory
            / f"session_{session_number:02d}"
        )

        if not session_dir.exists():

            session_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            return session_dir

        session_number += 1


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run random walkers on every image in an "
            "images directory and detect intersection points."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # ========================================================
    # INPUT
    # ========================================================

    parser.add_argument(
        "image_directory",
        type=Path,
        help=(
            "Directory containing input images."
        ),
    )

    # ========================================================
    # EDGE
    # ========================================================

    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_EDGE_THRESHOLD,
        help="Black-edge threshold.",
    )

    # ========================================================
    # RANDOM VERTICAL
    # ========================================================

    parser.add_argument(
        "--vertical-walkers",
        type=int,
        default=DEFAULT_VERTICAL_WALKERS,
    )

    parser.add_argument(
        "--vertical-consume-side",
        type=int,
        default=DEFAULT_VERTICAL_CONSUME_SIDE,
    )

    parser.add_argument(
        "--vertical-max-drift",
        type=int,
        default=DEFAULT_VERTICAL_MAX_DRIFT,
    )

    parser.add_argument(
        "--vertical-min-steps",
        type=int,
        default=DEFAULT_VERTICAL_MIN_STEPS,
    )

    # ========================================================
    # RANDOM HORIZONTAL
    # ========================================================

    parser.add_argument(
        "--horizontal-walkers",
        type=int,
        default=DEFAULT_HORIZONTAL_WALKERS,
    )

    parser.add_argument(
        "--horizontal-consume-side",
        type=int,
        default=DEFAULT_HORIZONTAL_CONSUME_SIDE,
    )

    parser.add_argument(
        "--horizontal-max-drift",
        type=int,
        default=DEFAULT_HORIZONTAL_MAX_DRIFT,
    )

    parser.add_argument(
        "--horizontal-min-steps",
        type=int,
        default=DEFAULT_HORIZONTAL_MIN_STEPS,
    )

    # ========================================================
    # INTERSECTION POINT SELECTION
    # ========================================================

    parser.add_argument(
        "--intersection-count-threshold",
        type=int,
        default=DEFAULT_INTERSECTION_COUNT_THRESHOLD,
        help=(
            "Minimum number of simulations in which an "
            "intersection must occur before it is considered "
            "an intersection point."
        ),
    )

    parser.add_argument(
        "--intersection-suppression",
        type=float,
        default=DEFAULT_INTERSECTION_SUPPRESSION,
        help=(
            "Minimum Euclidean distance between selected "
            "intersection points."
        ),
    )

    # ========================================================
    # RANDOM STATE
    # ========================================================

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    # ========================================================
    # DEBUG
    # ========================================================

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save combined horizontal and vertical "
            "random-walker debug overlays under "
            "session_XX/debug."
        ),
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_args(args):

    if not args.image_directory.is_dir():

        raise ValueError(
            f"Not a directory: "
            f"{args.image_directory}"
        )

    if args.threshold < 0:

        raise ValueError(
            "--threshold must be >= 0"
        )

    if args.vertical_walkers < 1:

        raise ValueError(
            "--vertical-walkers must be >= 1"
        )

    if args.horizontal_walkers < 1:

        raise ValueError(
            "--horizontal-walkers must be >= 1"
        )

    if args.vertical_consume_side < 0:

        raise ValueError(
            "--vertical-consume-side must be >= 0"
        )

    if args.horizontal_consume_side < 0:

        raise ValueError(
            "--horizontal-consume-side must be >= 0"
        )

    if args.vertical_max_drift < 0:

        raise ValueError(
            "--vertical-max-drift must be >= 0"
        )

    if args.horizontal_max_drift < 0:

        raise ValueError(
            "--horizontal-max-drift must be >= 0"
        )

    if args.vertical_min_steps < 0:

        raise ValueError(
            "--vertical-min-steps must be >= 0"
        )

    if args.horizontal_min_steps < 0:

        raise ValueError(
            "--horizontal-min-steps must be >= 0"
        )

    if args.intersection_count_threshold < 1:

        raise ValueError(
            "--intersection-count-threshold "
            "must be >= 1"
        )

    if args.intersection_suppression < 0:

        raise ValueError(
            "--intersection-suppression "
            "must be >= 0"
        )


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(directory):

    return [
        path
        for path in sorted(
            directory.iterdir()
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    heatmap_dir,
    counts_dir,
    intersections_dir,
    debug_horizontal_dir,
    debug_vertical_dir,
    args,
):
    """
    Process one image.

    Session outputs:

        session_XX/
            heatmaps/<stem>.png
            counts/<stem>.npy
            intersections/
                <stem>.npy
                <stem>.png

            debug/
                horizontal/<stem>.png
                vertical/<stem>.png
    """

    stem = image_path.stem

    # --------------------------------------------------------
    # Normal output paths
    # --------------------------------------------------------

    heatmap_output = (
        heatmap_dir
        / f"{stem}.png"
    )

    counts_output = (
        counts_dir
        / f"{stem}.npy"
    )

    intersection_points_output = (
        intersections_dir
        / f"{stem}.npy"
    )

    intersection_overlay_output = (
        intersections_dir
        / f"{stem}.png"
    )

    # --------------------------------------------------------
    # Debug output paths
    # --------------------------------------------------------

    horizontal_debug_output = (
        debug_horizontal_dir
        / f"{stem}.png"
    )

    vertical_debug_output = (
        debug_vertical_dir
        / f"{stem}.png"
    )

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    gray = load_image(
        image_path
    )

    # ========================================================
    # EDGE MASK
    # ========================================================

    edge_mask = make_edge_mask(
        gray,
        args.threshold,
    )

    edge_mask = cleanup_edge_mask(
        edge_mask
    )

    # ========================================================
    # STAGE 1
    # RANDOM WALKERS
    # ========================================================

    random_result = run_random_stage(
        edge_mask=edge_mask,

        num_simulations=(
            NUM_SIMULATIONS
        ),

        vertical_walkers=(
            args.vertical_walkers
        ),

        horizontal_walkers=(
            args.horizontal_walkers
        ),

        vertical_consume_side=(
            args.vertical_consume_side
        ),

        horizontal_consume_side=(
            args.horizontal_consume_side
        ),

        vertical_max_drift=(
            args.vertical_max_drift
        ),

        horizontal_max_drift=(
            args.horizontal_max_drift
        ),

        vertical_min_steps=(
            args.vertical_min_steps
        ),

        horizontal_min_steps=(
            args.horizontal_min_steps
        ),

        random_state=(
            args.random_state
        ),

        debug=args.debug,
    )

    # ========================================================
    # RANDOM WALKER DEBUG
    # ========================================================

    if args.debug:

        if not isinstance(
            random_result,
            tuple,
        ):

            raise RuntimeError(
                "Debug mode expected run_random_stage() "
                "to return the debug tuple."
            )

        if len(random_result) != 7:

            raise RuntimeError(
                "Debug mode expected "
                "run_random_stage() to return 7 values, "
                f"but received {len(random_result)}."
            )

        (
            intersection_counts,

            horizontal_left_paths,
            horizontal_right_paths,

            vertical_up_paths,
            vertical_down_paths,

            horizontal_consumed_mask,
            vertical_consumed_mask,

        ) = random_result

        save_random_debug_outputs(
            gray=gray,

            edge_mask=edge_mask,

            horizontal_left_paths=(
                horizontal_left_paths
            ),

            horizontal_right_paths=(
                horizontal_right_paths
            ),

            vertical_up_paths=(
                vertical_up_paths
            ),

            vertical_down_paths=(
                vertical_down_paths
            ),

            horizontal_consumed_mask=(
                horizontal_consumed_mask
            ),

            vertical_consumed_mask=(
                vertical_consumed_mask
            ),

            horizontal_output=(
                horizontal_debug_output
            ),

            vertical_output=(
                vertical_debug_output
            ),
        )

    else:

        intersection_counts = (
            random_result
        )

    # ========================================================
    # SAVE COUNTS
    # ========================================================

    np.save(
        str(counts_output),
        intersection_counts,
    )

    # ========================================================
    # SAVE HEATMAP
    # ========================================================

    Intersection.save_heatmap(
        gray=gray,

        intersection_counts=(
            intersection_counts
        ),

        num_simulations=(
            NUM_SIMULATIONS
        ),

        output=(
            heatmap_output
        ),
    )

    # ========================================================
    # GENERATE INTERSECTION POINTS
    # ========================================================

    intersection_points = (
        generate_intersection_points(
            intersection_counts=(
                intersection_counts
            ),

            min_count=(
                args.intersection_count_threshold
            ),

            suppression_distance=(
                args.intersection_suppression
            ),
        )
    )

    # ========================================================
    # SAVE INTERSECTION POINTS
    # ========================================================

    np.save(
        str(intersection_points_output),
        intersection_points,
    )

    # ========================================================
    # SAVE INTERSECTION VISUALIZATION
    # ========================================================

    save_intersection_overlay(
        gray=gray,

        intersection_points=(
            intersection_points
        ),

        output=(
            intersection_overlay_output
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    validate_args(
        args
    )

    image_directory = (
        args.image_directory.resolve()
    )

    # --------------------------------------------------------
    # Data directory is the parent of the image directory.
    #
    # Example:
    #
    # data/images
    #
    # becomes:
    #
    # data
    # --------------------------------------------------------

    data_directory = (
        image_directory.parent
    )

    # ========================================================
    # FIND IMAGES
    # ========================================================

    images = find_images(
        image_directory
    )

    if not images:

        print(
            f"No supported images found in "
            f"{image_directory}"
        )

        return

    # ========================================================
    # CREATE NEW SESSION
    # ========================================================

    session_dir = create_session_directory(
        data_directory
    )

    print(
        f"Created session: {session_dir}"
    )

    # ========================================================
    # SESSION OUTPUT DIRECTORIES
    # ========================================================

    heatmap_dir = (
        session_dir
        / "heatmaps"
    )

    counts_dir = (
        session_dir
        / "counts"
    )

    intersections_dir = (
        session_dir
        / "intersections"
    )

    # ========================================================
    # DEBUG DIRECTORIES
    # ========================================================

    debug_horizontal_dir = (
        session_dir
        / "debug"
        / "horizontal"
    )

    debug_vertical_dir = (
        session_dir
        / "debug"
        / "vertical"
    )

    # --------------------------------------------------------
    # Create normal directories
    # --------------------------------------------------------

    heatmap_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    intersections_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create debug directories only when requested
    # --------------------------------------------------------

    if args.debug:

        debug_horizontal_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_vertical_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # SHUFFLE IMAGES
    # ========================================================

    images = shuffle_images(
        images,
        args.random_state,
    )

    # ========================================================
    # PRINT RUN INFORMATION
    # ========================================================

    print(
        f"Input directory:  {image_directory}"
    )

    print(
        f"Output directory: {session_dir}"
    )

    print(
        f"Images:           {len(images):,}"
    )

    print(
        f"Random state:     {args.random_state}"
    )

    print(
        f"Simulations:      {NUM_SIMULATIONS}"
    )

    print(
        f"Intersection threshold: "
        f"{args.intersection_count_threshold}"
    )

    print(
        f"Intersection suppression: "
        f"{args.intersection_suppression}"
    )

    print()

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for image_path in tqdm(
        images,
        desc="Processing images",
        unit="image",
    ):

        process_image(
            image_path=image_path,

            heatmap_dir=(
                heatmap_dir
            ),

            counts_dir=(
                counts_dir
            ),

            intersections_dir=(
                intersections_dir
            ),

            debug_horizontal_dir=(
                debug_horizontal_dir
            ),

            debug_vertical_dir=(
                debug_vertical_dir
            ),

            args=args,
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print()

    print(
        f"Finished session: {session_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
