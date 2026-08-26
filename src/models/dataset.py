#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# DEFAULT DIRECTORIES
# ============================================================

DEFAULT_SAVED_WINGS_DIR = Path(
    "../../data/saved_wings"
)

DEFAULT_SEEDS_DIR = Path(
    "../../data/seeds"
)

DEFAULT_RAW_DIR = Path(
    "../../data/raw"
)


# ============================================================
# SAVED-WING SUFFIX
# ============================================================

BLACKWHITE_SUFFIX = "_blackwhite"


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
# FIND EXACT STEM MATCH
# ============================================================

def find_matching_file(
    directory,
    stem,
    extensions=None,
):
    """
    Find a file whose stem exactly matches `stem`.

    The extension is ignored unless `extensions` is provided.

    Example:

        stem = "wing_001"

    can match:

        wing_001.png
        wing_001.jpg
        wing_001.jpeg
    """

    directory = Path(
        directory
    )

    matches = []

    for path in directory.iterdir():

        if not path.is_file():
            continue

        if extensions is not None:

            if path.suffix.lower() not in extensions:
                continue

        if path.stem == stem:

            matches.append(
                path
            )

    if not matches:

        return None

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple files match "
            f"'{stem}' in {directory}:\n"
            + "\n".join(
                f"  {path.name}"
                for path in matches
            )
        )

    return matches[0]


# ============================================================
# LOAD IMAGE
# ============================================================

def load_grayscale(path):
    """
    Load an image as a grayscale numpy array.
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise RuntimeError(
            f"Could not load image: {path}"
        )

    return image


# ============================================================
# LOAD SEEDS
# ============================================================

def load_seeds(path):
    """
    Load intersection coordinates from an .npy file.

    Expected shape:

        (N, 2)

    Each row is:

        [x, y]
    """

    seeds = np.load(
        path
    )

    seeds = np.asarray(
        seeds
    )

    # --------------------------------------------------------
    # Empty seed file
    # --------------------------------------------------------

    if seeds.size == 0:

        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Single coordinate
    # --------------------------------------------------------

    if seeds.ndim == 1:

        if seeds.size != 2:

            raise ValueError(
                f"Invalid seed array in {path}. "
                f"Expected Nx2, got shape "
                f"{seeds.shape}"
            )

        seeds = seeds.reshape(
            1,
            2,
        )

    # --------------------------------------------------------
    # Validate Nx2
    # --------------------------------------------------------

    if (
        seeds.ndim != 2
        or seeds.shape[1] != 2
    ):

        raise ValueError(
            f"Invalid seed array in {path}. "
            f"Expected Nx2, got shape "
            f"{seeds.shape}"
        )

    return seeds.astype(
        np.float32
    )


# ============================================================
# GENERATE HEATMAP
# ============================================================

def make_heatmap(
    seeds,
    height,
    width,
    sigma=3.0,
):
    """
    Convert intersection coordinates into a Gaussian
    heatmap.

    Coordinates are:

        [x, y]

    Heatmap shape:

        (height, width)

    Each seed produces a Gaussian centered at that
    coordinate.
    """

    heatmap = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    if len(seeds) == 0:

        return heatmap

    # --------------------------------------------------------
    # Gaussian radius
    # --------------------------------------------------------

    radius = int(
        np.ceil(
            sigma * 3.0
        )
    )

    # --------------------------------------------------------
    # Precompute Gaussian
    # --------------------------------------------------------

    axis = np.arange(
        -radius,
        radius + 1,
        dtype=np.float32,
    )

    xx, yy = np.meshgrid(
        axis,
        axis,
    )

    gaussian = np.exp(
        -(
            xx * xx
            + yy * yy
        )
        / (
            2.0 * sigma * sigma
        )
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Add each seed
    # --------------------------------------------------------

    for seed in seeds:

        x = int(
            round(
                float(seed[0])
            )
        )

        y = int(
            round(
                float(seed[1])
            )
        )

        # ----------------------------------------------------
        # Ignore seeds outside canvas
        # ----------------------------------------------------

        if (
            x < 0
            or x >= width
            or y < 0
            or y >= height
        ):

            continue

        # ----------------------------------------------------
        # Image region
        # ----------------------------------------------------

        x0 = max(
            0,
            x - radius,
        )

        x1 = min(
            width,
            x + radius + 1,
        )

        y0 = max(
            0,
            y - radius,
        )

        y1 = min(
            height,
            y + radius + 1,
        )

        # ----------------------------------------------------
        # Gaussian region
        # ----------------------------------------------------

        gx0 = x0 - (
            x - radius
        )

        gx1 = gx0 + (
            x1 - x0
        )

        gy0 = y0 - (
            y - radius
        )

        gy1 = gy0 + (
            y1 - y0
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        heatmap[
            y0:y1,
            x0:x1,
        ] = np.maximum(
            heatmap[
                y0:y1,
                x0:x1,
            ],
            gaussian[
                gy0:gy1,
                gx0:gx1,
            ],
        )

    return heatmap


# ============================================================
# DATASET
# ============================================================

class IntersectionDataset(Dataset):
    """
    Dataset for full-wing intersection detection.

    Expected directory structure:

        data/
        ├── saved_wings/
        │   ├── foo_blackwhite.png
        │   └── bar_blackwhite.png
        │
        ├── seeds/
        │   ├── foo_blackwhite.npy
        │   └── bar_blackwhite.npy
        │
        └── raw/
            ├── foo.jpeg
            └── bar.jpeg


    Matching rules:

        saved_wings -> seeds
            EXACT stem match

            foo_blackwhite.png
                    ↓
            foo_blackwhite.npy


        saved_wings -> raw
            REMOVE "_blackwhite"

            foo_blackwhite.png
                    ↓
            foo.jpeg


    Input:

        Raw grayscale wing image.

    Target:

        Gaussian heatmap generated from the matching
        intersection coordinates.

    Padding:

        Images are padded on the bottom and right.

        No resizing is performed.

        No padding is added to the top or left.

        Therefore seed coordinates remain unchanged.
    """

    def __init__(
        self,
        saved_wings_dir=DEFAULT_SAVED_WINGS_DIR,
        seeds_dir=DEFAULT_SEEDS_DIR,
        raw_dir=DEFAULT_RAW_DIR,
        sigma=3.0,
        target_height=None,
        target_width=None,
    ):

        self.saved_wings_dir = Path(
            saved_wings_dir
        )

        self.seeds_dir = Path(
            seeds_dir
        )

        self.raw_dir = Path(
            raw_dir
        )

        self.sigma = float(
            sigma
        )

        # ====================================================
        # VALIDATE DIRECTORIES
        # ====================================================

        if not self.saved_wings_dir.is_dir():

            raise FileNotFoundError(
                "Saved wings directory does not exist: "
                f"{self.saved_wings_dir}"
            )

        if not self.seeds_dir.is_dir():

            raise FileNotFoundError(
                "Seeds directory does not exist: "
                f"{self.seeds_dir}"
            )

        if not self.raw_dir.is_dir():

            raise FileNotFoundError(
                "Raw image directory does not exist: "
                f"{self.raw_dir}"
            )

        # ====================================================
        # FIND SAVED WINGS
        # ====================================================

        self.wings = sorted(
            path
            for path in self.saved_wings_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )

        if not self.wings:

            raise RuntimeError(
                "No supported wing images found in "
                f"{self.saved_wings_dir}"
            )

        # ====================================================
        # BUILD SAMPLE MAPPINGS
        # ====================================================

        self.samples = []

        missing_seeds = []
        missing_raw = []

        for wing_path in self.wings:

            # =================================================
            # SAVED WING STEM
            # =================================================

            saved_stem = wing_path.stem

            # =================================================
            # SEED MATCH
            #
            # KEEP "_blackwhite"
            # =================================================

            seed_path = find_matching_file(
                self.seeds_dir,
                saved_stem,
                extensions={
                    ".npy"
                },
            )

            if seed_path is None:

                missing_seeds.append(
                    wing_path.name
                )

                continue

            # =================================================
            # RAW MATCH
            #
            # REMOVE "_blackwhite"
            # =================================================

            if not saved_stem.endswith(
                BLACKWHITE_SUFFIX
            ):

                raise ValueError(
                    "Saved wing does not end with "
                    f"'{BLACKWHITE_SUFFIX}': "
                    f"{wing_path.name}"
                )

            raw_stem = saved_stem[
                : -len(
                    BLACKWHITE_SUFFIX
                )
            ]

            # =================================================
            # FIND RAW IMAGE
            # =================================================

            raw_path = find_matching_file(
                self.raw_dir,
                raw_stem,
                extensions=SUPPORTED_EXTENSIONS,
            )

            if raw_path is None:

                missing_raw.append(
                    (
                        wing_path.name,
                        raw_stem,
                    )
                )

                continue

            # =================================================
            # COMPLETE SAMPLE
            # =================================================

            self.samples.append(
                {
                    "saved_wing": wing_path,
                    "seeds": seed_path,
                    "raw": raw_path,
                }
            )

        # ====================================================
        # REPORT MISSING SEEDS
        # ====================================================

        if missing_seeds:

            print()
            print(
                f"Warning: {len(missing_seeds):,} wings "
                "have no matching seed file."
            )

            for name in missing_seeds:

                print(
                    f"  Missing seeds: {name}"
                )

        # ====================================================
        # REPORT MISSING RAW
        # ====================================================

        if missing_raw:

            print()
            print(
                f"Warning: {len(missing_raw):,} wings "
                "have no matching raw image."
            )

            for wing_name, raw_stem in missing_raw:

                print(
                    f"  Saved wing: "
                    f"{wing_name}"
                )

                print(
                    f"  Searched raw stem: "
                    f"{raw_stem}"
                )

        # ====================================================
        # REQUIRE AT LEAST ONE SAMPLE
        # ====================================================

        if not self.samples:

            raise RuntimeError(
                "No complete training samples found."
            )

        # ====================================================
        # DETERMINE CANVAS SIZE
        # ====================================================

        if (
            target_height is None
            or target_width is None
        ):

            max_height = 0
            max_width = 0

            for sample in self.samples:

                image = load_grayscale(
                    sample["raw"]
                )

                height, width = (
                    image.shape
                )

                max_height = max(
                    max_height,
                    height,
                )

                max_width = max(
                    max_width,
                    width,
                )

            if target_height is None:

                target_height = max_height

            if target_width is None:

                target_width = max_width

        self.target_height = int(
            target_height
        )

        self.target_width = int(
            target_width
        )

        # ====================================================
        # VERIFY ALL IMAGES FIT
        # ====================================================

        for sample in self.samples:

            image = load_grayscale(
                sample["raw"]
            )

            height, width = (
                image.shape
            )

            if (
                height > self.target_height
                or width > self.target_width
            ):

                raise ValueError(
                    f"Image {sample['raw']} "
                    f"is {width}x{height}, "
                    f"which exceeds target canvas "
                    f"{self.target_width}x"
                    f"{self.target_height}."
                )

        # ====================================================
        # REPORT
        # ====================================================

        print()
        print(
            f"Loaded {len(self.samples):,} "
            "complete training samples."
        )

        print(
            f"Canvas size: "
            f"{self.target_width}x"
            f"{self.target_height}"
        )

        print(
            f"Heatmap sigma: "
            f"{self.sigma}"
        )

        # ====================================================
        # SHOW EXAMPLE MAPPING
        # ====================================================

        first = self.samples[0]

        print()
        print(
            "Example mapping:"
        )

        print(
            f"  saved wing : "
            f"{first['saved_wing'].name}"
        )

        print(
            f"  seed       : "
            f"{first['seeds'].name}"
        )

        print(
            f"  raw        : "
            f"{first['raw'].name}"
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.samples
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[
            index
        ]

        # ====================================================
        # LOAD RAW IMAGE
        # ====================================================

        image = load_grayscale(
            sample["raw"]
        )

        height, width = (
            image.shape
        )

        # ====================================================
        # PAD IMAGE
        #
        # Original image is placed at:
        #
        #     [0:height, 0:width]
        #
        # Nothing is added to the top or left.
        #
        # Therefore:
        #
        #     seed (x, y)
        #
        # continues to refer to exactly the same pixel.
        # ====================================================

        padded_image = np.zeros(
            (
                self.target_height,
                self.target_width,
            ),
            dtype=np.uint8,
        )

        padded_image[
            :height,
            :width,
        ] = image

        # ====================================================
        # LOAD SEEDS
        # ====================================================

        seeds = load_seeds(
            sample["seeds"]
        )

        # ====================================================
        # GENERATE TARGET HEATMAP
        # ====================================================

        heatmap = make_heatmap(
            seeds,
            self.target_height,
            self.target_width,
            sigma=self.sigma,
        )

        # ====================================================
        # NORMALIZE IMAGE
        # ====================================================

        image_float = (
            padded_image.astype(
                np.float32
            )
            / 255.0
        )

        # ====================================================
        # IMAGE TENSOR
        #
        # H x W
        #
        # ->
        #
        # 1 x H x W
        # ====================================================

        image_tensor = (
            torch.from_numpy(
                image_float
            )
            .unsqueeze(0)
        )

        # ====================================================
        # TARGET TENSOR
        # ====================================================

        heatmap_tensor = (
            torch.from_numpy(
                heatmap
            )
            .unsqueeze(0)
        )

        return (
            image_tensor,
            heatmap_tensor,
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Testing IntersectionDataset..."
    )

    print()

    dataset = IntersectionDataset()

    # ========================================================
    # DATASET SIZE
    # ========================================================

    print()
    print(
        f"Dataset length: "
        f"{len(dataset):,}"
    )

    # ========================================================
    # LOAD FIRST SAMPLE
    # ========================================================

    image, target = dataset[0]

    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    print()
    print(
        "Image:"
    )

    print(
        f"  shape = "
        f"{tuple(image.shape)}"
    )

    print(
        f"  dtype = "
        f"{image.dtype}"
    )

    print(
        f"  min   = "
        f"{image.min().item():.4f}"
    )

    print(
        f"  max   = "
        f"{image.max().item():.4f}"
    )

    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    print()
    print(
        "Target:"
    )

    print(
        f"  shape = "
        f"{tuple(target.shape)}"
    )

    print(
        f"  dtype = "
        f"{target.dtype}"
    )

    print(
        f"  min   = "
        f"{target.min().item():.4f}"
    )

    print(
        f"  max   = "
        f"{target.max().item():.4f}"
    )

    # ========================================================
    # FILE MAPPING
    # ========================================================

    sample = dataset.samples[0]

    print()
    print(
        "Files:"
    )

    print(
        f"  saved wing = "
        f"{sample['saved_wing']}"
    )

    print(
        f"  seeds      = "
        f"{sample['seeds']}"
    )

    print(
        f"  raw image  = "
        f"{sample['raw']}"
    )

    # ========================================================
    # SEED INFORMATION
    # ========================================================

    seeds = load_seeds(
        sample["seeds"]
    )

    print()
    print(
        f"Seed count: "
        f"{len(seeds):,}"
    )

    if len(seeds) > 0:

        print()
        print(
            "First 10 seeds:"
        )

        print(
            seeds[:10]
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "Dataset test successful."
    )
