#!/usr/bin/env python3

from pathlib import Path
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_SEEDS_DIR = Path("data/seeds")
DEFAULT_IMAGE_DIR = Path("data/rotate")
DEFAULT_BW_DIR = Path("data/blackwhite_rotate")

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


# ============================================================
# HEATMAP SYNTHESIS
# ============================================================

def make_heatmap(seeds, height, width, sigma=2.0):
    """
    Generate peak-normalized continuous Gaussian heatmap from discrete coordinates.
    Peak amplitude at each seed center is exactly 1.0.
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    if len(seeds) == 0:
        return heatmap

    radius = int(np.ceil(sigma * 3.0))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    gaussian = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)

    for seed in seeds:
        x = int(round(float(seed[0])))
        y = int(round(float(seed[1])))

        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)

        gx0 = x0 - (x - radius)
        gx1 = gx0 + (x1 - x0)
        gy0 = y0 - (y - radius)
        gy1 = gy0 + (y1 - y0)

        heatmap[y0:y1, x0:x1] = np.maximum(
            heatmap[y0:y1, x0:x1],
            gaussian[gy0:gy1, gx0:gx1],
        )

    return heatmap


# ============================================================
# JUNCTION DATASET
# ============================================================

class JunctionDataset(Dataset):
    """
    Dataset for dragonfly wing vein junction detection.
    Supports arbitrary resolution padding to multiples of 16,
    coordinate-preserving data augmentation, and patch extraction.
    """

    def __init__(
        self,
        seeds_dir=DEFAULT_SEEDS_DIR,
        image_dir=DEFAULT_IMAGE_DIR,
        bw_dir=DEFAULT_BW_DIR,
        sigma=2.0,
        augment=True,
        crop_size=None,  # (H, W) or None for full image
        max_samples=None,
    ):
        super().__init__()
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.seeds_dir = Path(seeds_dir) if Path(seeds_dir).exists() else repo_root / seeds_dir
        self.image_dir = Path(image_dir) if Path(image_dir).exists() else repo_root / image_dir
        self.bw_dir = Path(bw_dir) if Path(bw_dir).exists() else repo_root / bw_dir
        self.sigma = float(sigma)
        self.augment = augment
        self.crop_size = crop_size

        self.samples = []
        self._discover_samples(max_samples)

    def _discover_samples(self, max_samples):
        if not self.seeds_dir.exists():
            raise FileNotFoundError(f"Seeds directory not found: {self.seeds_dir}")

        seed_files = sorted(self.seeds_dir.glob("*.npy"))
        for seed_path in seed_files:
            stem = seed_path.stem
            clean_stem = stem.replace("_blackwhite", "")

            # Look for image in image_dir or bw_dir
            img_path = None
            for d in [self.image_dir, self.bw_dir]:
                for ext in SUPPORTED_EXTENSIONS:
                    candidate1 = d / f"{stem}{ext}"
                    candidate2 = d / f"{clean_stem}{ext}"
                    candidate3 = d / f"{clean_stem}_blackwhite{ext}"
                    if candidate1.exists():
                        img_path = candidate1
                        break
                    elif candidate2.exists():
                        img_path = candidate2
                        break
                    elif candidate3.exists():
                        img_path = candidate3
                        break
                if img_path is not None:
                    break

            if img_path is not None:
                self.samples.append((img_path, seed_path))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"JunctionDataset initialized with {len(self.samples)} paired samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, seed_path = self.samples[idx]

        # Load grayscale image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        # Load discrete seeds (x, y)
        seeds = np.load(str(seed_path))
        if seeds.ndim == 1:
            seeds = seeds.reshape(-1, 2)
        seeds = seeds.astype(np.float32)

        h, w = img.shape

        # ----------------------------------------------------
        # Data Augmentation
        # ----------------------------------------------------
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                if len(seeds) > 0:
                    seeds[:, 0] = (w - 1) - seeds[:, 0]

            # Vertical flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                if len(seeds) > 0:
                    seeds[:, 1] = (h - 1) - seeds[:, 1]

        # ----------------------------------------------------
        # Crop or Pad to fixed crop_size or multiple of 16
        # ----------------------------------------------------
        if self.crop_size is not None:
            ch, cw = self.crop_size
            # Height
            if h > ch:
                y0 = random.randint(0, h - ch) if self.augment else (h - ch) // 2
                img = img[y0:y0 + ch, :]
                if len(seeds) > 0:
                    valid_y = (seeds[:, 1] >= y0) & (seeds[:, 1] < y0 + ch)
                    seeds = seeds[valid_y]
                    seeds[:, 1] -= y0
                h = ch
            elif h < ch:
                pad_bottom = ch - h
                img = np.pad(img, ((0, pad_bottom), (0, 0)), mode="constant", constant_values=255)
                h = ch

            # Width
            if w > cw:
                x0 = random.randint(0, w - cw) if self.augment else (w - cw) // 2
                img = img[:, x0:x0 + cw]
                if len(seeds) > 0:
                    valid_x = (seeds[:, 0] >= x0) & (seeds[:, 0] < x0 + cw)
                    seeds = seeds[valid_x]
                    seeds[:, 0] -= x0
                w = cw
            elif w < cw:
                pad_right = cw - w
                img = np.pad(img, ((0, 0), (0, pad_right)), mode="constant", constant_values=255)
                w = cw
        else:
            # Pad to nearest multiple of 16
            pad_h = (16 - (h % 16)) % 16
            pad_w = (16 - (w % 16)) % 16
            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
                h, w = img.shape

        # Generate continuous heatmap
        heatmap = make_heatmap(seeds, h, w, sigma=self.sigma)

        # Normalize image to [0, 1]
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        heat_tensor = torch.from_numpy(heatmap).unsqueeze(0)

        return img_tensor, heat_tensor, str(img_path)


# Compatibility alias
IntersectionDataset = JunctionDataset
