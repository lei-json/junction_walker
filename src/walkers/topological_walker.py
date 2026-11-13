#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


class TopologicalWalker:
    """
    Omnidirectional topological vein walker for dragonfly wings.

    Traces the physical vein skeleton and detects genuine
    branch points, crossings, and bifurcations without cardinal
    direction bias or drift limitations.
    """

    def __init__(
        self,
        edge_mask,
        min_component_size=5,
        spur_threshold=3,
    ):
        """
        Parameters
        ----------
        edge_mask : numpy.ndarray (H, W)
            Binary edge mask where vein pixels are 1 (True)
            and background is 0 (False).
        min_component_size : int
            Remove connected components smaller than this size.
        spur_threshold : int
            Prune open boundary spurs shorter than this pixel distance.
        """
        self.edge_mask = edge_mask.astype(np.uint8)
        self.min_component_size = int(min_component_size)
        self.spur_threshold = int(spur_threshold)
        self.height, self.width = edge_mask.shape

    def clean_mask(self):
        """
        Remove isolated noise specs smaller than min_component_size.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            self.edge_mask, connectivity=8
        )
        cleaned = np.zeros_like(self.edge_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= self.min_component_size:
                cleaned[labels == label] = 1
        return cleaned

    def compute_skeleton(self, cleaned_mask=None):
        """
        Compute the 1-pixel-wide topological centerline.
        """
        if cleaned_mask is None:
            cleaned_mask = self.clean_mask()

        skeleton = skeletonize(cleaned_mask > 0).astype(np.uint8)

        # Clear borders to avoid border junction artifacts
        skeleton[0, :] = 0
        skeleton[-1, :] = 0
        skeleton[:, 0] = 0
        skeleton[:, -1] = 0

        return skeleton

    def find_junctions(self):
        """
        Detect exact junction coordinates using topological crossing number
        analysis and cluster contraction.

        Returns
        -------
        numpy.ndarray of shape (N, 2)
            Detected (x, y) coordinates of vein junctions.
        """
        cleaned = self.clean_mask()
        skeleton = self.compute_skeleton(cleaned)

        # 8-connected circular neighborhood offsets
        p = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0),
            (1, -1), (0, -1),
        ]
        pad = np.pad(skeleton, 1, mode="constant")
        h, w = self.height, self.width
        nbrs = [pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in p]

        # Rutovitz Crossing Number: 0.5 * sum(|p_i - p_{i+1}|)
        crossing_num = np.zeros_like(skeleton, dtype=np.float32)
        for i in range(8):
            crossing_num += np.abs(
                nbrs[i].astype(np.float32) - nbrs[(i + 1) % 8].astype(np.float32)
            )
        crossing_num = (0.5 * crossing_num) * skeleton

        # Degree (neighbor count)
        degree = np.zeros_like(skeleton, dtype=np.int32)
        for n in nbrs:
            degree += n
        degree = degree * skeleton

        # Junction candidates: crossing number >= 3 or (crossing number == 2 and degree >= 4)
        junc_cand = ((crossing_num >= 3) | ((crossing_num >= 2) & (degree >= 4))).astype(np.uint8)

        # Cluster connected adjacent candidate pixels into a single representative node
        num_j, j_labels, j_stats, centroids = cv2.connectedComponentsWithStats(
            junc_cand, connectivity=8
        )

        junctions = []
        for j in range(1, num_j):
            cx, cy = centroids[j]
            pts = np.argwhere(j_labels == j)  # (y, x)
            dists = (pts[:, 1] - cx) ** 2 + (pts[:, 0] - cy) ** 2
            best_pt = pts[np.argmin(dists)]
            junctions.append((int(best_pt[1]), int(best_pt[0])))

        if not junctions:
            return np.empty((0, 2), dtype=np.int32)

        junctions = np.array(junctions, dtype=np.int32)

        # Prune boundary roughness spurs: junctions connected to open endpoints within spur_threshold
        if self.spur_threshold > 0:
            endpoints = np.argwhere((crossing_num == 1) & (skeleton == 1))  # (y, x)
            if len(endpoints) > 0 and len(junctions) > 0:
                ep_xy = np.column_stack([endpoints[:, 1], endpoints[:, 0]])
                tree_ep = cKDTree(ep_xy)
                dists, _ = tree_ep.query(junctions)
                junctions = junctions[dists > self.spur_threshold]

        return junctions
