#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Intersection:
    """
    Center-line intersection detection and accumulation.

    Intersections are detected using ONLY the actual center-line
    pixels recorded in walker paths.

    The consumed footprint of a walker is NOT used for
    intersection detection.
    """

    # ========================================================
    # BUILD CENTER-LINE MASK
    # ========================================================

    @staticmethod
    def paths_to_center_mask(
        paths,
        image_shape,
    ):
        """
        Convert walker paths into a boolean center-line mask.

        Every (x, y) coordinate appearing in a path becomes
        True in the resulting image.

        Only actual path positions are included.
        """

        center_mask = np.zeros(
            image_shape,
            dtype=bool,
        )

        for path in paths:

            for x, y in path:

                # Paths produced by the walkers should already
                # be inside the image, but keep the bounds check
                # here so this utility is safe to use independently.

                if (
                    0 <= x < image_shape[1]
                    and 0 <= y < image_shape[0]
                ):

                    center_mask[
                        y,
                        x,
                    ] = True

        return center_mask

    # ========================================================
    # FIND INTERSECTIONS
    # ========================================================

    @classmethod
    def find(
        cls,
        vertical_paths,
        horizontal_paths,
        image_shape,
    ):
        """
        Find exact pixel intersections between vertical and
        horizontal walker center-lines.

        An intersection occurs when the same image coordinate
        appears in both a vertical path and a horizontal path.

        Returns
        -------
        numpy.ndarray
            Boolean image mask containing the intersection
            pixels.
        """

        # ----------------------------------------------------
        # Vertical center-line mask
        # ----------------------------------------------------

        vertical_mask = cls.paths_to_center_mask(
            paths=vertical_paths,
            image_shape=image_shape,
        )

        # ----------------------------------------------------
        # Horizontal center-line mask
        # ----------------------------------------------------

        horizontal_mask = cls.paths_to_center_mask(
            paths=horizontal_paths,
            image_shape=image_shape,
        )

        # ----------------------------------------------------
        # Exact pixel intersection
        # ----------------------------------------------------

        intersections = (
            vertical_mask
            & horizontal_mask
        )

        return intersections

    # ========================================================
    # ACCUMULATE
    # ========================================================

    @staticmethod
    def accumulate(
        counts,
        intersections,
    ):
        """
        Add one to the accumulated intersection count at every
        pixel where an intersection occurred in the current
        simulation.

        Parameters
        ----------
        counts : numpy.ndarray
            Accumulated intersection-frequency image.

        intersections : numpy.ndarray
            Boolean mask for intersections from one simulation.
        """

        counts[
            intersections
        ] += 1

    # ========================================================
    # SAVE HEATMAP
    # ========================================================

    @staticmethod
    def save_heatmap(
        gray,
        intersection_counts,
        num_simulations,
        output,
    ):
        """
        Save an accumulated intersection heatmap over the
        original grayscale image.

        The heatmap represents how frequently each pixel was
        identified as an intersection across simulations.

        Zero-count pixels are transparent so the original image
        remains visible underneath.

        The color scale is fixed to:

            1  -> minimum visible intersection frequency
            30 -> maximum displayed frequency

        Values above 30 are clipped to the maximum color.
        """

        output = Path(
            output
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # FIGURE
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(12, 10)
        )

        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================

        ax.imshow(
            gray,
            cmap="gray",
            interpolation="nearest",
        )

        # ====================================================
        # HEATMAP
        # ====================================================

        heatmap = (
            intersection_counts.astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Hide pixels with zero intersections.
        # ----------------------------------------------------

        masked_heatmap = np.ma.masked_where(
            heatmap == 0,
            heatmap,
        )

        image = ax.imshow(
            masked_heatmap,
            cmap="turbo",
            interpolation="nearest",
            alpha=1.0,
            vmin=1,
            vmax=30,
        )

        # ====================================================
        # COLORBAR
        # ====================================================

        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        colorbar.set_label(
            "Number of simulations containing an intersection",
            rotation=270,
            labelpad=20,
        )

        colorbar.set_ticks(
            [
                0,
                5,
                10,
                15,
                20,
                25,
                30,
            ]
        )

        colorbar.set_ticklabels(
            [
                "0",
                "5",
                "10",
                "15",
                "20",
                "25",
                "30+",
            ]
        )

        # ====================================================
        # STATISTICS
        # ====================================================

        total_intersection_pixels = int(
            np.count_nonzero(
                intersection_counts
            )
        )

        max_count = int(
            intersection_counts.max()
        )

        # ====================================================
        # TITLE
        # ====================================================

        ax.set_title(
            "Accumulated Vertical/Horizontal Walker Intersections\n"
            f"{num_simulations} simulations | "
            f"max frequency = "
            f"{max_count}/{num_simulations} | "
            f"intersection pixels = "
            f"{total_intersection_pixels:,}"
        )

        ax.set_axis_off()

        # ====================================================
        # LEGEND
        # ====================================================

        legend_text = (
            "Intersection frequency\n"
            "0 = no intersection\n"
            f"1 = intersection in 1/{num_simulations} simulations\n"
            f"{num_simulations // 2} = intersection in "
            f"{num_simulations // 2}/{num_simulations} simulations\n"
            f"{num_simulations} = intersection in every simulation"
        )

        ax.text(
            0.01,
            0.01,
            legend_text,
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="black",
                alpha=0.7,
            ),
        )

        # ====================================================
        # SAVE
        # ====================================================

        plt.tight_layout()

        fig.savefig(
            str(output),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )
