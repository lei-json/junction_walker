#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np


# ============================================================
# RANDOM WALKER DEBUG OVERLAY
# ============================================================

def save_random_combined_overlay(
    gray,
    edge_mask,
    paths,
    output,
    color,
):
    """
    Draw all random-walker paths over the original image.

    The edge mask is shown in orange and the walker paths
    are shown using the supplied BGR color.
    """

    image = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    # Original edge mask
    image[
        edge_mask > 0
    ] = (
        255,
        180,
        0,
    )

    # Walker paths
    for path in paths:

        if not path:
            continue

        for i in range(
            1,
            len(path),
        ):

            x1, y1 = path[i - 1]
            x2, y2 = path[i]

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
