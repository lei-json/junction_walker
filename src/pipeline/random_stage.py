import numpy as np

from walkers.random_walkers.horizontal_random import HorizontalRandom
from walkers.random_walkers.vertical_random import VerticalRandom
from walkers.intersection.intersection import Intersection


def run_random_stage(
    edge_mask,
    num_simulations,
    vertical_walkers,
    horizontal_walkers,
    vertical_consume_side,
    horizontal_consume_side,
    vertical_max_drift,
    horizontal_max_drift,
    vertical_min_steps,
    horizontal_min_steps,
    random_state,
    debug=False,
):
    """
    Run random walkers and accumulate intersection counts.

    Normal mode:
        Returns:
            intersection_counts

    Debug mode:
        Returns:
            (
                intersection_counts,

                horizontal_left_paths,
                horizontal_right_paths,

                vertical_up_paths,
                vertical_down_paths,

                horizontal_consumed_mask,
                vertical_consumed_mask,
            )

    Debug mode records ONLY the first simulation's paths and
    consumed masks.

    All simulations are still executed, and all simulations
    still contribute to intersection_counts.

    Therefore debug mode does not change the actual random
    walker or intersection behavior of the pipeline.
    """

    # ========================================================
    # INTERSECTION COUNTS
    # ========================================================

    intersection_counts = np.zeros(
        edge_mask.shape,
        dtype=np.uint32,
    )

    # ========================================================
    # DEBUG STORAGE
    #
    # Only created when debug=True.
    #
    # These contain ONLY simulation 0.
    # ========================================================

    if debug:

        all_horizontal_left_paths = []
        all_horizontal_right_paths = []

        all_vertical_up_paths = []
        all_vertical_down_paths = []

        horizontal_consumed_mask = np.zeros(
            edge_mask.shape,
            dtype=bool,
        )

        vertical_consumed_mask = np.zeros(
            edge_mask.shape,
            dtype=bool,
        )

    # ========================================================
    # SIMULATIONS
    # ========================================================

    for simulation_index in range(
        num_simulations
    ):

        # ----------------------------------------------------
        # Separate deterministic RNGs.
        #
        # These are identical regardless of debug mode.
        # ----------------------------------------------------

        vertical_rng = np.random.default_rng(
            random_state
            + simulation_index * 2
        )

        horizontal_rng = np.random.default_rng(
            random_state
            + simulation_index * 2
            + 1
        )

        # ====================================================
        # VERTICAL RANDOM WALKERS
        # ====================================================

        vertical_random = VerticalRandom(
            edge_mask=edge_mask,
            walkers=vertical_walkers,
            consume_side=vertical_consume_side,
            max_drift=vertical_max_drift,
            min_steps=vertical_min_steps,
            rng=vertical_rng,
        )

        (
            vertical_up_paths,
            vertical_down_paths,
            vertical_consumed,
        ) = vertical_random.run()

        # ----------------------------------------------------
        # DEBUG ONLY
        #
        # Record ONLY the first simulation.
        #
        # simulation_index == 0
        # ----------------------------------------------------

        if debug and simulation_index == 0:

            all_vertical_up_paths.extend(
                vertical_up_paths
            )

            all_vertical_down_paths.extend(
                vertical_down_paths
            )

            vertical_consumed_mask |= (
                vertical_consumed > 0
            )

        # ----------------------------------------------------
        # Combine directions for intersection detection.
        #
        # This happens for EVERY simulation.
        # ----------------------------------------------------

        vertical_paths = (
            vertical_up_paths
            + vertical_down_paths
        )

        # ====================================================
        # HORIZONTAL RANDOM WALKERS
        # ====================================================

        horizontal_random = HorizontalRandom(
            edge_mask=edge_mask,
            walkers=horizontal_walkers,
            consume_side=horizontal_consume_side,
            max_drift=horizontal_max_drift,
            min_steps=horizontal_min_steps,
            rng=horizontal_rng,
        )

        (
            horizontal_left_paths,
            horizontal_right_paths,
            horizontal_consumed,
        ) = horizontal_random.run()

        # ----------------------------------------------------
        # DEBUG ONLY
        #
        # Record ONLY the first simulation.
        #
        # simulation_index == 0
        # ----------------------------------------------------

        if debug and simulation_index == 0:

            all_horizontal_left_paths.extend(
                horizontal_left_paths
            )

            all_horizontal_right_paths.extend(
                horizontal_right_paths
            )

            horizontal_consumed_mask |= (
                horizontal_consumed > 0
            )

        # ----------------------------------------------------
        # Combine directions for intersection detection.
        #
        # This happens for EVERY simulation.
        # ----------------------------------------------------

        horizontal_paths = (
            horizontal_left_paths
            + horizontal_right_paths
        )

        # ====================================================
        # FIND INTERSECTIONS
        # ====================================================

        intersections = Intersection.find(
            vertical_paths=vertical_paths,
            horizontal_paths=horizontal_paths,
            image_shape=edge_mask.shape,
        )

        # ====================================================
        # ACCUMULATE
        #
        # Every simulation contributes.
        # ====================================================

        Intersection.accumulate(
            intersection_counts,
            intersections,
        )

    # ========================================================
    # RETURN
    # ========================================================

    if debug:

        return (
            intersection_counts,

            all_horizontal_left_paths,
            all_horizontal_right_paths,

            all_vertical_up_paths,
            all_vertical_down_paths,

            horizontal_consumed_mask,
            vertical_consumed_mask,
        )

    # ========================================================
    # NORMAL MODE
    # ========================================================

    return intersection_counts
