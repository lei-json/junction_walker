import numpy as np


class HorizontalRandom:
    """Random horizontal walker simulation.

    Walkers move one pixel horizontally per step and may drift by -1, 0, or
    +1 pixels vertically. The left and right halves of a path share a working
    consumption mask.
    """

    def __init__(self, edge_mask, walkers=5000, consume_side=3,
                 max_drift=1, min_steps=5, rng=None):
        self.edge_mask = edge_mask.astype(bool, copy=False)
        self.walkers = int(walkers)
        self.consume_side = int(consume_side)
        self.max_drift = int(max_drift)
        self.min_steps = int(min_steps)
        self.rng = rng if rng is not None else np.random.default_rng()

    def consumption_pixels(self, position, consumed_mask):
        x, y = position
        h, w = consumed_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return []

        y0 = max(0, y - self.consume_side)
        y1 = min(h - 1, y + self.consume_side)
        return [(x, py) for py in range(y0, y1 + 1)]

    def footprint_is_free(self, position, consumed_mask):
        pixels = self.consumption_pixels(position, consumed_mask)
        if not pixels:
            return False
        return all(not consumed_mask[y, x] for x, y in pixels)

    def consume_position(self, position, consumed_mask):
        for x, y in self.consumption_pixels(position, consumed_mask):
            consumed_mask[y, x] = True

    def generate_seeds(self, consumed_mask):
        ys, xs = np.where(self.edge_mask)
        if len(xs) == 0:
            return []

        indices = np.arange(len(xs))
        self.rng.shuffle(indices)
        seeds = []

        for index in indices:
            seed = (int(xs[index]), int(ys[index]))
            if not self.footprint_is_free(seed, consumed_mask):
                continue
            seeds.append(seed)
            if len(seeds) >= self.walkers:
                break

        return seeds

    def choose_next(self, current, seed, direction, consumed_mask):
        x, y = current
        seed_x, seed_y = seed

        if direction == "left":
            dx = -1
        elif direction == "right":
            dx = 1
        else:
            raise ValueError(f"Invalid direction: {direction}")

        h, w = self.edge_mask.shape
        candidates = []

        for dy in (-1, 0, 1):
            nx = x + dx
            ny = y + dy

            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if abs(ny - seed_y) > self.max_drift:
                continue
            if not self.edge_mask[ny, nx]:
                continue

            candidate = (nx, ny)
            if not self.footprint_is_free(candidate, consumed_mask):
                continue
            candidates.append(candidate)

        if not candidates:
            return None
        return candidates[self.rng.integers(len(candidates))]

    def walk_direction(self, start, seed, direction, consumed_mask):
        position = start
        path = [position]

        while True:
            next_position = self.choose_next(
                current=position,
                seed=seed,
                direction=direction,
                consumed_mask=consumed_mask,
            )
            if next_position is None:
                break

            dx = next_position[0] - position[0]
            dy = next_position[1] - position[1]
            assert dx == (-1 if direction == "left" else 1)
            assert dy in (-1, 0, 1)
            assert abs(next_position[1] - seed[1]) <= self.max_drift

            position = next_position
            path.append(position)
            self.consume_position(position, consumed_mask)

        return path

    def teleport_right_of_seed(self, seed, consumed_mask):
        seed_x, seed_y = seed
        h, w = self.edge_mask.shape
        candidates = []

        for dy in (-1, 0, 1):
            x = seed_x + 1
            y = seed_y + dy
            if not (0 <= x < w and 0 <= y < h):
                continue
            if not self.edge_mask[y, x]:
                continue

            position = (x, y)
            if self.footprint_is_free(position, consumed_mask):
                candidates.append(position)

        if not candidates:
            return None
        return candidates[self.rng.integers(len(candidates))]

    def walk_combined_path(self, seed, global_consumed_mask):
        working_mask = global_consumed_mask.copy()

        if not self.footprint_is_free(seed, working_mask):
            return None, None, None, "seed_collision"

        left_path = self.walk_direction(seed, seed, "left", working_mask)
        left_steps = len(left_path) - 1

        right_start = self.teleport_right_of_seed(seed, working_mask)
        if right_start is None:
            return None, left_path, None, "teleport_collision"

        right_path = self.walk_direction(
            right_start, seed, "right", working_mask
        )
        right_steps = len(right_path) - 1

        return working_mask, left_path, right_path, left_steps + right_steps

    def run(self):
        global_consumed_mask = np.zeros(self.edge_mask.shape, dtype=bool)
        seeds = self.generate_seeds(global_consumed_mask)
        left_paths = []
        right_paths = []

        for seed in seeds:
            working_mask, left_path, right_path, result = \
                self.walk_combined_path(seed, global_consumed_mask)

            if result in ("seed_collision", "teleport_collision"):
                continue

            total_steps = result
            if total_steps < self.min_steps:
                continue

            left_paths.append(left_path)
            right_paths.append(right_path)
            global_consumed_mask = working_mask

        return left_paths, right_paths, global_consumed_mask
