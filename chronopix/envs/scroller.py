"""
InfiniteScrollerEnv - a side-scrolling terrain of blocks that never
runs out, without ever storing more than what's on screen.

The trick: terrain height for a given column is a *pure function* of
that column's index (via a seeded hash -> deterministic RNG), not
something computed by simulating forward from column 0. That means:
  - the world never needs an unbounded history array
  - you can jump the camera to an arbitrary x (env.camera_x = 1e9) and
    the terrain there is instantly correct and always the same for a
    given seed
  - discrete and real-time stepping produce the exact same terrain,
    just sampled at different camera positions
"""

from __future__ import annotations
import numpy as np
from ..engine import Environment, paint_rect

_HASH_MULT = 2654435761  # Knuth's multiplicative hash constant


class InfiniteScrollerEnv(Environment):
    def __init__(
        self,
        width: int = 64,
        height: int = 32,
        block_size: int = 4,
        scroll_speed: float = 20.0,   # pixels per unit of world-time
        channels: int = 1,
        seed=0,
    ):
        self.block_size = block_size
        self.scroll_speed = scroll_speed
        self._terrain_seed = 0 if seed is None else int(seed)
        super().__init__(width, height, channels, seed)

    def reset(self) -> None:
        self.t = 0
        self.time = 0.0
        self.camera_x = 0.0  # world x-coordinate, in pixels, of the left edge

    def _block_height(self, block_index: int) -> int:
        """Deterministic terrain height (in blocks) for a given column index.
        Pure function of block_index + seed - no dependency on history,
        so any column can be computed on demand regardless of camera_x."""
        h = (int(block_index) * _HASH_MULT + self._terrain_seed) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        max_blocks_high = max(1, (self.height // self.block_size) // 2)
        return int(rng.integers(1, max_blocks_high + 1))

    def step(self, dt: float = 1.0) -> None:
        self.camera_x += self.scroll_speed * dt
        self.t += 1
        self.time += dt

    def draw(self, canvas: np.ndarray) -> None:
        b = self.block_size
        ground_colour = 255 if self.channels == 1 else (210, 210, 210)

        cam_block = self.camera_x / b
        first_block = int(np.floor(cam_block))
        sub_px = int(round((cam_block - first_block) * b))  # sub-block scroll offset

        n_cols = self.width // b + 2  # +2 covers partial columns at both edges
        for i in range(n_cols):
            block_idx = first_block + i
            h_px = self._block_height(block_idx) * b
            x0 = i * b - sub_px
            x1 = x0 + b
            paint_rect(canvas, self.height - h_px, self.height, x0, x1, ground_colour)
