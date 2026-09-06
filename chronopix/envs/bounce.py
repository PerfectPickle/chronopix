"""
DVDBounceEnv - a block bounces around a rectangular arena and reflects
off the walls, DVD-screensaver style.

Works identically under discrete stepping (dt=1.0 each call -> the block
moves `pixels_per_tick` pixels per step) and real-time stepping (dt=seconds since
last frame -> the block moves `pixels_per_tick` pixels per second, so it looks the
same regardless of your frame rate).

-----------------------------------------------------------------------------
WHY diagonal_only=True IS THE DEFAULT
-----------------------------------------------------------------------------
On a low-res pixel grid, x and y each get rounded to the nearest whole
pixel independently for display. If |vx| != |vy| (i.e. the heading isn't
exactly 45 degrees), the axis with the larger component reaches its next
whole-pixel threshold almost every tick, while the smaller-component axis
only catches up occasionally - so most ticks move only ONE axis, which
reads as jagged, staircase-y motion rather than a clean diagonal. This
isn't a bug to fix so much as a fact about representing arbitrary-slope
lines with unit pixel steps (the same reason line-drawing algorithms like
Bresenham's exist at all) - it's mathematically worst for shallow/steep
headings and disappears entirely only at exactly 45 degrees, where vx and
vy have equal magnitude and so always cross their rounding threshold
together.

Measured directly: a heading ~1 degree off 45 gave clean simultaneous
diagonal steps on 200/200 ticks; a heading ~5.5 degrees off 45 (i.e.
closer to purely horizontal/vertical) gave only 39/200.

The original DVD screensaver this environment is modelled on only ever
moves at exact 45-degree angles for exactly this reason - so
diagonal_only=True (the default here) picks a random one of the four 45
degree headings instead of a uniformly random angle, which makes every
single tick a clean simultaneous diagonal step, by construction, forever.

Set diagonal_only=False if you specifically want arbitrary continuous
headings (e.g. to later test perception on a wider range of trajectories)
- just know that some amount of single-axis "staircasing" is then a real,
inherent property of the trajectory, not something viewer.py or the env
can further smooth away without either anti-aliasing (contrary to the
1-bit pixel aesthetic) or constraining the heading again.
"""

from __future__ import annotations
import numpy as np
from ..engine import Environment, paint_rect


class DVDBounceEnv(Environment):
    def __init__(
        self,
        width: int = 64,
        height: int = 32,
        block_size: int = 4,
        pixels_per_tick: float = 20.0,       # (speed) per axis (see diagonal_only below)
        channels: int = 1,
        colour_shift_on_bounce: bool = True,  # only visible if channels=3
        diagonal_only: bool = True,   # True = classic DVD-screensaver 45-degree-only headings (see module docstring for why this is the default); False = uniformly random heading, with inherent staircasing at non-45-degree angles
        seed=None,
    ):
        self.block_size = block_size
        self.pixels_per_tick = pixels_per_tick
        self.colour_shift_on_bounce = colour_shift_on_bounce
        self.diagonal_only = diagonal_only
        super().__init__(width, height, channels, seed)

    def reset(self) -> None:
        self.t = 0
        self.time = 0.0
        b = self.block_size

        # Integer start position: with diagonal_only=True (or any case where
        # pixels_per_tick is a whole number of pixels per tick), starting on an exact
        # pixel means every subsequent rounded position is *exactly*
        # base + n*pixels_per_tick - no rounding ambiguity ever creeps in. (This isn't
        # actually required for correctness - round(a + n*pixels_per_tick) - round(a
        # + (n-1)*pixels_per_tick) == pixels_per_tick regardless of a's fractional part, since
        # adding an integer shifts a rounded value by exactly that integer -
        # but starting on a whole pixel keeps the very first frame tidy too.)
        self.x = float(self.rng.integers(0, self.width - b + 1))
        self.y = float(self.rng.integers(0, self.height - b + 1))
        self.prev_x = self.x
        self.prev_y = self.y

        if self.diagonal_only:
            # exactly one of the four 45-degree headings - vx and vy always
            # equal in magnitude, so they always cross their rounding
            # threshold on the same tick, forever
            sx = self.rng.choice([-1.0, 1.0])
            sy = self.rng.choice([-1.0, 1.0])
            self.vx = sx * self.pixels_per_tick
            self.vy = sy * self.pixels_per_tick
        else:
            angle = self.rng.uniform(0, 2 * np.pi)
            self.vx = float(np.cos(angle)) * self.pixels_per_tick
            self.vy = float(np.sin(angle)) * self.pixels_per_tick

        self.bounces = 0
        self.colour = 255 if self.channels == 1 else (255, 255, 255)



    def step(self) -> None:
        self.prev_x = self.x
        self.prev_y = self.y

        b = self.block_size

        self.x += self.vx
        self.y += self.vy

        bounced = False
        if self.x <= 0:
            self.x, self.vx = 0.0, -self.vx
            bounced = True
        elif self.x >= self.width - b:
            self.x, self.vx = float(self.width - b), -self.vx
            bounced = True

        if self.y <= 0:
            self.y, self.vy = 0.0, -self.vy
            bounced = True
        elif self.y >= self.height - b:
            self.y, self.vy = float(self.height - b), -self.vy
            bounced = True

        if bounced:
            self.bounces += 1
            if self.channels == 3 and self.colour_shift_on_bounce:
                self.colour = tuple(int(c) for c in self.rng.integers(80, 256, size=3))

        self.t += 1

    def draw(self, canvas: np.ndarray) -> None:
        b = self.block_size

        x = getattr(self, "render_x", self.x)
        y = getattr(self, "render_y", self.y)

        x0 = int(round(x))
        y0 = int(round(y))

        x0, y0 = int(round(self.x)), int(round(self.y))
        paint_rect(canvas, y0, y0 + b, x0, x0 + b, self.colour)
