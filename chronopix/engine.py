"""
pixelworld.engine
==================

Tiny, dependency-light "game engine" for generating low-res pixel
environments, aimed at perception tests for AI models.

Design goals
------------
1. Simulation logic never knows about wall-clock time or windows.
   `step(dt)` advances the world by `dt` units of world-time. Nothing
   else. This is what lets the *same* environment be driven either
   discretely (dt is always exactly 1.0, one call = one frame, fully
   reproducible) or in real time (dt = seconds elapsed since the last
   call, so motion looks the same regardless of frame rate).

2. Rendering always returns an (H, W, C) uint8 array, regardless of
   whether the environment is conceptually black-and-white, grayscale,
   or colour. C is fixed per-environment (1 or 3). This means:
     - BW is just C=1 with values in {0, 255}
     - grayscale is C=1 with values in [0, 255]
     - colour is C=3, standard RGB
   so switching an environment from BW -> grayscale -> colour is a
   change to *how it paints*, not a change to the interface. Anything
   consuming frames (dataset writer, pygame viewer, etc.) doesn't care
   which mode it's in.

3. Environments should be *state-jumpable* where possible: given a
   time index / camera position, you should be able to compute what's
   there without having replayed the whole history. This matters for
   "infinite" environments (e.g. a side-scroller) - see envs/scroller.py
   for how this is done with a deterministic per-column hash instead of
   an ever-growing array.
"""

from __future__ import annotations
import numpy as np


class Environment:
    """
    Base class for all pixel environments.

    Subclasses implement:
        reset(self)              -> set up self.t, self.time, and any state
        step(self, dt=1.0)       -> advance state by dt units of world-time
        draw(self, canvas)       -> paint current state into canvas in-place

    `canvas` passed to draw() always has shape (height, width, channels)
    and dtype uint8. Use paint_rect() below for a channel-agnostic way
    to fill a rectangular region with a colour.
    """

    def __init__(self, width: int, height: int, channels: int = 1, seed=None):
        assert channels in (1, 3), "channels must be 1 (bw/grayscale) or 3 (rgb)"
        self.width = width
        self.height = height
        self.channels = channels
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.t = 0        # discrete step counter (increments by 1 every step() call)
        self.time = 0.0   # continuous world-time (sum of dt passed to step())
        self.reset()

    # ---- subclasses implement these three -----------------------------

    def reset(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def draw(self, canvas: np.ndarray) -> None:
        raise NotImplementedError

    # ---- provided ------------------------------------------------------

    def render(self) -> np.ndarray:
        """Return the current frame as an (H, W, C) uint8 array."""
        canvas = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        self.draw(canvas)
        return canvas


def paint_rect(canvas: np.ndarray, y0: int, y1: int, x0: int, x1: int, colour) -> None:
    """
    Fill canvas[y0:y1, x0:x1] with `colour`, clipped to canvas bounds.
    `colour` can be a scalar (broadcasts to any channel count) or a
    tuple matching canvas.shape[-1]. Silently no-ops if the rect is
    fully outside the canvas - this is what makes it safe to draw
    off-screen blocks (e.g. a scroller column that's partly clipped).
    """
    h, w = canvas.shape[0], canvas.shape[1]
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    if y0c >= y1c or x0c >= x1c:
        return
    canvas[y0c:y1c, x0c:x1c] = colour


# ---- drivers ------------------------------------------------------------
#
# These are the only two places that know anything about "time" as
# experienced by a human or a dataset. Everything above is oblivious
# to it, by design.

def run_discrete(env: Environment, n_steps: int) -> np.ndarray:
    """
    Advance `env` by exactly n_steps ticks (dt each), deterministically.
    Returns frames stacked as (n_steps + 1, H, W, C) uint8 - includes the
    initial frame before any stepping, so you get n_steps transitions.

    This is the mode you want for generating reproducible datasets:
    same seed + same n_steps -> byte-identical output, no wall clock
    involved anywhere.
    """
    frames = [env.render()]
    for _ in range(n_steps):
        env.step()
        frames.append(env.render())
    return np.stack(frames)


def save_frames(frames: np.ndarray, path: str, fps: int = 30) -> None:
    """Save an (T, H, W, C) uint8 frame stack as a gif/mp4 (by extension)."""
    import imageio.v2 as imageio
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    imageio.mimsave(path, frames, fps=fps)
