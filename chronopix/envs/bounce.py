"""
DVDBounceEnv - one or more blocks bounce around a rectangular arena,
DVD-screensaver style, plus four optional complexity mechanics aimed at
long-range credit assignment:

  0. MULTIPLE BOXES + BOX-BOX COLLISIONS
     n_boxes > 1 spawns several white boxes that bounce off each other
     (in addition to the walls) using the same "reflect, don't smear"
     logic as wall bounces.

  1. PARTIAL VISUAL OCCLUSION
     Larger dark-grey (85,85,85) panels drift across the arena in a
     straight line (N/S/E/W or diagonal) and off the far side, optionally
     pausing once around the midpoint of their path. New panels spawn at
     random intervals and can overlap each other and the boxes.

  2. FLOOR-IMPACT COUNTER -> FULL REVERSAL
     A run-global counter increments once per FRAME in which at least one
     box touches the floor (never double-counted within a frame). When it
     reaches a fixed threshold, every box that touched the floor on that
     particular frame has its full velocity reversed (both axes, not just
     the usual vertical bounce) so it retraces its incoming path, and the
     counter resets to zero.

  3. SWEEPING FREEZE-LINE
     A 1px light-grey (170,170,170) line sweeps across the arena at a
     constant 1px/frame (no delay) in one of the four cardinal sweep
     directions. Touching a white box freezes it in place (velocity
     preserved, untouched) for a random number of frames, after which it
     resumes exactly as if it had just been paused. This mechanic only
     ever touches the white boxes - it doesn't know or care about
     occluders, the floor counter, or box-box collisions.

All four are opt-in and default to the original single-box behaviour, so
existing configs (n_boxes=1, everything else disabled) run byte-identical
to the old DVDBounceEnv, including RNG draw order.

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

The original DVD screensaver this environment is modelled on only ever
moves at exact 45-degree angles for exactly this reason - so
diagonal_only=True (the default here) picks a random one of the four 45
degree headings instead of a uniformly random angle, which makes every
single tick a clean simultaneous diagonal step, by construction, forever.

Set diagonal_only=False if you specifically want arbitrary continuous
headings - just know that some amount of single-axis "staircasing" is
then a real, inherent property of the trajectory.

-----------------------------------------------------------------------------
NEW MECHANICS - CONFIG REFERENCE
-----------------------------------------------------------------------------
(0) n_boxes=1, box_sizes=None, box_box_collisions=True
    - box_sizes, if given, is a list of length n_boxes overriding
      block_size per box. Collisions resolve like a wall bounce: the
      axis with the smaller overlap is treated as the "hit" axis (or
      both, on a corner hit) and both boxes have that velocity
      component flipped and their position reverted to before this
      tick's move, so they never visibly interpenetrate.

    With 2-4 boxes, each one is placed in its own arena quadrant with a
    distinct starting heading (a permutation of the 4 diagonal headings
    when diagonal_only=True, or an angle drawn from its own slice of the
    circle otherwise), so they start decorrelated instead of risking a
    near-duplicate start that keeps two boxes shadowing each other for
    the whole run. n_boxes=1 is unaffected (full arena, as before); with
    5+ boxes quadrants run out and placement falls back to uniform
    random (with overlap avoidance) like before.

(1) occlusion_enabled=False
    occlusion_interval_range=(20, 40)        # frames between new panels
    occlusion_size_range=(3, 8)              # panel side length, px
    occlusion_delay_range=(0, 2)             # frames waited between each
                                              # 1px move of a given panel
    occlusion_directions=None                # subset of Occluder.DIRS
                                              # keys; None = all 8
    occlusion_pause_prob=0.0                 # chance a given panel
                                              # pauses once near mid-path
    occlusion_pause_duration_range=(5, 5)    # frames it pauses for, if it does
    occlusion_colour=(85, 85, 85)
    Each new panel independently samples its own direction/size/delay/
    pause from the ranges above, so within one run these quantities vary
    quasi-randomly panel to panel, per the brief. Colour (and, if you
    want it fixed, the pause duration - just set the range to (k, k)) is
    the one thing meant to stay constant across a run.

(2) floor_reversal_enabled=False, floor_reversal_threshold=5 (fixed for the run)

(3) sweep_enabled=False
    sweep_interval_range=(15, 30)            # frames between new sweeps
    sweep_freeze_duration_range=(6, 6)       # frames a hit box is frozen
    sweep_direction="random"                 # "left_to_right" |
                                              # "right_to_left" |
                                              # "top_to_bottom" |
                                              # "bottom_to_top" | "random"
    sweep_colour=(170, 170, 170)
"""

from __future__ import annotations
import numpy as np
from ..engine import Environment, paint_rect


# =============================================================================
# Small state objects. These are plain data holders; all the logic that
# reads/writes them lives on DVDBounceEnv so that every mechanic can see
# and coordinate with the environment's rng, dimensions, etc.
# =============================================================================

class Box:
    """One white bouncing box."""
    __slots__ = (
        "size", "x", "y", "prev_x", "prev_y", "vx", "vy",
        "colour", "bounces", "frozen_remaining",
    )

    def __init__(self, size: int):
        self.size = size
        self.x = self.y = 0.0
        self.prev_x = self.prev_y = 0.0
        self.vx = self.vy = 0.0
        self.colour = 255
        self.bounces = 0
        self.frozen_remaining = 0  # >0 while paused by a sweep-line hit


class Occluder:
    """A dark-grey panel drifting in a straight line across the arena."""

    # 8 directions: N/S/E/W plus diagonals, as (dx, dy) unit steps.
    # (0,0) origin is top-left, +x is right, +y is down.
    DIRS = {
        "N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0),
        "NE": (1, -1), "NW": (-1, -1), "SE": (1, 1), "SW": (-1, 1),
    }

    def __init__(self, dx, dy, size, delay, start_x, start_y, pause_duration, colour):
        self.dx, self.dy = dx, dy
        self.size = int(size)
        self.delay = max(0, int(delay))
        self.x = float(start_x)
        self.y = float(start_y)
        self.pause_duration = max(0, int(pause_duration))
        self.colour = colour

        self._delay_counter = 0
        self._pause_counter = 0
        self._has_paused = False

    def step(self, width: int, height: int) -> None:
        if self._pause_counter > 0:
            self._pause_counter -= 1
            return

        if self._delay_counter < self.delay:
            self._delay_counter += 1
            return
        self._delay_counter = 0

        old_x, old_y = self.x, self.y
        self.x += self.dx
        self.y += self.dy

        if (not self._has_paused) and self.pause_duration > 0:
            cx, cy = width / 2.0, height / 2.0
            if self.dx != 0:
                crossed = (old_x - cx) * (self.x - cx) <= 0
            else:
                crossed = (old_y - cy) * (self.y - cy) <= 0
            if crossed:
                self._has_paused = True
                self._pause_counter = self.pause_duration

    def is_offscreen(self, width: int, height: int) -> bool:
        return (
            self.x + self.size <= 0 or self.x >= width or
            self.y + self.size <= 0 or self.y >= height
        )

    def draw(self, canvas: np.ndarray) -> None:
        x0, y0 = int(round(self.x)), int(round(self.y))
        paint_rect(canvas, y0, y0 + self.size, x0, x0 + self.size, self.colour)


class SweepLine:
    """A 1px line sweeping across the whole width or height at 1px/frame."""

    _VALID = ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")

    def __init__(self, direction: str, width: int, height: int, colour):
        assert direction in self._VALID, f"unknown sweep direction {direction!r}"
        self.direction = direction
        self.width = width
        self.height = height
        self.colour = colour

        if direction == "left_to_right":
            self.orientation, self.pos, self._step = "vertical", 0, 1
        elif direction == "right_to_left":
            self.orientation, self.pos, self._step = "vertical", width - 1, -1
        elif direction == "top_to_bottom":
            self.orientation, self.pos, self._step = "horizontal", 0, 1
        else:  # bottom_to_top
            self.orientation, self.pos, self._step = "horizontal", height - 1, -1

        self._prev_pos = self.pos

    def step(self) -> None:
        self._prev_pos = self.pos
        self.pos += self._step

    def is_offscreen(self) -> bool:
        lim = self.width if self.orientation == "vertical" else self.height
        return self.pos < 0 or self.pos >= lim

    def swept_hits(self, box: Box) -> bool:
        """
        True if the line's path THIS TICK (from its previous pixel to its
        new one - normally adjacent, but checked as a small swept range
        for robustness) overlaps the box's footprint. Only cares about
        the axis it moves along; it spans the full extent of the other
        axis by construction.
        """
        lo, hi = (self._prev_pos, self.pos) if self._prev_pos <= self.pos else (self.pos, self._prev_pos)
        if self.orientation == "vertical":
            b0 = int(round(box.x))
            b1 = b0 + box.size - 1
        else:
            b0 = int(round(box.y))
            b1 = b0 + box.size - 1
        return not (b1 < lo or b0 > hi)

    def draw(self, canvas: np.ndarray) -> None:
        if self.orientation == "vertical":
            paint_rect(canvas, 0, self.height, self.pos, self.pos + 1, self.colour)
        else:
            paint_rect(canvas, self.pos, self.pos + 1, 0, self.width, self.colour)


# =============================================================================
# Main environment
# =============================================================================

class DVDBounceEnv(Environment):
    def __init__(
        self,
        width: int = 64,
        height: int = 32,
        block_size: int = 4,
        pixels_per_tick: float = 20.0,
        channels: int = 1,
        colour_shift_on_bounce: bool = True,
        diagonal_only: bool = True,
        # ---- (0) multiple boxes + box-box collisions ----
        n_boxes: int = 1,
        box_sizes=None,
        box_box_collisions: bool = True,
        # ---- (2) floor-impact counter -> full reversal ----
        floor_reversal_enabled: bool = False,
        floor_reversal_threshold: int = 5,
        # ---- (1) partial visual occlusion ----
        occlusion_enabled: bool = False,
        occlusion_interval_range=(20, 40),
        occlusion_size_range=(3, 8),
        occlusion_delay_range=(0, 2),
        occlusion_directions=None,
        occlusion_pause_prob: float = 0.0,
        occlusion_pause_duration_range=(5, 5),
        occlusion_colour=(85, 85, 85),
        # ---- (3) sweeping freeze-line ----
        sweep_enabled: bool = False,
        sweep_interval_range=(15, 30),
        sweep_freeze_duration_range=(6, 6),
        sweep_direction: str = "random",
        sweep_colour=(170, 170, 170),
        seed=None,
    ):
        self.block_size = block_size
        self.pixels_per_tick = pixels_per_tick
        self.colour_shift_on_bounce = colour_shift_on_bounce
        self.diagonal_only = diagonal_only

        # (0)
        self.n_boxes = int(n_boxes)
        assert self.n_boxes >= 1
        if box_sizes is None:
            self._box_sizes = [block_size] * self.n_boxes
        else:
            assert len(box_sizes) == self.n_boxes, "box_sizes must have length n_boxes"
            self._box_sizes = list(box_sizes)
        self.box_box_collisions = box_box_collisions

        # (2)
        self.floor_reversal_enabled = floor_reversal_enabled
        self.floor_reversal_threshold = int(floor_reversal_threshold)
        assert self.floor_reversal_threshold >= 1

        # (1)
        self.occlusion_enabled = occlusion_enabled
        self.occlusion_interval_range = tuple(occlusion_interval_range)
        self.occlusion_size_range = tuple(occlusion_size_range)
        self.occlusion_delay_range = tuple(occlusion_delay_range)
        if occlusion_directions is None:
            occlusion_directions = list(Occluder.DIRS.keys())
        elif isinstance(occlusion_directions, str):
            occlusion_directions = [occlusion_directions]
        for d in occlusion_directions:
            assert d in Occluder.DIRS, f"unknown occlusion direction {d!r}"
        self.occlusion_directions = list(occlusion_directions)
        self.occlusion_pause_prob = float(occlusion_pause_prob)
        self.occlusion_pause_duration_range = tuple(occlusion_pause_duration_range)
        self.occlusion_colour = tuple(occlusion_colour)

        # (3)
        self.sweep_enabled = sweep_enabled
        self.sweep_interval_range = tuple(sweep_interval_range)
        self.sweep_freeze_duration_range = tuple(sweep_freeze_duration_range)
        assert sweep_direction == "random" or sweep_direction in SweepLine._VALID
        self.sweep_direction = sweep_direction
        self.sweep_colour = tuple(sweep_colour)

        super().__init__(width, height, channels, seed)

    # ---- colour helpers -------------------------------------------------

    def _white(self):
        return 255 if self.channels == 1 else (255, 255, 255)

    def _neutral(self, rgb):
        """rgb is an (r,g,b) tuple of equal components; collapse to a
        scalar for grayscale envs, keep as tuple for RGB envs."""
        return rgb[0] if self.channels == 1 else tuple(int(c) for c in rgb)

    # ---- setup ------------------------------------------------------------

    # Quadrant index -> (qx, qy), where qx/qy select the left/right or
    # top/bottom half of the arena respectively.
    _QUADRANTS = [(0, 0), (1, 0), (0, 1), (1, 1)]

    def reset(self) -> None:
        self.t = 0
        self.time = 0.0

        self.boxes = [Box(size) for size in self._box_sizes]
        n = len(self.boxes)

        # With 2-4 boxes, spread them across distinct quadrants and give
        # each a distinct starting heading, so a run doesn't end up with
        # two boxes shadowing each other in a tight cluster the whole
        # time just because their random start/velocity happened to be
        # similar. A single box (n=1) is left exactly as before - full
        # arena, no heading restriction - so old seeded configs are
        # unaffected.
        use_quadrants = 2 <= n <= 4
        if use_quadrants:
            quadrant_ids = self.rng.permutation(4)[:n]
            if self.diagonal_only:
                diag_headings = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
                headings = [diag_headings[k] for k in self.rng.permutation(4)[:n]]
            else:
                sector_width = 2 * np.pi / n
                sector_order = self.rng.permutation(n)

        for i, box in enumerate(self.boxes):
            if use_quadrants:
                qx, qy = self._QUADRANTS[quadrant_ids[i]]
                self._place_box(box, *self._quadrant_bounds(qx, qy, box.size))
            else:
                self._place_box(box, 0, max(0, self.width - box.size), 0, max(0, self.height - box.size))

            if self.diagonal_only:
                if use_quadrants:
                    sx, sy = headings[i]
                else:
                    sx = self.rng.choice([-1.0, 1.0])
                    sy = self.rng.choice([-1.0, 1.0])
                box.vx, box.vy = sx * self.pixels_per_tick, sy * self.pixels_per_tick
            else:
                if use_quadrants:
                    angle = sector_order[i] * sector_width + self.rng.uniform(0, sector_width)
                else:
                    angle = self.rng.uniform(0, 2 * np.pi)
                box.vx = float(np.cos(angle)) * self.pixels_per_tick
                box.vy = float(np.sin(angle)) * self.pixels_per_tick

            box.colour = self._white()

        self._sync_legacy_attrs()

        # (2)
        self.floor_hit_counter = 0

        # (1)
        self.occluders = []
        if self.occlusion_enabled:
            lo, hi = self.occlusion_interval_range
            self._occ_next_spawn_t = int(self.rng.integers(lo, hi + 1))

        # (3)
        self.sweep_lines = []
        if self.sweep_enabled:
            lo, hi = self.sweep_interval_range
            self._sweep_next_spawn_t = int(self.rng.integers(lo, hi + 1))

    def _quadrant_bounds(self, qx: int, qy: int, size: int):
        half_w = self.width // 2
        half_h = self.height // 2
        if qx == 0:
            x_lo, x_hi = 0, max(0, half_w - size)
        else:
            x_lo, x_hi = half_w, max(half_w, self.width - size)
        if qy == 0:
            y_lo, y_hi = 0, max(0, half_h - size)
        else:
            y_lo, y_hi = half_h, max(half_h, self.height - size)
        return x_lo, x_hi, y_lo, y_hi

    def _place_box(self, box: Box, x_lo: int, x_hi: int, y_lo: int, y_hi: int, max_tries: int = 200) -> None:
        size = box.size
        x = y = 0.0
        for _ in range(max_tries):
            x = float(self.rng.integers(x_lo, x_hi + 1))
            y = float(self.rng.integers(y_lo, y_hi + 1))
            if not any(
                self._aabb_overlap_xy(x, y, size, o.x, o.y, o.size)
                for o in self.boxes if o is not box and o.size
            ):
                break
        box.x, box.y = x, y
        box.prev_x, box.prev_y = x, y

    def _sync_legacy_attrs(self) -> None:
        """Mirror box[0]'s state onto env.x/y/vx/vy/colour/bounces so any
        code written against the old single-box DVDBounceEnv keeps working."""
        b0 = self.boxes[0]
        self.x, self.y = b0.x, b0.y
        self.vx, self.vy = b0.vx, b0.vy
        self.colour = b0.colour
        self.bounces = b0.bounces

    # ---- per-tick update ----------------------------------------------

    def step(self) -> None:
        for box in self.boxes:
            box.prev_x, box.prev_y = box.x, box.y

        # ---- move + wall bounce (skips frozen boxes) ----
        floor_touched = []
        for box in self.boxes:
            if box.frozen_remaining > 0:
                box.frozen_remaining -= 1
                continue
            box.x += box.vx
            box.y += box.vy
            if self._wall_bounce(box):
                floor_touched.append(box)

        # ---- (0) box-box collisions ----
        if len(self.boxes) > 1 and self.box_box_collisions:
            self._resolve_box_collisions()

        # ---- (2) floor-impact counter -> full reversal ----
        if self.floor_reversal_enabled and floor_touched:
            self.floor_hit_counter += 1
            if self.floor_hit_counter >= self.floor_reversal_threshold:
                for box in floor_touched:
                    # vy was already flipped by the wall bounce above;
                    # flipping vx too gives a full reversal of the
                    # original incoming velocity, i.e. it retraces its path.
                    box.vx = -box.vx
                self.floor_hit_counter = 0

        # ---- (1) occlusion panels ----
        if self.occlusion_enabled:
            self._update_occluders()

        # ---- (3) sweeping freeze-line ----
        if self.sweep_enabled:
            self._update_sweep_lines()

        self._sync_legacy_attrs()
        self.t += 1

    def _wall_bounce(self, box: Box) -> bool:
        """Reflect box off the arena walls. Returns True iff it hit the floor."""
        bounced = False
        bottom = False

        if box.x <= 0:
            box.x, box.vx = 0.0, -box.vx
            bounced = True
        elif box.x >= self.width - box.size:
            box.x, box.vx = float(self.width - box.size), -box.vx
            bounced = True

        if box.y <= 0:
            box.y, box.vy = 0.0, -box.vy
            bounced = True
        elif box.y >= self.height - box.size:
            box.y, box.vy = float(self.height - box.size), -box.vy
            bounced = True
            bottom = True

        if bounced:
            box.bounces += 1
            if self.channels == 3 and self.colour_shift_on_bounce:
                box.colour = tuple(int(c) for c in self.rng.integers(80, 256, size=3))

        return bottom

    # ---- (0) box-box collisions -----------------------------------------

    @staticmethod
    def _aabb_overlap_xy(ax, ay, asize, bx, by, bsize) -> bool:
        return not (
            ax + asize <= bx or bx + bsize <= ax or
            ay + asize <= by or by + bsize <= ay
        )

    def _aabb_overlap(self, a: Box, b: Box) -> bool:
        return self._aabb_overlap_xy(a.x, a.y, a.size, b.x, b.y, b.size)

    def _resolve_box_collisions(self) -> None:
        """
        Resolve all pairwise overlaps. A single left-to-right pass isn't
        enough: fixing pair (1,2) can move box 1 back into a box 0 that
        was already checked earlier in the same pass. Since each
        resolution reverts the offending boxes to their (necessarily
        non-overlapping) start-of-tick position, repeating the full scan
        converges quickly - a handful of passes comfortably covers any
        chain reachable with this many boxes.
        """
        boxes = self.boxes
        n = len(boxes)
        for _ in range(max(4, n)):
            any_resolved = False
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = boxes[i], boxes[j]
                    if self._aabb_overlap(a, b):
                        self._resolve_pair(a, b)
                        any_resolved = True
            if not any_resolved:
                break

    def _resolve_pair(self, a: Box, b: Box) -> None:
        a_frozen = a.frozen_remaining > 0
        b_frozen = b.frozen_remaining > 0
        if a_frozen and b_frozen:
            return  # neither can react this tick

        overlap_x = min(a.x + a.size, b.x + b.size) - max(a.x, b.x)
        overlap_y = min(a.y + a.size, b.y + b.size) - max(a.y, b.y)
        # Smaller overlap = the axis along which they actually collided
        # (mirrors how a wall bounce flips just the perpendicular axis).
        # Equal overlap = a corner hit -> flip both.
        hit_x = overlap_x <= overlap_y
        hit_y = overlap_y <= overlap_x

        for box in (a, b):
            if box.frozen_remaining > 0:
                continue  # frozen boxes act as immovable obstacles
            box.x, box.y = box.prev_x, box.prev_y
            if hit_x:
                box.vx = -box.vx
            if hit_y:
                box.vy = -box.vy

    # ---- (1) occlusion panels -------------------------------------------

    def _occluder_start(self, dx, dy, size):
        if dx > 0:
            x = -size
        elif dx < 0:
            x = self.width
        else:
            x = int(self.rng.integers(0, max(1, self.width - size + 1)))

        if dy > 0:
            y = -size
        elif dy < 0:
            y = self.height
        else:
            y = int(self.rng.integers(0, max(1, self.height - size + 1)))

        return x, y

    def _spawn_occluder(self) -> None:
        direction = self.rng.choice(self.occlusion_directions)
        dx, dy = Occluder.DIRS[direction]

        lo, hi = self.occlusion_size_range
        size = int(self.rng.integers(lo, hi + 1))

        lo, hi = self.occlusion_delay_range
        delay = int(self.rng.integers(lo, hi + 1))

        will_pause = self.rng.random() < self.occlusion_pause_prob
        if will_pause:
            lo, hi = self.occlusion_pause_duration_range
            pause_duration = int(self.rng.integers(lo, hi + 1))
        else:
            pause_duration = 0

        sx, sy = self._occluder_start(dx, dy, size)
        colour = self._neutral(self.occlusion_colour)
        self.occluders.append(Occluder(dx, dy, size, delay, sx, sy, pause_duration, colour))

    def _update_occluders(self) -> None:
        if self.t >= self._occ_next_spawn_t:
            self._spawn_occluder()
            lo, hi = self.occlusion_interval_range
            self._occ_next_spawn_t = self.t + int(self.rng.integers(lo, hi + 1))

        for occ in self.occluders:
            occ.step(self.width, self.height)
        self.occluders = [o for o in self.occluders if not o.is_offscreen(self.width, self.height)]

    # ---- (3) sweeping freeze-line -----------------------------------------

    def _spawn_sweep_line(self) -> "SweepLine":
        direction = self.sweep_direction
        if direction == "random":
            direction = self.rng.choice(list(SweepLine._VALID))
        colour = self._neutral(self.sweep_colour)
        line = SweepLine(direction, self.width, self.height, colour)
        self.sweep_lines.append(line)
        return line

    def _update_sweep_lines(self) -> None:
        newly_spawned = None
        if self.t >= self._sweep_next_spawn_t:
            newly_spawned = self._spawn_sweep_line()
            lo, hi = self.sweep_interval_range
            self._sweep_next_spawn_t = self.t + int(self.rng.integers(lo, hi + 1))

        for line in self.sweep_lines:
            # A line spawned this very tick hasn't been drawn yet, so it
            # must render at its starting edge pixel first - stepping it
            # immediately would skip that pixel and make it look like it
            # started one pixel further in than it actually did.
            if line is not newly_spawned:
                line.step()
            for box in self.boxes:
                if line.swept_hits(box):
                    lo, hi = self.sweep_freeze_duration_range
                    box.frozen_remaining = int(self.rng.integers(lo, hi + 1))

        self.sweep_lines = [l for l in self.sweep_lines if not l.is_offscreen()]

    # ---- rendering ---------------------------------------------------------

    def draw(self, canvas: np.ndarray) -> None:
        # Sweep-line paints first, i.e. UNDER the boxes - a box passing
        # over it hides the segment beneath it, which visually reads as
        # "the line is behind the box" and keeps it distinct from the
        # occlusion panels below.
        for line in self.sweep_lines:
            line.draw(canvas)

        for box in self.boxes:
            x0, y0 = int(round(box.x)), int(round(box.y))
            paint_rect(canvas, y0, y0 + box.size, x0, x0 + box.size, box.colour)

        # Occluders paint OVER the boxes - that's what makes them occlude.
        for occ in self.occluders:
            occ.draw(canvas)
