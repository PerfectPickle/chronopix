"""
pixelworld.viewer - real-time visualisation for any Environment.

This is the *only* file in the package that imports pygame or knows
about wall-clock time for display purposes. The environment itself
never imports pygame, so envs stay usable headlessly (dataset
generation, CI, notebooks) with zero GUI dependency.

Run this on a machine with an actual display - it opens a real window.
Use run_discrete() from engine.py instead if you just want frame arrays
with no window at all (e.g. for saving a gif/dataset over SSH).

-----------------------------------------------------------------------------
WHY THERE ARE TWO MODES ("discrete" vs "continuous")
-----------------------------------------------------------------------------
Earlier versions of this file called env.step(dt) once per *display*
frame, with dt = however many real seconds had actually elapsed since the
last one (~1/60s at 60fps). That looks fine for fast motion, but for
anything moving only a few pixels per second it produces a visibly jagged
"staircase" motion: since x and y are each independently rounded to the
nearest integer pixel for display, and they very rarely cross their next
whole-pixel threshold in exactly the same 1/60s frame, you get long runs
of frames where only ONE axis's on-screen position moves, then a frame
where only the OTHER axis moves, never a clean simultaneous diagonal step.
(Measured directly: at DVDBounceEnv's default speed, ~60fps, only ~7% of
frames moved both axes together - most moved exactly one axis, which is
what reads as "staggered".)

mode="discrete" (the default below) fixes this by decoupling the
SIMULATION rate from the DISPLAY refresh rate: env.step(1.0) - the exact
same fixed step run_discrete() uses - is called at a controlled, steady
rate (`ticks_per_second`), while the window itself still redraws at `fps`
for a smooth, flicker-free picture (just re-showing the same frame between
ticks). Because both x and y are updated together, atomically, once per
tick, you get clean simultaneous diagonal pixel steps - and as a bonus,
this is now a genuine LIVE PREVIEW of the discrete/reproducible sequence
run_discrete() would generate for a dataset, just watched instead of
pre-computed - same dt=1.0 steps, same determinism, only difference is
pacing them against a real clock instead of dumping them all at once.

mode="continuous" keeps the old real-elapsed-dt-every-frame behaviour, for
environments where you deliberately want dt to vary continuously with wall
clock (e.g. once you're driving genuinely continuous physics, like a
pymunk-backed env, where sub-tick smoothness matters more than clean pixel
steps).
"""

from __future__ import annotations
import time
import numpy as np

def play_frames(
    frames: np.ndarray,
    fps: int = 12,
    scale: int = 10,
    window_title: str = "pixelworld",
):
    import pygame

    # frames: (T, H, W, C)
    T, H, W, C = frames.shape

    pygame.init()
    pygame.display.set_caption(window_title)
    screen = pygame.display.set_mode((W * scale, H * scale))
    clock = pygame.time.Clock()

    surface = pygame.Surface((W, H))
    scaled = pygame.Surface((W * scale, H * scale))

    running = True
    i = 0

    while running and i < T:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        frame = frames[i]
        rgb = np.repeat(frame, 3, axis=-1) if C == 1 else frame

        pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
        pygame.transform.scale(surface, scaled.get_size(), scaled)
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        i += 1
        clock.tick(fps)

    pygame.quit()

def run_realtime(
    env,
    fps: int = 60,
    scale: int = 10,
    window_title: str = "pixelworld",
    max_steps: int | None = None,
    record_path: str | None = None,
    mode: str = "discrete",           # "discrete" (recommended default - see module docstring) or "continuous"
    ticks_per_second: float = 12.0,   # only used when mode="discrete": how many env.step(1.0) calls happen per real second
):
    """
    Open a window and step `env` in real time until closed (or Esc pressed).

    scale: integer upscale factor. Pygame's transform.scale does a plain
    nearest-neighbour-style stretch (no smoothing), which is what you
    want here - it keeps individual pixels crisp instead of blurring
    them, so low-res frames still read as low-res.

    mode / ticks_per_second: see the module docstring above for why
    "discrete" is the default and what problem it fixes. ticks_per_second
    controls how fast the *world* moves (independent of fps, which only
    controls how smoothly the *window* redraws) - lower it for a more
    deliberate/retro feel, raise it for snappier motion. Ignored entirely
    in mode="continuous".

    max_steps: optional cap, mainly useful for automated testing/capture
    (a real interactive session would normally leave this as None).

    record_path: if given, also saves every frame shown to a gif/mp4 at
    that path (via imageio) so you have a file to review even if you
    can't see the live window - e.g. running headless / over SSH.
    """
    assert mode in ("discrete", "continuous"), "mode must be 'discrete' or 'continuous'"
    import pygame

    pygame.init()
    pygame.display.set_caption(window_title)
    screen = pygame.display.set_mode((env.width * scale, env.height * scale))
    clock = pygame.time.Clock()

    recorded = [] if record_path else None
    running = True
    last = time.perf_counter()
    steps = 0
    tick_interval = 1.0 / ticks_per_second  # only meaningful in discrete mode
    accumulator = 0.0                        # only meaningful in discrete mode

    surface = pygame.Surface((env.width, env.height))
    scaled = pygame.Surface((env.width * scale, env.height * scale))

    while running:
        real_dt = clock.tick(fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if mode == "continuous":
            pass
        else:
            # Advance the simulation in whole, fixed dt=1.0 ticks, paced by
            # a real-time accumulator - the "fixed timestep" pattern. Both
            # x and y (or whatever the env's state is) move together, once
            # per tick, exactly as run_discrete() would produce - this loop
            # can even fire zero or multiple times in a single display
            # frame depending on how real_dt lines up with tick_interval,
            # which is exactly what keeps simulation rate independent of
            # (and unaffected by) any jitter in the display's frame rate.
            accumulator += real_dt
            while accumulator >= tick_interval:
                accumulator -= tick_interval

                print(int(round(env.x)), int(round(env.y)))
                

        frame = env.render()  # (H, W, C) uint8

        rgb = np.repeat(frame, 3, axis=-1) if frame.shape[-1] == 1 else frame
        if recorded is not None:
            recorded.append(rgb)

        pygame.surfarray.blit_array(surface, rgb.transpose(1, 0, 2))
        pygame.transform.scale(surface, scaled.get_size(), scaled)
        screen.blit(scaled, (0, 0))
        pygame.display.update()

        steps += 1
        if max_steps is not None and steps >= max_steps:
            running = False

    pygame.quit()

    if recorded is not None and recorded:
        import imageio.v2 as imageio
        imageio.mimsave(record_path, np.stack(recorded), fps=fps)



import re
import glob
import imageio.v2 as imageio


def _natural_key(path: str):
    """Sort key that orders 'frame_2' before 'frame_10' regardless of
    zero-padding width, so mixed-padding runs still sort correctly."""
    fname = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok
            for tok in re.split(r'(\d+)', fname)]


def compile_videos_from_frames(
        output_dir: str = "visual_predictions",
        fps: int = 15,
        subdirs: list[str] | None = None,
        quality: int = 6,
    ):
    """
    Compiles each subfolder of frame images inside `output_dir` into its
    own mp4, saved back into `output_dir`.

    e.g. visual_predictions/ground_truth/frame_*.png ->
         visual_predictions/ground_truth.mp4

    Parameters
    ----------
    output_dir : str
        Base directory containing per-array subfolders of frames
        (as produced by plot_visual_prediction).
    fps : int
        Frames per second for the output video. Default 15.
    subdirs : list of str, optional
        Which subfolders to compile (e.g. ["ground_truth", "prior_pred"]).
        If None, auto-detects all subfolders of `output_dir` that contain
        at least one frame_*.png file.
    quality : int
        imageio/ffmpeg quality setting, 0 (worst) to 10 (best).
        Default 6 is decent/reasonable, not maximal.

    Requires: pip install imageio[ffmpeg]
    """
    if subdirs is None:
        subdirs = [
            d for d in sorted(os.listdir(output_dir))
            if os.path.isdir(os.path.join(output_dir, d))
            and glob.glob(os.path.join(output_dir, d, "frame_*.png"))
        ]

    for sub in subdirs:
        frame_paths = sorted(
            glob.glob(os.path.join(output_dir, sub, "frame_*.png")),
            key=_natural_key,
        )
        if not frame_paths:
            print(f"Skipping '{sub}': no frames found.")
            continue

        video_path = os.path.join(output_dir, f"{sub}.mp4")
        with imageio.get_writer(
            video_path, fps=fps, codec='libx264', quality=quality
        ) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))

        print(f"Saved {len(frame_paths)} frames -> {video_path}")