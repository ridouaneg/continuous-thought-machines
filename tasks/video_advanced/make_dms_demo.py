"""Generate a side-by-side DMS demo (match + non-match) as a GIF and a
static frame grid. Mirrors the dataset logic in
tasks/video_advanced/delayed_match_to_sample.ipynb so anyone can see
exactly what a "video DMS benchmark" clip looks like.

Outputs:
  assets/dms_example.gif       (animated side-by-side, looped)
  assets/dms_example_grid.png  (static frame grid)
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../..")

# --- DMS dataset, copied from the notebook -------------------------------- #

SHAPES = ["circle", "square", "triangle", "cross"]
COLORS = np.array(
    [[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.3, 0.4, 1.0], [1.0, 0.9, 0.2]],
    dtype=np.float32,
)


def _draw(canvas, shape_id, color, cx, cy, radius):
    H, W = canvas.shape[1:]
    ys, xs = np.mgrid[0:H, 0:W]
    dx, dy = xs - cx, ys - cy
    if shape_id == 0:
        mask = dx * dx + dy * dy <= radius * radius
    elif shape_id == 1:
        mask = (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
    elif shape_id == 2:
        mask = (dy <= radius) & (dy >= -radius) & (np.abs(dx) <= (radius - dy) / 2 + 1)
    else:
        bar = max(1, radius // 3)
        mask = ((np.abs(dx) <= radius) & (np.abs(dy) <= bar)) | (
            (np.abs(dy) <= radius) & (np.abs(dx) <= bar)
        )
    for c in range(3):
        canvas[c][mask] = color[c]


def make_clip(idx, sample_len, delay_len, probe_len, image_size, force_match=None):
    rng = np.random.default_rng(idx)
    H = W = image_size
    radius = H // 6
    cue_shape = int(rng.integers(0, 4))
    cue_color = int(rng.integers(0, len(COLORS)))
    is_match = bool(rng.integers(0, 2)) if force_match is None else force_match
    if is_match:
        probe_shape, probe_color = cue_shape, cue_color
    else:
        probe_shape, probe_color = cue_shape, cue_color
        while probe_shape == cue_shape and probe_color == cue_color:
            probe_shape = int(rng.integers(0, 4))
            probe_color = int(rng.integers(0, len(COLORS)))
    n_frames = sample_len + delay_len + probe_len
    clip = np.zeros((n_frames, 3, H, W), dtype=np.float32)
    for t in range(n_frames):
        for c in range(3):
            clip[t, c] = float(rng.uniform(0, 0.15))
    cx_cue, cy_cue = rng.uniform(radius, W - radius), rng.uniform(radius, H - radius)
    for t in range(sample_len):
        _draw(clip[t], cue_shape, COLORS[cue_color], cx_cue, cy_cue, radius)
    for t in range(sample_len, sample_len + delay_len):
        ds = int(rng.integers(0, 4))
        dc = int(rng.integers(0, len(COLORS)))
        cx, cy = rng.uniform(radius, W - radius), rng.uniform(radius, H - radius)
        _draw(clip[t], ds, COLORS[dc], cx, cy, radius)
    cx_p, cy_p = rng.uniform(radius, W - radius), rng.uniform(radius, H - radius)
    for t in range(sample_len + delay_len, n_frames):
        _draw(clip[t], probe_shape, COLORS[probe_color], cx_p, cy_p, radius)
    return clip, is_match


# --- Compose the demo ---------------------------------------------------- #

SAMPLE_LEN, DELAY_LEN, PROBE_LEN = 2, 8, 2
N_FRAMES = SAMPLE_LEN + DELAY_LEN + PROBE_LEN
IMG = 64  # render at 64 to make the static grid readable; clips are 32 in the dataset
UPSCALE = 4  # 32 -> 128 for the GIF

# Find a match clip and a non-match clip with the same first cue (so the
# difference is unambiguously the probe identity, not the cue).
match_idx = None
nonmatch_idx = None
i = 0
while match_idx is None or nonmatch_idx is None:
    clip, is_m = make_clip(i, SAMPLE_LEN, DELAY_LEN, PROBE_LEN, 32)
    if is_m and match_idx is None:
        match_idx = i
    elif (not is_m) and nonmatch_idx is None:
        nonmatch_idx = i
    i += 1

match_clip, _ = make_clip(match_idx, SAMPLE_LEN, DELAY_LEN, PROBE_LEN, 32, force_match=True)
nonmatch_clip, _ = make_clip(nonmatch_idx, SAMPLE_LEN, DELAY_LEN, PROBE_LEN, 32, force_match=False)


def to_pil(clip_arr, scale):
    """clip_arr: (T, 3, H, W) in [0,1]. Returns list of PIL upscaled images."""
    out = []
    for f in clip_arr:
        f = np.clip(np.moveaxis(f, 0, -1) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(f, "RGB").resize(
            (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST
        )
        out.append(img)
    return out


# --- Animated GIF: match on top, non-match below, with phase header strip --- #

match_imgs = to_pil(match_clip, UPSCALE)
nonmatch_imgs = to_pil(nonmatch_clip, UPSCALE)
W = match_imgs[0].width
H = match_imgs[0].height
HEADER = 26
LABEL_W = 110
gif_frames = []
phase_colors = {
    "sample": (255, 215, 80),
    "delay": (200, 200, 200),
    "probe": (130, 220, 130),
}
font = ImageFont.load_default()
total_w = LABEL_W + W
total_h = HEADER + H * 2 + 18
for t in range(N_FRAMES):
    canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    if t < SAMPLE_LEN:
        phase = "sample"
    elif t < SAMPLE_LEN + DELAY_LEN:
        phase = "delay"
    else:
        phase = "probe"
    # Header strip across the full clip width
    draw.rectangle([(LABEL_W, 0), (LABEL_W + W, HEADER)], fill=phase_colors[phase])
    draw.text((LABEL_W + 6, 6), f"phase: {phase}   frame {t}/{N_FRAMES - 1}", fill=(0, 0, 0), font=font)
    # Row labels
    draw.text((6, HEADER + H // 2 - 8), "MATCH", fill=(0, 110, 0), font=font)
    draw.text((6, HEADER + H + 18 + H // 2 - 8), "NON-MATCH", fill=(140, 0, 0), font=font)
    # Frames
    canvas.paste(match_imgs[t], (LABEL_W, HEADER))
    canvas.paste(nonmatch_imgs[t], (LABEL_W, HEADER + H + 18))
    gif_frames.append(canvas)

os.makedirs(os.path.join(REPO_ROOT, "assets"), exist_ok=True)
gif_path = os.path.join(REPO_ROOT, "assets", "dms_example.gif")
gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=350,
    loop=0,
    optimize=True,
)
print("wrote", gif_path, os.path.getsize(gif_path), "bytes")


# --- Static frame grid PNG (for rendering in commit messages / docs) ------ #

# Tile the 12 frames horizontally for each row, with phase strip above.
GW = match_imgs[0].width
GH = match_imgs[0].height
LBL = 110
HSTRIP = 22
strip_y = 0
grid_w = LBL + GW * N_FRAMES
grid_h = HSTRIP + GH * 2 + 22
grid = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
draw = ImageDraw.Draw(grid)
for t in range(N_FRAMES):
    if t < SAMPLE_LEN:
        c = phase_colors["sample"]
        ph = "sample"
    elif t < SAMPLE_LEN + DELAY_LEN:
        c = phase_colors["delay"]
        ph = "delay"
    else:
        c = phase_colors["probe"]
        ph = "probe"
    draw.rectangle([(LBL + GW * t, 0), (LBL + GW * (t + 1), HSTRIP)], fill=c)
    draw.text((LBL + GW * t + 4, 4), f"{ph} t={t}", fill=(0, 0, 0), font=font)
    grid.paste(match_imgs[t], (LBL + GW * t, HSTRIP))
    grid.paste(nonmatch_imgs[t], (LBL + GW * t, HSTRIP + GH + 22))
draw.text((6, HSTRIP + GH // 2 - 8), "MATCH", fill=(0, 110, 0), font=font)
draw.text((6, HSTRIP + GH + 22 + GH // 2 - 8), "NON-MATCH", fill=(140, 0, 0), font=font)
png_path = os.path.join(REPO_ROOT, "assets", "dms_example_grid.png")
grid.save(png_path, optimize=True)
print("wrote", png_path, os.path.getsize(png_path), "bytes")
