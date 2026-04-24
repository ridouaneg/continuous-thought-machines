"""Video datasets for the CTM action-recognition PoC.

Three backends:

- ``SyntheticMovingShapes``: fully synthetic "coloured shape bounces around
  a canvas" dataset. No downloads, no video decoding. Class = shape type.
  Deterministic per-index, so the same index always yields the same clip.

- ``UCF101Clips``: standard UCF-101 directory layout
  (``<data_root>/UCF-101/<ClassName>/<video>.avi``) plus the canonical
  split files (``trainlist0{1,2,3}.txt``, ``testlist0{1,2,3}.txt``) under
  ``<data_root>/ucfTrainTestlist/``.

- ``HMDB51Clips``: standard HMDB-51 layout
  (``<data_root>/hmdb51_org/<ClassName>/<video>.avi``) plus the split txt
  files under ``<data_root>/testTrainMulti_7030_splits/``.

Both real datasets use ``torchvision.io.read_video`` for decoding. Frames
are uniformly subsampled to ``n_frames`` (TSN-style segment sampling at
train time: pick a random frame inside each of ``n_frames`` equal-length
segments; uniform mid-segment sampling at test time).
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Synthetic moving shapes
# --------------------------------------------------------------------------- #

SYNTHETIC_CLASS_LABELS = ["circle", "square", "triangle", "cross"]


class SyntheticMovingShapes(Dataset):
    """A tiny synthetic video dataset: one coloured shape bouncing around.

    The class is the shape type. Colour and trajectory are randomised per
    clip but seeded by the clip index, so ``dataset[42]`` is always the
    same clip (useful for deterministic visualisation).

    Args:
        n_samples: Number of distinct clips.
        n_frames: Frames per clip.
        image_size: Spatial resolution (H = W = image_size).
        split: Only affects which random seeds are used; otherwise identical.
    """

    def __init__(self, n_samples=1024, n_frames=16, image_size=64, split="train"):
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.image_size = image_size
        self.split = split
        self.offset = 0 if split == "train" else 10_000_000

    def __len__(self):
        return self.n_samples

    def _draw_shape(self, canvas, shape_id, cx, cy, radius, color):
        H, W = canvas.shape[1:]
        ys, xs = np.mgrid[0:H, 0:W]
        dx = xs - cx
        dy = ys - cy
        if shape_id == 0:  # circle
            mask = (dx * dx + dy * dy) <= radius * radius
        elif shape_id == 1:  # square
            mask = (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
        elif shape_id == 2:  # triangle (pointing up)
            mask = (dy <= radius) & (dy >= -radius) & (
                np.abs(dx) <= (radius - dy) / 2 + 1
            )
        else:  # cross
            bar = radius // 3 + 1
            mask = ((np.abs(dx) <= radius) & (np.abs(dy) <= bar)) | (
                (np.abs(dy) <= radius) & (np.abs(dx) <= bar)
            )
        for c in range(3):
            canvas[c][mask] = color[c]

    def __getitem__(self, index):
        rng = np.random.default_rng(index + self.offset)
        shape_id = int(rng.integers(0, len(SYNTHETIC_CLASS_LABELS)))

        # Bouncing trajectory.
        H = W = self.image_size
        radius = H // 8
        cx = float(rng.uniform(radius, W - radius))
        cy = float(rng.uniform(radius, H - radius))
        vx = float(rng.uniform(-2.5, 2.5)) + 0.1
        vy = float(rng.uniform(-2.5, 2.5)) + 0.1
        color = rng.uniform(0.4, 1.0, size=3).astype(np.float32)

        clip = np.zeros((self.n_frames, 3, H, W), dtype=np.float32)
        # Background tint to make it non-trivial for the backbone.
        bg = rng.uniform(0.0, 0.2, size=3).astype(np.float32)
        for t in range(self.n_frames):
            for c in range(3):
                clip[t, c] = bg[c]
            # Bounce off walls.
            if cx - radius < 0 or cx + radius > W - 1:
                vx = -vx
            if cy - radius < 0 or cy + radius > H - 1:
                vy = -vy
            cx = float(np.clip(cx + vx, radius, W - 1 - radius))
            cy = float(np.clip(cy + vy, radius, H - 1 - radius))
            self._draw_shape(clip[t], shape_id, cx, cy, radius, color)

        # Normalise to [-1, 1]-ish, centred around 0.5.
        clip = (clip - 0.5) / 0.5
        return torch.from_numpy(clip), shape_id


# --------------------------------------------------------------------------- #
# UCF-101 / HMDB-51
# --------------------------------------------------------------------------- #


@dataclass
class ClipRecord:
    path: str
    label: int


def _tsn_segment_indices(num_video_frames: int, n_frames: int, train: bool) -> np.ndarray:
    """Pick ``n_frames`` indices from a video using TSN segment sampling."""
    if num_video_frames <= 0:
        return np.zeros(n_frames, dtype=np.int64)
    if num_video_frames < n_frames:
        # Repeat what we have if the video is too short.
        idxs = np.linspace(0, num_video_frames - 1, n_frames)
        return np.round(idxs).astype(np.int64)
    seg_len = num_video_frames / n_frames
    if train:
        offsets = np.random.uniform(0, seg_len, size=n_frames)
    else:
        offsets = np.full(n_frames, seg_len / 2.0)
    starts = np.arange(n_frames) * seg_len
    return np.clip(np.floor(starts + offsets), 0, num_video_frames - 1).astype(np.int64)


def _decode_clip(path: str, n_frames: int, image_size: int, train: bool) -> torch.Tensor:
    """Read a video file and return a (T, C, H, W) float tensor in [-1, 1]."""
    from torchvision.io import read_video  # imported lazily so synthetic works w/o ffmpeg

    try:
        frames, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {path}: {exc}") from exc

    num = frames.shape[0]
    if num == 0:
        raise RuntimeError(f"Empty video: {path}")

    idxs = _tsn_segment_indices(num, n_frames, train=train)
    frames = frames[idxs].float() / 255.0  # (T, C, H, W)

    # Resize to a square image_size.
    frames = F.interpolate(
        frames, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    # Standard ImageNet-ish normalisation (the existing backbones expect it).
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    if train and random.random() < 0.5:
        frames = torch.flip(frames, dims=[3])
    return frames


class _ClipDatasetBase(Dataset):
    """Shared logic for UCF-101 and HMDB-51."""

    records: List[ClipRecord]
    class_labels: List[str]

    def __init__(self, n_frames: int, image_size: int, train: bool):
        self.n_frames = n_frames
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        clip = _decode_clip(rec.path, self.n_frames, self.image_size, self.train)
        return clip, rec.label


class UCF101Clips(_ClipDatasetBase):
    """UCF-101 clip dataset.

    Expects ``<root>/UCF-101/<Class>/<video>.avi`` plus
    ``<root>/ucfTrainTestlist/{trainlist,testlist}0{1,2,3}.txt``.

    Split files map ``<Class>/<video>.avi`` (with or without a label
    integer) to train/test membership. The class index is defined by
    ``classInd.txt`` if present; otherwise alphabetical class order.
    """

    def __init__(self, root, split="train", fold=1, n_frames=16, image_size=112):
        super().__init__(n_frames, image_size, train=(split == "train"))
        self.root = root
        self.fold = fold
        self.split = split

        videos_root = os.path.join(root, "UCF-101")
        splits_root = os.path.join(root, "ucfTrainTestlist")

        class_ind = os.path.join(splits_root, "classInd.txt")
        if os.path.isfile(class_ind):
            with open(class_ind) as f:
                labels = []
                for line in f:
                    _, cname = line.strip().split()
                    labels.append(cname)
            self.class_labels = labels
        else:
            self.class_labels = sorted(
                d for d in os.listdir(videos_root)
                if os.path.isdir(os.path.join(videos_root, d))
            )
        class_to_idx = {c: i for i, c in enumerate(self.class_labels)}

        split_file = os.path.join(
            splits_root, f"{'trainlist' if split == 'train' else 'testlist'}{fold:02d}.txt"
        )
        self.records = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                rel = parts[0]
                cname = rel.split("/")[0]
                if cname not in class_to_idx:
                    continue
                path = os.path.join(videos_root, rel)
                if os.path.isfile(path):
                    self.records.append(ClipRecord(path=path, label=class_to_idx[cname]))


class HMDB51Clips(_ClipDatasetBase):
    """HMDB-51 clip dataset.

    Expects ``<root>/hmdb51_org/<Class>/<video>.avi`` plus
    ``<root>/testTrainMulti_7030_splits/<Class>_test_split{fold}.txt``
    where each line is ``<video>.avi <flag>`` with flag 1=train, 2=test,
    0=unused.
    """

    def __init__(self, root, split="train", fold=1, n_frames=16, image_size=112):
        super().__init__(n_frames, image_size, train=(split == "train"))
        self.root = root
        self.fold = fold
        self.split = split

        videos_root = os.path.join(root, "hmdb51_org")
        splits_root = os.path.join(root, "testTrainMulti_7030_splits")
        class_dirs = sorted(
            d for d in os.listdir(videos_root)
            if os.path.isdir(os.path.join(videos_root, d))
        )
        self.class_labels = class_dirs
        class_to_idx = {c: i for i, c in enumerate(class_dirs)}

        target_flag = "1" if split == "train" else "2"
        self.records = []
        for cname in class_dirs:
            split_file = os.path.join(
                splits_root, f"{cname}_test_split{fold}.txt"
            )
            if not os.path.isfile(split_file):
                continue
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    vname, flag = parts[0], parts[1]
                    if flag != target_flag:
                        continue
                    path = os.path.join(videos_root, cname, vname)
                    if os.path.isfile(path):
                        self.records.append(ClipRecord(path=path, label=class_to_idx[cname]))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_datasets(
    dataset: str, data_root: str, n_frames: int, image_size: int, fold: int = 1
) -> Tuple[Dataset, Dataset, List[str]]:
    """Build (train_data, test_data, class_labels) for the requested dataset."""
    if dataset == "synthetic":
        train = SyntheticMovingShapes(
            n_samples=2048, n_frames=n_frames, image_size=image_size, split="train"
        )
        test = SyntheticMovingShapes(
            n_samples=256, n_frames=n_frames, image_size=image_size, split="test"
        )
        return train, test, list(SYNTHETIC_CLASS_LABELS)
    if dataset == "ucf101":
        train = UCF101Clips(
            data_root, split="train", fold=fold, n_frames=n_frames, image_size=image_size
        )
        test = UCF101Clips(
            data_root, split="test", fold=fold, n_frames=n_frames, image_size=image_size
        )
        return train, test, train.class_labels
    if dataset == "hmdb51":
        train = HMDB51Clips(
            data_root, split="train", fold=fold, n_frames=n_frames, image_size=image_size
        )
        test = HMDB51Clips(
            data_root, split="test", fold=fold, n_frames=n_frames, image_size=image_size
        )
        return train, test, train.class_labels
    raise ValueError(f"Unknown dataset: {dataset}")
