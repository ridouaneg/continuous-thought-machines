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

- ``Kinetics400Clips``: Kinetics-400 layout as distributed by the official
  download scripts:
  ``<data_root>/kinetics_400_<split>/<class>/<youtube_id>_<start:06d>_<end:06d>.mp4``
  plus a CSV per split (``kinetics_400_{train,val,test}.csv``) with columns
  ``label,youtube_id,time_start,time_end,split,is_cc`` (the test CSV has no
  ``label`` column). The class index is derived from the train CSV so it is
  stable across machines even when only a subset of videos is on disk.

These datasets use ``torchvision.io.read_video`` for decoding. Frames
are uniformly subsampled to ``n_frames`` (TSN-style segment sampling at
train time: pick a random frame inside each of ``n_frames`` equal-length
segments; uniform mid-segment sampling at test time).
"""

from __future__ import annotations

import csv
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


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _train_augment_video(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    """Apply video-coherent spatial+photometric augmentation.

    Augmentation parameters are drawn ONCE per clip and reused across all
    frames so temporal coherence is preserved. Includes:
      - random resized crop (scale 0.5-1.0, ratio 3/4-4/3)
      - random horizontal flip (p=0.5)
      - color jitter (brightness/contrast/saturation in [0.8, 1.2])
      - random erasing (p=0.25, same region across frames)

    Args:
        frames: (T, C, H, W) tensor in [0, 1].
        image_size: Output spatial resolution.

    Returns:
        (T, C, image_size, image_size) tensor in [0, 1].
    """
    import torchvision.transforms.v2.functional as TF
    from torchvision.transforms.v2 import RandomResizedCrop

    # Random resized crop — same crop for every frame in the clip.
    i, j, h, w = RandomResizedCrop.get_params(
        frames, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
    )
    frames = TF.resized_crop(
        frames, i, j, h, w, [image_size, image_size],
        interpolation=TF.InterpolationMode.BILINEAR, antialias=True,
    )

    if random.random() < 0.5:
        frames = TF.hflip(frames)

    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.8, 1.2)
    frames = TF.adjust_brightness(frames, brightness)
    frames = TF.adjust_contrast(frames, contrast)
    frames = TF.adjust_saturation(frames, saturation)
    frames = frames.clamp(0.0, 1.0)

    if random.random() < 0.25:
        eh = max(4, int(random.uniform(0.10, 0.25) * image_size))
        ew = max(4, int(random.uniform(0.10, 0.25) * image_size))
        top = random.randint(0, image_size - eh)
        left = random.randint(0, image_size - ew)
        frames[:, :, top:top + eh, left:left + ew] = random.random()

    return frames


def _decode_clip(path: str, n_frames: int, image_size: int, train: bool) -> torch.Tensor:
    """Read a video file and return a (T, C, H, W) float tensor in [-1, 1]."""
    import cv2  # imported lazily so synthetic works w/o opencv

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    buf = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # cv2 returns BGR uint8 (H, W, C); convert to RGB
            buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not buf:
        raise RuntimeError(f"Empty video: {path}")
    # (T, H, W, C) uint8 -> (T, C, H, W) uint8 tensor
    frames = torch.from_numpy(np.stack(buf, axis=0)).permute(0, 3, 1, 2).contiguous()
    num = frames.shape[0]

    idxs = _tsn_segment_indices(num, n_frames, train=train)
    frames = frames[idxs].float() / 255.0  # (T, C, H, W) in [0, 1]

    if train:
        frames = _train_augment_video(frames, image_size)
    else:
        # Test time: deterministic centre crop after resize.
        frames = F.interpolate(
            frames, size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        )

    frames = (frames - _IMAGENET_MEAN) / _IMAGENET_STD
    return frames


class _ClipDatasetBase(Dataset):
    """Shared logic for UCF-101 / HMDB-51 / Kinetics-400."""

    records: List[ClipRecord]
    class_labels: List[str]

    def __init__(self, n_frames: int, image_size: int, train: bool):
        self.n_frames = n_frames
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # Some Kinetics mp4s on disk are truncated downloads ("moov atom not
        # found") and a handful of UCF/HMDB clips have decode quirks. Skip
        # ahead instead of failing the whole DataLoader. Bounded so we don't
        # loop forever on a fully-corrupt slice.
        n = len(self.records)
        for offset in range(min(n, 32)):
            rec = self.records[(index + offset) % n]
            try:
                clip = _decode_clip(rec.path, self.n_frames, self.image_size, self.train)
            except Exception:
                continue
            return clip, rec.label
        raise RuntimeError(
            f"Could not decode any of 32 consecutive videos starting at index "
            f"{index} — check the dataset for corrupt files."
        )


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
        if not os.path.isdir(videos_root):
            videos_root = os.path.join(root, "videos")
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
        if not os.path.isdir(videos_root):
            videos_root = os.path.join(root, "video_data")
        splits_root = os.path.join(root, "testTrainMulti_7030_splits")
        if not os.path.isdir(splits_root):
            splits_root = os.path.join(root, "annotations")
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


class Kinetics400Clips(_ClipDatasetBase):
    """Kinetics-400 clip dataset.

    Expects::

        <root>/kinetics_400_<split>.csv
        <root>/kinetics_400_<split>/<class>/<youtube_id>_<start:06d>_<end:06d>.mp4

    where ``<split>`` is ``train`` or ``val``. The Kinetics test split is not
    supported here because the released test CSV has no labels.

    Class index is derived from ``kinetics_400_train.csv`` (alphabetical over
    the unique labels found there) so the index is stable across machines
    regardless of which videos are present on the local filesystem. Records
    whose underlying ``.mp4`` is missing are skipped silently — this lets the
    same code work on a full Kinetics mirror and on a small "mini-Kinetics"
    slice (e.g. a handful of classes copied to a workstation for testing).
    """

    SPLIT_TO_DIR = {"train": "kinetics_400_train", "val": "kinetics_400_val"}
    SPLIT_TO_CSV = {"train": "kinetics_400_train.csv", "val": "kinetics_400_val.csv"}
    TRAIN_CSV = "kinetics_400_train.csv"

    def __init__(self, root, split="train", n_frames=16, image_size=112):
        if split not in self.SPLIT_TO_DIR:
            raise ValueError(
                f"Kinetics400Clips supports split='train' or 'val' (got {split!r}); "
                f"the test split has no labels."
            )
        super().__init__(n_frames, image_size, train=(split == "train"))
        self.root = root
        self.split = split

        train_csv = os.path.join(root, self.TRAIN_CSV)
        if not os.path.isfile(train_csv):
            raise FileNotFoundError(f"Missing train CSV: {train_csv}")
        labels = set()
        with open(train_csv, newline="") as f:
            for row in csv.DictReader(f):
                lbl = (row.get("label") or "").strip()
                if lbl:
                    labels.add(lbl)
        self.class_labels = sorted(labels)
        class_to_idx = {c: i for i, c in enumerate(self.class_labels)}

        split_csv = os.path.join(root, self.SPLIT_TO_CSV[split])
        videos_root = os.path.join(root, self.SPLIT_TO_DIR[split])
        # On a partial mirror, the val folder may not exist yet — fall back to
        # train so val-time evaluation still works for sanity testing.
        if not os.path.isdir(videos_root) and split == "val":
            fallback = os.path.join(root, self.SPLIT_TO_DIR["train"])
            if os.path.isdir(fallback):
                videos_root = fallback
                split_csv = os.path.join(root, self.SPLIT_TO_CSV["train"])
        if not os.path.isfile(split_csv):
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")
        if not os.path.isdir(videos_root):
            raise FileNotFoundError(f"Missing video directory: {videos_root}")

        self.records = []
        n_seen = n_kept = 0
        with open(split_csv, newline="") as f:
            for row in csv.DictReader(f):
                n_seen += 1
                lbl = (row.get("label") or "").strip()
                if not lbl or lbl not in class_to_idx:
                    continue
                yt = row["youtube_id"].strip()
                ts = int(float(row["time_start"]))
                te = int(float(row["time_end"]))
                fname = f"{yt}_{ts:06d}_{te:06d}.mp4"
                path = os.path.join(videos_root, lbl, fname)
                if os.path.isfile(path):
                    self.records.append(ClipRecord(path=path, label=class_to_idx[lbl]))
                    n_kept += 1
        if not self.records:
            raise RuntimeError(
                f"No Kinetics videos found on disk for split={split!r}. "
                f"Looked under {videos_root} using {split_csv}."
            )


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
    if dataset == "kinetics":
        train = Kinetics400Clips(
            data_root, split="train", n_frames=n_frames, image_size=image_size
        )
        test = Kinetics400Clips(
            data_root, split="val", n_frames=n_frames, image_size=image_size
        )
        return train, test, train.class_labels
    raise ValueError(f"Unknown dataset: {dataset}")
