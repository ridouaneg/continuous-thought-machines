"""Video datasets for CTM repetition counting.

Four backends:

- ``SyntheticOscillatingDots``: fully synthetic. A bright dot oscillates
  vertically with a pure sine wave N times across n_frames. The count is N.
  No downloads, no video decoding. Deterministic per index, ideal for
  unit-testing whether per-neuron FFT peaks align with the ground-truth count.

- ``CountixDataset``: Countix (from the RepNet paper). CSV annotation with
  Kinetics-400 video IDs + per-clip start/end timestamps and repetition counts.
  Expects videos pre-downloaded as MP4 files.

  Expected layout::

      <data_root>/
      ├── countix_train.csv
      ├── countix_val.csv
      └── videos/
          ├── <kinetics_id>.mp4
          └── ...

  CSV columns: kinetics_id, repetition_count, start_time, end_time
  (column names are configurable via __init__ kwargs).

- ``RepCountADataset``: RepCount-A (from the TransRAC paper). CSV annotation
  with video paths and integer counts.

  Expected layout::

      <data_root>/
      ├── annotation/
      │   ├── train.csv
      │   ├── valid.csv
      │   └── test.csv
      └── videos/
          └── <video_name>

  CSV columns: name (filename), count
  (column names are configurable via __init__ kwargs).

- ``UCFRepDataset``: UCFRep — UCF-101 videos re-annotated with repetition
  counts. JSON annotation file listing video paths, counts, and splits.

  Expected layout::

      <data_root>/
      ├── ucfrep_annotations.json
      └── UCF-101/
          └── <ClassName>/
              └── <video>.avi

  JSON: list of {"video_name": "<Class>/<video>.avi", "count": N, "split": "train|test"}

All real-video datasets share the same TSN-style segment sampling and
ImageNet normalisation as tasks/video/dataset.py.
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Shared video utilities (mirrors tasks/video/dataset.py)
# --------------------------------------------------------------------------- #

def _tsn_segment_indices(num_video_frames: int, n_frames: int, train: bool) -> np.ndarray:
    """Pick n_frames indices from a video using TSN segment sampling."""
    if num_video_frames <= 0:
        return np.zeros(n_frames, dtype=np.int64)
    if num_video_frames < n_frames:
        idxs = np.linspace(0, num_video_frames - 1, n_frames)
        return np.round(idxs).astype(np.int64)
    seg_len = num_video_frames / n_frames
    if train:
        offsets = np.random.uniform(0, seg_len, size=n_frames)
    else:
        offsets = np.full(n_frames, seg_len / 2.0)
    starts = np.arange(n_frames) * seg_len
    return np.clip(np.floor(starts + offsets), 0, num_video_frames - 1).astype(np.int64)


def _decode_clip(
    path: str,
    n_frames: int,
    image_size: int,
    train: bool,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> torch.Tensor:
    """Read a video file and return a (T, C, H, W) float tensor in ImageNet norm.

    Args:
        path:      Path to the video file.
        n_frames:  Number of frames to sample.
        image_size: Spatial resolution (H = W = image_size).
        train:     If True, random TSN sampling + horizontal flip; else uniform.
        start_sec: Optional clip start time in seconds (for trimmed clips like Countix).
        end_sec:   Optional clip end time in seconds.
    """
    from torchvision.io import read_video  # lazy import: synthetic works without ffmpeg

    kwargs = {"pts_unit": "sec", "output_format": "TCHW"}
    if start_sec is not None:
        kwargs["start_pts"] = start_sec
    if end_sec is not None:
        kwargs["end_pts"] = end_sec

    try:
        frames, _, _ = read_video(path, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {path}: {exc}") from exc

    num = frames.shape[0]
    if num == 0:
        raise RuntimeError(f"Empty video (or empty time range): {path}")

    idxs = _tsn_segment_indices(num, n_frames, train=train)
    frames = frames[idxs].float() / 255.0                              # (T, C, H, W)

    frames = F.interpolate(
        frames, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    if train and random.random() < 0.5:
        frames = torch.flip(frames, dims=[3])
    return frames


# --------------------------------------------------------------------------- #
# Synthetic oscillating dots
# --------------------------------------------------------------------------- #

class SyntheticOscillatingDots(Dataset):
    """Synthetic clip: a bright dot bouncing N times vertically.

    The dot follows a pure cosine trajectory so the video's motion spectrum
    has a clean peak at frequency N / n_frames. This directly verifies the
    phase-locked-loop hypothesis: if the CTM NLMs learn oscillators, the
    dominant per-neuron FFT bin (expressed in cycle counts) should match N.

    Args:
        n_samples:  Number of distinct clips.
        n_frames:   Frames per clip (also == total CTM ticks when iterations_per_frame=1).
        image_size: Spatial resolution H = W.
        max_count:  Maximum oscillation count; labels are uniform in [1, max_count].
        split:      'train' or 'test' — selects a different random-seed offset.
    """

    def __init__(
        self,
        n_samples: int = 2048,
        n_frames: int = 64,
        image_size: int = 64,
        max_count: int = 16,
        split: str = "train",
    ):
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.image_size = image_size
        self.max_count = max_count
        self.offset = 0 if split == "train" else 10_000_000

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rng = np.random.default_rng(index + self.offset)
        count = int(rng.integers(1, self.max_count + 1))

        H = W = self.image_size
        radius = max(2, H // 12)
        cx = W // 2

        # Pure cosine trajectory: count full oscillations over n_frames.
        phase = np.linspace(0, 2 * np.pi * count, self.n_frames, endpoint=False)
        # Map [-1, 1] → [radius, H - radius]
        cy_arr = ((H - 2 * radius) / 2 * (1 - np.cos(phase)) + radius).astype(np.int32)

        color = rng.uniform(0.6, 1.0, size=3).astype(np.float32)
        bg = rng.uniform(0.0, 0.15, size=3).astype(np.float32)

        ys, xs = np.mgrid[0:H, 0:W]
        clip = np.zeros((self.n_frames, 3, H, W), dtype=np.float32)
        for t_idx in range(self.n_frames):
            cy = cy_arr[t_idx]
            dx = xs - cx
            dy = ys - cy
            mask = (dx * dx + dy * dy) <= radius * radius
            for c in range(3):
                frame = np.full((H, W), bg[c], dtype=np.float32)
                frame[mask] = color[c]
                clip[t_idx, c] = frame

        # Normalise to [-1, 1] to match the synthetic convention in tasks/video.
        clip = (clip - 0.5) / 0.5
        return torch.from_numpy(clip), count


# --------------------------------------------------------------------------- #
# Countix (RepNet paper)
# --------------------------------------------------------------------------- #

class CountixDataset(Dataset):
    """Countix repetition-counting dataset (Dwibedi et al., 2020).

    Countix is a subset of Kinetics-400 re-annotated with repetition counts.
    Videos must be downloaded independently (e.g. via the Kinetics download
    scripts). Each CSV row specifies the Kinetics video ID, the repetition
    count, and start/end timestamps within the full Kinetics clip.

    Args:
        data_root:      Root directory (see module-level docstring for layout).
        split:          'train' or 'val'.
        n_frames:       Frames to sample per clip.
        image_size:     Spatial resolution H = W.
        id_col:         CSV column name for the Kinetics video ID.
        count_col:      CSV column name for the integer repetition count.
        start_col:      CSV column name for clip start time (seconds), or None.
        end_col:        CSV column name for clip end time (seconds), or None.
        video_ext:      Extension used when looking up downloaded video files.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        n_frames: int = 64,
        image_size: int = 112,
        id_col: str = "kinetics_id",
        count_col: str = "repetition_count",
        start_col: str = "start_time",
        end_col: str = "end_time",
        video_ext: str = ".mp4",
    ):
        self.n_frames = n_frames
        self.image_size = image_size
        self.train = split == "train"
        self.start_col = start_col
        self.end_col = end_col

        csv_name = f"countix_{split}.csv"
        csv_path = os.path.join(data_root, csv_name)
        videos_dir = os.path.join(data_root, "videos")

        self.records: List[dict] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid_id = row[id_col].strip()
                count = int(float(row[count_col]))
                start = float(row[start_col]) if start_col and start_col in row else None
                end = float(row[end_col]) if end_col and end_col in row else None
                path = os.path.join(videos_dir, vid_id + video_ext)
                if os.path.isfile(path):
                    self.records.append(
                        {"path": path, "count": count, "start": start, "end": end}
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[index]
        clip = _decode_clip(
            rec["path"], self.n_frames, self.image_size, self.train,
            start_sec=rec["start"], end_sec=rec["end"],
        )
        return clip, rec["count"]


# --------------------------------------------------------------------------- #
# RepCount-A (TransRAC paper)
# --------------------------------------------------------------------------- #

class RepCountADataset(Dataset):
    """RepCount-A repetition-counting dataset (Hu et al., 2022).

    CSV annotation with one row per video. Videos can be MP4 or AVI files
    stored under <data_root>/videos/.

    Args:
        data_root:   Root directory (see module-level docstring for layout).
        split:       'train', 'valid', or 'test'.
        n_frames:    Frames to sample per clip.
        image_size:  Spatial resolution H = W.
        name_col:    CSV column containing the video filename (with or without extension).
        count_col:   CSV column containing the integer repetition count.
        video_subdir: Subdirectory under data_root where video files live.
        video_exts:  File extensions to try when locating a video.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        n_frames: int = 64,
        image_size: int = 112,
        name_col: str = "name",
        count_col: str = "count",
        video_subdir: str = "videos",
        video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov"),
    ):
        self.n_frames = n_frames
        self.image_size = image_size
        self.train = split == "train"

        csv_path = os.path.join(data_root, "annotation", f"{split}.csv")
        videos_dir = os.path.join(data_root, video_subdir)

        self.records: List[dict] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row[name_col].strip()
                count = int(float(row[count_col]))
                # Try to find the video file with any supported extension.
                path = self._resolve_path(videos_dir, name, video_exts)
                if path is not None:
                    self.records.append({"path": path, "count": count})

    @staticmethod
    def _resolve_path(
        videos_dir: str, name: str, exts: Tuple[str, ...]
    ) -> Optional[str]:
        # If name already has an extension, try it directly.
        if os.path.isfile(os.path.join(videos_dir, name)):
            return os.path.join(videos_dir, name)
        stem = os.path.splitext(name)[0]
        for ext in exts:
            candidate = os.path.join(videos_dir, stem + ext)
            if os.path.isfile(candidate):
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[index]
        clip = _decode_clip(rec["path"], self.n_frames, self.image_size, self.train)
        return clip, rec["count"]


# --------------------------------------------------------------------------- #
# UCFRep
# --------------------------------------------------------------------------- #

class UCFRepDataset(Dataset):
    """UCFRep — UCF-101 videos annotated with repetition counts.

    JSON annotation file with a list of records, each containing
    "video_name" (<ClassName>/<video>.avi), "count", and "split".

    Args:
        data_root:       Root directory (see module-level docstring for layout).
        split:           'train' or 'test'.
        n_frames:        Frames to sample per clip.
        image_size:      Spatial resolution H = W.
        annotation_file: JSON filename relative to data_root.
        videos_subdir:   Subdirectory under data_root containing UCF-101 videos.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        n_frames: int = 64,
        image_size: int = 112,
        annotation_file: str = "ucfrep_annotations.json",
        videos_subdir: str = "UCF-101",
    ):
        self.n_frames = n_frames
        self.image_size = image_size
        self.train = split == "train"

        ann_path = os.path.join(data_root, annotation_file)
        videos_root = os.path.join(data_root, videos_subdir)

        with open(ann_path) as f:
            annotations = json.load(f)

        self.records: List[dict] = []
        for entry in annotations:
            if entry.get("split", split) != split:
                continue
            rel = entry["video_name"]
            count = int(entry["count"])
            path = os.path.join(videos_root, rel)
            if os.path.isfile(path):
                self.records.append({"path": path, "count": count})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[index]
        clip = _decode_clip(rec["path"], self.n_frames, self.image_size, self.train)
        return clip, rec["count"]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_datasets(
    dataset: str,
    data_root: str,
    n_frames: int,
    image_size: int,
    max_count: int = 16,
    fold: int = 1,
) -> Tuple[Dataset, Dataset]:
    """Build (train_dataset, test_dataset) for the requested backend.

    Args:
        dataset:    One of 'synthetic', 'countix', 'repcount', 'ucfrep'.
        data_root:  Root directory for real datasets (ignored for 'synthetic').
        n_frames:   Frames sampled per clip.
        image_size: Spatial resolution H = W.
        max_count:  Upper bound on oscillation count for 'synthetic'.
        fold:       Unused (kept for API consistency with tasks/video).

    Returns:
        (train_dataset, test_dataset)
    """
    if dataset == "synthetic":
        train = SyntheticOscillatingDots(
            n_samples=2048, n_frames=n_frames, image_size=image_size,
            max_count=max_count, split="train",
        )
        test = SyntheticOscillatingDots(
            n_samples=256, n_frames=n_frames, image_size=image_size,
            max_count=max_count, split="test",
        )
        return train, test

    if dataset == "countix":
        train = CountixDataset(data_root, split="train", n_frames=n_frames, image_size=image_size)
        test = CountixDataset(data_root, split="val", n_frames=n_frames, image_size=image_size)
        return train, test

    if dataset == "repcount":
        train = RepCountADataset(data_root, split="train", n_frames=n_frames, image_size=image_size)
        test = RepCountADataset(data_root, split="valid", n_frames=n_frames, image_size=image_size)
        return train, test

    if dataset == "ucfrep":
        train = UCFRepDataset(data_root, split="train", n_frames=n_frames, image_size=image_size)
        test = UCFRepDataset(data_root, split="test", n_frames=n_frames, image_size=image_size)
        return train, test

    raise ValueError(f"Unknown dataset: {dataset!r}. "
                     f"Choose from 'synthetic', 'countix', 'repcount', 'ucfrep'.")
