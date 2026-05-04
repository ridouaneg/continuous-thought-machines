"""Video datasets for CTM repetition counting.

Four backends:

- ``SyntheticOscillatingDots``: fully synthetic. A bright dot oscillates
  vertically with a pure sine wave N times across n_frames. The count is N.
  No downloads, no video decoding. Deterministic per index, ideal for
  unit-testing whether per-neuron FFT peaks align with the ground-truth count.

- ``CountixDataset``: Countix (from the RepNet paper). CSV annotation with
  Kinetics-400 video IDs + per-clip start/end timestamps and repetition counts.
  Expects videos pre-downloaded as MP4 files.

  Two video layouts are supported:

    1. Flat ``<data_root>/videos/<kinetics_id>.mp4`` (the original Countix
       distribution).
    2. Official Kinetics-400 layout, where files live under
       ``<kinetics_root>/kinetics_400_<split>/<class>/<youtube_id>_<start:06d>_<end:06d>.mp4``.
       Pass ``kinetics_root=<path to Kinetics root>`` to enable this; the
       loader builds a ``youtube_id -> path`` index by stripping the trailing
       ``_<start>_<end>`` from each filename.

  CSV columns (official Countix release):
    video_id, class, kinetics_start, kinetics_end,
    repetition_start, repetition_end, count
  Column names are configurable via __init__ kwargs. Repetition timestamps
  are absolute (relative to the original YouTube video). When the videos on
  disk are already trimmed to the Kinetics clip span (the standard Kinetics
  download layout), the loader subtracts ``kinetics_start`` so the seek
  range is relative to the trimmed file — set ``time_offset_col=None`` to
  disable that subtraction.

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
import re
from typing import Dict, List, Optional, Tuple

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


def _read_video_frames(
    path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> torch.Tensor:
    """Decode a video into a (T, C, H, W) uint8 tensor.

    Prefers ``torchvision.io.read_video`` when available; falls back to
    OpenCV otherwise. ``read_video`` was removed from torchvision in
    version 0.24, so the OpenCV path is the working backend on newer envs.
    """
    try:
        from torchvision.io import read_video
    except ImportError:
        return _read_video_opencv(path, start_sec, end_sec)

    kwargs = {"pts_unit": "sec", "output_format": "TCHW"}
    if start_sec is not None:
        kwargs["start_pts"] = start_sec
    if end_sec is not None:
        kwargs["end_pts"] = end_sec
    frames, _, _ = read_video(path, **kwargs)
    return frames  # (T, C, H, W) uint8


def _read_video_opencv(
    path: str,
    start_sec: Optional[float],
    end_sec: Optional[float],
) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = 0 if start_sec is None else max(0, int(start_sec * fps))
        end_frame = total if end_sec is None else min(total, int(end_sec * fps))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        out = []
        for _ in range(start_frame, end_frame if end_frame > start_frame else total):
            ok, bgr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not out:
        raise RuntimeError(f"Empty video (or empty time range): {path}")
    arr = np.stack(out, axis=0)  # (T, H, W, C) uint8
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


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
    try:
        frames = _read_video_frames(path, start_sec, end_sec)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {path}: {exc}") from exc

    num = frames.shape[0]
    if num == 0:
        raise RuntimeError(f"Empty video (or empty time range): {path}")

    idxs = _tsn_segment_indices(num, n_frames, train=train)
    frames = frames[idxs].float() / 255.0                              # (T, C, H, W)

    if train:
        # Reuse the temporally-coherent augmentation pipeline from tasks/video:
        # random resized crop + horizontal flip + colour jitter + random
        # erasing, all parameters drawn once per clip.
        from tasks.video.dataset import _train_augment_video
        frames = _train_augment_video(frames, image_size)
    else:
        frames = F.interpolate(
            frames, size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
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
# Kinetics-400 youtube_id index
# --------------------------------------------------------------------------- #

# Trailing ``_<6-digit>_<6-digit>`` suffix produced by the official Kinetics
# download scripts (e.g. ``-qAC8YY5F1Q_000040_000050.mp4`` → id ``-qAC8YY5F1Q``).
_KINETICS_TIMESPAN_RE = re.compile(r"_(\d{6})_(\d{6})$")


def _build_kinetics_youtube_id_index(kinetics_root: str) -> Dict[str, str]:
    """Index a Kinetics-400 mirror by youtube_id.

    Walks ``<kinetics_root>/kinetics_400_*/`` (falling back to a full walk of
    ``<kinetics_root>/`` if those subfolders are absent) and maps each video's
    youtube_id to its full path. Per-clip start/end is intentionally ignored —
    Countix's start/end times are already relative to the trimmed Kinetics
    clip, and there is exactly one Kinetics clip per youtube_id.

    Args:
        kinetics_root: Root of the Kinetics mirror.

    Returns:
        ``{youtube_id: video_path}``. Train wins over val if both have the id.
    """
    if not os.path.isdir(kinetics_root):
        raise FileNotFoundError(f"kinetics_root does not exist: {kinetics_root}")

    # Prefer the standard kinetics_400_train / kinetics_400_val subtrees so we
    # don't accidentally walk unrelated siblings of the data root.
    candidate_roots = [
        os.path.join(kinetics_root, "kinetics_700_train"),
        os.path.join(kinetics_root, "kinetics_700_val"),
    ]
    candidate_roots = [r for r in candidate_roots if os.path.isdir(r)]
    if not candidate_roots:
        candidate_roots = [kinetics_root]

    index: Dict[str, str] = {}
    for root in candidate_roots:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                stem, ext = os.path.splitext(fname)
                if ext.lower() != ".mp4":
                    continue
                m = _KINETICS_TIMESPAN_RE.search(stem)
                yt_id = stem[: m.start()] if m else stem
                # Don't overwrite a hit from train with one from val.
                index.setdefault(yt_id, os.path.join(dirpath, fname))
    return index


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
        data_root:       Root directory (see module-level docstring for layout).
        split:           'train', 'val', or 'test'.
        n_frames:        Frames to sample per clip.
        image_size:      Spatial resolution H = W.
        id_col:          CSV column name for the YouTube/Kinetics video ID.
        count_col:       CSV column name for the integer repetition count.
        start_col:       CSV column name for repetition start time (seconds), or None.
        end_col:         CSV column name for repetition end time (seconds), or None.
        time_offset_col: CSV column name whose value is subtracted from
                         start/end (seconds). Use 'kinetics_start' when the
                         videos on disk are already trimmed to the Kinetics
                         clip span; set to None for raw-YouTube videos.
        video_ext:       Extension used when looking up downloaded video files.
        kinetics_root:   Optional root of an official Kinetics-400 mirror
                         (``<root>/kinetics_400_<split>/<class>/<id>_<start>_<end>.mp4``).
                         When given, the loader indexes that tree by youtube_id
                         instead of looking under ``<data_root>/videos/``.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        n_frames: int = 64,
        image_size: int = 112,
        id_col: str = "video_id",
        count_col: str = "count",
        start_col: str = "repetition_start",
        end_col: str = "repetition_end",
        time_offset_col: Optional[str] = "kinetics_start",
        video_ext: str = ".mp4",
        kinetics_root: Optional[str] = None,
    ):
        self.n_frames = n_frames
        self.image_size = image_size
        self.train = split == "train"
        self.start_col = start_col
        self.end_col = end_col

        csv_name = f"countix_{split}.csv"
        csv_path = os.path.join(data_root, csv_name)

        if kinetics_root is not None:
            id_to_path = _build_kinetics_youtube_id_index(kinetics_root)
            def lookup(vid_id: str) -> Optional[str]:
                return id_to_path.get(vid_id)
        else:
            videos_dir = os.path.join(data_root, "videos")
            def lookup(vid_id: str) -> Optional[str]:
                p = os.path.join(videos_dir, vid_id + video_ext)
                return p if os.path.isfile(p) else None

        self.records: List[dict] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid_id = row[id_col].strip()
                count = int(float(row[count_col]))
                start = float(row[start_col]) if start_col and start_col in row else None
                end = float(row[end_col]) if end_col and end_col in row else None
                if time_offset_col and time_offset_col in row:
                    offset = float(row[time_offset_col])
                    if start is not None:
                        start = max(0.0, start - offset)
                    if end is not None:
                        end = max(0.0, end - offset)
                path = lookup(vid_id)
                if path is not None:
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
    kinetics_root: Optional[str] = None,
) -> Tuple[Dataset, Dataset]:
    """Build (train_dataset, test_dataset) for the requested backend.

    Args:
        dataset:       One of 'synthetic', 'countix', 'repcount', 'ucfrep'.
        data_root:     Root directory for real datasets (ignored for 'synthetic').
        n_frames:      Frames sampled per clip.
        image_size:    Spatial resolution H = W.
        max_count:     Upper bound on oscillation count for 'synthetic'.
        fold:          Unused (kept for API consistency with tasks/video).
        kinetics_root: For 'countix' only — root of an official Kinetics-400
                       mirror used to look up videos by youtube_id when the
                       Countix CSVs are not co-located with the videos.

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
        train = CountixDataset(
            data_root, split="train", n_frames=n_frames, image_size=image_size,
            kinetics_root=kinetics_root,
        )
        test = CountixDataset(
            data_root, split="val", n_frames=n_frames, image_size=image_size,
            kinetics_root=kinetics_root,
        )
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
