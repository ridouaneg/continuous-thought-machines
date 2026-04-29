"""Datasets for the CTM multi-object tracking task.

Three backends share the same interface:
    frames  : (T, C, H, W)  float32
    targets : (T, N, 2)     int64 bin indices in [0, n_bins-1]
                            (-1 means the object is absent — real datasets only)

SyntheticTrackingDataset
    N Gaussian blobs move inside a unit box under constant-velocity dynamics
    with elastic wall reflections. All blobs are visually identical, so the
    model must use motion cues to maintain object identity across frames.

MOT17Dataset
    Download: https://motchallenge.net/data/MOT17.zip  (~5 GB)
    Layout::
        <data_root>/
        ├── train/
        │   ├── MOT17-02-DPM/
        │   │   ├── img1/  000001.jpg ...
        │   │   ├── gt/gt.txt
        │   │   └── seqinfo.ini
        │   └── ...
        └── test/   (no gt.txt)
    GT format (comma-separated):
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    Only class=1 (pedestrian) with conf=1 is used.

DanceTrackDataset
    Download: https://github.com/DanceTrack/DanceTrack
    Same MOT-format layout; train/, val/, test/ splits are pre-defined.

Object canonical ordering (all backends): tracks are sorted by their average
x-position within the window — leftmost track is object 0. The model must
maintain this identity even after paths cross.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ImageNet statistics used by pretrained encoders.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticTrackingDataset(Dataset):
    """Synthetic multi-object tracking with bouncing Gaussian blobs.

    Frames are emitted with ``in_channels`` channels. With ``in_channels=3``
    the grayscale blob is replicated across RGB and ImageNet-normalised so
    the output matches what a frozen ImageNet-pretrained backbone expects.
    """

    def __init__(
        self,
        n_samples: int,
        n_objects: int = 2,
        n_frames: int = 8,
        img_size: int = 32,
        n_bins: int = 16,
        blob_sigma_px: float = 1.5,
        velocity_scale: float = 0.07,
        seed: int = 0,
        in_channels: int = 1,
    ):
        self.n_samples = n_samples
        self.n_objects = n_objects
        self.n_frames = n_frames
        self.img_size = img_size
        self.n_bins = n_bins
        self.blob_sigma_px = blob_sigma_px
        self.velocity_scale = velocity_scale
        self.in_channels = in_channels

        rng = np.random.default_rng(seed)
        pos = rng.uniform(0.1, 0.9, (n_samples, n_objects, 2)).astype(np.float32)
        vel = (rng.uniform(-1, 1, (n_samples, n_objects, 2)) * velocity_scale).astype(np.float32)

        # Sort objects canonically by initial x-position
        order = np.argsort(pos[:, :, 0], axis=1)           # (S, N)
        idx = np.arange(n_samples)[:, None]
        pos = pos[idx, order]
        vel = vel[idx, order]

        # Simulate full trajectories — (S, T, N, 2)
        traj = np.empty((n_samples, n_frames, n_objects, 2), dtype=np.float32)
        p, v = pos.copy(), vel.copy()
        for t in range(n_frames):
            traj[:, t] = p
            p = p + v
            lo, hi = p < 0.05, p > 0.95
            v[lo] = np.abs(v[lo])
            v[hi] = -np.abs(v[hi])
            p = np.clip(p, 0.05, 0.95)

        self.traj = traj  # positions in [0, 1]

        H = W = img_size
        xs = np.linspace(0, 1, W, dtype=np.float32)
        ys = np.linspace(0, 1, H, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)
        self.sigma2 = 2.0 * (blob_sigma_px / img_size) ** 2

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        traj = self.traj[idx]   # (T, N, 2)
        T, N = self.n_frames, self.n_objects
        H = W = self.img_size

        frames = np.empty((T, 1, H, W), dtype=np.float32)
        for t in range(T):
            frame = np.zeros((H, W), dtype=np.float32)
            for n in range(N):
                px, py = traj[t, n]
                frame += np.exp(-((self.grid_x - px) ** 2 + (self.grid_y - py) ** 2) / self.sigma2)
            frames[t, 0] = np.clip(frame, 0.0, 1.0)

        frames_t = torch.from_numpy(frames)
        if self.in_channels == 3:
            frames_t = frames_t.expand(T, 3, H, W).contiguous()
            frames_t = (frames_t - _IMAGENET_MEAN) / _IMAGENET_STD

        targets = np.clip(
            (traj * self.n_bins).astype(np.int64), 0, self.n_bins - 1
        )  # (T, N, 2)

        return frames_t, torch.from_numpy(targets)


# ---------------------------------------------------------------------------
# Real-data helpers (MOT format)
# ---------------------------------------------------------------------------

def _read_mot_gt(
    gt_path: str,
    valid_classes: Tuple[int, ...] = (1,),
) -> Dict[int, Dict[int, np.ndarray]]:
    """Parse a MOT-format gt.txt file.

    Returns
    -------
    tracks : dict[track_id → dict[frame_id → np.ndarray([cx, cy])]]
        Positions are in pixel coordinates (caller must divide by image size).
        Only rows with conf=1 and class in valid_classes are kept.
    """
    tracks: Dict[int, Dict[int, np.ndarray]] = {}
    with open(gt_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            bb_left  = float(parts[2])
            bb_top   = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            conf     = int(float(parts[6])) if len(parts) > 6 else 1
            cls      = int(float(parts[7])) if len(parts) > 7 else 1
            vis      = float(parts[8])      if len(parts) > 8 else 1.0

            if conf == 0:
                continue
            if valid_classes and cls not in valid_classes:
                continue
            if vis < 0.1:
                continue

            cx = bb_left + bb_width  / 2.0
            cy = bb_top  + bb_height / 2.0
            if track_id not in tracks:
                tracks[track_id] = {}
            tracks[track_id][frame_id] = np.array([cx, cy], dtype=np.float32)

    return tracks


def _load_seqinfo(seq_dir: Path) -> dict:
    """Parse seqinfo.ini; fall back to sane defaults if absent."""
    info_path = seq_dir / 'seqinfo.ini'
    cfg = configparser.ConfigParser()
    defaults = {
        'imdir': 'img1', 'imext': '.jpg',
        'seqlength': '0', 'imwidth': '1920', 'imheight': '1080',
    }
    if info_path.exists():
        cfg.read(info_path)
        sec = 'Sequence'
    else:
        cfg['Sequence'] = defaults
        sec = 'Sequence'

    for k, v in defaults.items():
        if not cfg.has_option(sec, k):
            cfg.set(sec, k, v)

    img_dir = seq_dir / cfg.get(sec, 'imDir')
    if not img_dir.exists():
        for alt in ['img1', 'images', 'frames']:
            if (seq_dir / alt).exists():
                img_dir = seq_dir / alt
                break

    seq_length = int(cfg.get(sec, 'seqLength'))
    if seq_length == 0:
        ext = cfg.get(sec, 'imExt')
        seq_length = len(list(img_dir.glob(f'*{ext}'))) if img_dir.exists() else 0

    return {
        'img_dir':    img_dir,
        'img_ext':    cfg.get(sec, 'imExt'),
        'seq_length': seq_length,
        'img_width':  int(cfg.get(sec, 'imWidth')),
        'img_height': int(cfg.get(sec, 'imHeight')),
    }


class _Sequence:
    """Wraps one tracking sequence and extracts (start_frame, track_ids) windows."""

    def __init__(
        self,
        seq_dir: Path,
        n_frames: int,
        n_objects: int,
        n_bins: int,
        stride: int,
        frame_range: Optional[Tuple[int, int]],
        valid_classes: Tuple[int, ...],
    ):
        self.seq_dir   = seq_dir
        self.n_frames  = n_frames
        self.n_objects = n_objects
        self.n_bins    = n_bins

        info = _load_seqinfo(seq_dir)
        self.img_dir    = info['img_dir']
        self.img_ext    = info['img_ext']
        self.img_width  = info['img_width']
        self.img_height = info['img_height']
        seq_length      = info['seq_length']

        first = (frame_range[0] if frame_range else 1)
        last  = (frame_range[1] if frame_range else seq_length)
        self.first_frame = max(1, first)
        self.last_frame  = min(seq_length, last) if seq_length > 0 else last

        gt_path = seq_dir / 'gt' / 'gt.txt'
        raw_tracks = _read_mot_gt(str(gt_path), valid_classes=valid_classes)
        self.tracks: Dict[int, Dict[int, np.ndarray]] = {}
        for tid, frames in raw_tracks.items():
            self.tracks[tid] = {}
            for fid, pos in frames.items():
                if self.first_frame <= fid <= self.last_frame:
                    cx_n = float(np.clip(pos[0] / self.img_width,  0.0, 1.0))
                    cy_n = float(np.clip(pos[1] / self.img_height, 0.0, 1.0))
                    self.tracks[tid][fid] = np.array([cx_n, cy_n], dtype=np.float32)

        self.windows = self._extract_windows(stride)

    def _extract_windows(self, stride: int) -> List[Tuple[int, List[int]]]:
        """Return list of (start_frame, ordered_track_ids) for every valid window."""
        windows = []
        for start in range(self.first_frame,
                           self.last_frame - self.n_frames + 2,
                           stride):
            end   = start + self.n_frames - 1
            fspan = list(range(start, end + 1))

            presence = {
                tid: sum(1 for f in fspan if f in t_frames)
                for tid, t_frames in self.tracks.items()
            }
            candidates = [
                tid for tid, cnt in presence.items()
                if cnt >= max(1, self.n_frames // 2)
            ]
            if len(candidates) < self.n_objects:
                continue

            def sort_key(tid):
                visible_xs = [
                    self.tracks[tid][f][0]
                    for f in fspan if f in self.tracks[tid]
                ]
                avg_x = float(np.mean(visible_xs)) if visible_xs else 0.5
                return (-presence[tid], avg_x)

            candidates.sort(key=sort_key)
            chosen = candidates[:self.n_objects]

            # Canonical ordering: leftmost average x = object 0 (matches synthetic)
            def avg_x(tid):
                xs = [self.tracks[tid][f][0] for f in fspan if f in self.tracks[tid]]
                return float(np.mean(xs)) if xs else 0.5

            chosen.sort(key=avg_x)
            windows.append((start, chosen))

        return windows

    def get_sample(self, window_idx: int, img_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start, track_ids = self.windows[window_idx]
        T, N, K = self.n_frames, self.n_objects, self.n_bins

        frames  = []
        targets = np.full((T, N, 2), -1, dtype=np.int64)

        for ti, fid in enumerate(range(start, start + T)):
            img_name = f'{fid:06d}{self.img_ext}'
            img_path = self.img_dir / img_name
            if not img_path.exists():
                # DanceTrack uses 8-digit zero-padding
                img_name = f'{fid:08d}{self.img_ext}'
                img_path = self.img_dir / img_name

            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size), Image.BILINEAR)
            frame_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
            frame_t = frame_t.permute(2, 0, 1)
            frame_t = (frame_t - _IMAGENET_MEAN) / _IMAGENET_STD
            frames.append(frame_t)

            for ni, tid in enumerate(track_ids):
                if fid in self.tracks[tid]:
                    pos = self.tracks[tid][fid]
                    bx  = int(np.clip(pos[0] * K, 0, K - 1))
                    by  = int(np.clip(pos[1] * K, 0, K - 1))
                    targets[ti, ni] = [bx, by]

        return torch.stack(frames), torch.from_numpy(targets)


class _MOTFormatDataset(Dataset):
    """Base dataset for any MOT-format sequence collection."""

    def __init__(self, sequences: List[_Sequence], img_size: int):
        self.sequences = sequences
        self.img_size  = img_size

        self._index: List[Tuple[int, int]] = []
        for si, seq in enumerate(sequences):
            for wi in range(len(seq.windows)):
                self._index.append((si, wi))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        si, wi = self._index[idx]
        return self.sequences[si].get_sample(wi, self.img_size)


class MOT17Dataset(_MOTFormatDataset):
    """MOT17 pedestrian tracking. Validation is a temporal split of training sequences."""

    _TRAIN_SEQS = [2, 4, 5, 9, 10, 11, 13]

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        n_frames: int = 8,
        img_size: int = 128,
        n_objects: int = 8,
        n_bins: int = 16,
        stride: int = 4,
        val_ratio: float = 0.2,
        detector: str = 'DPM',
    ):
        train_root = Path(data_root) / 'train'
        sequences = []
        for seq_id in self._TRAIN_SEQS:
            seq_name = f'MOT17-{seq_id:02d}-{detector}'
            seq_dir  = train_root / seq_name
            if not seq_dir.exists():
                continue

            info = _load_seqinfo(seq_dir)
            L = info['seq_length']
            val_start = max(1, int(L * (1 - val_ratio)) + 1)

            if split == 'train':
                frame_range = (1, val_start - 1)
            else:
                frame_range = (val_start, L)

            seq = _Sequence(
                seq_dir, n_frames, n_objects, n_bins, stride,
                frame_range=frame_range, valid_classes=(1,),
            )
            if seq.windows:
                sequences.append(seq)

        super().__init__(sequences, img_size)


class DanceTrackDataset(_MOTFormatDataset):
    """DanceTrack dancer-tracking dataset. Uses pre-defined train/val/test splits."""

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        n_frames: int = 8,
        img_size: int = 128,
        n_objects: int = 8,
        n_bins: int = 16,
        stride: int = 4,
    ):
        split_root = Path(data_root) / split
        if not split_root.exists():
            raise FileNotFoundError(f'DanceTrack split directory not found: {split_root}')

        sequences = []
        for seq_dir in sorted(split_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_path = seq_dir / 'gt' / 'gt.txt'
            if not gt_path.exists():
                continue   # test split — no GT
            seq = _Sequence(
                seq_dir, n_frames, n_objects, n_bins, stride,
                frame_range=None, valid_classes=(),
            )
            if seq.windows:
                sequences.append(seq)

        super().__init__(sequences, img_size)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_datasets(
    dataset: str,
    data_root: str,
    n_frames: int,
    img_size: int,
    n_objects: int,
    n_bins: int,
    stride: int = 4,
    val_ratio: float = 0.2,
    seed: int = 0,
    # synthetic-only
    n_train: int = 50_000,
    n_test: int = 5_000,
    blob_sigma_px: float = 1.5,
    velocity_scale: float = 0.07,
    in_channels: int = 1,
) -> Tuple[Dataset, Dataset]:
    """Return (train_dataset, val_dataset) for the requested backend.

    Args:
        dataset:   One of 'synthetic', 'mot17', 'dancetrack'.
        data_root: Root directory for real datasets (ignored for 'synthetic').
        n_frames:  Frames per window / sequence.
        img_size:  Spatial resolution H = W.
        n_objects: Number of tracked objects per sample.
        n_bins:    Position discretisation bins per axis.
        stride:    Window stride for real datasets (frames).
        val_ratio: Fraction of frames used for validation in MOT17.
        seed, n_train, n_test, blob_sigma_px, velocity_scale:
                   Synthetic-dataset parameters (ignored for real datasets).
    """
    if dataset == 'synthetic':
        train = SyntheticTrackingDataset(
            n_samples=n_train, n_objects=n_objects, n_frames=n_frames,
            img_size=img_size, n_bins=n_bins,
            blob_sigma_px=blob_sigma_px, velocity_scale=velocity_scale, seed=seed,
            in_channels=in_channels,
        )
        val = SyntheticTrackingDataset(
            n_samples=n_test, n_objects=n_objects, n_frames=n_frames,
            img_size=img_size, n_bins=n_bins,
            blob_sigma_px=blob_sigma_px, velocity_scale=velocity_scale, seed=seed + 1,
            in_channels=in_channels,
        )
        return train, val

    if dataset == 'mot17':
        train = MOT17Dataset(
            data_root, split='train', n_frames=n_frames, img_size=img_size,
            n_objects=n_objects, n_bins=n_bins, stride=stride, val_ratio=val_ratio,
        )
        val = MOT17Dataset(
            data_root, split='val', n_frames=n_frames, img_size=img_size,
            n_objects=n_objects, n_bins=n_bins, stride=stride * 2, val_ratio=val_ratio,
        )
        return train, val

    if dataset == 'dancetrack':
        train = DanceTrackDataset(
            data_root, split='train', n_frames=n_frames, img_size=img_size,
            n_objects=n_objects, n_bins=n_bins, stride=stride,
        )
        val = DanceTrackDataset(
            data_root, split='val', n_frames=n_frames, img_size=img_size,
            n_objects=n_objects, n_bins=n_bins, stride=stride * 2,
        )
        return train, val

    raise ValueError(
        f"Unknown dataset {dataset!r}. Choose from: 'synthetic', 'mot17', 'dancetrack'."
    )
