"""Datasets for predictive-coding pre-training and downstream fine-tuning.

Pre-training: ``VideoFolderDataset`` walks a directory tree and emits clips —
no labels needed. Suitable for Kinetics-{200,400,600,700} as long as videos
live somewhere under ``<root>``. For the official Kinetics-400 layout
(``<root>/kinetics_400_train/<class>/<id>_<start>_<end>.mp4``), point
``data_root`` at the ``kinetics_400_train`` subfolder.

Fine-tuning: re-uses the existing UCF-101 / HMDB-51 / Kinetics-400 / synthetic
builders from ``tasks/video/dataset.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from tasks.video.dataset import (
    SYNTHETIC_CLASS_LABELS,
    HMDB51Clips,
    Kinetics400Clips,
    SyntheticMovingShapes,
    UCF101Clips,
    _decode_clip,
)


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class _VideoRecord:
    path: str


class VideoFolderDataset(Dataset):
    """Recursively gather every video under ``root`` and return clips.

    No labels required. A single dummy label is returned alongside each clip
    so the downstream collation stays uniform with the action-recognition
    datasets.
    """

    def __init__(self, root: str, n_frames: int = 16, image_size: int = 112,
                 train: bool = True, max_videos: int | None = None):
        self.root = root
        self.n_frames = n_frames
        self.image_size = image_size
        self.train = train

        if not os.path.isdir(root):
            raise FileNotFoundError(f"VideoFolderDataset root does not exist: {root}")

        records: List[_VideoRecord] = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(VIDEO_EXTS):
                    records.append(_VideoRecord(path=os.path.join(dirpath, f)))
        records.sort(key=lambda r: r.path)
        if max_videos is not None:
            records = records[:max_videos]
        if not records:
            raise RuntimeError(f"No videos found under {root} with extensions {VIDEO_EXTS}")
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        try:
            clip = _decode_clip(rec.path, self.n_frames, self.image_size, self.train)
        except Exception:
            # Skip unreadable files by falling back to a neighbour.
            return self.__getitem__((index + 1) % len(self.records))
        return clip, 0


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def build_pretrain_dataset(
    dataset: str, data_root: str, n_frames: int, image_size: int,
    max_videos: int | None = None,
) -> Dataset:
    """Build the unlabeled clip dataset used for pre-training."""
    if dataset == "synthetic":
        return SyntheticMovingShapes(
            n_samples=2048, n_frames=n_frames, image_size=image_size, split="train"
        )
    if dataset in ("kinetics", "video_folder"):
        return VideoFolderDataset(
            data_root, n_frames=n_frames, image_size=image_size,
            train=True, max_videos=max_videos
        )
    raise ValueError(f"Unknown pretrain dataset: {dataset}")


def build_finetune_datasets(
    dataset: str, data_root: str, n_frames: int, image_size: int, fold: int = 1
) -> Tuple[Dataset, Dataset, List[str]]:
    """Build (train, test, class_labels) for downstream action recognition."""
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
    raise ValueError(f"Unknown finetune dataset: {dataset}")
