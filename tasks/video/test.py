"""Smoke test for tasks/video datasets.

Auto-detects whether we are on JZ (lustre) or titan (/geovic) based on which
data roots exist on disk, then reports:

  - whether each annotation/split file is present
  - number of entries listed in the annotation
  - number of videos actually present on disk
  - whether the Dataset class instantiates and returns a sample of the
    expected shape

Run from anywhere:
    python tasks/video/test.py
or limit to one dataset:
    python tasks/video/test.py --only ucf101
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
import traceback
from typing import Optional

# Ensure the repo root is importable when running this file directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tasks.video.dataset import (  # noqa: E402
    HMDB51Clips,
    Kinetics400Clips,
    SyntheticMovingShapes,
    UCF101Clips,
)


# --------------------------------------------------------------------------- #
# Cluster detection
# --------------------------------------------------------------------------- #

JZ_ROOTS = {
    "ucf101":     "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/UCF-101",
    "hmdb51":     "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/hmdb51",
    "kinetics":   "/lustre/fsmisc/dataset/kinetics",
}

TITAN_ROOTS = {
    "ucf101":     "/geovic/geovic/UCF-101",
    "hmdb51":     "/geovic/ghermi/data/hmdb51",
    "kinetics":   "/geovic/ghermi/data/kinetics",
}


def detect_cluster() -> str:
    if os.path.isdir("/lustre/fsn1/projects/rech/kcn"):
        return "jz"
    if os.path.isdir("/geovic"):
        return "titan"
    return "unknown"


def get_roots(cluster: str) -> dict:
    if cluster == "jz":
        return JZ_ROOTS
    if cluster == "titan":
        return TITAN_ROOTS
    return {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _err(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _try_sample(ds, name: str) -> None:
    try:
        t0 = time.time()
        sample = ds[0]
        dt = time.time() - t0
    except Exception as exc:
        _err(f"{name}: failed to fetch sample[0]: {exc}")
        traceback.print_exc()
        return
    if isinstance(sample, tuple) and len(sample) == 2:
        x, y = sample
        shape = tuple(getattr(x, "shape", ()))
        _ok(f"{name}: sample[0] -> frames {shape}, label={y!r} ({dt*1000:.0f} ms)")
    else:
        _ok(f"{name}: sample[0] -> {type(sample).__name__} ({dt*1000:.0f} ms)")


# --------------------------------------------------------------------------- #
# Per-dataset checks
# --------------------------------------------------------------------------- #

def check_synthetic() -> None:
    _hr("synthetic (SyntheticMovingShapes)")
    try:
        ds = SyntheticMovingShapes(n_samples=8, n_frames=8, image_size=32, split="train")
        _ok(f"instantiated, len={len(ds)}")
        _try_sample(ds, "synthetic")
    except Exception as exc:
        _err(f"synthetic: {exc}")
        traceback.print_exc()


def check_ucf101(root: Optional[str]) -> None:
    _hr(f"ucf101  (root={root})")
    if not root or not os.path.isdir(root):
        _warn("data root not found — skipping")
        return

    videos_root = os.path.join(root, "UCF-101")
    if not os.path.isdir(videos_root):
        videos_root = os.path.join(root, "videos")
    splits_root = os.path.join(root, "ucfTrainTestlist")

    print(f"  videos_root = {videos_root} (exists={os.path.isdir(videos_root)})")
    print(f"  splits_root = {splits_root} (exists={os.path.isdir(splits_root)})")

    class_ind = os.path.join(splits_root, "classInd.txt")
    if os.path.isfile(class_ind):
        with open(class_ind) as f:
            n_classes = sum(1 for line in f if line.strip())
        print(f"  classInd.txt -> {n_classes} classes")
    else:
        _warn("classInd.txt missing")

    for fold in (1, 2, 3):
        for split in ("trainlist", "testlist"):
            p = os.path.join(splits_root, f"{split}{fold:02d}.txt")
            if not os.path.isfile(p):
                _warn(f"{p} missing")
                continue
            with open(p) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            n_listed = len(lines)
            n_present = 0
            for ln in lines:
                rel = ln.split()[0]
                if os.path.isfile(os.path.join(videos_root, rel)):
                    n_present += 1
            print(f"  {split}{fold:02d}.txt: {n_present}/{n_listed} videos present")

    for split in ("train", "test"):
        try:
            ds = UCF101Clips(root, split=split, fold=1, n_frames=8, image_size=64)
            _ok(f"UCF101Clips(split={split}, fold=1) len={len(ds)} classes={len(ds.class_labels)}")
            if len(ds):
                _try_sample(ds, f"ucf101[{split}]")
        except Exception as exc:
            _err(f"UCF101Clips(split={split}): {exc}")
            traceback.print_exc()


def check_hmdb51(root: Optional[str]) -> None:
    _hr(f"hmdb51  (root={root})")
    if not root or not os.path.isdir(root):
        _warn("data root not found — skipping")
        return

    videos_root = os.path.join(root, "hmdb51_org")
    if not os.path.isdir(videos_root):
        videos_root = os.path.join(root, "video_data")
    splits_root = os.path.join(root, "testTrainMulti_7030_splits")
    if not os.path.isdir(splits_root):
        splits_root = os.path.join(root, "annotations")

    print(f"  videos_root = {videos_root} (exists={os.path.isdir(videos_root)})")
    print(f"  splits_root = {splits_root} (exists={os.path.isdir(splits_root)})")

    if os.path.isdir(videos_root):
        classes = sorted(d for d in os.listdir(videos_root)
                         if os.path.isdir(os.path.join(videos_root, d)))
        print(f"  found {len(classes)} class directories")
        n_total_videos = sum(
            len([f for f in os.listdir(os.path.join(videos_root, c))
                 if f.lower().endswith((".avi", ".mp4"))])
            for c in classes
        )
        print(f"  total video files on disk: {n_total_videos}")

    if os.path.isdir(splits_root):
        split_files = [f for f in os.listdir(splits_root)
                       if f.endswith(".txt") and "_split" in f]
        print(f"  found {len(split_files)} split files")

    for split in ("train", "test"):
        try:
            ds = HMDB51Clips(root, split=split, fold=1, n_frames=8, image_size=64)
            _ok(f"HMDB51Clips(split={split}, fold=1) len={len(ds)} classes={len(ds.class_labels)}")
            if len(ds):
                _try_sample(ds, f"hmdb51[{split}]")
        except Exception as exc:
            _err(f"HMDB51Clips(split={split}): {exc}")
            traceback.print_exc()


def check_kinetics(root: Optional[str]) -> None:
    _hr(f"kinetics  (root={root})")
    if not root or not os.path.isdir(root):
        _warn("data root not found — skipping")
        return

    train_csv = os.path.join(root, "kinetics_400_train.csv")
    val_csv = os.path.join(root, "kinetics_400_val.csv")
    train_dir = os.path.join(root, "kinetics_400_train")
    val_dir = os.path.join(root, "kinetics_400_val")

    print(f"  train_csv  = {train_csv} (exists={os.path.isfile(train_csv)})")
    print(f"  val_csv    = {val_csv}   (exists={os.path.isfile(val_csv)})")
    print(f"  train_dir  = {train_dir} (exists={os.path.isdir(train_dir)})")
    print(f"  val_dir    = {val_dir}   (exists={os.path.isdir(val_dir)})")

    for label, csv_path, vid_dir in [
        ("train", train_csv, train_dir),
        ("val",   val_csv,   val_dir),
    ]:
        if not os.path.isfile(csv_path):
            _warn(f"{label}: csv missing")
            continue
        n_rows = 0
        n_with_label = 0
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                n_rows += 1
                if (row.get("label") or "").strip():
                    n_with_label += 1
        print(f"  {label} csv: {n_rows} rows, {n_with_label} with labels")

        if not os.path.isdir(vid_dir):
            continue
        # Count mp4 files (cheaply): one walk per split.
        n_mp4 = 0
        n_classes = 0
        for cls in os.listdir(vid_dir):
            cls_dir = os.path.join(vid_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            n_classes += 1
            n_mp4 += sum(1 for f in os.listdir(cls_dir) if f.endswith(".mp4"))
        print(f"  {label} dir: {n_classes} class dirs, {n_mp4} mp4 files")

    for split in ("train", "val"):
        try:
            ds = Kinetics400Clips(root, split=split, n_frames=8, image_size=64)
            _ok(f"Kinetics400Clips(split={split}) len={len(ds)} classes={len(ds.class_labels)}")
            if len(ds):
                _try_sample(ds, f"kinetics[{split}]")
        except Exception as exc:
            _err(f"Kinetics400Clips(split={split}): {exc}")
            traceback.print_exc()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", choices=("auto", "jz", "titan"), default="auto",
                        help="select default data roots (overridden by --*-root flags)")
    parser.add_argument("--only", choices=("synthetic", "ucf101", "hmdb51", "kinetics"),
                        default=None, help="limit to one dataset")
    parser.add_argument("--ucf101-root",   default=None)
    parser.add_argument("--hmdb51-root",   default=None)
    parser.add_argument("--kinetics-root", default=None)
    args = parser.parse_args()

    cluster = detect_cluster() if args.cluster == "auto" else args.cluster
    print(f"hostname  : {socket.gethostname()}")
    print(f"cluster   : {cluster}")
    print(f"cwd       : {os.getcwd()}")
    print(f"repo root : {_REPO_ROOT}")

    defaults = get_roots(cluster)
    roots = {
        "ucf101":   args.ucf101_root   or defaults.get("ucf101"),
        "hmdb51":   args.hmdb51_root   or defaults.get("hmdb51"),
        "kinetics": args.kinetics_root or defaults.get("kinetics"),
    }
    print("data roots:")
    for k, v in roots.items():
        exists = os.path.isdir(v) if v else False
        print(f"  {k:10s} -> {v}  (exists={exists})")

    todo = [args.only] if args.only else ["synthetic", "ucf101", "hmdb51", "kinetics"]
    if "synthetic" in todo:
        check_synthetic()
    if "ucf101" in todo:
        check_ucf101(roots.get("ucf101"))
    if "hmdb51" in todo:
        check_hmdb51(roots.get("hmdb51"))
    if "kinetics" in todo:
        check_kinetics(roots.get("kinetics"))


if __name__ == "__main__":
    main()
