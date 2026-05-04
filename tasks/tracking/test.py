"""Smoke test for tasks/tracking datasets.

Auto-detects whether we are on JZ (lustre) or titan (/geovic) and reports,
for each backend (synthetic, mot17, dancetrack):

  - presence of the expected directory layout
  - number of sequences found
  - per-sequence: frame count, gt rows, number of unique tracks
  - whether the Dataset class instantiates and returns a sample of the
    expected shape

Per-dataset roots can be overridden with CLI flags:
    python tasks/tracking/test.py --mot17-root /path/to/MOT17 \\
        --dancetrack-root /path/to/DanceTrack
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tasks.tracking.dataset import (  # noqa: E402
    DanceTrackDataset,
    MOT17Dataset,
    SyntheticTrackingDataset,
    _load_seqinfo,
    _read_mot_gt,
)


# --------------------------------------------------------------------------- #
# Cluster detection / defaults
# --------------------------------------------------------------------------- #

JZ_ROOTS = {
    "mot17":      "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/MOT17",
    "dancetrack": "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/DanceTrack",
}

TITAN_ROOTS = {
    "mot17":      "/geovic/ghermi/data/MOT17",
    "dancetrack": "/geovic/ghermi/data/DanceTrack",
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
        x_shape = tuple(getattr(x, "shape", ()))
        y_shape = tuple(getattr(y, "shape", ()))
        _ok(f"{name}: sample[0] -> frames {x_shape}, targets {y_shape} ({dt*1000:.0f} ms)")
    else:
        _ok(f"{name}: sample[0] -> {type(sample).__name__} ({dt*1000:.0f} ms)")


def _summarise_seq(seq_dir: Path, max_print: int = 8) -> dict:
    """Return a small summary dict for a MOT-format sequence directory."""
    info = _load_seqinfo(seq_dir)
    summary = {
        "name":       seq_dir.name,
        "img_dir":    info["img_dir"],
        "img_ext":    info["img_ext"],
        "seq_length": info["seq_length"],
        "img_wh":     (info["img_width"], info["img_height"]),
        "n_images":   0,
        "gt_rows":    0,
        "n_tracks":   0,
    }
    if info["img_dir"].exists():
        summary["n_images"] = sum(
            1 for f in info["img_dir"].iterdir()
            if f.is_file() and f.suffix.lower() == info["img_ext"].lower()
        )
    gt_path = seq_dir / "gt" / "gt.txt"
    if gt_path.exists():
        with open(gt_path) as fh:
            summary["gt_rows"] = sum(1 for ln in fh if ln.strip())
        try:
            tracks = _read_mot_gt(str(gt_path), valid_classes=())
            summary["n_tracks"] = len(tracks)
        except Exception:
            summary["n_tracks"] = -1
    return summary


# --------------------------------------------------------------------------- #
# Per-dataset checks
# --------------------------------------------------------------------------- #

def check_synthetic() -> None:
    _hr("synthetic (SyntheticTrackingDataset)")
    try:
        ds = SyntheticTrackingDataset(
            n_samples=8, n_objects=2, n_frames=8, img_size=32, n_bins=16,
        )
        _ok(f"instantiated, len={len(ds)}")
        _try_sample(ds, "synthetic")
    except Exception as exc:
        _err(f"synthetic: {exc}")
        traceback.print_exc()


def check_mot17(root: Optional[str]) -> None:
    _hr(f"mot17  (root={root})")
    if not root or not os.path.isdir(root):
        _warn("data root not found — skipping")
        return

    train_dir = Path(root) / "train"
    test_dir = Path(root) / "test"
    print(f"  train dir = {train_dir} (exists={train_dir.is_dir()})")
    print(f"  test dir  = {test_dir}  (exists={test_dir.is_dir()})")

    if train_dir.is_dir():
        seqs = sorted(d for d in train_dir.iterdir() if d.is_dir())
        print(f"  train: {len(seqs)} sequence directories")
        for seq_dir in seqs:
            s = _summarise_seq(seq_dir)
            print(f"    {s['name']:24s} frames={s['n_images']:5d}/{s['seq_length']:5d}  "
                  f"gt_rows={s['gt_rows']:6d}  tracks={s['n_tracks']:4d}  "
                  f"wh={s['img_wh']}")

    for split in ("train", "val"):
        try:
            ds = MOT17Dataset(
                root, split=split, n_frames=8, img_size=64,
                n_objects=4, n_bins=16, stride=8,
            )
            _ok(f"MOT17Dataset(split={split}) len={len(ds)} "
                f"sequences={len(ds.sequences)}")
            if len(ds):
                _try_sample(ds, f"mot17[{split}]")
        except Exception as exc:
            _err(f"MOT17Dataset(split={split}): {exc}")
            traceback.print_exc()


def check_dancetrack(root: Optional[str]) -> None:
    _hr(f"dancetrack  (root={root})")
    if not root or not os.path.isdir(root):
        _warn("data root not found — skipping")
        return

    for split in ("train", "val", "test"):
        split_dir = Path(root) / split
        exists = split_dir.is_dir()
        print(f"  {split} dir = {split_dir} (exists={exists})")
        if not exists:
            continue
        seqs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        n_with_gt = sum(1 for s in seqs if (s / "gt" / "gt.txt").exists())
        print(f"    sequences: {len(seqs)}  (with gt: {n_with_gt})")
        for seq_dir in seqs[:5]:
            s = _summarise_seq(seq_dir)
            print(f"      {s['name']:24s} frames={s['n_images']:5d}/{s['seq_length']:5d}  "
                  f"gt_rows={s['gt_rows']:6d}  tracks={s['n_tracks']:4d}  "
                  f"wh={s['img_wh']}")
        if len(seqs) > 5:
            print(f"      ... ({len(seqs) - 5} more)")

    for split in ("train", "val"):
        try:
            ds = DanceTrackDataset(
                root, split=split, n_frames=8, img_size=64,
                n_objects=4, n_bins=16, stride=8,
            )
            _ok(f"DanceTrackDataset(split={split}) len={len(ds)} "
                f"sequences={len(ds.sequences)}")
            if len(ds):
                _try_sample(ds, f"dancetrack[{split}]")
        except Exception as exc:
            _err(f"DanceTrackDataset(split={split}): {exc}")
            traceback.print_exc()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", choices=("auto", "jz", "titan"), default="auto",
                        help="select default data roots (overridden by --*-root flags)")
    parser.add_argument("--only", choices=("synthetic", "mot17", "dancetrack"),
                        default=None, help="limit to one dataset")
    parser.add_argument("--mot17-root",      default=None)
    parser.add_argument("--dancetrack-root", default=None)
    args = parser.parse_args()

    cluster = detect_cluster() if args.cluster == "auto" else args.cluster
    print(f"hostname  : {socket.gethostname()}")
    print(f"cluster   : {cluster}")
    print(f"cwd       : {os.getcwd()}")
    print(f"repo root : {_REPO_ROOT}")

    defaults = get_roots(cluster)
    roots = {
        "mot17":      args.mot17_root      or defaults.get("mot17"),
        "dancetrack": args.dancetrack_root or defaults.get("dancetrack"),
    }
    print("data roots:")
    for k, v in roots.items():
        exists = os.path.isdir(v) if v else False
        print(f"  {k:12s} -> {v}  (exists={exists})")

    todo = [args.only] if args.only else ["synthetic", "mot17", "dancetrack"]
    if "synthetic" in todo:
        check_synthetic()
    if "mot17" in todo:
        check_mot17(roots.get("mot17"))
    if "dancetrack" in todo:
        check_dancetrack(roots.get("dancetrack"))


if __name__ == "__main__":
    main()
