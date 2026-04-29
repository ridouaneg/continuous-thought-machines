"""Convert raw UCFRep annotations to the JSON format expected by ``UCFRepDataset``.

The raw archive (``ucf526_annotations.zip``) is laid out as::

    <data_root>/annotations/
        repetition_label/<Class>.txt   # blocks of (video_name, "f1 f2 ...") per video
        train/<video>.mat              # 421 train videos
        val/<video>.mat                # 105 val videos

This script produces ``<data_root>/ucfrep_annotations.json`` with one entry per
video::

    [{"video_name": "<Class>/<video>.avi", "count": N, "split": "train"|"test"}, ...]

Repetition count is computed as ``max(0, len(keypoints) - 1)`` — the standard
UCFRep convention where keypoints are cycle boundaries (N+1 boundaries → N cycles).

Optionally creates a symlink ``<data_root>/UCF-101 -> <ucf101_videos>`` so the
dataset can resolve ``<Class>/<video>.avi`` paths.

Usage::

    python -m tasks.repetition.scripts.prepare_ucfrep \\
        --data_root /geovic/ghermi/data/ucfrep \\
        --ucf101_videos /geovic/geovic/UCF-101/videos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_label_file(path: Path):
    """Yield ``(video_name, count)`` tuples from one repetition_label/<Class>.txt."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) % 2 != 0:
        print(f"WARN: odd number of non-empty lines in {path.name}; truncating",
              file=sys.stderr)
        lines = lines[:len(lines) - (len(lines) % 2)]
    for i in range(0, len(lines), 2):
        name = lines[i]
        kp = lines[i + 1].split()
        # UCFRep keypoints mark cycle boundaries: N+1 boundaries → N cycles.
        count = max(0, len(kp) - 1)
        yield name, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="UCFRep data root containing annotations/")
    p.add_argument("--ucf101_videos", type=str, default=None,
                   help="Existing UCF-101 video dir (with <Class>/<video>.avi). "
                        "If given, creates <data_root>/UCF-101 symlink.")
    p.add_argument("--video_ext", type=str, default=".avi")
    p.add_argument("--output", type=str, default="ucfrep_annotations.json")
    args = p.parse_args()

    data_root = Path(args.data_root)
    ann_dir = data_root / "annotations"
    label_dir = ann_dir / "repetition_label"
    train_dir = ann_dir / "train"
    val_dir = ann_dir / "val"

    for d in (label_dir, train_dir, val_dir):
        if not d.is_dir():
            sys.exit(f"Expected directory not found: {d}")

    # 1) Parse counts from repetition_label/*.txt — keyed by video stem.
    counts = {}
    class_of = {}
    for txt in sorted(label_dir.glob("*.txt")):
        cls = txt.stem
        for name, count in _parse_label_file(txt):
            counts[name] = count
            class_of[name] = cls

    # 2) Determine split membership from train/val .mat filenames.
    train_stems = {p.stem for p in train_dir.glob("*.mat")}
    val_stems = {p.stem for p in val_dir.glob("*.mat")}

    # 3) Build entries; verify each annotated video has a known split.
    entries = []
    missing_split = []
    missing_video = []
    ucf101_root = Path(args.ucf101_videos) if args.ucf101_videos else None

    for name, count in counts.items():
        if name in train_stems:
            split = "train"
        elif name in val_stems:
            split = "test"
        else:
            missing_split.append(name)
            continue

        cls = class_of[name]
        rel = f"{cls}/{name}{args.video_ext}"
        if ucf101_root is not None and not (ucf101_root / rel).is_file():
            missing_video.append(rel)
        entries.append({"video_name": rel, "count": count, "split": split})

    # 4) Write JSON.
    out_path = data_root / args.output
    with out_path.open("w") as f:
        json.dump(entries, f, indent=2)

    # 5) Optionally create the symlink expected by UCFRepDataset.
    if ucf101_root is not None:
        link = data_root / "UCF-101"
        if link.is_symlink() or link.exists():
            print(f"Symlink/dir already exists: {link} (leaving untouched)")
        else:
            link.symlink_to(ucf101_root.resolve())
            print(f"Created symlink: {link} -> {ucf101_root.resolve()}")

    n_train = sum(1 for e in entries if e["split"] == "train")
    n_test = sum(1 for e in entries if e["split"] == "test")
    print(f"Wrote {len(entries)} entries to {out_path}  "
          f"(train={n_train}, test={n_test})")
    if missing_split:
        print(f"WARN: {len(missing_split)} annotated videos had no split membership "
              f"(e.g. {missing_split[:3]})", file=sys.stderr)
    if missing_video:
        print(f"WARN: {len(missing_video)} videos referenced by annotations were "
              f"NOT found at {ucf101_root} (e.g. {missing_video[:3]})", file=sys.stderr)


if __name__ == "__main__":
    main()
