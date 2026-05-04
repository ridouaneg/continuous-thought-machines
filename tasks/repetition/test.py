"""Preflight check for tasks/repetition on titan / JZ.

Auto-detects the cluster (lustre vs /geovic) and runs two blocks:

1. Environment — everything the train script needs to even start:
     - Python deps + versions (torch, torchvision, numpy, cv2, matplotlib,
       seaborn, tqdm)
     - Repo-internal modules (tasks.repetition.{model,losses,utils},
       tasks.video.model, utils.{housekeeping,run,schedulers})
     - CUDA availability + device info + a tiny AMP autocast matmul
       (mirrors the --use_amp path used in the JZ scripts)
     - System ffmpeg on PATH
     - Writable log dir, plus the JZ slurm output dir when on JZ

2. Datasets — for each backend (synthetic, countix, repcount, ucfrep):
     - presence of the annotation/CSV/JSON files
     - number of records listed in annotation
     - number of videos actually present on disk
     - whether the Dataset class instantiates and returns a sample of the
       expected shape

Per-dataset roots can be overridden with CLI flags:
    python tasks/repetition/test.py --countix-root /path/to/countix \\
        --kinetics-root /path/to/kinetics --repcount-root ... --ucfrep-root ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
import traceback
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tasks.repetition.dataset import (  # noqa: E402
    CountixDataset,
    RepCountADataset,
    SyntheticOscillatingDots,
    SyntheticOscillatingDotsV2,
    UCFRepDataset,
    _build_kinetics_youtube_id_index,
)


# --------------------------------------------------------------------------- #
# Cluster detection / defaults
# --------------------------------------------------------------------------- #

JZ_ROOTS = {
    "countix":  "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/countix",
    "kinetics": "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/kinetics",
    "repcount": "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/repcount",
    "ucfrep":   "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/ucfrep",
}

TITAN_ROOTS = {
    "countix":  "/geovic/ghermi/data/countix",
    "kinetics": "/geovic/ghermi/data/kinetics",
    "repcount": "/geovic/ghermi/data/repcount",
    "ucfrep":   "/geovic/ghermi/data/ucfrep",
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
# Environment checks
# --------------------------------------------------------------------------- #

PY_DEPS = [
    "torch", "torchvision", "numpy", "cv2",
    "matplotlib", "seaborn", "tqdm",
]

INTERNAL_MODULES = [
    "tasks.repetition.dataset",
    "tasks.repetition.model",
    "tasks.repetition.losses",
    "tasks.repetition.utils",
    "tasks.video.model",
    "utils.housekeeping",
    "utils.run",
    "utils.schedulers",
]


def check_python() -> None:
    _hr("python")
    print(f"  executable : {sys.executable}")
    print(f"  version    : {sys.version.split()[0]}")


def check_python_deps() -> None:
    _hr("python deps")
    import importlib
    for name in PY_DEPS:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            _ok(f"{name:12s} {ver}")
        except Exception as exc:
            _err(f"{name}: import failed ({exc})")


def check_internal_modules() -> None:
    _hr("repo internal modules")
    import importlib
    for name in INTERNAL_MODULES:
        try:
            importlib.import_module(name)
            _ok(name)
        except Exception as exc:
            _err(f"{name}: import failed ({exc})")
            traceback.print_exc()


def check_cuda_and_amp() -> None:
    _hr("cuda + amp")
    try:
        import torch
    except Exception as exc:
        _err(f"torch import failed: {exc}")
        return
    if not torch.cuda.is_available():
        _warn("CUDA not available — train scripts assume a GPU")
        return
    print(f"  cuda runtime : {torch.version.cuda}")
    print(f"  device count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"    [{i}] {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}")
    try:
        device = torch.device("cuda:0")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x = torch.randn(64, 64, device=device)
            y = x @ x
        torch.cuda.synchronize()
        _ok(f"AMP autocast matmul ok (out dtype={y.dtype})")
    except Exception as exc:
        _err(f"AMP autocast matmul failed: {exc}")
        traceback.print_exc()


def check_ffmpeg() -> None:
    _hr("ffmpeg")
    import shutil
    import subprocess
    bin_path = shutil.which("ffmpeg")
    if not bin_path:
        _err("ffmpeg not on PATH (on JZ: 'module load ffmpeg/6.1.1')")
        return
    print(f"  binary : {bin_path}")
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5,
        )
        line = (out.stdout or out.stderr).splitlines()
        _ok(line[0] if line else "ffmpeg ran but produced no output")
    except Exception as exc:
        _err(f"ffmpeg call failed: {exc}")


def check_writable_paths(cluster: str) -> None:
    _hr("writable paths (logs / slurm)")
    paths = [os.path.join(_REPO_ROOT, "logs", "repetition")]
    if cluster == "jz":
        paths.append("/lustre/fsn1/projects/rech/kcn/ucm72yx/slurm/ctm")
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
            probe = os.path.join(p, ".smoketest_write")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            _ok(f"writable: {p}")
        except Exception as exc:
            _err(f"not writable: {p} ({exc})")


# --------------------------------------------------------------------------- #
# Per-dataset checks
# --------------------------------------------------------------------------- #

def check_synthetic() -> None:
    _hr("synthetic (SyntheticOscillatingDots)")
    try:
        ds = SyntheticOscillatingDots(
            n_samples=8, target_fps=8.0, duration_s_min=2.0, duration_s_max=4.0,
            image_size=32, max_count=8, split="train",
        )
        _ok(f"instantiated, len={len(ds)}")
        _try_sample(ds, "synthetic")
    except Exception as exc:
        _err(f"synthetic: {exc}")
        traceback.print_exc()


def check_synthetic_v2() -> None:
    _hr("synthetic-v2 (SyntheticOscillatingDotsV2)")
    try:
        ds = SyntheticOscillatingDotsV2(
            n_samples=8, target_fps=8.0, duration_s_min=2.0, duration_s_max=4.0,
            image_size=32, max_count=8, min_active_s=1.5, max_segments=3,
            noise_std=0.02, split="train",
        )
        _ok(f"instantiated, len={len(ds)}")
        _try_sample(ds, "synthetic-v2")
    except Exception as exc:
        _err(f"synthetic-v2: {exc}")
        traceback.print_exc()


def check_countix(countix_root: Optional[str], kinetics_root: Optional[str]) -> None:
    _hr(f"countix  (root={countix_root}, kinetics_root={kinetics_root})")
    if not countix_root or not os.path.isdir(countix_root):
        _warn("countix root not found — skipping")
        return

    csvs = {
        "train": os.path.join(countix_root, "countix_train.csv"),
        "val":   os.path.join(countix_root, "countix_val.csv"),
        "test":  os.path.join(countix_root, "countix_test.csv"),
    }
    for split, p in csvs.items():
        if not os.path.isfile(p):
            _warn(f"{split} csv missing: {p}")
            continue
        with open(p, newline="") as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
        print(f"  {split} csv: {n_rows} rows  ({p})")

    flat_videos = os.path.join(countix_root, "videos")
    if os.path.isdir(flat_videos):
        n_flat = sum(1 for f in os.listdir(flat_videos) if f.endswith((".mp4", ".webm")))
        print(f"  flat videos dir: {flat_videos} -> {n_flat} files")

    yt_index: dict = {}
    if kinetics_root and os.path.isdir(kinetics_root):
        try:
            t0 = time.time()
            yt_index = _build_kinetics_youtube_id_index(kinetics_root)
            print(f"  kinetics youtube_id index: {len(yt_index)} entries "
                  f"(built in {time.time()-t0:.1f}s)")
        except Exception as exc:
            _err(f"failed to build kinetics index: {exc}")
            traceback.print_exc()
    else:
        _warn("kinetics_root not provided/missing — Countix will only see flat videos")

    # For each split, count how many CSV rows resolve to an existing video.
    for split, p in csvs.items():
        if not os.path.isfile(p) or split == "test":
            continue
        n_listed = 0
        n_resolved = 0
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                n_listed += 1
                vid = (row.get("video_id") or "").strip()
                if not vid:
                    continue
                if vid in yt_index:
                    n_resolved += 1
                elif os.path.isfile(os.path.join(flat_videos, vid + ".mp4")):
                    n_resolved += 1
        print(f"  {split}: {n_resolved}/{n_listed} videos resolvable on disk")

    for split in ("train", "val"):
        try:
            ds = CountixDataset(
                countix_root, split=split, n_frames=16, image_size=64,
                kinetics_root=kinetics_root if kinetics_root and os.path.isdir(kinetics_root) else None,
            )
            _ok(f"CountixDataset(split={split}) len={len(ds)}")
            if len(ds):
                _try_sample(ds, f"countix[{split}]")
        except Exception as exc:
            _err(f"CountixDataset(split={split}): {exc}")
            traceback.print_exc()


def check_repcount(root: Optional[str]) -> None:
    _hr(f"repcount  (root={root})")
    if not root or not os.path.isdir(root):
        _warn("data root not found — skipping")
        return

    ann_dir = os.path.join(root, "annotation")
    videos_dir = os.path.join(root, "videos")
    print(f"  annotation dir = {ann_dir} (exists={os.path.isdir(ann_dir)})")
    print(f"  videos dir     = {videos_dir} (exists={os.path.isdir(videos_dir)})")

    if os.path.isdir(videos_dir):
        n_files = sum(1 for f in os.listdir(videos_dir)
                      if f.lower().endswith((".mp4", ".avi", ".mov")))
        print(f"  videos on disk: {n_files}")

    for split in ("train", "valid", "test"):
        p = os.path.join(ann_dir, f"{split}.csv")
        if not os.path.isfile(p):
            _warn(f"{split}.csv missing: {p}")
            continue
        n_rows = 0
        n_resolved = 0
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                n_rows += 1
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                stem = os.path.splitext(name)[0]
                for ext in (".mp4", ".avi", ".mov"):
                    if os.path.isfile(os.path.join(videos_dir, stem + ext)):
                        n_resolved += 1
                        break
                else:
                    if os.path.isfile(os.path.join(videos_dir, name)):
                        n_resolved += 1
        print(f"  {split} csv: {n_resolved}/{n_rows} videos resolvable")

    for split in ("train", "valid"):
        try:
            ds = RepCountADataset(root, split=split, n_frames=16, image_size=64)
            _ok(f"RepCountADataset(split={split}) len={len(ds)}")
            if len(ds):
                _try_sample(ds, f"repcount[{split}]")
        except Exception as exc:
            _err(f"RepCountADataset(split={split}): {exc}")
            traceback.print_exc()


def check_ucfrep(root: Optional[str]) -> None:
    _hr(f"ucfrep  (root={root})")
    if not root or not os.path.isdir(root):
        _warn("data root not found — skipping")
        return

    ann_path = os.path.join(root, "ucfrep_annotations.json")
    videos_root = os.path.join(root, "UCF-101")
    print(f"  annotation = {ann_path} (exists={os.path.isfile(ann_path)})")
    print(f"  UCF-101 dir = {videos_root} (exists={os.path.isdir(videos_root)})")

    if os.path.isfile(ann_path):
        try:
            with open(ann_path) as f:
                data = json.load(f)
            print(f"  annotation: {len(data)} entries")
            from collections import Counter
            split_counts = Counter(e.get("split", "?") for e in data)
            print(f"  split breakdown: {dict(split_counts)}")
            n_resolved = 0
            for e in data:
                rel = e.get("video_name")
                if rel and os.path.isfile(os.path.join(videos_root, rel)):
                    n_resolved += 1
            print(f"  resolvable: {n_resolved}/{len(data)}")
        except Exception as exc:
            _err(f"failed to read annotation: {exc}")

    for split in ("train", "test"):
        try:
            ds = UCFRepDataset(root, split=split, n_frames=16, image_size=64)
            _ok(f"UCFRepDataset(split={split}) len={len(ds)}")
            if len(ds):
                _try_sample(ds, f"ucfrep[{split}]")
        except Exception as exc:
            _err(f"UCFRepDataset(split={split}): {exc}")
            traceback.print_exc()


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", choices=("auto", "jz", "titan"), default="auto",
                        help="select default data roots (overridden by --*-root flags)")
    parser.add_argument("--only",
                        choices=("synthetic", "synthetic-v2", "countix", "repcount", "ucfrep"),
                        default=None, help="limit to one dataset")
    parser.add_argument("--countix-root",  default=None)
    parser.add_argument("--kinetics-root", default=None,
                        help="root of kinetics mirror (used by countix lookup)")
    parser.add_argument("--repcount-root", default=None)
    parser.add_argument("--ucfrep-root",   default=None)
    args = parser.parse_args()

    cluster = detect_cluster() if args.cluster == "auto" else args.cluster
    print(f"hostname  : {socket.gethostname()}")
    print(f"cluster   : {cluster}")
    print(f"cwd       : {os.getcwd()}")
    print(f"repo root : {_REPO_ROOT}")

    defaults = get_roots(cluster)
    roots = {
        "countix":  args.countix_root  or defaults.get("countix"),
        "kinetics": args.kinetics_root or defaults.get("kinetics"),
        "repcount": args.repcount_root or defaults.get("repcount"),
        "ucfrep":   args.ucfrep_root   or defaults.get("ucfrep"),
    }
    print("data roots:")
    for k, v in roots.items():
        exists = os.path.isdir(v) if v else False
        print(f"  {k:10s} -> {v}  (exists={exists})")

    # Environment preflight — always on, light, runs before dataset checks.
    check_python()
    check_python_deps()
    check_internal_modules()
    check_cuda_and_amp()
    check_ffmpeg()
    check_writable_paths(cluster)

    todo = ([args.only] if args.only
            else ["synthetic", "synthetic-v2", "countix", "repcount", "ucfrep"])
    if "synthetic" in todo:
        check_synthetic()
    if "synthetic-v2" in todo:
        check_synthetic_v2()
    if "countix" in todo:
        check_countix(roots.get("countix"), roots.get("kinetics"))
    if "repcount" in todo:
        check_repcount(roots.get("repcount"))
    if "ucfrep" in todo:
        check_ucfrep(roots.get("ucfrep"))


if __name__ == "__main__":
    main()
