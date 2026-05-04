"""Copy only the Kinetics videos referenced by Countix into a smaller mirror.

The full JZ Kinetics mirror at /lustre/fsmisc/dataset/kinetics/ contains
~570k clips and is slow to read on cold-cache. Countix (RepNet) only references
~8k of those IDs, so copying the relevant subset into the project's lustre area
gives faster + more predictable IO during training.

What it does:
  1. Reads countix_{train,val,test}.csv and collects unique youtube_id values.
  2. Indexes the source Kinetics tree by youtube_id (same logic as
     ``tasks.repetition.dataset._build_kinetics_youtube_id_index``).
  3. For each needed id that exists on disk, copies the source file to the
     destination preserving its relative subdirectory (so the destination is
     usable directly as ``--kinetics_root`` for ``CountixDataset``).

Defaults are JZ-specific. Override with --countix-root / --src / --dst.

Usage:
    # Dry-run first to see what would happen:
    python -m tasks.repetition.scripts.copy_countix_kinetics --dry-run

    # Real copy with 16 parallel workers:
    python -m tasks.repetition.scripts.copy_countix_kinetics --workers 16

    # Use symlinks instead of copying (cheap, but ties dst to src lifetime):
    python -m tasks.repetition.scripts.copy_countix_kinetics --symlink
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple

# --- Defaults (JZ) --------------------------------------------------------- #

DEFAULT_COUNTIX_ROOT = "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/countix"
DEFAULT_SRC          = "/lustre/fsmisc/dataset/kinetics"
DEFAULT_DST          = "/lustre/fsn1/projects/rech/kcn/ucm72yx/data/kinetics"
DEFAULT_SPLITS       = ("train", "val", "test")

# Trailing ``_<6-digit>_<6-digit>`` suffix produced by the Kinetics download
# scripts (e.g. ``-qAC8YY5F1Q_000040_000050.mp4`` → id ``-qAC8YY5F1Q``).
_KINETICS_TIMESPAN_RE = re.compile(r"_(\d{6})_(\d{6})$")


# --- Helpers --------------------------------------------------------------- #

def collect_required_ids(countix_root: str, splits) -> Dict[str, Set[str]]:
    """Return ``{split: set(youtube_id)}`` parsed from countix_<split>.csv."""
    result: Dict[str, Set[str]] = {}
    for split in splits:
        csv_path = os.path.join(countix_root, f"countix_{split}.csv")
        if not os.path.isfile(csv_path):
            print(f"  [WARN] missing {csv_path} — skipping split {split!r}",
                  file=sys.stderr)
            continue
        ids: Set[str] = set()
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                vid = (row.get("video_id") or "").strip()
                if vid:
                    ids.add(vid)
        result[split] = ids
        print(f"  {split}: {len(ids)} unique youtube_ids in {csv_path}")
    return result


def build_src_index(src: str) -> Dict[str, str]:
    """Walk ``src`` and map each Kinetics youtube_id to its absolute path.

    Mirrors ``tasks.repetition.dataset._build_kinetics_youtube_id_index``: prefer
    common per-split subtrees, fall back to a full walk. Train wins over val
    when both contain the same id.
    """
    if not os.path.isdir(src):
        raise FileNotFoundError(f"src does not exist: {src}")

    # Try the standard layouts first; the "fsmisc" mirror on JZ stores
    # Kinetics-700, but we also accept 400 and a flat root.
    candidate_subdirs = [
        "kinetics_700_train", "kinetics_700_val",
        "kinetics_400_train", "kinetics_400_val",
    ]
    candidate_roots = [os.path.join(src, sd) for sd in candidate_subdirs
                       if os.path.isdir(os.path.join(src, sd))]
    if not candidate_roots:
        candidate_roots = [src]

    print(f"  walking {len(candidate_roots)} root(s) under {src}:")
    for r in candidate_roots:
        print(f"    {r}")

    t0 = time.time()
    index: Dict[str, str] = {}
    for root in candidate_roots:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                stem, ext = os.path.splitext(fname)
                if ext.lower() != ".mp4":
                    continue
                m = _KINETICS_TIMESPAN_RE.search(stem)
                yt = stem[: m.start()] if m else stem
                index.setdefault(yt, os.path.join(dirpath, fname))
    print(f"  indexed {len(index):,} youtube_ids in {time.time() - t0:.1f}s")
    return index


def plan_copies(
    needed: Dict[str, Set[str]],
    index: Dict[str, str],
    src: str,
    dst: str,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """Build the list of (yt_id, src_path, dst_path) triples to copy.

    The dst path mirrors the source's relative location under ``src`` so the
    destination is directly usable as the Kinetics root for CountixDataset.
    """
    triples: List[Tuple[str, str, str]] = []
    seen: Set[str] = set()
    stats = {"needed_unique": 0, "found": 0, "missing": 0}

    all_ids: Set[str] = set()
    for ids in needed.values():
        all_ids.update(ids)
    stats["needed_unique"] = len(all_ids)

    for yt in all_ids:
        if yt in seen:
            continue
        seen.add(yt)
        sp = index.get(yt)
        if sp is None:
            stats["missing"] += 1
            continue
        rel = os.path.relpath(sp, start=src)
        dp = os.path.join(dst, rel)
        triples.append((yt, sp, dp))
        stats["found"] += 1
    return triples, stats


def _copy_one(
    src_path: str, dst_path: str, mode: str, overwrite: bool
) -> Tuple[bool, str, str]:
    """Materialise one file. Returns (did_work, status, dst_path)."""
    try:
        if os.path.exists(dst_path) and not overwrite:
            # Treat existing files as success; useful for resume after crash.
            return False, "exists", dst_path
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # Atomic-ish write: stage to <dst>.partial then rename.
        tmp = dst_path + ".partial"
        if os.path.exists(tmp):
            os.remove(tmp)
        if mode == "symlink":
            os.symlink(src_path, tmp)
        else:
            shutil.copyfile(src_path, tmp)
            shutil.copystat(src_path, tmp, follow_symlinks=True)
        os.replace(tmp, dst_path)
        return True, "ok", dst_path
    except Exception as exc:
        return False, f"error: {exc}", dst_path


# --- Entrypoint ------------------------------------------------------------ #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--countix-root", default=DEFAULT_COUNTIX_ROOT,
                   help="Dir containing countix_<split>.csv (default: JZ).")
    p.add_argument("--src", default=DEFAULT_SRC,
                   help="Source Kinetics root to copy from.")
    p.add_argument("--dst", default=DEFAULT_DST,
                   help="Destination root (will be created).")
    p.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                   help="Which Countix splits to include.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel copy workers (IO-bound).")
    p.add_argument("--symlink", action="store_true",
                   help="Create symlinks instead of copying.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-copy even if the destination already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan + first few entries, do nothing.")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap on number of files to materialise (0 = no cap).")
    args = p.parse_args()

    if args.symlink and args.overwrite:
        print("[INFO] --symlink + --overwrite: existing symlinks will be replaced")

    print(f"countix_root : {args.countix_root}")
    print(f"src          : {args.src}")
    print(f"dst          : {args.dst}")
    print(f"splits       : {args.splits}")
    print(f"mode         : {'symlink' if args.symlink else 'copy'}")
    print(f"workers      : {args.workers}")
    print()

    print("Step 1/3 — read Countix CSVs:")
    needed = collect_required_ids(args.countix_root, args.splits)
    if not needed:
        print("[FAIL] no CSVs found — aborting", file=sys.stderr)
        return 1
    print()

    print("Step 2/3 — index source Kinetics tree:")
    index = build_src_index(args.src)
    print()

    print("Step 3/3 — plan + materialise:")
    triples, stats = plan_copies(needed, index, args.src, args.dst)
    print(f"  required (unique) : {stats['needed_unique']}")
    print(f"  found in src      : {stats['found']}")
    print(f"  missing in src    : {stats['missing']}")

    if not triples:
        print("[INFO] nothing to do.")
        return 0

    if args.limit > 0:
        triples = triples[: args.limit]
        print(f"  --limit applied → planning {len(triples)} files")

    if args.dry_run:
        print("\nDry-run preview (first 10):")
        for yt, sp, dp in triples[:10]:
            print(f"  {yt}\n    src={sp}\n    dst={dp}")
        print(f"\n[OK] dry run — would have processed {len(triples)} files")
        return 0

    os.makedirs(args.dst, exist_ok=True)

    mode = "symlink" if args.symlink else "copy"
    n_total = len(triples)
    n_ok = n_skipped = n_failed = 0
    t0 = time.time()
    last_report = t0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(_copy_one, sp, dp, mode, args.overwrite)
            for _, sp, dp in triples
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            did_work, status, dp = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "exists":
                n_skipped += 1
            else:
                n_failed += 1
                print(f"  [FAIL] {dp}: {status}", file=sys.stderr)
            now = time.time()
            if now - last_report >= 5.0 or i == n_total:
                rate = i / max(1e-6, now - t0)
                print(f"  progress: {i}/{n_total}  "
                      f"ok={n_ok} skipped={n_skipped} failed={n_failed}  "
                      f"({rate:.1f} files/s)")
                last_report = now

    print()
    print(f"Done in {time.time() - t0:.1f}s.")
    print(f"  copied/symlinked : {n_ok}")
    print(f"  already present  : {n_skipped}")
    print(f"  failed           : {n_failed}")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
