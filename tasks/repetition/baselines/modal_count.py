"""Predict-modal-count baseline for repetition counting.

Always predicts the modal (most frequent) count from the train set on every
sample. This is the chance floor every learned model has to beat.

Counts ``>= n_count_buckets`` are clamped to ``n_count_buckets - 1`` to mirror
``count_loss`` / ``count_from_buckets`` behaviour in ``tasks.repetition.train``.

Reads only the integer count labels (no video decoding) so it returns in
seconds even on Countix.

Usage:
    # Synthetic (cheap sanity check):
    python -m tasks.repetition.baselines.modal_count --dataset synthetic

    # Countix on JZ (decodes nothing, so kinetics_root is only needed to
    # filter records to the videos that exist on disk):
    python -m tasks.repetition.baselines.modal_count \\
        --dataset countix \\
        --data_root  /lustre/fsn1/projects/rech/kcn/ucm72yx/data/countix \\
        --kinetics_root /lustre/fsn1/projects/rech/kcn/ucm72yx/data/kinetics

    # UCFRep / RepCount on JZ:
    python -m tasks.repetition.baselines.modal_count \\
        --dataset ucfrep --data_root /lustre/fsn1/projects/rech/kcn/ucm72yx/data/ucfrep
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

from tasks.repetition.dataset import build_datasets
from tasks.repetition.utils import extract_counts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="synthetic",
                   choices=["synthetic", "countix", "repcount", "ucfrep"])
    p.add_argument("--data_root", type=str, default="data/repetition")
    p.add_argument("--kinetics_root", type=str, default=None,
                   help="For --dataset countix: root of the Kinetics mirror.")
    # Only used so Synthetic / clamping match the trainer exactly.
    p.add_argument("--n_frames", type=int, default=64)
    p.add_argument("--image_size", type=int, default=112)
    p.add_argument("--max_count", type=int, default=16,
                   help="Synthetic only — uniform[1, max_count] labels.")
    p.add_argument("--n_count_buckets", type=int, default=32,
                   help="Counts >= n_count_buckets are clamped, mirroring "
                        "count_loss in the trainer.")
    # Synthetic-only sampler args; build_datasets() requires them for the
    # synthetic backend (variable-length FPS sampling).
    p.add_argument("--target_fps", type=float, default=8.0)
    p.add_argument("--clip_duration_s_min", type=float, default=4.0)
    p.add_argument("--clip_duration_s_max", type=float, default=12.0)
    p.add_argument("--save_json", type=str, default=None,
                   help="Optional path to dump the result dict as JSON.")
    return p.parse_args()


def modal_baseline(
    train_counts: np.ndarray, test_counts: np.ndarray, n_count_buckets: int,
) -> dict:
    """Return modal count + OBO/MAE against train and test targets.

    Args:
        train_counts:    (N_train,) int array of labels.
        test_counts:     (N_test,)  int array of labels.
        n_count_buckets: clamp ceiling — counts >= this are mapped to it - 1.

    Returns a dict::

        {"modal": int,
         "train": {"n": int, "obo": float, "mae": float, "clamped": int},
         "test":  {"n": int, "obo": float, "mae": float, "clamped": int}}
    """
    cap = max(1, n_count_buckets) - 1

    def _clamp(x):
        return np.clip(x, 0, cap)

    train_clamped = _clamp(train_counts)
    test_clamped = _clamp(test_counts) if test_counts.size else test_counts

    if train_clamped.size == 0:
        return {"modal": 0,
                "train": {"n": 0, "obo": 0.0, "mae": 0.0, "clamped": 0},
                "test":  {"n": 0, "obo": 0.0, "mae": 0.0, "clamped": 0}}

    vals, freqs = np.unique(train_clamped, return_counts=True)
    modal = int(vals[freqs.argmax()])

    def _eval(targets: np.ndarray, raw: np.ndarray) -> dict:
        if targets.size == 0:
            return {"n": 0, "obo": 0.0, "mae": 0.0, "clamped": 0}
        diff = np.abs(targets - modal)
        return {
            "n": int(targets.size),
            "obo": float((diff <= 1).mean()),
            "mae": float(diff.mean()),
            "clamped": int((raw >= n_count_buckets).sum()),
        }

    return {
        "modal": modal,
        "train": _eval(train_clamped, train_counts),
        "test":  _eval(test_clamped,  test_counts),
    }


def main() -> int:
    args = parse_args()
    print(f"Dataset       : {args.dataset}")
    print(f"data_root     : {args.data_root}")
    print(f"kinetics_root : {args.kinetics_root}")
    print(f"n_count_buckets : {args.n_count_buckets}")
    print()

    extra = {}
    if args.dataset == "synthetic":
        extra = {"target_fps": args.target_fps,
                 "duration_s_min": args.clip_duration_s_min,
                 "duration_s_max": args.clip_duration_s_max}
    train_data, test_data = build_datasets(
        args.dataset, args.data_root, args.n_frames, args.image_size,
        max_count=args.max_count, kinetics_root=args.kinetics_root, **extra,
    )
    print(f"train: len={len(train_data)}")
    print(f"test : len={len(test_data)}")

    train_counts = extract_counts(train_data)
    test_counts  = extract_counts(test_data)
    result = modal_baseline(train_counts, test_counts, args.n_count_buckets)

    print()
    print(f"Modal count (from train) : {result['modal']}")
    print(f"  train: n={result['train']['n']:>6d}  "
          f"OBO={result['train']['obo']:.3f}  MAE={result['train']['mae']:.3f}  "
          f"clamped={result['train']['clamped']}")
    print(f"  test : n={result['test']['n']:>6d}  "
          f"OBO={result['test']['obo']:.3f}  MAE={result['test']['mae']:.3f}  "
          f"clamped={result['test']['clamped']}")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({
                "args": {k: getattr(args, k) for k in vars(args)},
                "result": result,
            }, f, indent=2)
        print(f"\nSaved to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
