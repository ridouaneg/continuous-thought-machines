"""Task-specific evaluation helpers for repetition counting."""

from typing import Iterable

import numpy as np
import torch


def extract_counts(ds, max_iter: int = 4096) -> np.ndarray:
    """Return a numpy array of integer counts for every sample in ``ds``.

    Real datasets (Countix / RepCount / UCFRep) expose ``.records`` with a
    ``'count'`` key — we use that directly so we don't decode any videos.
    For the synthetic dataset (deterministic + cheap) we iterate up to
    ``max_iter`` samples.

    Used by both training (for dataset-stat logging) and the baselines
    in ``tasks/repetition/baselines/``.
    """
    if hasattr(ds, "records") and ds.records and "count" in ds.records[0]:
        return np.asarray([int(r["count"]) for r in ds.records], dtype=np.int64)
    n = min(len(ds), max_iter)
    return np.asarray([int(ds[i][1]) for i in range(n)], dtype=np.int64)


def count_from_buckets(logits):
    """Convert count-bucket logits to predicted counts.

    Args:
        logits: (B, n_buckets) tensor of raw logits.

    Returns:
        expected: (B,) float tensor — soft expected count E[count] = sum(i * p_i).
        argmax:   (B,) long tensor  — hard argmax count.
    """
    probs = torch.softmax(logits, dim=-1)
    buckets = torch.arange(logits.size(-1), device=logits.device, dtype=torch.float32)
    expected = (probs * buckets).sum(dim=-1)
    argmax = probs.argmax(dim=-1)
    return expected, argmax


def obo_accuracy(pred_counts, gt_counts):
    """Off-by-one accuracy: fraction of samples where |pred - gt| <= 1.

    Args:
        pred_counts: (B,) int or float tensor of predicted counts.
        gt_counts:   (B,) int or float tensor of ground-truth counts.

    Returns:
        Scalar float in [0, 1].
    """
    return ((pred_counts.float() - gt_counts.float()).abs() <= 1).float().mean().item()


def mae(pred_counts, gt_counts):
    """Mean absolute error between predicted and ground-truth counts.

    Args:
        pred_counts: (B,) tensor.
        gt_counts:   (B,) tensor.

    Returns:
        Scalar float.
    """
    return (pred_counts.float() - gt_counts.float()).abs().mean().item()


def count_from_hazards(logits):
    """Convert hazard logits to predicted counts (CORN survival head).

    Args:
        logits: (..., K) tensor of raw logits. Sigmoided into hazards
                ``h_k = P(N = k | N >= k)``.

    Returns:
        expected: (...) float tensor — soft expected count Σ k · p_k where
                  p_k is the induced PMF (tail absorbed at K-1 so it sums to 1).
        argmax:   (...) long tensor  — first k with sigmoid(logits[k]) > 0.5
                  (rank-consistent CORN decoding); falls back to K-1 if no
                  hazard fires.

    Same return signature as ``count_from_buckets`` so callers can swap them.
    """
    hazards = torch.sigmoid(logits)
    K = hazards.size(-1)

    log_one_minus_h = torch.log((1 - hazards).clamp_min(1e-6))
    # log S_k = sum_{j<k} log(1 - h_j); S_0 = 1.
    log_S = torch.cat([
        torch.zeros_like(log_one_minus_h[..., :1]),
        torch.cumsum(log_one_minus_h, dim=-1)[..., :-1],
    ], dim=-1)
    # Absorb the tail at K-1 so the PMF sums to 1 by construction.
    h_for_pmf = hazards.clone()
    h_for_pmf[..., -1] = 1.0
    log_pmf = torch.log(h_for_pmf.clamp_min(1e-6)) + log_S
    pmf = log_pmf.exp()

    bins = torch.arange(K, device=logits.device, dtype=torch.float32)
    expected = (pmf * bins).sum(dim=-1)

    fired = hazards > 0.5
    first_fired = fired.long().argmax(dim=-1)                        # 0 if none fired
    argmax = torch.where(
        fired.any(dim=-1), first_fired,
        torch.full_like(first_fired, K - 1),
    )
    return expected, argmax
