"""Task-specific evaluation helpers for repetition counting."""

import torch


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
