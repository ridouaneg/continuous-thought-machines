"""Repetition-counting loss (local to tasks/repetition; not added to utils/losses.py).

Uses bucket classification over discrete counts so the existing CTM certainty
mechanism (entropy-based) stays intact. The loss is structurally identical to
image_classification_loss: min-CE over all ticks + CE at the most-certain tick.

Framing counts as classification rather than regression avoids redesigning
the certainty head and lets the model produce a full distribution over counts
at each tick, enabling soft expected-count decoding for MAE evaluation.
"""

import torch
import torch.nn.functional as F


def count_loss(predictions, certainties, targets, use_most_certain=True):
    """Min-CE + most-certain-tick CE over count-bucket logits.

    Args:
        predictions:      (B, n_buckets, T) — count-bucket logits per tick.
        certainties:      (B, 2, T) — [normalised_entropy, 1-normalised_entropy].
        targets:          (B,) — ground-truth integer counts (will be clamped to
                          [0, n_buckets-1] to handle any out-of-range labels).
        use_most_certain: bool — if False, uses the final tick instead of the
                          most-certain tick (mirrors the API of image_classification_loss).

    Returns:
        loss (scalar), where_certain (B,) — selected tick index per sample.
    """
    B, C, T = predictions.shape
    targets_clipped = targets.clamp(0, C - 1).long()               # (B,)
    targets_exp = targets_clipped.unsqueeze(-1).expand(B, T)        # (B, T)

    # Cross-entropy at every tick: reshape to (B*T, C) for F.cross_entropy.
    losses = F.cross_entropy(
        predictions.permute(0, 2, 1).reshape(B * T, C),
        targets_exp.reshape(B * T),
        reduction="none",
    ).reshape(B, T)                                                 # (B, T)

    loss_index_min = losses.argmin(dim=1)                           # (B,)
    loss_index_cert = certainties[:, 1].argmax(dim=-1)              # (B,)
    if not use_most_certain:
        loss_index_cert = torch.full_like(loss_index_cert, T - 1)   # last tick

    batch_idx = torch.arange(B, device=predictions.device)
    loss_min_ce = losses[batch_idx, loss_index_min].mean()
    loss_certain = losses[batch_idx, loss_index_cert].mean()

    return (loss_min_ce + loss_certain) / 2, loss_index_cert
