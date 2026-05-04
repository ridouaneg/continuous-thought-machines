"""Repetition-counting losses (local to tasks/repetition).

Two head types share the same (B, n_buckets, T) prediction tensor:

  - 'classification' (default): softmax over buckets, categorical CE.
    See ``count_loss``. Keeps the existing CTM certainty mechanism
    (entropy of the categorical distribution) intact.

  - 'survival': sigmoid per bucket interpreted as the discrete hazard
    h_k = P(N = k | N >= k). CORN-style masked BCE (Cao et al., 2020).
    See ``count_loss_survival``. Recomputes certainty from the induced
    PMF so the dual-tick selection still has a meaningful signal.

Both losses keep the dual-tick structure (min-loss tick + most-certain
tick) so the rest of the CTM machinery is untouched.
"""

import math

import torch
import torch.nn.functional as F


def count_loss(predictions, certainties, targets, use_most_certain=True, tick_mask=None):
    """Min-CE + most-certain-tick CE over count-bucket logits.

    Args:
        predictions:      (B, n_buckets, T) — count-bucket logits per tick.
        certainties:      (B, 2, T) — [normalised_entropy, 1-normalised_entropy].
        targets:          (B,) — ground-truth integer counts (clamped to
                          [0, n_buckets-1] to handle any out-of-range labels).
        use_most_certain: bool — if False, uses the last *valid* tick instead
                          of the most-certain tick.
        tick_mask:        Optional (B, T) bool. ``True`` = tick reads a real
                          (non-padded) frame. The argmin / argmax that pick
                          ``loss_index_min`` and ``loss_index_cert`` ignore
                          masked-out ticks, so loss never gets credit for
                          predictions made past the end of a clip.

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

    if tick_mask is not None:
        # +inf at padded ticks so argmin avoids them; -inf in certainty so
        # argmax avoids them.
        invalid = ~tick_mask
        losses_for_min = losses.masked_fill(invalid, float("inf"))
        cert_for_argmax = certainties[:, 1].masked_fill(invalid, float("-inf"))
    else:
        losses_for_min = losses
        cert_for_argmax = certainties[:, 1]

    loss_index_min = losses_for_min.argmin(dim=1)                   # (B,)
    loss_index_cert = cert_for_argmax.argmax(dim=-1)                # (B,)
    if not use_most_certain:
        if tick_mask is not None:
            # Last tick where the sample still has a real frame.
            last_valid = tick_mask.long().sum(dim=1) - 1
            loss_index_cert = last_valid.clamp(min=0)
        else:
            loss_index_cert = torch.full_like(loss_index_cert, T - 1)

    batch_idx = torch.arange(B, device=predictions.device)
    loss_min_ce = losses[batch_idx, loss_index_min].mean()
    loss_certain = losses[batch_idx, loss_index_cert].mean()

    return (loss_min_ce + loss_certain) / 2, loss_index_cert


def count_loss_survival(predictions, certainties, targets, use_most_certain=True, tick_mask=None):
    """CORN-style discrete-hazard loss for ordinal count targets.

    Treats each of the K bins as a discrete-time hazard
    ``h_k = sigmoid(predictions[:, k, :]) = P(N = k | N >= k)``. Per sample
    with true count N, bins ``k <= N`` contribute BCE with target
    ``1[k == N]``; bins ``k > N`` are masked out (CORN's conditional masking).

    Args:
        predictions:      (B, K, T) — raw logits, sigmoided into hazards.
        certainties:      (B, 2, T) — ignored. Certainty is recomputed from
                          the induced PMF so it reflects the survival head.
        targets:          (B,) — ground-truth integer counts (clamped to [0, K-1]).
        use_most_certain: bool — if False, uses the last *valid* tick.
        tick_mask:        Optional (B, T) bool — True = tick reads a real
                          (non-padded) frame. Same semantics as ``count_loss``.

    Returns:
        loss (scalar), where_certain (B,) — selected tick index per sample.
    """
    B, K, T = predictions.shape
    targets_clipped = targets.clamp(0, K - 1).long()                 # (B,)

    bin_idx = torch.arange(K, device=predictions.device)             # (K,)
    target_bk = (bin_idx[None, :] == targets_clipped[:, None]).float()   # (B, K)
    mask_bk   = (bin_idx[None, :] <= targets_clipped[:, None]).float()   # (B, K)

    target_bkt = target_bk.unsqueeze(-1).expand(B, K, T)             # (B, K, T)
    mask_bkt   = mask_bk.unsqueeze(-1).expand(B, K, T)               # (B, K, T)

    bce = F.binary_cross_entropy_with_logits(
        predictions, target_bkt, reduction="none",
    )                                                                 # (B, K, T)
    losses = (bce * mask_bkt).sum(dim=1) / mask_bkt.sum(dim=1).clamp_min(1.0)  # (B, T)

    # Recompute certainty from induced PMF (entropy of the survival head's
    # implied categorical distribution over counts).
    with torch.no_grad():
        hazards = torch.sigmoid(predictions)                          # (B, K, T)
        log_one_minus_h = torch.log((1 - hazards).clamp_min(1e-6))
        # log S_k = sum_{j<k} log(1 - h_j); S_0 = 1 by convention.
        log_S = torch.cat([
            torch.zeros_like(log_one_minus_h[:, :1, :]),
            torch.cumsum(log_one_minus_h, dim=1)[:, :-1, :],
        ], dim=1)
        # Absorb the tail at K-1 (force h_{K-1} = 1) so PMF sums to 1.
        h_for_pmf = hazards.clone()
        h_for_pmf[:, -1, :] = 1.0
        log_pmf = torch.log(h_for_pmf.clamp_min(1e-6)) + log_S        # (B, K, T)
        pmf = log_pmf.exp()
        entropy = -(pmf * log_pmf).sum(dim=1) / math.log(K)           # (B, T) ∈ [0, 1]
        cert_for_argmax = 1.0 - entropy                                # (B, T)

    if tick_mask is not None:
        invalid = ~tick_mask
        losses_for_min = losses.masked_fill(invalid, float("inf"))
        cert_for_argmax = cert_for_argmax.masked_fill(invalid, float("-inf"))
    else:
        losses_for_min = losses

    loss_index_min = losses_for_min.argmin(dim=1)                     # (B,)
    loss_index_cert = cert_for_argmax.argmax(dim=-1)                  # (B,)
    if not use_most_certain:
        if tick_mask is not None:
            last_valid = tick_mask.long().sum(dim=1) - 1
            loss_index_cert = last_valid.clamp(min=0)
        else:
            loss_index_cert = torch.full_like(loss_index_cert, T - 1)

    batch_idx = torch.arange(B, device=predictions.device)
    loss_min = losses[batch_idx, loss_index_min].mean()
    loss_certain = losses[batch_idx, loss_index_cert].mean()
    return (loss_min + loss_certain) / 2, loss_index_cert
