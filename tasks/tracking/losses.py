import torch
import torch.nn as nn


def tracking_loss(
    predictions, certainties, targets, iterations_per_frame, use_most_certain=True
):
    """Per-frame loss for the on-the-fly tracking CTM.

    Each internal tick ``t`` predicts the positions for frame
    ``f = t // iterations_per_frame``. Within each frame's block of
    ``iterations_per_frame`` ticks the standard CTM loss is applied:
    average of (min-CE-tick loss, most-certain-tick loss). The final loss
    is the mean over frames.

    predictions : (B, N*2, K, iters)  logits over K position bins,
                  ordering matches targets reshape below
    certainties : (B, 2, iters)
    targets     : (B, T, N, 2)  bin indices in [0, K-1], -1 for absent
    iterations_per_frame : ipf, with iters = T * ipf

    Returns
    -------
    loss : scalar
    where_certain_per_frame : (B, T) tick index of the most-certain tick
        chosen within each frame's block (absolute, not relative to block).
    """
    B, n_per_frame, K, iters = predictions.shape
    T = targets.shape[1]
    ipf = iterations_per_frame
    assert iters == T * ipf, f"iters={iters} but T*ipf={T * ipf}"

    # (B, T, N, 2) -> (B, T, N*2) keeping (N, axis) ordering.
    targets_per_frame = targets.reshape(B, T, n_per_frame)            # (B, T, n_per_frame)
    valid_per_frame = (targets_per_frame >= 0)                        # (B, T, n_per_frame)

    # Repeat each frame's target across its ipf ticks → (B, iters, n_per_frame).
    targets_per_tick = targets_per_frame.repeat_interleave(ipf, dim=1)  # (B, iters, n_per_frame)
    valid_per_tick   = valid_per_frame.repeat_interleave(ipf, dim=1)    # (B, iters, n_per_frame)

    # Per-(sample, position, tick) cross-entropy.
    safe_targets = targets_per_tick.clamp(min=0)                      # (B, iters, n_per_frame)
    # CE expects logits (N, C, *) and targets (N, *). Build with shape
    # (B*n_per_frame, K, iters) vs (B*n_per_frame, iters).
    preds_flat = predictions.flatten(0, 1)                            # (B*n_per_frame, K, iters)
    tgts_flat  = safe_targets.transpose(1, 2).reshape(-1, iters)      # (B*n_per_frame, iters)

    losses = nn.CrossEntropyLoss(reduction="none")(preds_flat, tgts_flat)
    losses = losses.reshape(B, n_per_frame, iters)                    # (B, n_per_frame, iters)

    # Mask out absent positions, then average over valid positions per (sample, tick).
    valid_3d = valid_per_tick.transpose(1, 2).float()                 # (B, n_per_frame, iters)
    losses   = losses * valid_3d
    n_valid  = valid_3d.sum(dim=1).clamp(min=1)                       # (B, iters)
    losses_per_tick = losses.sum(dim=1) / n_valid                     # (B, iters)

    # Group ticks by frame: (B, T, ipf).
    losses_blocked = losses_per_tick.reshape(B, T, ipf)
    cert_blocked   = certainties[:, 1].reshape(B, T, ipf)             # (B, T, ipf)

    # Per-frame min-CE and most-certain ticks within each block.
    rel_min  = losses_blocked.argmin(dim=2)                           # (B, T)
    rel_cert = cert_blocked.argmax(dim=2)                             # (B, T)
    if not use_most_certain:
        rel_cert = torch.full_like(rel_cert, ipf - 1)

    abs_min  = torch.arange(T, device=predictions.device) * ipf + rel_min      # (B, T)
    abs_cert = torch.arange(T, device=predictions.device) * ipf + rel_cert     # (B, T)

    b_idx = torch.arange(B, device=predictions.device).unsqueeze(1).expand(-1, T)
    f_idx = torch.arange(T, device=predictions.device).unsqueeze(0).expand(B, -1)
    loss_min  = losses_blocked[b_idx, f_idx, rel_min]                 # (B, T)
    loss_cert = losses_blocked[b_idx, f_idx, rel_cert]                # (B, T)

    per_frame = (loss_min + loss_cert) / 2                            # (B, T)

    # Skip frames whose targets are entirely absent (no valid positions).
    frame_has_valid = valid_per_frame.any(dim=2).float()              # (B, T)
    denom = frame_has_valid.sum().clamp(min=1)
    loss = (per_frame * frame_has_valid).sum() / denom

    return loss, abs_cert


def position_mae(predictions, targets, n_bins, where_certain_per_frame):
    """Mean absolute error in normalised [0, 1] coordinates.

    predictions             : (B, N*2, K, iters)
    targets                 : (B, T, N, 2) bin indices (-1 = absent)
    where_certain_per_frame : (B, T) absolute tick index per frame

    Returns scalar MAE averaged over valid (B, T, N, 2) positions.
    """
    B, n_per_frame, K, iters = predictions.shape
    T = targets.shape[1]

    b_idx = torch.arange(B, device=predictions.device).unsqueeze(1).expand(-1, T)
    # predictions[b, :, :, where] -> (B, T, n_per_frame, K)
    preds_at_tick = predictions.permute(0, 3, 1, 2)[b_idx, where_certain_per_frame]
    pred_bins = preds_at_tick.argmax(dim=-1).float()                  # (B, T, n_per_frame)

    targets_pf = targets.reshape(B, T, n_per_frame)                   # (B, T, n_per_frame)
    valid = (targets_pf >= 0)
    if valid.sum() == 0:
        return 0.0

    tgt_bins = targets_pf.float().clamp(min=0)
    pred_coord = (pred_bins + 0.5) / n_bins
    tgt_coord  = (tgt_bins  + 0.5) / n_bins
    diff = (pred_coord - tgt_coord).abs()
    return (diff * valid.float()).sum().item() / valid.float().sum().item()
