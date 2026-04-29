import torch
import torch.nn as nn


def tracking_loss(predictions, certainties, targets, use_most_certain=True):
    """Loss for discretised coordinate prediction.

    predictions : (B, N*T*2, K, iterations)  logits over K position bins
    certainties : (B, 2, iterations)
    targets     : (B, N*T*2)  bin indices in [0, K-1], or -1 for absent objects

    Absent positions (target == -1) are masked out of the loss so the model
    is not penalised for predicting anything there.

    Follows the standard CTM pattern:
        loss = (min-CE-tick loss + most-certain-tick loss) / 2
    """
    B, n_coords, K, iters = predictions.shape

    # -1 sentinel → valid mask
    valid = (targets >= 0)                                          # (B, N*T*2)

    # Replace -1 with 0 so cross-entropy indexing never sees a negative class.
    safe_targets = targets.clamp(min=0)                             # (B, N*T*2)

    # CrossEntropyLoss: logits (N, C, *), targets (N, *)
    preds_flat = predictions.flatten(0, 1)                          # (B*n_coords, K, iters)
    tgts_flat  = safe_targets.flatten(0, 1)                        # (B*n_coords,)
    tgts_flat  = tgts_flat.unsqueeze(-1).expand(-1, iters)         # (B*n_coords, iters)

    losses = nn.CrossEntropyLoss(reduction='none')(preds_flat, tgts_flat)
    losses = losses.reshape(B, n_coords, iters)                     # (B, N*T*2, iters)

    # Zero out absent positions
    valid_3d = valid.unsqueeze(-1).expand_as(losses).float()        # (B, N*T*2, iters)
    losses   = losses * valid_3d

    # Average over the valid coordinates per sample (avoid /0 with clamp)
    n_valid = valid.float().sum(dim=1, keepdim=True).clamp(min=1)   # (B, 1)
    losses  = losses.sum(dim=1) / n_valid                            # (B, iters)

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:, 1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1

    batch_idx    = torch.arange(B, device=predictions.device)
    loss_min_ce  = losses[batch_idx, loss_index_1].mean()
    loss_certain = losses[batch_idx, loss_index_2].mean()

    return (loss_min_ce + loss_certain) / 2, loss_index_2


def position_mae(predictions, targets, n_bins):
    """Mean absolute error in normalised [0, 1] coordinates.

    predictions : (B, N*T*2, K)  logits at a single tick
    targets     : (B, N*T*2)     bin indices (-1 = absent)
    n_bins      : K

    Returns scalar MAE averaged over valid (B, N, T, 2) positions.
    """
    valid = (targets >= 0)                                          # (B, N*T*2)
    if valid.sum() == 0:
        return 0.0

    pred_bins  = predictions.argmax(dim=-1).float()                 # (B, N*T*2)
    tgt_bins   = targets.float().clamp(min=0)

    pred_coord = (pred_bins + 0.5) / n_bins
    tgt_coord  = (tgt_bins  + 0.5) / n_bins

    diff = (pred_coord - tgt_coord).abs()
    return (diff * valid.float()).sum().item() / valid.float().sum().item()
