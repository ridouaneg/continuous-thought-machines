import torch

from tasks.tracking.model import ContinuousThoughtMachineTracking


def prepare_model(args, device):
    """Build and return the on-the-fly tracking CTM.

    The CTM attends to one frame per internal tick:
        frame_idx = stepi // iterations_per_frame
        total iterations = n_frames * iterations_per_frame
    At each tick the model emits a prediction over (N objects × 2 axes ×
    n_bins) for the *current* frame.

    out_dims            = n_objects * 2 * n_bins
    prediction_reshaper = [n_objects * 2, n_bins]
    """
    n_per_frame = args.n_objects * 2

    model = ContinuousThoughtMachineTracking(
        n_frames=args.n_frames,
        iterations_per_frame=args.iterations_per_frame,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_normalisation,
        backbone_type=args.backbone_type,
        positional_embedding_type=args.positional_embedding_type,
        out_dims=n_per_frame * args.n_bins,
        prediction_reshaper=[n_per_frame, args.n_bins],
        dropout=args.dropout,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    return model


def decode_predictions(predictions, where_certain_per_frame, n_objects, n_bins):
    """Decode per-frame predictions at each frame's most-certain tick.

    predictions             : (B, N*2, n_bins, iterations)
    where_certain_per_frame : (B, T) absolute tick index per frame

    Returns (B, T, N, 2) predicted bin indices.
    """
    B, n_per_frame, _, _ = predictions.shape
    T = where_certain_per_frame.shape[1]
    b_idx = torch.arange(B, device=predictions.device).unsqueeze(1).expand(-1, T)
    # → (B, T, N*2, K)
    preds = predictions.permute(0, 3, 1, 2)[b_idx, where_certain_per_frame]
    pred_bins = preds.argmax(dim=-1)                                  # (B, T, N*2)
    return pred_bins.reshape(B, T, n_objects, 2)


# Re-export so callers can do: from tasks.tracking.utils import build_datasets
from tasks.tracking.dataset import build_datasets                     # noqa: E402, F401
