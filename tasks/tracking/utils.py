import torch

from models.ctm import ContinuousThoughtMachine
from tasks.tracking.model import FrameEncoder, TrackingCTM


def prepare_model(args, device):
    """Build and return the full TrackingCTM model (frame encoder + CTM).

    The CTM receives a T-length sequence of frame tokens and outputs
    discretised (x, y) position bins for each (object, frame) pair.

    out_dims            = n_objects * n_frames * 2 * n_bins
    prediction_reshaper = [n_objects * n_frames * 2, n_bins]
    """
    n_coords = args.n_objects * args.n_frames * 2

    in_channels  = getattr(args, 'in_channels',  3)
    encoder_type = getattr(args, 'encoder_type', 'resnet18')

    frame_encoder = FrameEncoder(
        in_channels=in_channels,
        img_size=args.img_size,
        d_feat=args.d_feat,
        n_frames=args.n_frames,
        encoder_type=encoder_type,
    )

    ctm = ContinuousThoughtMachine(
        iterations=args.iterations,
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
        backbone_type='none',
        positional_embedding_type='none',
        out_dims=n_coords * args.n_bins,
        prediction_reshaper=[n_coords, args.n_bins],
        dropout=args.dropout,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
    )

    model = TrackingCTM(frame_encoder, ctm).to(device)
    return model


def decode_predictions(predictions, n_objects, n_frames, n_bins, where_certain):
    """Decode predictions at the most-certain tick.

    predictions   : (B, N*T*2, n_bins, iterations)
    where_certain : (B,) tick indices

    Returns (B, T, N, 2) predicted bin indices.
    """
    B = predictions.shape[0]
    batch_idx     = torch.arange(B, device=predictions.device)
    preds_at_tick = predictions[batch_idx, :, :, where_certain]    # (B, N*T*2, n_bins)
    pred_bins     = preds_at_tick.argmax(-1)                       # (B, N*T*2)
    return pred_bins.reshape(B, n_objects, n_frames, 2).permute(0, 2, 1, 3)  # (B, T, N, 2)


# Re-export so callers can do: from tasks.tracking.utils import build_datasets
from tasks.tracking.dataset import build_datasets                   # noqa: E402, F401
