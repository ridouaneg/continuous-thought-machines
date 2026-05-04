"""CTM for video action recognition.

Subclasses the base ContinuousThoughtMachine so that the internal tick axis
is coupled to the video frame axis: at internal tick `t`, the attention
query reads from the spatial features of frame `t // iterations_per_frame`.

All recurrent state (pre-activation trace, NLM outputs, leaky sync
accumulators) carries forward across frames naturally. Training is full
BPTT through every tick.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from models.ctm import ContinuousThoughtMachine


class ContinuousThoughtMachineVideo(ContinuousThoughtMachine):
    """CTM that attends to a different frame at each internal tick.

    Iterations are derived per-call from the input clip length, not fixed at
    construction time: ``iterations = T_input * iterations_per_frame``. This
    lets a single model serve clips of different durations (e.g. variable-FPS
    sampling, or true variable-length batches via the
    ``video_count_collate`` mask).

    Args:
        n_frames: Default frame count. Used only as a sentinel for the base
            class's ``iterations`` arg — the actual loop length is read from
            the input on every forward.
        iterations_per_frame: CTM internal ticks spent on each frame
            (total ticks per call = T_input * iterations_per_frame).
        All other args are forwarded to ``ContinuousThoughtMachine``.
    """

    def __init__(self, n_frames, iterations_per_frame=1, **kwargs):
        # Base class wants an ``iterations`` arg up-front; we pass the
        # default here but override per-forward below.
        kwargs.setdefault("iterations", n_frames * iterations_per_frame)
        super().__init__(**kwargs)
        self.n_frames = n_frames
        self.iterations_per_frame = iterations_per_frame

    def compute_features(self, x):
        """Encode every frame with the backbone in a single batched call.

        Args:
            x: Clip tensor of shape (B, T, C, H, W) in [0, 1] after normalisation.

        Returns:
            kv_all: Tensor of shape (B, T, N_tokens, d_input) — per-frame
                key/value features ready for attention.
        """
        B, T, C, H, W = x.shape
        flat = x.reshape(B * T, C, H, W)
        flat = self.initial_rgb(flat)
        feats = self.backbone(flat)
        pos_emb = self.positional_embedding(feats)
        combined = feats + pos_emb
        _, Cp, Hp, Wp = feats.shape
        combined = combined.flatten(2).transpose(1, 2)
        kv = self.kv_proj(combined)
        kv = kv.reshape(B, T, Hp * Wp, self.d_input)
        # Store the per-frame spatial shape so visualisation code can
        # reshape attention back to (H', W').
        self.kv_spatial_shape = (Hp, Wp)
        return kv

    def forward(self, x, frame_mask=None, track=False):
        """Run the frame-coupled CTM loop.

        Args:
            x:          (B, T, C, H, W) clip tensor.
            frame_mask: Optional (B, T) bool tensor. ``True`` = real frame,
                        ``False`` = padding from ``video_count_collate``. When
                        a sample's nominal frame index for tick ``t`` falls on
                        a padded slot, the attention reads from that sample's
                        last *real* frame instead — so attention never sees
                        all-masked keys (no NaN) and the model "lingers" on
                        the latest valid input. The trailing ticks for that
                        sample are flagged invalid in the returned tick_mask
                        so the loss / certainty selector can ignore them.
            track:      If True, returns activation/sync/attention traces.

        Returns:
            predictions: (B, out_dims, iterations).
            certainties: (B, 2, iterations).
            sync_out: latest output-side synchronisation tensor.
            tick_mask: (B, iterations) bool — True at ticks corresponding to
                       real (non-padded) frames. Always returned so callers
                       can filter argmin/argmax over valid ticks only.
        """
        B, T_in = x.size(0), x.size(1)
        device = x.device
        iterations = T_in * self.iterations_per_frame

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        frame_index_tracking = []

        kv_all = self.compute_features(x)  # (B, T_frames, N_tokens, d_input)

        if frame_mask is not None:
            assert frame_mask.shape == (B, T_in), (
                f"frame_mask shape {tuple(frame_mask.shape)} != ({B}, {T_in})"
            )
            T_per_sample = frame_mask.sum(dim=1).clamp_min(1)  # (B,) long
        else:
            T_per_sample = torch.full((B,), T_in, device=device, dtype=torch.long)

        # tick_mask[b, t] = True iff tick t reads a real (non-padded) frame for sample b.
        ticks_per_frame = self.iterations_per_frame
        tick_arange = torch.arange(iterations, device=device)
        nominal_frame_per_tick = tick_arange // ticks_per_frame  # (iterations,)
        tick_mask = nominal_frame_per_tick.unsqueeze(0) < T_per_sample.unsqueeze(1)  # (B, iter)

        batch_idx = torch.arange(B, device=device)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(
            B, self.out_dims, iterations, device=device, dtype=torch.float32
        )
        certainties = torch.empty(
            B, 2, iterations, device=device, dtype=torch.float32
        )

        # Learnable per-pair decays -> leaky integrators for sync.
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        decay_alpha_action, decay_beta_action = None, None
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type="out"
        )

        for stepi in range(iterations):
            nominal_frame_idx = stepi // self.iterations_per_frame
            # Per-sample clamp: shorter clips reuse their last real frame.
            if frame_mask is not None:
                frame_indices = torch.minimum(
                    torch.full((B,), nominal_frame_idx, device=device, dtype=torch.long),
                    T_per_sample - 1,
                )
                kv = kv_all[batch_idx, frame_indices]  # (B, N_tokens, d_input)
            else:
                kv = kv_all[:, nominal_frame_idx]

            synchronisation_action, decay_alpha_action, decay_beta_action = (
                self.compute_synchronisation(
                    activated_state,
                    decay_alpha_action,
                    decay_beta_action,
                    r_action,
                    synch_type="action",
                )
            )

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(
                q, kv, kv, average_attn_weights=False, need_weights=True
            )
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat(
                (state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1
            )
            activated_state = self.trace_processor(state_trace)

            synchronisation_out, decay_alpha_out, decay_beta_out = (
                self.compute_synchronisation(
                    activated_state,
                    decay_alpha_out,
                    decay_beta_out,
                    r_out,
                    synch_type="out",
                )
            )

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            if track:
                pre_activations_tracking.append(
                    state_trace[:, :, -1].detach().cpu().numpy()
                )
                post_activations_tracking.append(
                    activated_state.detach().cpu().numpy()
                )
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(
                    synchronisation_action.detach().cpu().numpy()
                )
                frame_index_tracking.append(nominal_frame_idx)

        if track:
            return (
                predictions,
                certainties,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
                np.array(frame_index_tracking),
                self.kv_spatial_shape,
                tick_mask,
            )
        return predictions, certainties, synchronisation_out, tick_mask
