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

    Args:
        n_frames: Number of frames sampled per clip.
        iterations_per_frame: CTM internal ticks spent on each frame
            (total iterations = n_frames * iterations_per_frame).
        All other args are forwarded to ``ContinuousThoughtMachine``.
    """

    def __init__(self, n_frames, iterations_per_frame=1, **kwargs):
        kwargs["iterations"] = n_frames * iterations_per_frame
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

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        frame_index_tracking = []

        kv_all = self.compute_features(x)  # (B, T_frames, N_tokens, d_input)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(
            B, self.out_dims, self.iterations, device=device, dtype=torch.float32
        )
        certainties = torch.empty(
            B, 2, self.iterations, device=device, dtype=torch.float32
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

        for stepi in range(self.iterations):
            frame_idx = stepi // self.iterations_per_frame
            kv = kv_all[:, frame_idx]  # (B, N_tokens, d_input)

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
                frame_index_tracking.append(frame_idx)

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
            )
        return predictions, certainties, synchronisation_out
