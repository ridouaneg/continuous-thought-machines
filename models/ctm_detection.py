import numpy as np
import torch
import torch.nn as nn

from models.ctm import ContinuousThoughtMachine


class ContinuousThoughtMachineDetection(ContinuousThoughtMachine):
    """CTM for object detection via DETR-style set prediction.

    Augments the base CTM with a separate bounding-box regression head that
    consumes the same synchronisation_out vector as the class head at every
    internal tick. Certainty is derived from the class logits exactly as in
    classification tasks: the model is "certain" at the tick where its class
    distribution across all slots has lowest entropy.

    Args:
        n_slots:   Number of object query slots (max detectable objects).
        n_classes: Number of foreground classes (excluding background).
        **kwargs:  Forwarded verbatim to ContinuousThoughtMachine.
                   Do NOT pass out_dims or prediction_reshaper — they are
                   set internally.

    Forward returns (class_preds, box_preds, certainties) where:
        class_preds: (B, n_slots, n_classes+1, T)  raw class logits per tick
        box_preds:   (B, n_slots, 4, T)            (cx,cy,w,h) in [0,1], sigmoided
        certainties: (B, 2, T)                     [norm_entropy, 1-norm_entropy]
    """

    def __init__(self, n_slots: int, n_classes: int, **kwargs):
        super().__init__(
            out_dims=n_slots * (n_classes + 1),
            prediction_reshaper=[n_slots, n_classes + 1],
            **kwargs,
        )
        self.n_slots = n_slots
        self.n_classes = n_classes
        # Sigmoid keeps box coordinates in (0, 1) — avoids gradient explosion
        # for coordinates that wander far from the image.
        self.bbox_projector = nn.Sequential(
            nn.LazyLinear(n_slots * 4),
            nn.Sigmoid(),
        )

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device
        T = self.iterations

        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        kv = self.compute_features(x)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        class_predictions = torch.empty(B, self.n_slots * (self.n_classes + 1), T, device=device)
        box_predictions = torch.empty(B, self.n_slots * 4, T, device=device)
        certainties = torch.empty(B, 2, T, device=device)

        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type='out'
        )

        for stepi in range(T):
            synchronisation_action, decay_alpha_action, decay_beta_action = (
                self.compute_synchronisation(
                    activated_state, decay_alpha_action, decay_beta_action,
                    r_action, synch_type='action',
                )
            )

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(
                q, kv, kv, average_attn_weights=False, need_weights=True
            )
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1)

            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated_state = self.trace_processor(state_trace)

            synchronisation_out, decay_alpha_out, decay_beta_out = (
                self.compute_synchronisation(
                    activated_state, decay_alpha_out, decay_beta_out,
                    r_out, synch_type='out',
                )
            )

            current_class = self.output_projector(synchronisation_out)
            current_boxes = self.bbox_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_class)

            class_predictions[..., stepi] = current_class
            box_predictions[..., stepi] = current_boxes
            certainties[..., stepi] = current_certainty

            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        class_preds = class_predictions.reshape(B, self.n_slots, self.n_classes + 1, T)
        box_preds = box_predictions.reshape(B, self.n_slots, 4, T)

        if track:
            return (
                class_preds, box_preds, certainties,
                (np.array(synch_out_tracking), np.array(synch_action_tracking)),
                np.array(pre_activations_tracking),
                np.array(post_activations_tracking),
                np.array(attention_tracking),
            )
        return class_preds, box_preds, certainties
