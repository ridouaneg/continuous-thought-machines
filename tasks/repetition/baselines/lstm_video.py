"""LSTM baseline for repetition counting — frame-coupled video LSTM.

Mirrors ContinuousThoughtMachineVideo's frame-tick coupling so the comparison
is apples-to-apples: at internal tick t, the LSTM attends to spatial features
of frame t // iterations_per_frame, and the hidden state carries forward
across ticks (and frames) via standard BPTT.

The output / certainty / mask conventions match
ContinuousThoughtMachineVideo.forward exactly so the existing repetition
trainer (count_loss, evaluate, plotting) consumes LSTM outputs unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from models.modules import (
    LearnableFourierPositionalEncoding,
    MultiLearnableFourierPositionalEncoding,
    CustomRotationalEmbedding,
    CustomRotationalEmbedding1D,
)
from models.resnet import (
    prepare_resnet_backbone,
    prepare_pretrained_resnet_backbone,
)
from models.utils import compute_normalized_entropy


class LSTMBaselineVideoRepCount(nn.Module):
    """Frame-coupled LSTM baseline for repetition counting.

    Parallels ContinuousThoughtMachineVideo: per-frame ResNet features →
    per-tick attention into the current frame's spatial tokens → LSTM step
    → output projection. Hidden state carries forward across all ticks.

    Args mirror the CTM-RepCount construction so a single dispatch in the
    trainer can swap between the two (the LSTM ignores synch / NLM / depth
    knobs that have no analogue here).
    """

    def __init__(
        self,
        n_frames: int,
        iterations_per_frame: int = 1,
        d_model: int = 256,
        d_input: int = 64,
        heads: int = 16,
        num_layers: int = 2,
        backbone_type: str = "resnet18-1",
        positional_embedding_type: str = "none",
        out_dims: int = 32,
        prediction_reshaper=(-1,),
        dropout: float = 0.0,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.iterations_per_frame = iterations_per_frame
        self.d_model = d_model
        self.d_input = d_input
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.positional_embedding_type = positional_embedding_type
        self.out_dims = out_dims
        self.prediction_reshaper = list(prediction_reshaper)
        self.pretrained_backbone = pretrained_backbone
        self.freeze_backbone = freeze_backbone

        d_backbone = self._get_d_backbone()
        self._set_initial_rgb()
        self._set_backbone()
        self.positional_embedding = self._get_positional_embedding(d_backbone)

        self.kv_proj = nn.Sequential(nn.LazyLinear(d_input), nn.LayerNorm(d_input))
        self.q_proj = nn.LazyLinear(d_input)
        self.attention = nn.MultiheadAttention(
            d_input, heads, dropout=dropout, batch_first=True,
        )
        # nn.LSTM disallows dropout when num_layers == 1.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            d_input, d_model, num_layers,
            batch_first=True, dropout=lstm_dropout,
        )
        self.output_projector = nn.Sequential(nn.LazyLinear(out_dims))

        bound = math.sqrt(1.0 / d_model)
        self.register_parameter(
            "start_hidden_state",
            nn.Parameter(torch.empty(num_layers, d_model).uniform_(-bound, bound)),
        )
        self.register_parameter(
            "start_cell_state",
            nn.Parameter(torch.empty(num_layers, d_model).uniform_(-bound, bound)),
        )

    # ---------- backbone / projector setup (parallels CTM) ----------

    def _set_initial_rgb(self):
        if "resnet" in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1)
        else:
            self.initial_rgb = nn.Identity()

    def _get_d_backbone(self) -> int:
        if "resnet" not in self.backbone_type:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")
        family_18_34 = "18" in self.backbone_type or "34" in self.backbone_type
        keep = self.backbone_type.split("-")[1]
        if family_18_34:
            return {"1": 64, "2": 128, "3": 256, "4": 512}[keep]
        return {"1": 256, "2": 512, "3": 1024, "4": 2048}[keep]

    def _set_backbone(self):
        if self.pretrained_backbone:
            self.backbone = prepare_pretrained_resnet_backbone(
                self.backbone_type, freeze=self.freeze_backbone,
            )
        else:
            self.backbone = prepare_resnet_backbone(self.backbone_type)

    def _get_positional_embedding(self, d_backbone):
        t = self.positional_embedding_type
        if t == "learnable-fourier":
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        if t == "multi-learnable-fourier":
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        if t == "custom-rotational":
            return CustomRotationalEmbedding(d_backbone)
        if t == "custom-rotational-1d":
            return CustomRotationalEmbedding1D(d_backbone)
        if t == "none":
            return lambda x: 0
        raise ValueError(f"Invalid positional_embedding_type: {t}")

    # ---------- features / certainty (parallels CTM-Video) ----------

    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode every frame in a single batched backbone call.

        Input  x: (B, T, C, H, W).
        Output kv_all: (B, T, N_tokens, d_input).
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
        self.kv_spatial_shape = (Hp, Wp)
        return kv

    def compute_certainty(self, current_prediction: torch.Tensor) -> torch.Tensor:
        B = current_prediction.size(0)
        reshaped = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped)
        return torch.stack((ne, 1 - ne), dim=-1)

    # ---------- forward (frame-coupled tick loop) ----------

    def forward(self, x: torch.Tensor, frame_mask=None, track: bool = False):
        """Run the frame-coupled LSTM loop.

        Returns the same 4-tuple as ContinuousThoughtMachineVideo.forward:
        (predictions, certainties, sync_out_or_None, tick_mask).
        """
        B, T_in = x.size(0), x.size(1)
        device = x.device
        iterations = T_in * self.iterations_per_frame

        kv_all = self.compute_features(x)  # (B, T, N_tokens, d_input)

        if frame_mask is not None:
            assert frame_mask.shape == (B, T_in)
            T_per_sample = frame_mask.sum(dim=1).clamp_min(1)  # (B,) long
        else:
            T_per_sample = torch.full((B,), T_in, device=device, dtype=torch.long)

        tick_arange = torch.arange(iterations, device=device)
        nominal_frame_per_tick = tick_arange // self.iterations_per_frame
        tick_mask = nominal_frame_per_tick.unsqueeze(0) < T_per_sample.unsqueeze(1)

        batch_idx = torch.arange(B, device=device)

        hn = self.start_hidden_state.unsqueeze(1).expand(-1, B, -1).contiguous()
        cn = self.start_cell_state.unsqueeze(1).expand(-1, B, -1).contiguous()

        predictions = torch.empty(
            B, self.out_dims, iterations, device=device, dtype=torch.float32,
        )
        certainties = torch.empty(
            B, 2, iterations, device=device, dtype=torch.float32,
        )

        activations_tracking, attention_tracking, frame_index_tracking = [], [], []

        for stepi in range(iterations):
            nominal_frame_idx = stepi // self.iterations_per_frame
            if frame_mask is not None:
                frame_indices = torch.minimum(
                    torch.full((B,), nominal_frame_idx, device=device, dtype=torch.long),
                    T_per_sample - 1,
                )
                kv = kv_all[batch_idx, frame_indices]
            else:
                kv = kv_all[:, nominal_frame_idx]

            q = self.q_proj(hn[-1].unsqueeze(1))
            attn_out, attn_weights = self.attention(
                q, kv, kv, average_attn_weights=False, need_weights=True,
            )
            hidden_state, (hn, cn) = self.lstm(attn_out, (hn, cn))
            hidden_state = hidden_state.squeeze(1)

            current_prediction = self.output_projector(hidden_state)
            current_certainty = self.compute_certainty(current_prediction)
            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            if track:
                activations_tracking.append(hidden_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                frame_index_tracking.append(nominal_frame_idx)

        if track:
            return (
                predictions, certainties, None,
                np.array(activations_tracking),
                np.array(attention_tracking),
                np.array(frame_index_tracking),
                self.kv_spatial_shape,
                tick_mask,
            )
        return predictions, certainties, None, tick_mask
