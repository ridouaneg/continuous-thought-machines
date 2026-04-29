"""CTM with predictive-coding pre-training head.

Subclasses ``ContinuousThoughtMachineVideo`` and:

1. Replaces the random-init backbone with an ImageNet-pretrained ResNet from
   torchvision, truncated at the same depth as the CTM's ``backbone_type`` flag
   (e.g. ``resnet18-2`` -> conv1+bn1+layer1+layer2).
2. Freezes the backbone in both pre-training and fine-tuning. Only the CTM
   core (kv_proj, q_proj, attention, synapses, NLMs, sync params, start
   states) and the new predictor / output heads are trained.
3. Adds a ``predictor_head`` used only at pre-training time. At the last
   internal tick of each frame ``f``, the head reads ``synch_out`` and predicts
   the mean-pooled key-value features of frame ``f+1``. Loss is cosine
   similarity with stop-gradient on the target (BYOL-style asymmetry).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from models.modules import Identity
from tasks.video.model import ContinuousThoughtMachineVideo


# Output channel count after each ResNet block, by architecture.
_RESNET_BLOCK_CHANNELS = {
    "resnet18": [64, 128, 256, 512],
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
}

_RESNET_WEIGHTS = {
    "resnet18": tv_models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": tv_models.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": tv_models.ResNet50_Weights.IMAGENET1K_V1,
}


def build_imagenet_backbone(backbone_type: str) -> Tuple[nn.Sequential, int]:
    """Return (truncated ImageNet-pretrained ResNet, output channels)."""
    arch, scale_str = backbone_type.split("-")
    n_blocks = int(scale_str)
    if arch not in _RESNET_WEIGHTS:
        raise ValueError(
            f"Unsupported backbone {backbone_type}; supported: resnet18-N, resnet34-N, resnet50-N"
        )
    if not 1 <= n_blocks <= 4:
        raise ValueError(f"Invalid backbone scale: {scale_str}")

    full = getattr(tv_models, arch)(weights=_RESNET_WEIGHTS[arch])
    layers = [full.conv1, full.bn1, full.relu, full.maxpool, full.layer1]
    if n_blocks >= 2:
        layers.append(full.layer2)
    if n_blocks >= 3:
        layers.append(full.layer3)
    if n_blocks >= 4:
        layers.append(full.layer4)
    out_channels = _RESNET_BLOCK_CHANNELS[arch][n_blocks - 1]
    return nn.Sequential(*layers), out_channels


class CTMVideoPredictiveCoding(ContinuousThoughtMachineVideo):
    """Video CTM with an ImageNet-pretrained frozen backbone and a predictor head.

    The base ``ContinuousThoughtMachineVideo`` is initialised normally; we then
    swap out the backbone for an ImageNet-pretrained one and freeze it. The
    predictor head is small and only used at pre-training time — discard it
    when fine-tuning.
    """

    def __init__(self, n_frames, iterations_per_frame=1, **kwargs):
        super().__init__(n_frames=n_frames, iterations_per_frame=iterations_per_frame, **kwargs)

        # Replace channel-adapter + backbone with the ImageNet-pretrained one.
        self.initial_rgb = Identity()
        self.backbone, _ = build_imagenet_backbone(self.backbone_type)

        # Predictor head: synch_out -> next-frame mean-pooled feature in d_input space.
        self.predictor_head = nn.Sequential(
            nn.Linear(self.synch_representation_size_out, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_input),
        )

        self._apply_freeze_backbone()

    # --- Backbone freezing -------------------------------------------------
    # Note: the parent ContinuousThoughtMachine stores ``freeze_backbone`` as a
    # bool attribute on ``self``, which would shadow any method named
    # ``freeze_backbone``. We use a different name here.

    def _apply_freeze_backbone(self):
        for p in self.initial_rgb.parameters():
            p.requires_grad_(False)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def encoder_eval(self):
        """Force backbone into eval mode (BatchNorm uses running stats)."""
        self.initial_rgb.eval()
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep BN frozen even when the rest of the model is in train mode.
        self.encoder_eval()
        return self

    # --- Pre-training forward ---------------------------------------------

    def forward_pretrain(self, x):
        """Forward pass returning the per-frame end-of-frame sync_out and the kv features.

        Args:
            x: (B, n_frames, C, H, W) clip tensor.

        Returns:
            sync_outs: (B, n_frames, synch_representation_size_out) — the
                synch_out at the last internal tick of each frame.
            kv_all: (B, n_frames, n_tokens, d_input).
        """
        B = x.size(0)

        kv_all = self.compute_features(x)

        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        decay_alpha_action, decay_beta_action = None, None
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type="out"
        )

        sync_outs_per_frame = []
        for stepi in range(self.iterations):
            frame_idx = stepi // self.iterations_per_frame
            kv = kv_all[:, frame_idx]

            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action,
                r_action, synch_type="action"
            )
            q = self.q_proj(sync_action).unsqueeze(1)
            attn_out, _ = self.attention(
                q, kv, kv, average_attn_weights=False, need_weights=True
            )
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.cat((attn_out, activated_state), dim=-1)
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat(
                (state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1
            )
            activated_state = self.trace_processor(state_trace)
            sync_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out,
                r_out, synch_type="out"
            )

            # End-of-frame tick: collect sync_out as the source for next-frame prediction.
            if (stepi + 1) % self.iterations_per_frame == 0:
                sync_outs_per_frame.append(sync_out)

        sync_outs = torch.stack(sync_outs_per_frame, dim=1)
        return sync_outs, kv_all

    def predictive_coding_loss(self, x):
        """Cosine-similarity loss between predictor(sync_out_f) and mean-pooled kv_{f+1}.

        Stop-gradient on the target prevents collapse (BYOL-style asymmetry).
        """
        sync_outs, kv_all = self.forward_pretrain(x)
        target = kv_all[:, 1:].mean(dim=2).detach()        # (B, n_frames-1, d_input)
        pred = self.predictor_head(sync_outs[:, :-1])       # (B, n_frames-1, d_input)

        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        cos = (pred_n * target_n).sum(dim=-1)               # (B, n_frames-1)
        loss = (1 - cos).mean()
        return loss, cos.detach()

    # --- Checkpoint helpers -----------------------------------------------

    def core_state_dict(self):
        """State dict with backbone keys removed.

        The backbone is always reloaded from torchvision, so we don't need to
        save its weights. The predictor_head IS saved so that pre-trained
        checkpoints are self-describing — drop it on load if not needed.
        """
        sd = self.state_dict()
        return {k: v for k, v in sd.items() if not (k.startswith("backbone.") or k.startswith("initial_rgb."))}

    def load_core_state_dict(self, state_dict, drop_predictor_head=True,
                              drop_output_projector=True):
        """Load a previously saved core state dict. Backbone is left as-is (frozen ImageNet).

        ``drop_output_projector`` defaults to True because the number of classes
        almost always differs between pre-training (placeholder) and the
        downstream task — we want a fresh head.
        """
        if drop_predictor_head:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("predictor_head.")}
        if drop_output_projector:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("output_projector.")}
        return self.load_state_dict(state_dict, strict=False)
