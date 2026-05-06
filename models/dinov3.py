"""DINOv3 backbone wrapper for the CTM.

Exposes ImageNet-aligned (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
DINOv3 ViTs as drop-in CTM backbones that return per-patch spatial feature
maps shaped ``(B, D, H', W')`` — matching the ResNet contract used by
``compute_features`` (see ``models/ctm.py``).

DINOv3 weights are gated; Meta requires accepting a license before
downloading. To use this backbone on a JZ compute node:

  1. On a host with internet access, ``git clone`` facebookresearch/dinov3 to
     ``$LUSTRE/code/dinov3``, run ``torch.hub.load(...)`` once with the
     appropriate ``DINOV3_TOKEN`` (or download weights manually), and copy
     the resulting weight file (e.g. ``dinov3_vits16_pretrain_lvd1689m.pth``)
     to ``$LUSTRE/checkpoints/dinov3/``.
  2. Export ``DINOV3_REPO_PATH=$LUSTRE/code/dinov3`` and
     ``DINOV3_WEIGHTS_PATH=$LUSTRE/checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m.pth``
     before invoking ``train.py`` — these env vars are picked up here.

The class accepts only square inputs whose side is a multiple of the patch
size; with ``image_size=112`` and ``patch_size=16`` this gives a 7×7 spatial
grid (49 tokens per frame), comfortably under typical attention budgets.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


DINOV3_VARIANTS = {
    # Smallest DINOv3 ViT — the "tiny" of the family in everyday usage.
    'dinov3-vits16': {
        'hub_entry': 'dinov3_vits16',
        'embed_dim': 384,
        'patch_size': 16,
    },
    'dinov3-vitb16': {
        'hub_entry': 'dinov3_vitb16',
        'embed_dim': 768,
        'patch_size': 16,
    },
    'dinov3-vitl16': {
        'hub_entry': 'dinov3_vitl16',
        'embed_dim': 1024,
        'patch_size': 16,
    },
}


def is_dinov3_backbone(backbone_type: str) -> bool:
    return backbone_type in DINOV3_VARIANTS


def get_dinov3_embed_dim(backbone_type: str) -> int:
    return DINOV3_VARIANTS[backbone_type]['embed_dim']


class _DinoV3Spatial(nn.Module):
    """Adapter that exposes a DINOv3 ViT as a (B, D, H', W') spatial encoder.

    Patch tokens come back from ``forward_features`` as ``(B, N_patches, D)``;
    we transpose+reshape them onto the input grid. The CLS / register tokens
    are dropped — the CTM's own attention reads from the patch grid.
    """

    def __init__(self, vit: nn.Module, patch_size: int, freeze: bool = True):
        super().__init__()
        self.vit = vit
        self.patch_size = patch_size
        self.frozen = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(
                f"DINOv3 ViT requires input H,W to be multiples of patch_size "
                f"{self.patch_size}; got {H}x{W}."
            )
        Hp, Wp = H // self.patch_size, W // self.patch_size

        out = self.vit.forward_features(x)
        if isinstance(out, dict):
            # Standard DINOv2/3 contract.
            tokens = out.get('x_norm_patchtokens', out.get('x_patchtokens'))
            if tokens is None:
                raise RuntimeError(
                    "DINOv3 forward_features returned a dict without "
                    "'x_norm_patchtokens'/'x_patchtokens' keys: "
                    f"{list(out.keys())}"
                )
        else:
            # Some forks return tokens directly; assume CLS at position 0.
            tokens = out[:, 1:] if out.dim() == 3 else out

        N = Hp * Wp
        if tokens.shape[1] != N:
            # Drop any extra register/CLS tokens that come bundled in.
            tokens = tokens[:, -N:]
        D = tokens.shape[-1]
        feats = tokens.transpose(1, 2).reshape(B, D, Hp, Wp).contiguous()
        return feats


def prepare_dinov3_backbone(
    backbone_type: str,
    weights_path: str | None = None,
    repo_path: str | None = None,
    freeze: bool = True,
) -> nn.Module:
    """Build a DINOv3 ViT and wrap it as a CTM-compatible spatial encoder.

    Args:
        backbone_type: Key into ``DINOV3_VARIANTS`` (e.g. ``"dinov3-vits16"``).
        weights_path: Optional local checkpoint. If ``None``, falls back to
            the ``DINOV3_WEIGHTS_PATH`` env var. If still ``None``, the hub
            entry is loaded with ``pretrained=True`` (which requires Meta
            credentials).
        repo_path: Optional local clone of facebookresearch/dinov3. If
            ``None``, falls back to the ``DINOV3_REPO_PATH`` env var. If
            still ``None``, ``torch.hub.load`` will fetch the source from
            GitHub at runtime.
        freeze: If True, parameters are frozen and the ViT is locked to
            ``eval`` mode (so dropout / no BN-running-stats drift).
    """
    if backbone_type not in DINOV3_VARIANTS:
        raise ValueError(
            f"Unknown DINOv3 variant {backbone_type!r}. "
            f"Choices: {list(DINOV3_VARIANTS)}"
        )
    cfg = DINOV3_VARIANTS[backbone_type]

    repo_path = repo_path if repo_path is not None else os.environ.get('DINOV3_REPO_PATH')
    weights_path = weights_path if weights_path is not None else os.environ.get('DINOV3_WEIGHTS_PATH')

    pretrained_via_hub = weights_path is None
    if repo_path is not None:
        vit = torch.hub.load(
            repo_path, cfg['hub_entry'],
            source='local', pretrained=pretrained_via_hub,
        )
    else:
        vit = torch.hub.load(
            'facebookresearch/dinov3', cfg['hub_entry'],
            pretrained=pretrained_via_hub,
        )

    if weights_path is not None:
        sd = torch.load(weights_path, map_location='cpu')
        if isinstance(sd, dict) and 'model' in sd and not any(
            k.startswith(('blocks.', 'patch_embed.', 'norm.')) for k in sd
        ):
            sd = sd['model']
        missing, unexpected = vit.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"[dinov3] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if len(unexpected) > 0:
            print(f"[dinov3] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    return _DinoV3Spatial(vit, patch_size=cfg['patch_size'], freeze=freeze)
