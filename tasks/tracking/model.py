from typing import Optional

import torch
import torch.nn as nn

from models.ctm import ContinuousThoughtMachine


class FrameEncoder(nn.Module):
    """
    Encodes each video frame independently, then adds learned temporal
    positional embeddings so the CTM can distinguish frame order.

    Input  : (B*T, C, H, W)
    Output : (B, d_feat, T)  — transposed so the CTM backbone='none' pipeline
             sees (B, d_feat, T) → flatten(2).transpose(1,2) → (B, T, d_feat)
             → kv_proj → kv.

    encoder_type options
    --------------------
    'tiny'
        3-layer Conv + BN + ReLU, AdaptiveAvgPool → 64-d → Linear.
        Suitable for small synthetic frames (1 channel, 32 px).
    'medium'
        5-layer Conv + BN + ReLU, AdaptiveAvgPool → 128-d → Linear.
        A reasonable choice for mid-size real images (3 channels, ≤128 px).
    'resnet18'
        Pretrained ResNet-18 up to layer2 (stride-8, 128 channels), then
        AdaptiveAvgPool → 128-d → Linear. The pretrained ResNet stack is
        fully frozen with BatchNorm locked to eval mode; only the
        Linear(128, d_feat) projection on top is trainable. Requires
        torchvision.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        d_feat: int,
        n_frames: int,
        encoder_type: str = 'resnet18',
    ):
        super().__init__()
        self.n_frames     = n_frames
        self.d_feat       = d_feat
        self.encoder_type = encoder_type
        self._frozen_resnet_stack: Optional[nn.Module] = None

        if encoder_type == 'resnet18':
            import torchvision.models as tvm
            resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
            resnet_stack = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2,
            )
            for p in resnet_stack.parameters():
                p.requires_grad_(False)
            resnet_stack.eval()
            self._frozen_resnet_stack = resnet_stack
            self.cnn = nn.Sequential(
                resnet_stack,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, d_feat),
                nn.ReLU(),
            )

        elif encoder_type == 'medium':
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, d_feat),
                nn.ReLU(),
            )

        else:  # 'tiny' — default for synthetic
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, d_feat),
                nn.ReLU(),
            )

        # Learned temporal positional embedding: one vector per frame slot
        self.temporal_emb = nn.Embedding(n_frames, d_feat)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen_resnet_stack is not None:
            self._frozen_resnet_stack.eval()
        return self

    def forward(self, flat_frames: torch.Tensor) -> torch.Tensor:
        """
        flat_frames : (B*T, C, H, W)
        returns     : (B, d_feat, T)
        """
        B_T = flat_frames.shape[0]
        T   = self.n_frames
        B   = B_T // T

        feats = self.cnn(flat_frames)                               # (B*T, d_feat)
        feats = feats.view(B, T, self.d_feat)                      # (B, T, d_feat)
        t_idx = torch.arange(T, device=flat_frames.device)
        feats = feats + self.temporal_emb(t_idx)                   # broadcast over B
        return feats.transpose(1, 2)                               # (B, d_feat, T)


class TrackingCTM(nn.Module):
    """
    End-to-end tracking model: frame encoder + CTM.

    The CTM receives T frame tokens (one per frame) as a sequence via
    cross-attention and iteratively refines object tracks across its internal
    thought ticks.
    """

    def __init__(self, frame_encoder: FrameEncoder, ctm: ContinuousThoughtMachine):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.ctm           = ctm

    def forward(self, frames: torch.Tensor, track: bool = False):
        """
        frames : (B, T, C, H, W)
        """
        B, T, C, H, W = frames.shape
        flat_frames    = frames.reshape(B * T, C, H, W)
        kv             = self.frame_encoder(flat_frames)   # (B, d_feat, T)
        return self.ctm(kv, track=track)
