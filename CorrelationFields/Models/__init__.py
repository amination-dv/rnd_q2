"""
CorrelationNet — Per-track shift regression for ILI track alignment.

Simplified architecture following the ParamNet pattern from PerspectiveFields:
    Backbone (MiT) → multi-scale global average pooling → MLP → (B, num_tracks)

No dense decoder needed — we directly regress the 22-element shift vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BACKBONES


class CorrelationNet(nn.Module):
    """
    Per-track shift regression network for ILI track correlation.

    Architecture:
        1. MiT backbone → multi-scale features at [1/4, 1/8, 1/16, 1/32].
        2. Global average pooling at each scale → concatenate.
        3. MLP regression head → (B, num_tracks) shift vector.

    This follows the ParamNet pattern from PerspectiveFields: the task is a
    simple geometric regression (22 shift values), so a global pooling +
    MLP head suffices — no dense decoder required.

    Args:
        in_channels: Input image channels (1 for grayscale).
        backbone_name: 'mit_b0' (lightweight) or 'mit_b3' (full).
        num_tracks: Number of sensor tracks to predict shifts for.
        hidden_dim: Hidden dimension of the MLP head.
        freeze_backbone: Freeze backbone weights.
    """

    def __init__(
        self,
        in_channels=1,
        backbone_name='mit_b0',
        num_tracks=22,
        hidden_dim=256,
        freeze_backbone=False,
    ):
        super().__init__()
        self.num_tracks = num_tracks

        # --- Backbone ---
        assert backbone_name in BACKBONES, (
            f"Unknown backbone '{backbone_name}'. Choose from {list(BACKBONES)}"
        )
        self.backbone = BACKBONES[backbone_name](in_chans=in_channels)
        self.embed_dims = list(self.backbone.embed_dims)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- Regression head ---
        # Pool each scale globally and concatenate
        total_feat_dim = sum(self.embed_dims)  # e.g. 32+64+160+256=512 for mit_b0

        self.head = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tracks),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input tensor (grayscale ILI tubeview).

        Returns:
            shifts: (B, num_tracks) predicted per-track horizontal shifts.
        """
        features = self.backbone(x)  # [c1, c2, c3, c4]

        # Global average pooling at each scale and concatenate
        pooled = []
        for feat in features:
            pooled.append(F.adaptive_avg_pool2d(feat, 1).flatten(1))
        x = torch.cat(pooled, dim=1)  # (B, sum(embed_dims))

        shifts = self.head(x)  # (B, num_tracks)
        return shifts
