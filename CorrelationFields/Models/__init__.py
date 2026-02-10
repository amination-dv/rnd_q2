"""
CorrelationNet — Dense Shift Field prediction for ILI track alignment.

Follows the PerspectiveFields architecture:
    Backbone (MiT) → multi-scale features
    Low-Level Encoder → edge/texture features at 1/2 scale
    ShiftFieldDecoder → dense (B, 1, H, W) shift field

At inference the dense field is averaged per-track band to yield a
(B, 22) shift vector for image reconstruction.
"""

import torch
import torch.nn as nn

from .backbone import BACKBONES
from .decoder import ShiftFieldDecoder


class LowLevelEncoder(nn.Module):
    """Simple convolutional encoder that captures low-level details at 1/2 scale."""

    def __init__(self, in_channels=1, feat_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, feat_dim, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(feat_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))


class CorrelationNet(nn.Module):
    """
    Dense Shift Field network for ILI track correlation.

    Architecture (following PerspectiveFields):
        1. MiT backbone → multi-scale features at [1/4, 1/8, 1/16, 1/32].
        2. Low-level encoder → edge features at 1/2 scale.
        3. ShiftFieldDecoder → fuses all scales → dense (B, 1, H, W) shift field.

    The shift field predicts, for every pixel, the horizontal shift (in pixels)
    required to align that location. Since all pixels in the same track share
    one shift, the network learns a structured step-like output.

    Args:
        in_channels: Input image channels (1 for grayscale).
        backbone_name: 'mit_b0' (lightweight) or 'mit_b3' (full).
        num_tracks: Number of sensor tracks (for per-track extraction).
        embedding_dim: Common projection dim in the decoder.
        freeze_backbone: Freeze backbone weights (e.g. after loading pretrained).
    """

    def __init__(
        self,
        in_channels=1,
        backbone_name='mit_b0',
        num_tracks=22,
        embedding_dim=768,
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

        # --- Low-level encoder ---
        self.ll_enc = LowLevelEncoder(in_channels=in_channels, feat_dim=64)

        # --- Shift field decoder ---
        self.decoder = ShiftFieldDecoder(
            in_channels=self.embed_dims,
            embedding_dim=embedding_dim,
            ll_feat_dim=64,
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input tensor (grayscale ILI tubeview).

        Returns:
            shift_field: (B, 1, H, W) dense shift field.
        """
        hl_features = self.backbone(x)    # [c1, c2, c3, c4]
        ll_features = self.ll_enc(x)      # (B, 64, H/2, W/2)
        shift_field = self.decoder(hl_features, ll_features)  # (B, 1, H, W)
        return shift_field

    def extract_track_shifts(self, shift_field):
        """
        Average the dense shift field within each track band to obtain
        a per-track shift vector.

        Args:
            shift_field: (B, 1, H, W) dense predictions.

        Returns:
            shifts: (B, num_tracks) per-track average shift.
        """
        B, _, H, W = shift_field.shape
        track_height = H // self.num_tracks
        # Reshape into (B, num_tracks, track_height, W) and average
        field = shift_field.squeeze(1)  # (B, H, W)
        field = field[:, :self.num_tracks * track_height, :]  # trim if not exact
        field = field.reshape(B, self.num_tracks, track_height, W)
        shifts = field.mean(dim=(2, 3))  # (B, num_tracks)
        return shifts
