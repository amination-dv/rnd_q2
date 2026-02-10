"""
Shift Field Decoder — dense per-pixel shift regression.

Architecture follows the PerspectiveFields Latitude/Gravity decoders:
    1. MLP projection of each backbone scale to a common embedding dim.
    2. 3×3 Conv refinement at each level.
    3. Progressive 2× upsampling via FeatureFusionBlocks.
    4. Concatenation with low-level features at 1/2 scale.
    5. Final Conv layers → 1-channel dense shift field.

The key insight from the paper:
    *Dense predictions are better than global regressions.*
    Instead of asking the network for one shift scalar per track,
    we ask it to predict a Shift Field — explaining at every pixel
    where it sees misalignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Supporting modules (ported from PerspectiveFields decode_head.py)
# ---------------------------------------------------------------------------

class LinearProjection(nn.Module):
    """Project flattened feature map tokens to a common embedding dimension."""

    def __init__(self, input_dim, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) → (B, HW, C) → project → (B, HW, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ResidualConvUnit(nn.Module):
    """Two-layer residual 3×3 conv block."""

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """
    Fuse two feature maps (optionally) and upsample 2×.

    When ``unit2only=True`` only the residual unit is applied (no fusion
    with a second input), used at the coarsest level.
    """

    def __init__(self, features, unit2only=False):
        super().__init__()
        if not unit2only:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode='bilinear',
                               align_corners=False)
        return output


# ---------------------------------------------------------------------------
# Shift Field Decoder
# ---------------------------------------------------------------------------

class ShiftFieldDecoder(nn.Module):
    """
    Dense shift field decoder following the PerspectiveFields architecture.

    Takes multi-scale features from the backbone (4 levels) plus low-level
    features and produces a full-resolution dense shift field (B, 1, H, W).

    Args:
        in_channels: Channel counts at each backbone stage, e.g. [64,128,320,512].
        embedding_dim: Common projection dimension for the MLP heads.
        ll_feat_dim: Channel count of low-level encoder output (default 64).
    """

    def __init__(self, in_channels=(64, 128, 320, 512), embedding_dim=768,
                 ll_feat_dim=64):
        super().__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels

        # --- MLP projection per scale ---
        self.linear_c4 = LinearProjection(c4_in, embedding_dim)
        self.linear_c3 = LinearProjection(c3_in, embedding_dim)
        self.linear_c2 = LinearProjection(c2_in, embedding_dim)
        self.linear_c1 = LinearProjection(c1_in, embedding_dim)

        # --- 3×3 conv refinement per scale ---
        self.conv_c4 = nn.Conv2d(embedding_dim, 256, 3, 1, 1)
        self.conv_c3 = nn.Conv2d(embedding_dim, 256, 3, 1, 1)
        self.conv_c2 = nn.Conv2d(embedding_dim, 256, 3, 1, 1)
        self.conv_c1 = nn.Conv2d(embedding_dim, 256, 3, 1, 1)

        # --- Progressive feature fusion with 2× upsampling ---
        self.fusion4 = FeatureFusionBlock(256, unit2only=True)
        self.fusion3 = FeatureFusionBlock(256)
        self.fusion2 = FeatureFusionBlock(256)
        self.fusion1 = FeatureFusionBlock(256)

        # --- Fuse with low-level features and refine ---
        self.conv_fuse0 = nn.Sequential(
            nn.Conv2d(256 + ll_feat_dim, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv_fuse1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # --- Final prediction: 1-channel shift field ---
        self.pred_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, hl_features, ll_features):
        """
        Args:
            hl_features: list [c1, c2, c3, c4] from backbone.
                c1: (B, C1, H/4,  W/4)
                c2: (B, C2, H/8,  W/8)
                c3: (B, C3, H/16, W/16)
                c4: (B, C4, H/32, W/32)
            ll_features: (B, 64, H/2, W/2) from LowLevelEncoder.

        Returns:
            shift_field: (B, 1, H, W) dense shift prediction.
        """
        c1, c2, c3, c4 = hl_features
        n = c4.shape[0]

        # Project + reshape back to spatial
        def _project(linear, conv, feat):
            B, C, Hf, Wf = feat.shape
            out = linear(feat)                                # (B, Hf*Wf, embed_dim)
            out = out.permute(0, 2, 1).reshape(B, -1, Hf, Wf)  # (B, embed_dim, Hf, Wf)
            out = conv(out)                                    # (B, 256, Hf, Wf)
            return out

        _c4 = _project(self.linear_c4, self.conv_c4, c4)
        _c3 = _project(self.linear_c3, self.conv_c3, c3)
        _c2 = _project(self.linear_c2, self.conv_c2, c2)
        _c1 = _project(self.linear_c1, self.conv_c1, c1)

        # Progressive fusion: coarse → fine, each stage doubles resolution
        _c4 = self.fusion4(_c4)           # → 2× of c4
        _c3 = self.fusion3(_c4, _c3)      # → 2× of c3
        _c2 = self.fusion2(_c3, _c2)      # → 2× of c2
        _c1 = self.fusion1(_c2, _c1)      # → H/2, W/2

        # Fuse with low-level features
        x = torch.cat([_c1, ll_features], dim=1)  # (B, 256+64, H/2, W/2)
        x = self.conv_fuse0(x)                      # (B, 64, H/2, W/2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv_fuse1(x)                      # (B, 32, H, W)

        shift_field = self.pred_head(x)              # (B, 1, H, W)
        return shift_field
