"""
Strip-DINOv2: Multi-Task Vision Transformer for ILI Component Detection & Track Alignment.

Architecture:
- Strip Tokenizer: Custom patch embedding where each token represents a segment of
  exactly one sensor track, preventing the network from baking misalignment into features.
- Pre-trained DINOv2 transformer blocks for feature extraction.
- Dual Heads:
    - DetectionHead: Classification from final CLS token (Tee, Stopple, Weld, Background).
    - AlignmentHead: Regression from intermediate features to predict per-track shifts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StripTokenizer(nn.Module):
    """
    Custom patch embedding aligned with ILI sensor track physics.

    Instead of standard square patches (14x14), this creates tokens where each
    token covers exactly one track's height and a fixed width (14px). This prevents
    visual discontinuities from track misalignment from appearing inside tokens.

    kernel_size = (track_height, patch_width)
    stride      = (track_height, patch_width)

    Output token grid shape: (num_tracks, W // patch_width)
    """

    def __init__(self, in_channels, embed_dim, track_height, patch_width=14):
        super().__init__()
        self.track_height = track_height
        self.patch_width = patch_width
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=(track_height, patch_width),
            stride=(track_height, patch_width),
            bias=True,
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image tensor.
        Returns:
            tokens: (B, N, embed_dim) where N = num_tracks * (W // patch_width).
            grid_size: (grid_h, grid_w) tuple.
        """
        x = self.proj(x)  # (B, embed_dim, num_tracks, W // patch_width)
        B, D, H_t, W_t = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H_t * W_t, embed_dim)
        return tokens, (H_t, W_t)


class AlignmentHead(nn.Module):
    """
    Regression head for predicting per-track horizontal pixel shifts.

    Uses intermediate-layer features (e.g., Block 8) because they retain more
    geometric/spatial information compared to the final layer which is too
    semantic/invariant for precise geometric regression.

    Input:  Concatenation of [CLS_intermediate, MeanPool(patch_tokens_intermediate)]
    Output: (B, num_tracks) shift predictions in pixels.
    """

    def __init__(self, embed_dim, num_tracks=22, hidden_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_tracks),
        )

    def forward(self, cls_token, patch_tokens):
        """
        Args:
            cls_token: (B, embed_dim) CLS token from intermediate block.
            patch_tokens: (B, N, embed_dim) patch tokens from intermediate block.
        Returns:
            shifts: (B, num_tracks) predicted pixel shifts per track.
        """
        pooled = patch_tokens.mean(dim=1)  # (B, embed_dim)
        x = torch.cat([cls_token, pooled], dim=-1)  # (B, embed_dim * 2)
        return self.head(x)


class DetectionHead(nn.Module):
    """
    Classification head for component detection.

    Uses the CLS token from the final transformer block, which captures the
    most abstract semantic representation of the input.

    Input:  CLS token from final block (B, embed_dim).
    Output: (B, num_classes) logits.
    """

    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token):
        return self.head(cls_token)


class StripViT(nn.Module):
    """
    Strip-DINOv2: Multi-task Vision Transformer for ILI component detection
    and track alignment.

    Architecture:
        1. Strip Tokenizer replaces the standard patch embedding, creating
           tokens aligned to the 22 sensor tracks.
        2. Pre-trained DINOv2 transformer blocks process the token sequence.
        3. Dual heads:
            - DetectionHead: uses final CLS token for classification.
            - AlignmentHead: uses intermediate CLS + pooled patch tokens
              for shift regression.

    Args:
        img_height: Input image height (must be divisible by num_tracks).
        num_tracks: Number of sensor tracks (default: 22).
        num_classes: Number of component classes (default: 4).
        in_channels: Input channels (default: 1 for grayscale).
        patch_width: Width of each token in pixels (default: 14).
        backbone_name: DINOv2 model variant (default: 'dinov2_vits14').
        train_backbone: Whether to fine-tune backbone blocks.
        unfreeze_last_n_blocks: Number of final blocks to unfreeze.
        intermediate_block_idx: 0-indexed block for alignment features
            (default: 7 = "Block 8" in 1-indexed notation).
    """

    def __init__(
        self,
        img_height,
        num_tracks=22,
        num_classes=4,
        in_channels=1,
        patch_width=14,
        backbone_name="dinov2_vits14",
        train_backbone=True,
        unfreeze_last_n_blocks=4,
        intermediate_block_idx=7,
    ):
        super().__init__()
        self.num_tracks = num_tracks
        self.track_height = img_height // num_tracks
        self.patch_width = patch_width
        self.intermediate_block_idx = intermediate_block_idx

        assert img_height % num_tracks == 0, (
            f"img_height ({img_height}) must be divisible by num_tracks ({num_tracks})"
        )

        # --- Load pre-trained DINOv2 backbone ---
        backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        self.embed_dim = backbone.embed_dim
        self.num_blocks = len(backbone.blocks)

        assert 0 <= intermediate_block_idx < self.num_blocks, (
            f"intermediate_block_idx ({intermediate_block_idx}) must be in "
            f"[0, {self.num_blocks})"
        )

        # --- Custom Strip Tokenizer (replaces patch_embed) ---
        self.strip_tokenizer = StripTokenizer(
            in_channels=in_channels,
            embed_dim=self.embed_dim,
            track_height=self.track_height,
            patch_width=patch_width,
        )

        # --- Transfer pre-trained components ---
        self.cls_token = nn.Parameter(backbone.cls_token.data.clone())
        self.blocks = backbone.blocks
        self.norm = backbone.norm

        # --- Learnable positional embeddings ---
        # Initialized fresh since the token grid differs from the pre-trained model.
        # Default grid: (num_tracks, default_w_patches). Interpolated at runtime
        # if the input width differs.
        default_w_patches = 14  # for default training width = 14 * 14 = 196
        num_patches = num_tracks * default_w_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._default_grid_h = num_tracks
        self._default_grid_w = default_w_patches

        # --- Freeze / unfreeze backbone ---
        for p in self.blocks.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False

        if train_backbone and unfreeze_last_n_blocks > 0:
            for p in self.blocks[-unfreeze_last_n_blocks:].parameters():
                p.requires_grad = True
            for p in self.norm.parameters():
                p.requires_grad = True

        # --- Task heads ---
        self.alignment_head = AlignmentHead(
            self.embed_dim, num_tracks, hidden_dim=512
        )
        self.detection_head = DetectionHead(
            self.embed_dim, num_classes
        )

    def interpolate_pos_encoding(self, num_patches, grid_h, grid_w):
        """
        Interpolate positional embeddings for variable-size inputs.

        The pos_embed is stored for the default grid size and resized via
        bilinear interpolation when the actual input grid differs.
        """
        N = self.pos_embed.shape[1] - 1  # exclude CLS position
        if num_patches == N and grid_h == self._default_grid_h:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1, :]    # (1, 1, D)
        patch_pos = self.pos_embed[:, 1:, :]  # (1, N, D)

        dim = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(
            1, self._default_grid_h, self._default_grid_w, dim
        ).permute(0, 3, 1, 2)  # (1, D, H_default, W_default)

        patch_pos = F.interpolate(
            patch_pos.float(),
            size=(grid_h, grid_w),
            mode='bilinear',
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x):
        """
        Forward pass for training (single-window per-image predictions).

        Args:
            x: (B, C, H, W) input tensor (grayscale ILI tubeview).

        Returns:
            class_logits: (B, num_classes) classification logits.
            shift_preds:  (B, num_tracks) predicted shift values.
        """
        B = x.shape[0]

        # 1. Strip tokenization
        tokens, (grid_h, grid_w) = self.strip_tokenizer(x)  # (B, N, D)
        num_patches = tokens.shape[1]

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+N, D)

        # 3. Add positional encoding (interpolated if size differs)
        pos_embed = self.interpolate_pos_encoding(num_patches, grid_h, grid_w)
        tokens = tokens + pos_embed.to(tokens.dtype)

        # 4. Pass through transformer blocks, capturing intermediate output
        intermediate_output = None
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i == self.intermediate_block_idx:
                # Capture output of this block for alignment head.
                # No .clone() needed since `tokens = blk(tokens)` rebinds the name.
                intermediate_output = tokens

        # 5. Final layer norm
        tokens = self.norm(tokens)

        # 6. Extract features for each head
        # Detection: CLS from final layer (most semantic)
        cls_final = tokens[:, 0]  # (B, D)

        # Alignment: CLS + patch tokens from intermediate block (more geometric)
        intermediate_cls = intermediate_output[:, 0]       # (B, D)
        intermediate_patches = intermediate_output[:, 1:]  # (B, N, D)

        # 7. Head predictions
        class_logits = self.detection_head(cls_final)                           # (B, num_classes)
        shift_preds = self.alignment_head(intermediate_cls, intermediate_patches)  # (B, num_tracks)

        return class_logits, shift_preds
