"""
Loss functions for the ILI Strip-ViT pipeline.

Losses:
    - MultiClassCrossEntropy: Multi-class classification loss for detection head.
    - MSGLoss: Multi-Scale Gradient loss (auxiliary, image-level alignment quality).
      Referenced from PerspectiveFields; L1 loss is preferred for geometric regression
      as it is less sensitive to outliers and stabilises convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .msg_loss import MSGLoss


class MultiClassCrossEntropy(nn.Module):
    """
    Multiclass cross-entropy for mutually exclusive classes.

    Args:
        reduction: Loss reduction mode ('mean' | 'sum' | 'none').
        class_weight: Optional (C,) tensor to re-balance classes.
        label_smoothing: Label smoothing factor.
    """

    def __init__(self, reduction: str = "mean", class_weight=None, label_smoothing: float = 0.0):
        super().__init__()
        if class_weight is None:
            self.register_buffer("class_weight", None, persistent=False)
        else:
            cw = torch.as_tensor(class_weight, dtype=torch.float32)
            self.register_buffer("class_weight", cw)
        self.loss = nn.CrossEntropyLoss(
            weight=self.class_weight,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw class logits.
            targets: (B,) long tensor with class indices in [0, C-1].
        """
        return self.loss(logits, targets)
