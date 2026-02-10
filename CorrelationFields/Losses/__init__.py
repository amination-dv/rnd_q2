"""
Losses for the CorrelationFields regression pipeline.

- MSGLoss: Multi-Scale Gradient loss (kornia-based, from existing codebase).
- msgil_norm_loss: Multi-Scale Gradient Image Loss (from PerspectiveFields).
- L1Loss: standard torch.nn.L1Loss (used directly in training).
"""

from .msg_loss import MSGLoss
from .loss_fns import msgil_norm_loss, one_scale_gradient_loss
