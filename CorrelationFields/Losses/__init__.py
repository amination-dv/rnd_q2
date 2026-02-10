"""
Losses for the CorrelationFields regression pipeline.

- MSGLoss: Multi-Scale Gradient loss on reconstructed images (kornia-based).
- L1Loss: standard torch.nn.L1Loss on shift vectors (used directly in training).
"""

from .msg_loss import MSGLoss
