"""
Loss functions ported from PerspectiveFields.

- one_scale_gradient_loss: Single-scale gradient loss.
- msgil_norm_loss: Multi-Scale Gradient Image Loss (GT normalised).

Reference: https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS
"""

import torch


def one_scale_gradient_loss(pred_scale, gt, mask):
    """
    Computes the mean absolute gradient difference between prediction and GT
    at a single scale, masked by ``mask``.
    """
    mask_float = mask.to(dtype=pred_scale.dtype, device=pred_scale.device)
    d_diff = pred_scale - gt

    # Vertical gradients
    v_mask = torch.mul(mask_float[:, :, :-2, :], mask_float[:, :, 2:, :])
    v_gradient = torch.abs(d_diff[:, :, :-2, :] - d_diff[:, :, 2:, :])
    v_gradient = v_gradient[v_mask.to(dtype=mask.dtype)]

    # Horizontal gradients
    h_mask = torch.mul(mask_float[:, :, :, :-2], mask_float[:, :, :, 2:])
    h_gradient = torch.abs(d_diff[:, :, :, :-2] - d_diff[:, :, :, 2:])
    h_gradient = h_gradient[h_mask.to(dtype=mask.dtype)]

    valid_num = torch.sum(h_mask) + torch.sum(v_mask)
    gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient)) / (valid_num + 1e-8)
    return gradient_loss


def msgil_norm_loss(pred, gt, valid_mask, scales_num=4):
    """
    Multi-Scale Gradient Image Loss.

    Penalises gradient differences between prediction and ground truth at
    multiple downsampled scales, encouraging smooth predictions with correct
    edges.

    Args:
        pred: (B, C, H, W) predicted field.
        gt: (B, C, H, W) ground truth field.
        valid_mask: (B, C, H, W) boolean mask.
        scales_num: number of scales (powers of 2).

    Returns:
        Scalar loss.
    """
    grad_term = 0.0
    for i in range(scales_num):
        step = pow(2, i)
        d_gt = gt[:, :, ::step, ::step]
        d_pred = pred[:, :, ::step, ::step]
        d_mask = valid_mask[:, :, ::step, ::step]
        grad_term += one_scale_gradient_loss(d_pred, d_gt, d_mask)
    return grad_term
