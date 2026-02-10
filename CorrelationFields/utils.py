"""
Utility functions for the CorrelationFields pipeline.
"""

import os
import re
import random
from glob import glob
from datetime import datetime

import wandb
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def current_timestamp():
    return datetime.now().strftime('%y-%m-%d-%H-%M-%S')


def createDirectory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_last_checkpoint(checkpoint_dir, return_best=False, keyword='best'):
    checkpoint_paths = glob(f"{checkpoint_dir}/*.ckpt")
    if return_best:
        best_checkpoints = [
            ckpt for ckpt in checkpoint_paths if keyword in os.path.basename(ckpt)
        ]

        def parse_loss(ckpt):
            match = re.search(r'val_loss_epoch=([0-9]+\.[0-9]+)', ckpt)
            return float(match.group(1)) if match else float('inf')

        best_sorted = sorted(best_checkpoints, key=parse_loss)
        return best_sorted[0] if best_sorted else None
    for ckpt in checkpoint_paths:
        if os.path.basename(ckpt) == 'last.ckpt':
            return ckpt
    return None


def load_config(config_file):
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def fix_random_seed(seed: int):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Track shift operations
# ---------------------------------------------------------------------------

def apply_track_shifts(image, shift_vector, num_tracks):
    """
    Apply horizontal shifts to each track using torch.roll (non-differentiable).
    Used for data augmentation in the dataset.

    Since ILI data represents a cylindrical pipe surface, wrapping via torch.roll
    is physically correct.

    Args:
        image: (C, H, W) tensor.
        shift_vector: (num_tracks,) tensor of pixel shifts (will be rounded to int).
        num_tracks: number of vertical sensor tracks.

    Returns:
        shifted: (C, H, W) tensor with each track shifted horizontally.
    """
    C, H, W = image.shape
    track_height = H // num_tracks
    shifted = image.clone()

    for i in range(num_tracks):
        start = i * track_height
        end = start + track_height
        shift = int(shift_vector[i].item())
        if shift != 0:
            shifted[:, start:end, :] = torch.roll(
                image[:, start:end, :], shifts=shift, dims=-1
            )

    return shifted


def apply_track_shifts_differentiable(image, shift_vector, num_tracks):
    """
    Apply horizontal shifts using grid_sample (differentiable, per-track).
    Used for MSGLoss computation so gradients flow through the shift predictions.

    Args:
        image: (B, C, H, W) tensor.
        shift_vector: (B, num_tracks) tensor of pixel shifts.
        num_tracks: number of vertical sensor tracks.

    Returns:
        shifted: (B, C, H, W) tensor.
    """
    B, C, H, W = image.shape
    track_height = H // num_tracks

    y_coords = torch.linspace(-1, 1, H, device=image.device, dtype=image.dtype)
    x_coords = torch.linspace(-1, 1, W, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, H, W, 2)

    pixel_to_norm = 2.0 / (W - 1) if W > 1 else 0.0
    track_indices = torch.arange(H, device=image.device) // track_height
    track_indices = track_indices.clamp(max=num_tracks - 1)
    row_shifts = shift_vector[:, track_indices] * pixel_to_norm  # (B, H)
    grid[:, :, :, 0] = grid[:, :, :, 0] - row_shifts.unsqueeze(-1)

    shifted = F.grid_sample(image, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
    return shifted


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _tensor_to_numpy_img(tensor):
    """Convert a (1, H, W) or (H, W) float tensor in [0,1] to uint8 numpy."""
    img = tensor.squeeze().cpu().detach().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def log_correlation_predictions_to_wandb(
    shifted_images,
    original_images,
    pred_shifts,
    target_shifts,
    num_tracks,
    paths,
    phase="train",
):
    """
    Log CorrelationFields predictions to WandB:
      - Uncorrelated (shifted / broken) input image
      - Ground-truth correlated (aligned) image
      - Predicted correlated image (input + predicted shifts applied)

    A random sample from the batch is selected.

    Args:
        shifted_images: (B, 1, H, W) model input (uncorrelated).
        original_images: (B, 1, H, W) ground truth aligned.
        pred_shifts: (B, num_tracks) predicted per-track shifts.
        target_shifts: (B, num_tracks) ground truth per-track shifts.
        num_tracks: number of sensor tracks.
        paths: list of file paths.
        phase: 'train' or 'val'.

    Returns:
        dict: wandb loggable image dict.
    """
    idx = random.randint(0, shifted_images.shape[0] - 1)

    shifted_img = shifted_images[idx]        # (1, H, W)
    original_img = original_images[idx]       # (1, H, W)
    pred_shift = pred_shifts[idx]             # (num_tracks,)
    gt_shift = target_shifts[idx]             # (num_tracks,)

    mae = (pred_shift - gt_shift).abs().mean().item()

    # Reconstruct predicted aligned image (non-differentiable for viz)
    predicted_aligned = apply_track_shifts(
        shifted_img.cpu(), pred_shift.detach().cpu().round(), num_tracks,
    )

    images_dict = {
        f"{phase}/uncorrelated_input": wandb.Image(
            _tensor_to_numpy_img(shifted_img),
            caption=f"Input (uncorrelated) | {os.path.basename(paths[idx])}",
        ),
        f"{phase}/correlated_original": wandb.Image(
            _tensor_to_numpy_img(original_img),
            caption="Ground Truth (aligned)",
        ),
        f"{phase}/predicted_correlated": wandb.Image(
            _tensor_to_numpy_img(predicted_aligned),
            caption=f"Predicted (aligned) | MAE={mae:.2f}px",
        ),
    }
    return images_dict
