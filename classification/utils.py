"""
Utility functions for the ILI Strip-ViT classification and alignment pipeline.
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
            match = re.search(r'val_total_loss_epoch=([0-9]+\.[0-9]+)', ckpt)
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
    Apply horizontal shifts using grid_sample (differentiable).
    Used for computing image-level losses (e.g., MSGLoss) during training so
    that gradients can flow back through the predicted shift values.

    Args:
        image: (B, C, H, W) tensor.
        shift_vector: (B, num_tracks) tensor of pixel shifts.
        num_tracks: number of vertical sensor tracks.

    Returns:
        shifted: (B, C, H, W) tensor.
    """
    B, C, H, W = image.shape
    track_height = H // num_tracks

    # Create base normalised coordinate grid [-1, 1]
    y_coords = torch.linspace(-1, 1, H, device=image.device, dtype=image.dtype)
    x_coords = torch.linspace(-1, 1, W, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, H, W, 2)

    # Convert pixel shifts to normalised coordinate shifts
    # With align_corners=True: pixel range [0, W-1] maps to [-1, 1]
    pixel_to_norm = 2.0 / (W - 1) if W > 1 else 0.0

    # Assign each row to its track index
    track_indices = torch.arange(H, device=image.device) // track_height
    track_indices = track_indices.clamp(max=num_tracks - 1)

    # Per-row normalised shifts: (B, H)
    row_shifts = shift_vector[:, track_indices] * pixel_to_norm

    # Apply shift to x-coordinates (subtract to shift content right when shift > 0)
    grid[:, :, :, 0] = grid[:, :, :, 0] - row_shifts.unsqueeze(-1)

    shifted = F.grid_sample(
        image, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )

    return shifted


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _tensor_to_numpy_img(tensor):
    """Convert a (1, H, W) or (H, W) float tensor in [0,1] to uint8 numpy."""
    img = tensor.squeeze().cpu().detach().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def log_ili_predictions_to_wandb(
    shifted_images,
    original_images,
    shift_preds,
    labels,
    preds,
    target_shifts,
    has_component,
    paths,
    class_names,
    num_tracks,
    phase="train",
):
    """
    Log ILI predictions to Weights & Biases, including:
      - The uncorrelated (shifted / broken) input image
      - The ground-truth correlated (aligned) image
      - The predicted correlated image (input + predicted shifts applied)

    A random sample from the batch is selected for logging.

    Args:
        shifted_images: (B, 1, H, W) model input (uncorrelated).
        original_images: (B, 1, H, W) ground truth aligned (correlated).
        shift_preds: (B, num_tracks) predicted shifts.
        labels: (B,) ground truth class indices.
        preds: (B,) predicted class indices.
        target_shifts: (B, num_tracks) ground truth inverse shifts.
        has_component: (B,) 1 if component, 0 if background.
        paths: list of file paths.
        class_names: list of class name strings.
        num_tracks: number of sensor tracks.
        phase: 'train' or 'val'.

    Returns:
        dict: wandb loggable image dict.
    """
    idx = random.randint(0, shifted_images.shape[0] - 1)

    shifted_img = shifted_images[idx]   # (1, H, W)
    original_img = original_images[idx]  # (1, H, W)
    pred_shift = shift_preds[idx]        # (num_tracks,)
    label = labels[idx].item()
    pred = preds[idx].item()

    # Reconstruct predicted aligned image by applying predicted shifts
    predicted_aligned = apply_track_shifts(
        shifted_img.cpu(),
        pred_shift.detach().cpu().round(),
        num_tracks,
    )

    label_name = class_names[label] if label < len(class_names) else str(label)
    pred_name = class_names[pred] if pred < len(class_names) else str(pred)

    images_dict = {
        f"{phase}/uncorrelated_input": wandb.Image(
            _tensor_to_numpy_img(shifted_img),
            caption=f"Input (uncorrelated) | GT: {label_name}",
        ),
        f"{phase}/correlated_original": wandb.Image(
            _tensor_to_numpy_img(original_img),
            caption=f"Ground Truth (correlated) | GT: {label_name}",
        ),
        f"{phase}/predicted_correlated": wandb.Image(
            _tensor_to_numpy_img(predicted_aligned),
            caption=f"Predicted (correlated) | GT: {label_name}, Pred: {pred_name}",
        ),
    }

    return images_dict
