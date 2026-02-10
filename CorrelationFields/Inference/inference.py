"""
CorrelationFields Inference Pipeline.

Given an uncorrelated ILI tubeview image (already cropped by a separate
object detection model), predict per-track shifts and reconstruct the
aligned image.

Usage:
    python inference.py --image path/to/image.png --config config.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Models import CorrelationNet
from utils import load_config, apply_track_shifts

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config):
    """Load CorrelationNet from config and checkpoint."""
    model = CorrelationNet(
        in_channels=config.get('in_channels', 1),
        backbone_name=config['backbone'],
        num_tracks=config['num_tracks'],
        hidden_dim=config.get('hidden_dim', 256),
    )
    state_dict = torch.load(config['model_path'], map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path, img_height, img_width):
    """
    Load a grayscale image, resize, and convert to tensor.

    Returns:
        tensor: (1, 1, H, W) float tensor.
        original: PIL Image (original resolution).
    """
    original = Image.open(image_path).convert('L')
    resized = original.resize((img_width, img_height), Image.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor.to(DEVICE), original


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_shifts(model, image_tensor):
    """
    Run the model to predict per-track shifts.

    Args:
        model: CorrelationNet in eval mode.
        image_tensor: (1, 1, H, W) input.

    Returns:
        track_shifts: (num_tracks,) per-track shifts in pixels.
    """
    shifts = model(image_tensor)  # (1, num_tracks)
    return shifts.squeeze(0)


def reconstruct_aligned(image_tensor, track_shifts, num_tracks):
    """
    Apply predicted per-track shifts to reconstruct the aligned image.

    Args:
        image_tensor: (1, 1, H, W) uncorrelated input.
        track_shifts: (num_tracks,) shifts in pixels.
        num_tracks: number of tracks.

    Returns:
        aligned: (1, 1, H, W) reconstructed aligned image.
    """
    aligned = apply_track_shifts(
        image_tensor.squeeze(0).cpu(),  # (1, H, W)
        track_shifts.cpu().round(),
        num_tracks,
    )
    return aligned.unsqueeze(0)  # (1, 1, H, W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config)

    print("Loading model...")
    model = load_model(config)

    print(f"Processing: {args.image}")
    image_tensor, original_img = preprocess_image(
        args.image, config['img_height'], config['img_width'],
    )

    track_shifts = predict_shifts(model, image_tensor)

    print(f"Predicted per-track shifts (pixels):")
    for i, s in enumerate(track_shifts.cpu().numpy()):
        print(f"  Track {i:2d}: {s:+.2f}")

    # Reconstruct aligned image
    aligned = reconstruct_aligned(image_tensor, track_shifts, config['num_tracks'])

    # Save outputs
    out_dir = args.output_dir or os.path.dirname(args.image)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]

    aligned_np = (aligned.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(aligned_np).save(os.path.join(out_dir, f"{base}_aligned.png"))

    shifts_np = track_shifts.cpu().numpy()
    np.save(os.path.join(out_dir, f"{base}_shifts.npy"), shifts_np)

    print(f"Saved aligned image and shifts to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CorrelationFields Inference")
    parser.add_argument('--image', required=True, help='Path to uncorrelated image')
    parser.add_argument('--config', default='config.yaml', help='Config YAML file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    args = parser.parse_args()
    main(args)
