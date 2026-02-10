"""
CorrelationFields Inference Pipeline.

Given an uncorrelated ILI tubeview image (already cropped by a separate
object detection model), predict the dense shift field and extract per-track
shifts to reconstruct the aligned image.

Usage:
    python inference.py --image path/to/image.png --config config.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
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
        embedding_dim=config.get('embedding_dim', 768),
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

    Args:
        image_path: path to the image.
        img_height: target height.
        img_width: target width.

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
def predict_shift_field(model, image_tensor):
    """
    Run the model to predict the dense shift field.

    Args:
        model: CorrelationNet in eval mode.
        image_tensor: (1, 1, H, W) input.

    Returns:
        shift_field: (1, 1, H, W) predicted dense shift field.
        track_shifts: (num_tracks,) per-track average shifts.
    """
    shift_field = model(image_tensor)  # (1, 1, H, W)
    track_shifts = model.extract_track_shifts(shift_field)  # (1, num_tracks)
    return shift_field, track_shifts.squeeze(0)


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
# Sliding-window inference for long strips
# ---------------------------------------------------------------------------

@torch.no_grad()
def inference_on_strip(model, long_image, config, stride=None):
    """
    Sliding-window inference on a long image strip.

    Args:
        model: CorrelationNet in eval mode.
        long_image: (1, 1, H, W_long) preprocessed long strip.
        config: configuration dict.
        stride: step size in pixels (default: img_width // 2).

    Returns:
        all_fields: list of (1, H, window_W) shift field predictions.
        all_track_shifts: list of (num_tracks,) shift vectors.
        positions: list of window center x-positions.
    """
    window_width = config['img_width']
    if stride is None:
        stride = window_width // 2

    _, _, H, W = long_image.shape
    if W < window_width:
        long_image = F.pad(long_image, (0, window_width - W), mode='constant', value=0)
        W = window_width

    all_fields = []
    all_track_shifts = []
    positions = []

    for start_x in range(0, W - window_width + 1, stride):
        window = long_image[:, :, :, start_x:start_x + window_width]
        shift_field, track_shifts = predict_shift_field(model, window)
        all_fields.append(shift_field.cpu())
        all_track_shifts.append(track_shifts.cpu())
        positions.append(start_x + window_width // 2)

    return all_fields, all_track_shifts, positions


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

    shift_field, track_shifts = predict_shift_field(model, image_tensor)

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

    field_np = shift_field.squeeze().cpu().numpy()
    np.save(os.path.join(out_dir, f"{base}_shift_field.npy"), field_np)

    print(f"Saved aligned image and shift field to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CorrelationFields Inference")
    parser.add_argument('--image', required=True, help='Path to uncorrelated image')
    parser.add_argument('--config', default='config.yaml', help='Config YAML file')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    args = parser.parse_args()
    main(args)
