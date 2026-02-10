"""
ILI Strip-ViT Inference Pipeline.

Runs sliding-window inference on long strips of pipe, then applies Gaussian
smoothing to class probabilities and peak detection to locate components.

Usage:
    python inference.py --input_path long_strip.png --config_path config.yaml

Pipeline:
    1. Load trained StripViT model.
    2. Slide a window across the long strip (stride = patch_width).
    3. Collect per-position class logits and shift predictions.
    4. Smooth class probabilities with a 1D Gaussian filter.
    5. Detect peaks in the non-background probability channels.
    6. Return peak locations and their corresponding shift vectors.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from PIL import Image

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Models import StripViT
from utils import load_config, apply_track_shifts

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config):
    """Load a trained StripViT model from config."""
    model = StripViT(
        img_height=config['img_height'],
        num_tracks=config['num_tracks'],
        num_classes=config['num_classes'],
        in_channels=config.get('in_channels', 1),
        patch_width=config.get('patch_width', 14),
        backbone_name=config['backbone'],
        train_backbone=False,
        unfreeze_last_n_blocks=0,
        intermediate_block_idx=config.get('intermediate_block_idx', 7),
    )
    state_dict = torch.load(config['model_path'], map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_strip(image_array, img_height, patch_width=14):
    """
    Preprocess a grayscale strip image for inference.

    - Resizes height to ``img_height``.
    - Pads width to the nearest multiple of ``patch_width``.
    - Normalises to [0, 1].

    Args:
        image_array: (H, W) uint8 numpy array.
        img_height: target height.
        patch_width: token width (default 14).

    Returns:
        tensor: (1, 1, H, W_padded) float tensor on DEVICE.
        original_width: original width before padding.
    """
    h_orig, w_orig = image_array.shape[:2]

    # Resize height, keep width proportional
    image = Image.fromarray(image_array).resize((w_orig, img_height), Image.BILINEAR)
    image = np.array(image)

    # Pad width to multiple of patch_width
    w = image.shape[1]
    pad_w = (patch_width - w % patch_width) % patch_width
    if pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_w)), mode='constant', constant_values=0)

    # To tensor
    tensor = torch.from_numpy(image).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor.to(DEVICE), w  # return original (unpadded) width


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def inference_on_strip(model, long_image, config, stride=None):
    """
    Run sliding-window inference on a long strip.

    The Strip Tokenizer naturally slides across the width: each window of
    ``img_width`` pixels produces one classification + one shift prediction.
    Adjacent windows overlap by ``img_width - stride``.

    Args:
        model: trained StripViT in eval mode.
        long_image: (1, 1, H, W) tensor.
        config: configuration dict.
        stride: sliding stride in pixels (default: patch_width = 14).

    Returns:
        class_probs: (N, num_classes) ndarray of class probabilities.
        shifts: (N, num_tracks) ndarray of shift predictions.
        positions: (N,) ndarray of centre x-coordinates in pixels.
    """
    window_width = config['img_width']
    patch_width = config.get('patch_width', 14)
    if stride is None:
        stride = patch_width

    _, _, H, W = long_image.shape

    # Handle strips narrower than a single window
    if W < window_width:
        pad_w = window_width - W
        long_image = F.pad(long_image, (0, pad_w), mode='constant', value=0)
        W = window_width

    all_logits = []
    all_shifts = []
    positions = []

    for start_x in range(0, W - window_width + 1, stride):
        window = long_image[:, :, :, start_x:start_x + window_width]
        with torch.no_grad():
            class_logits, shift_preds = model(window)
        all_logits.append(class_logits.cpu())
        all_shifts.append(shift_preds.cpu())
        positions.append(start_x + window_width // 2)

    if len(all_logits) == 0:
        return None, None, None

    class_logits = torch.cat(all_logits, dim=0)  # (N, num_classes)
    shifts = torch.cat(all_shifts, dim=0)          # (N, num_tracks)
    class_probs = F.softmax(class_logits, dim=-1).numpy()
    shifts = shifts.numpy()
    positions = np.array(positions)

    return class_probs, shifts, positions


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def detect_components(class_probs, shifts, positions, config, sigma=5.0, prominence=0.3):
    """
    Smooth class probabilities and detect component peaks.

    For each non-background class:
        1. Apply 1D Gaussian filter to the probability channel.
        2. Run scipy peak detection.
        3. Extract the shift vector at each peak.

    Args:
        class_probs: (N, num_classes) probability array.
        shifts: (N, num_tracks) shift predictions.
        positions: (N,) centre x-coordinates.
        config: configuration dict.
        sigma: Gaussian smoothing sigma.
        prominence: minimum peak prominence for detection.

    Returns:
        detections: list of dicts with keys
            'position', 'class', 'class_idx', 'confidence', 'shifts'.
    """
    num_classes = class_probs.shape[1]
    class_names = config.get('class_names', None)
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    background_idx = config.get('background_class_idx', num_classes - 1)

    detections = []

    for cls_idx in range(num_classes):
        if cls_idx == background_idx:
            continue

        probs = class_probs[:, cls_idx]
        smoothed = gaussian_filter1d(probs, sigma=sigma)

        peaks, _ = find_peaks(smoothed, prominence=prominence, distance=10)

        for peak_idx in peaks:
            detections.append({
                'position': int(positions[peak_idx]),
                'class': class_names[cls_idx] if cls_idx < len(class_names) else f'class_{cls_idx}',
                'class_idx': int(cls_idx),
                'confidence': float(smoothed[peak_idx]),
                'shifts': shifts[peak_idx].tolist(),
            })

    detections.sort(key=lambda d: d['position'])
    return detections


def upsample_predictions(class_probs, shifts, original_width):
    """
    Upsample sparse predictions back to the original image width
    using linear interpolation for visualisation.

    Args:
        class_probs: (N, num_classes) probability array.
        shifts: (N, num_tracks) shift array.
        original_width: target width W.

    Returns:
        upsampled_probs: (W, num_classes) ndarray.
        upsampled_shifts: (W, num_tracks) ndarray.
    """
    probs_t = torch.from_numpy(class_probs).float().unsqueeze(0).permute(0, 2, 1)
    shifts_t = torch.from_numpy(shifts).float().unsqueeze(0).permute(0, 2, 1)

    up_probs = F.interpolate(probs_t, size=original_width, mode='linear', align_corners=True)
    up_shifts = F.interpolate(shifts_t, size=original_width, mode='linear', align_corners=True)

    up_probs = up_probs.squeeze(0).permute(1, 0).numpy()    # (W, num_classes)
    up_shifts = up_shifts.squeeze(0).permute(1, 0).numpy()  # (W, num_tracks)

    return up_probs, up_shifts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config_path)
    model = load_model(config)

    # Load long strip image
    image = np.array(Image.open(args.input_path).convert('L'))
    print(f"Loaded image: {image.shape}")

    # Preprocess
    image_tensor, orig_width = preprocess_strip(
        image,
        img_height=config['img_height'],
        patch_width=config.get('patch_width', 14),
    )
    print(f"Preprocessed tensor: {image_tensor.shape}")

    # Run inference
    class_probs, shifts, positions = inference_on_strip(
        model, image_tensor, config, stride=args.stride,
    )

    if class_probs is None:
        print("No predictions generated.")
        return

    print(f"Generated {len(positions)} position predictions")

    # Detect components
    detections = detect_components(
        class_probs, shifts, positions, config,
        sigma=args.sigma, prominence=args.prominence,
    )

    print(f"\nDetected {len(detections)} components:")
    for det in detections:
        print(
            f"  - {det['class']} at x={det['position']}, "
            f"confidence={det['confidence']:.3f}"
        )

    # Save outputs
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Upsample for visualisation
        up_probs, up_shifts = upsample_predictions(class_probs, shifts, orig_width)

        np.savez(
            os.path.join(args.output_dir, 'predictions.npz'),
            class_probs=class_probs,
            shifts=shifts,
            positions=positions,
            upsampled_probs=up_probs,
            upsampled_shifts=up_shifts,
            detections=np.array(detections, dtype=object),
        )
        print(f"Predictions saved to {args.output_dir}/predictions.npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ILI Strip-ViT Inference')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to long strip image')
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs')
    parser.add_argument('--stride', type=int, default=14,
                        help='Sliding window stride in pixels')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='Gaussian smoothing sigma for peak detection')
    parser.add_argument('--prominence', type=float, default=0.3,
                        help='Minimum peak prominence for detection')
    args = parser.parse_args()
    main(args)
