"""
ILI Dataset for Strip-ViT training.

Loads aligned (correlated) tubeview images and generates self-supervised
alignment training data by artificially shifting individual sensor tracks.
"""

import random

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from utils import apply_track_shifts


class ILIDataset(ImageFolder):
    """
    ILI (In-Line Inspection) dataset for multi-task training.

    Training data generation (self-supervised alignment):
        1. Load an aligned (correlated) grayscale image.
        2. Divide into ``num_tracks`` horizontal strips.
        3. For non-background classes: randomly shift each strip to create a
           "broken" (uncorrelated) image. The inverse of these shifts is the
           regression target.
        4. For background class: return the original image with zero shifts
           and mask=0 so the regression loss is masked out.

    Each sample returns::

        (shifted_image, label, target_shift, has_component, original_image, path)

    Args:
        root: Path to dataset directory (ImageFolder layout).
        img_height: Target image height after resize (must be divisible by num_tracks).
        img_width: Target image width after resize (must be divisible by patch_width=14).
        num_tracks: Number of sensor tracks (default: 22).
        max_shift: Maximum pixel shift per track (default: 15).
        background_class: Name of the background class folder (default: 'background').
        augment: Whether to apply additional augmentations (brightness, noise).
    """

    def __init__(
        self,
        root,
        img_height=308,
        img_width=196,
        num_tracks=22,
        max_shift=15,
        background_class='background',
        augment=False,
    ):
        super().__init__(root)
        self.img_height = img_height
        self.img_width = img_width
        self.num_tracks = num_tracks
        self.max_shift = max_shift
        self.augment = augment

        # Resolve background class index (-1 if not present)
        self.background_idx = self.class_to_idx.get(background_class, -1)

    def __getitem__(self, index):
        path, label = self.samples[index]

        # Load as grayscale and resize
        image = Image.open(path).convert('L')
        image = image.resize((self.img_width, self.img_height), Image.BILINEAR)

        # Convert to float tensor [0, 1] with shape (1, H, W)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.unsqueeze(0)  # (1, H, W)

        # Optional augmentations (brightness / noise)
        if self.augment:
            image = self._apply_augmentations(image)

        # Store original aligned image (for logging and optional MSGLoss)
        original_image = image.clone()

        # Determine whether the sample contains a component
        has_component = 1 if label != self.background_idx else 0

        if has_component:
            # Generate random per-track shifts (integer pixels)
            shift_vector = torch.FloatTensor(self.num_tracks).uniform_(
                -self.max_shift, self.max_shift,
            ).round()

            # Apply shifts to create the broken (uncorrelated) input
            shifted_image = apply_track_shifts(image, shift_vector, self.num_tracks)

            # Target: the inverse shift required to fix the image
            target_shift = -shift_vector
        else:
            shifted_image = image.clone()
            target_shift = torch.zeros(self.num_tracks)

        return (
            shifted_image,          # Model input  (1, H, W)
            label,                  # Class label  (int)
            target_shift,           # Shift target (num_tracks,)
            has_component,          # Mask flag    (int: 0 or 1)
            original_image,         # GT aligned   (1, H, W)
            path,                   # File path    (str)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_augmentations(image):
        """Simple augmentations suitable for grayscale ILI data."""
        # Random brightness
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            image = (image * factor).clamp(0, 1)

        # Random Gaussian noise
        if random.random() < 0.2:
            noise = torch.randn_like(image) * 0.02
            image = (image + noise).clamp(0, 1)

        return image
