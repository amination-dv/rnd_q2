"""
ILI Dataset for CorrelationFields training.

Loads aligned (correlated) tubeview images and generates self-supervised
alignment training data by artificially shifting individual sensor tracks.

Key difference from the classification variant: the regression target is now
a **dense shift field** (B, 1, H, W) — not a 22-element vector. Every pixel
in the field carries the shift value for its track band.
"""

import random

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from utils import apply_track_shifts


class ILIDataset(ImageFolder):
    """
    ILI dataset for dense shift field regression.

    Each sample returns::

        (shifted_image, target_shift_field, target_shift_vector,
         original_image, path)

    Args:
        root: ImageFolder-layout directory of aligned images.
        img_height: Target height (must be divisible by num_tracks AND 32).
        img_width: Target width (must be divisible by 32).
        num_tracks: Number of sensor tracks (default 22).
        max_shift: Maximum per-track pixel shift (default 15).
        augment: Apply extra augmentations (brightness, noise).
    """

    def __init__(self, root, img_height=704, img_width=480,
                 num_tracks=22, max_shift=15, augment=False):
        super().__init__(root)
        self.img_height = img_height
        self.img_width = img_width
        self.num_tracks = num_tracks
        self.max_shift = max_shift
        self.augment = augment
        self.track_height = img_height // num_tracks

    def __getitem__(self, index):
        path, _label = self.samples[index]

        # Load grayscale + resize
        image = Image.open(path).convert('L')
        image = image.resize((self.img_width, self.img_height), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.unsqueeze(0)  # (1, H, W)

        if self.augment:
            image = self._apply_augmentations(image)

        original_image = image.clone()

        # Generate random per-track shifts
        shift_vector = torch.FloatTensor(self.num_tracks).uniform_(
            -self.max_shift, self.max_shift,
        ).round()

        # Apply shifts → broken (uncorrelated) input
        shifted_image = apply_track_shifts(image, shift_vector, self.num_tracks)

        # Build dense GT shift field: inverse shifts expanded to (1, H, W)
        target_shift_vector = -shift_vector
        target_shift_field = self._vector_to_field(target_shift_vector)

        return (
            shifted_image,          # Model input       (1, H, W)
            target_shift_field,     # Dense GT           (1, H, W)
            target_shift_vector,    # Per-track GT       (num_tracks,)
            original_image,         # Aligned GT         (1, H, W)
            path,                   # File path          (str)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vector_to_field(self, shift_vector):
        """Expand a (num_tracks,) shift vector to a dense (1, H, W) field."""
        field = torch.zeros(1, self.img_height, self.img_width)
        for i in range(self.num_tracks):
            start = i * self.track_height
            end = start + self.track_height
            field[0, start:end, :] = shift_vector[i]
        return field

    @staticmethod
    def _apply_augmentations(image):
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            image = (image * factor).clamp(0, 1)
        if random.random() < 0.2:
            noise = torch.randn_like(image) * 0.02
            image = (image + noise).clamp(0, 1)
        return image
