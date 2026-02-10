"""
ILI Dataset for CorrelationFields training.

Loads aligned (correlated) tubeview images and generates self-supervised
alignment training data by artificially shifting individual sensor tracks.

Returns a per-track shift vector (num_tracks,) as the regression target.
"""

import random

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from utils import apply_track_shifts


class ILIDataset(ImageFolder):
    """
    ILI dataset for per-track shift regression.

    Each sample returns::

        (shifted_image, target_shift_vector, original_image, path)

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

        # Target: inverse shifts to undo the corruption
        target_shift_vector = -shift_vector

        return (
            shifted_image,          # Model input       (1, H, W)
            target_shift_vector,    # Per-track GT       (num_tracks,)
            original_image,         # Aligned GT         (1, H, W)
            path,                   # File path          (str)
        )

    @staticmethod
    def _apply_augmentations(image):
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            image = (image * factor).clamp(0, 1)
        if random.random() < 0.2:
            noise = torch.randn_like(image) * 0.02
            image = (image + noise).clamp(0, 1)
        return image
