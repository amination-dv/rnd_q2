import numpy as np
import random


class RandomHorizontalRoll:
    """
    Apply a random horizontal roll (circular shift) to a 2D NumPy image.
    This maintains all pixel values by wrapping overflow around to the other side.

    Args:
        max_shift (int): Maximum number of pixels to shift left or right.
    """

    def __init__(self, max_shift: int):
        self.max_shift = max_shift

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(img)}")
        if img.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {img.shape}")

        # Random horizontal shift between [-max_shift, +max_shift]
        shift_val = random.randint(-self.max_shift, self.max_shift)

        # Roll along axis=1 (horizontal)
        return np.roll(img, shift=shift_val, axis=1)


class RandomJitter:
    """
    Apply a random horizontal roll (circular shift) to each row of a 2D NumPy image independently.
    Each row is shifted by a different random value within the range [-max_shift, max_shift].

    Args:
        max_shift (int): Maximum number of pixels to shift left or right per row.
    """

    def __init__(self, max_jitter: int):
        self.max_jitter = max_jitter

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(img)}")
        if img.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {img.shape}")

        output = np.empty_like(img)
        for i in range(img.shape[0]):
            shift_val = random.randint(-self.max_jitter, self.max_jitter)
            output[i] = np.roll(img[i], shift=shift_val)
        return output


class RandomRowSwap:
    """
    Randomly swaps rows in a 2D NumPy image.

    Args:
        num_swaps (int): Number of row pairs to swap.
        max_distance (int): Max offset between rows to swap.
    """

    def __init__(self, num_swaps: int = 5):
        self.num_swaps = num_swaps

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(img)}")
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {img.shape}")

        h = img.shape[0]
        for _ in range(self.num_swaps):
            row1, row2 = random.sample(range(h), 2)
            img[[row1, row2]] = img[[row2, row1]]

        return img


class RandomApplyNp:
    """
    Apply a NumPy-based augmentation with a given probability.
    """

    def __init__(self, transform_fn, p=0.5):
        self.transform_fn = transform_fn
        self.p = p

    def __call__(self, img):
        return self.transform_fn(img) if random.random() < self.p else img
