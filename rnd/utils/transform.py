import cv2
import numpy as np
from PIL import Image


def resize_to_224(x: np.ndarray) -> np.ndarray:
    return np.stack([cv2.resize(xi, (224, 224), interpolation=cv2.INTER_AREA) for xi in x], axis=0)




def resize_to_224_pil(x: np.ndarray) -> np.ndarray:
    """
    Resize a numpy array of images to (224, 224) using PIL.

    Args:
        x (np.ndarray): Input array of shape (N, H, W) or (N, H, W, C)

    Returns:
        np.ndarray: Resized array of shape (N, 224, 224) or (N, 224, 224, C)
    """
    resized = []
    for xi in x:
        if xi.ndim == 2:  # grayscale (H, W)
            img = Image.fromarray((xi * 255).astype(np.uint8))  # scale back to 0–255
            img_resized = img.resize((224, 224), Image.BILINEAR)
            resized.append(np.array(img_resized) / 255.0)  # normalize back if needed
        else:  # RGB (H, W, C)
            img = Image.fromarray((xi * 255).astype(np.uint8))
            img_resized = img.resize((224, 224), Image.BILINEAR)
            resized.append(np.array(img_resized) / 255.0)

    return np.stack(resized, axis=0)
