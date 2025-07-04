import cv2
import numpy as np


def resize_to_224(x: np.ndarray) -> np.ndarray:
    return cv2.resize(x, (224, 224), interpolation=cv2.INTER_LINEAR)
