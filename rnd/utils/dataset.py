import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class NumpyImageFolder(Dataset):
    """
    Custom Dataset to load grayscale 2D .npy images from class-labeled subfolders (e.g., pos/, neg/).
    """

    def __init__(self, root_dir, transform=None, debug=False):
        self.samples = []
        self.debug = debug
        self.transform = transform
        root = Path(root_dir)
        for label, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = root / class_name
            for file in class_dir.glob("*.npy"):
                self.samples.append((file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.load(path)  # expected shape: (H, W)
        if self.transform:
            img = self.transform(img)
        if self.debug:
            return img, label, path.name
        return img, label
