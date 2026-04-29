import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import json
from statsmodels.tsa.seasonal import seasonal_decompose

class NumpyImageFolder(Dataset):
    """
    Custom Dataset to load grayscale 2D .npy images from class-labeled subfolders (e.g., pos/, neg/).
    """

    def __init__(self, root_dir, transform=None, debug=False):
        self.samples = []
        self.debug = debug
        self.transform = transform
        root = Path(root_dir)
        # #read json file
        # with open(root / "dent_tracks_indices.json", "r") as json_file:
        #     metadata = json.load(json_file)

        for label, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = root / class_name

            for file in (class_dir ).glob("*.npy"):
                if label == 0:
                    pass
                else:
                    label = int(file.stem.split("_")[2]) + 1
                self.samples.append((file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        X= np.load(path)  # expected shape: (H, W)
        mean_seq = np.mean(X, axis=1, keepdims=True)
        std_seq  = np.std(X, axis=1, keepdims=True)
        std_seq[std_seq == 0] = 1.0  # Avoid division by zero
        img = (X - mean_seq) / std_seq

        #img_std = np.load(Path(str(path).replace("patch_normalized", "std_normalized")))  # expected shape: (H, W)
        #img_patch_trend = seasonal_d`ecompose(img_patch.T, model='additive', period=30, extrapolate_trend=2).trend.T
        #img = np.stack((img_patch, img_patch_trend, img_std), axis=0)  # shape: (3, H, W)
        # shape: (1, H, W) # shape: (1, H, W)
        if self.transform:
            img = self.transform(img)
        if self.debug:
            return img, label, path.name
        return img, label
