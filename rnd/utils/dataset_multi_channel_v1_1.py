import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import json
import torch    
from statsmodels.tsa.seasonal import seasonal_decompose
Model_num_classes = 20 # 0: no dent, 1-20: dent classes
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

            for file in (class_dir / "patch_normalized").glob("*.npy"):
                if label == 0:
                    new_label = [0]
                else:
                    if "validation" in root_dir:
                        new_label = [1]
                    else:
                        label = json.loads(file.stem.split("_")[2])
                        new_label = [lab + 0 for lab in label]
                self.samples.append((file, new_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        #change lable to multi-label
        m_label = [0] * Model_num_classes
        if label == [0]:
            pass
        else:
            for lab in label:
                m_label[lab] = 1
        label = torch.tensor(m_label)
        img_patch = np.load(path)  # expected shape: (H, W)
        img_std = np.load(Path(str(path).replace("patch", "std")))  # expected shape: (H, W)
        img_patch_trend = seasonal_decompose(img_patch.T, model='additive', period=30, extrapolate_trend=2).trend.T
        img = np.stack((img_patch, img_patch_trend, img_std), axis=0)  # shape: (3, H, W)
        if self.transform:
            img = self.transform(img)
        if self.debug:
            return img, label, path.name
        return img, label
