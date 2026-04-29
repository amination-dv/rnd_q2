from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Default multi-label size (one logit per circumferential track index 0 .. num_tracks-1)
DEFAULT_NUM_TRACKS = 22


def _std_npy_for_patch(patch_npy_path: Path) -> Path:
    """Sibling std array for a patch_normalized/*.npy path."""
    return Path(str(patch_npy_path).replace("patch_normalized", "std_normalized"))


def _first_bracket_list(stem: str):
    """Parse first `[...]` in stem as a Python list (e.g. `['008','009']`)."""
    m = re.search(r"\[.*?\]", stem)
    if not m:
        return None
    try:
        v = ast.literal_eval(m.group(0))
        if isinstance(v, list):
            return [int(x) for x in v]
    except (ValueError, TypeError, SyntaxError):
        pass
    return None


class NumpyImageFolder(Dataset):
    """
    Dataset for ml_data_v4 layout:

    - **pos/** (positives): flat
        ``<root>/pos/patch_normalized/*.npy`` and matching
        ``<root>/pos/std_normalized/*.npy``

    - **neg/** (negatives): nested by source and inspection id
        ``<root>/neg/girth_weld/<iid>/patch_normalized/*.npy``
        ``<root>/neg/random/<iid>/patch_normalized/*.npy``
        (and sibling ``std_normalized/`` folders)

    Only ``neg`` and ``pos`` under ``root_dir`` are used (not ``dent``, ``fp``, etc.).

    Args:
        num_tracks: Length of multi-hot label (must match model output dim). Default ``DEFAULT_NUM_TRACKS``.
    """

    def __init__(self, root_dir, transform=None, debug=False, num_tracks: int | None = None):
        self.samples = []
        self.debug = debug
        self.transform = transform
        self.num_tracks = int(num_tracks) if num_tracks is not None else DEFAULT_NUM_TRACKS
        if self.num_tracks < 1:
            raise ValueError("num_tracks must be >= 1")
        root = Path(root_dir)

        neg_root = root / "neg"
        pos_root = root / "pos"
        skipped_unpaired = 0

        # --- Negatives: all patch .npy under neg/**/patch_normalized/
        # Empty list = "no tracks active". Don't use [0] as the all-negative sentinel —
        # track index 0 is a valid GT label, and a positive whose stem parses to [0]
        # would otherwise collide with the negative sentinel and be silently mislabeled.
        if neg_root.is_dir():
            for file in sorted(neg_root.glob("**/patch_normalized/*.npy")):
                if not _std_npy_for_patch(file).exists():
                    skipped_unpaired += 1
                    continue
                self.samples.append((file, []))

        # --- Positives: flat pos/patch_normalized/
        if pos_root.is_dir():
            pos_patch = pos_root / "patch_normalized"
            if pos_patch.is_dir():
                for file in sorted(pos_patch.glob("*.npy")):
                    if not _std_npy_for_patch(file).exists():
                        skipped_unpaired += 1
                        continue
                    parsed = _first_bracket_list(file.stem)
                    if parsed is not None and len(parsed) > 0:
                        new_label = parsed
                    else:
                        raise ValueError(f"No track list found in {file.stem}")
                    self.samples.append((file, new_label))

    def __len__(self):
        return len(self.samples)

    def _build_multi_hot(self, raw_label: list[int]) -> torch.Tensor:
        """Convert a stored raw label (list[int]) to a (num_tracks,) float32 multi-hot tensor.

        Convention: an empty list (``[]``) means "all-negative" (no track active).
        Any non-empty list lists the active track indices; ``[0]`` legitimately
        means track index 0 is active.
        """
        m = torch.zeros(self.num_tracks, dtype=torch.float32)
        for lab in raw_label:
            if 0 <= lab < self.num_tracks:
                m[lab] = 1.0
        return m

    def label_at(self, idx: int) -> torch.Tensor:
        """Return the multi-hot label for sample ``idx`` WITHOUT loading any .npy file.

        Use this for label-only loops (class balancing, pos_weight estimation,
        train/val statistics) to avoid the per-sample np.load that __getitem__ does.
        """
        _, raw_label = self.samples[idx]
        return self._build_multi_hot(raw_label)

    def __getitem__(self, idx):
        path, raw_label = self.samples[idx]
        label_tensor = self._build_multi_hot(raw_label)

        img_patch = np.load(path)
        std_path = _std_npy_for_patch(path)
        if not std_path.exists():
            raise FileNotFoundError(f"Missing std pair for {path}: {std_path}")
        img_std = np.load(std_path)
        img = np.stack((img_patch, img_std, img_std), axis=0)

        if self.transform:
            img = self.transform(img)
        if self.debug:
            # Full path string so consumers can locate the paired std_normalized file
            # without re-indexing back into self.samples.
            return img, label_tensor, str(path)
        return img, label_tensor
