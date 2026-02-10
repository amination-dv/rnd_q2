"""
Utility functions for the ILI component detection pipeline.
"""

import os
import re
import random
from glob import glob
from datetime import datetime

import cv2
import yaml
import numpy as np
import torch
import wandb
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def current_timestamp():
    return datetime.now().strftime('%y-%m-%d-%H-%M-%S')


def createDirectory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def get_last_checkpoint(checkpoint_dir, return_best=False, keyword='best'):
    """
    Get the last or best checkpoint from a directory.

    For Detectron2, 'best' defaults to ``model_final.pth`` and 'last'
    returns the checkpoint with the highest iteration number.
    """
    checkpoint_paths = glob(f"{checkpoint_dir}/*.pth")
    if not checkpoint_paths:
        return None

    def _parse_iter(ckpt):
        match = re.search(r'model_(\d+)\.pth', ckpt)
        return int(match.group(1)) if match else 0

    if return_best:
        # Prefer model_final.pth
        final = os.path.join(checkpoint_dir, 'model_final.pth')
        if os.path.exists(final):
            return final
        # Fall back to highest-iteration checkpoint
        best = [c for c in checkpoint_paths if keyword in os.path.basename(c)]
        if best:
            return sorted(best, key=_parse_iter, reverse=True)[0]
        return sorted(checkpoint_paths, key=_parse_iter, reverse=True)[0]

    # Return last (highest iter)
    final = os.path.join(checkpoint_dir, 'model_final.pth')
    if os.path.exists(final):
        return final
    sorted_ckpts = sorted(checkpoint_paths, key=_parse_iter, reverse=True)
    return sorted_ckpts[0] if sorted_ckpts else None


# ---------------------------------------------------------------------------
# Visualisation / WandB logging
# ---------------------------------------------------------------------------

def log_predictions_to_wandb(outputs, dataset_dict, metadata, phase="train"):
    """
    Create a side-by-side prediction vs ground-truth visualisation for WandB.

    Args:
        outputs: Detectron2 model output dict.
        dataset_dict: Single dataset entry with annotations.
        metadata: MetadataCatalog metadata.
        phase: Label prefix (train / val / test).

    Returns:
        Dict suitable for ``wandb.log()``.
    """
    img = cv2.imread(dataset_dict["file_name"])

    # Predictions
    v_pred = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
    pred_img = out_pred.get_image()[:, :, ::-1]

    # Ground truth
    v_gt = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out_gt = v_gt.draw_dataset_dict(dataset_dict)
    gt_img = out_gt.get_image()[:, :, ::-1]

    concat = np.concatenate((pred_img, gt_img), axis=1)

    return {
        f"{phase}/predictions": wandb.Image(
            concat, caption="Left: Predictions | Right: Ground Truth",
        ),
    }


def visualize_sample(dataset_dict, metadata, save_path=None):
    """Visualise a single dataset entry and optionally save to disk."""
    img = cv2.imread(dataset_dict["file_name"])
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = v.draw_dataset_dict(dataset_dict)
    vis = out.get_image()[:, :, ::-1]
    if save_path:
        cv2.imwrite(save_path, vis)
    return vis
