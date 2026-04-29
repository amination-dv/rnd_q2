import io
import os
import argparse
import re
from collections import defaultdict

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

from torchcam import methods
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rnd.utils.modeling import get_model
from rnd.utils.data_utils import set_num_tracks
from rnd.utils.transform import resize_to_224
from rnd.utils.dataset_multi_channel_v4 import NumpyImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import random

# Filename stem suffixes produced by v4 augmentation (see augment_v4.ipynb, dataset_v4.ipynb)
_ROLL_SUFFIX_RE = re.compile(r"_vroll_-?\d+_hroll_-?\d+$")


def strip_augment_suffixes(stem: str) -> str:
    """Remove trailing roll / noise / scale augmentation tokens (repeat until stable)."""
    s = stem
    while True:
        m = _ROLL_SUFFIX_RE.search(s)
        if m:
            s = s[: m.start()]
            continue
        if s.endswith("_aug_noise"):
            s = s[: -len("_aug_noise")]
            continue
        scale_m = re.search(r"_aug_scale_.+$", s)
        if scale_m:
            s = s[: scale_m.start()]
            continue
        break
    return s


def group_id_for_path(path: Path | str) -> str:
    """
    Group ID so the original patch and all its augmentations land in the same train/val split.

    - **pos/** (flat ``pos/patch_normalized/*.npy``): strip aug suffixes, then use defect id
      ``d<digits>`` prefix when the stem starts with that pattern (same physical defect across
      roll variants with different bracket labels).
    - **neg/** (``neg/<category>/<iid>/patch_normalized/*.npy``): category + iid + stripped stem.
    """
    p = Path(path)
    parts = p.parts
    stem = p.stem

    if "pos" in parts:
        stripped = strip_augment_suffixes(stem)
        base, _, _ = stripped.partition("_")
        if (
            len(base) >= 2
            and base[0] == "d"
            and base[1:].isdigit()
        ):
            return f"pos:{base}"
        return f"pos:{stripped}"

    if "neg" in parts:
        stripped = strip_augment_suffixes(stem)
        try:
            ni = parts.index("neg")
        except ValueError:
            return f"neg:unknown::unknown:{stripped}"
        category = parts[ni + 1] if ni + 1 < len(parts) else "unknown"
        iid = parts[ni + 2] if ni + 2 < len(parts) else "unknown"
        return f"neg:{category}:{iid}:{stripped}"

    return f"other:{p.as_posix()}"


def grouped_train_val_indices(
    full_ds: NumpyImageFolder,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, int]]:
    """
    Split by group_id so no group appears in both train and val.

    Returns (train_indices, val_indices, stats dict).
    If there is only one group, falls back to a random **index** split (same as legacy leakage
    risk); log a warning.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (path_p, _) in enumerate(full_ds.samples):
        p = Path(path_p) if not isinstance(path_p, Path) else path_p
        groups[group_id_for_path(p)].append(i)

    gids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(gids)

    n = len(full_ds)
    vf = min(max(val_fraction, 0.0), 1.0)
    stats: dict[str, int] = {
        "num_groups": len(gids),
        "largest_group": max((len(v) for v in groups.values()), default=0),
    }

    if len(gids) < 2:
        print(
            f"[Split] grouped: only {len(gids)} distinct group_id(s); "
            "falling back to random index split (consider --split-random for clarity)."
        )
        rng2 = random.Random(seed)
        order = list(range(n))
        rng2.shuffle(order)
        n_val = max(1, min(n - 1, int(round(vf * n)))) if n > 1 else 0
        val_idx = sorted(order[:n_val])
        train_idx = sorted(order[n_val:])
        return train_idx, val_idx, stats

    n_val_groups = int(round(vf * len(gids)))
    n_val_groups = max(1, min(n_val_groups, len(gids) - 1))

    val_gid_set = set(gids[:n_val_groups])
    train_gid_set = set(gids[n_val_groups:])

    train_idx = sorted(i for gid in train_gid_set for i in groups[gid])
    val_idx = sorted(i for gid in val_gid_set for i in groups[gid])
    stats["val_groups"] = n_val_groups
    stats["train_groups"] = len(gids) - n_val_groups
    return train_idx, val_idx, stats


def make_train_val_subsets(
    full_ds: NumpyImageFolder,
    val_fraction: float,
    split_seed: int,
    split_random: bool,
):
    """Build train/val Subsets; grouped by default, random index split if split_random."""
    n = len(full_ds)
    vf = min(max(val_fraction, 0.0), 1.0)
    if n <= 1:
        train_ds = torch.utils.data.Subset(full_ds, list(range(n)))
        val_ds = torch.utils.data.Subset(full_ds, [])
        return train_ds, val_ds, "trivial", {"num_groups": 0}

    if split_random:
        train_size = int((1.0 - vf) * n)
        train_size = max(1, min(n - 1, train_size))
        val_size = n - train_size
        g = torch.Generator().manual_seed(split_seed)
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size], generator=g
        )
        return train_ds, val_ds, "random", {"num_groups": 0}

    train_idx, val_idx, gstats = grouped_train_val_indices(full_ds, vf, split_seed)
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    return train_ds, val_ds, "grouped", gstats


def compute_per_track_pos_weight(
    full_ds: NumpyImageFolder,
    train_indices: list[int],
    num_tracks: int,
    eps: float = 1e-6,
    cap: float | None = 20.0,
) -> torch.Tensor:
    """
    For each track k: pos_weight[k] ≈ N_neg_k / N_pos_k on the training subset.

    Reads labels via full_ds.label_at(i) — no .npy IO.
    """
    pos_counts = torch.zeros(num_tracks, dtype=torch.float64)
    neg_counts = torch.zeros(num_tracks, dtype=torch.float64)
    for i in train_indices:
        y = full_ds.label_at(i).to(torch.float64)
        pos_counts += y
        neg_counts += 1.0 - y
    pw = neg_counts / (pos_counts + eps)
    if cap is not None and cap > 0:
        pw = torch.clamp(pw, max=float(cap))
    return pw.float()


def count_neg_pos_images(full_ds: NumpyImageFolder, train_indices: list[int]) -> tuple[int, int]:
    """All-negative vs any-positive sample counts (image-level). No .npy IO."""
    n_neg = n_pos = 0
    for i in train_indices:
        if float(full_ds.label_at(i).sum()) == 0.0:
            n_neg += 1
        else:
            n_pos += 1
    return n_neg, n_pos


def undersample_train_balanced_pos_neg(
    full_ds: NumpyImageFolder,
    train_indices: list[int],
    seed: int,
) -> list[int]:
    """
    Random undersample so training has equal counts of:
    - **negative** images (all-zero multi-hot), and
    - **positive** images (at least one track == 1).

    Uses min(n_pos, n_neg) per class. Returns a new shuffled index list (full_ds indices).
    Reads labels via full_ds.label_at(i) — no .npy IO.
    """
    pos_idx: list[int] = []
    neg_idx: list[int] = []
    for i in train_indices:
        if float(full_ds.label_at(i).sum()) == 0.0:
            neg_idx.append(i)
        else:
            pos_idx.append(i)
    n = min(len(pos_idx), len(neg_idx))
    if n == 0:
        return list(train_indices)
    rng = random.Random(seed)
    pos_sel = rng.sample(pos_idx, n)
    neg_sel = rng.sample(neg_idx, n)
    balanced = pos_sel + neg_sel
    rng.shuffle(balanced)
    return balanced


class CombinedMultiLabelBCELoss(nn.Module):
    """
    (1) BCE with logits + optional per-class pos_weight (PyTorch multi-label).
    (2) Per-sample weight: w_neg if all labels 0 else w_pos.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None,
        w_neg: float,
        w_pos: float,
        use_pos_weight: bool = True,
    ):
        super().__init__()
        self.use_pos_weight = bool(use_pos_weight) and pos_weight is not None
        if self.use_pos_weight:
            self.register_buffer("pos_weight", pos_weight.clone().float())
        self.w_neg = float(w_neg)
        self.w_pos = float(w_pos)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight if self.use_pos_weight else None
        loss_elem = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        is_neg_sample = (targets.sum(dim=1) == 0).float().unsqueeze(1)
        sample_w = self.w_neg * is_neg_sample + self.w_pos * (1.0 - is_neg_sample)
        return (loss_elem * sample_w).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure that CUDA operations are deterministic if using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # # slower but more reproducibility
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch: int = 0,
    global_step_start: int = 0,
    wandb_log_every: int = 1000,
    print_every: int = 100,
):
    """Train one epoch.

    Streams training loss to W&B every ``wandb_log_every`` batches (and at the
    end of the epoch). The global step counter increments per batch and is
    shared with end-of-epoch validation logging, so the W&B X-axis is a single
    monotonic timeline of optimizer steps.

    Returns ``(epoch_avg_loss, global_step_after_epoch)``.
    """
    model.train()
    running_loss = 0.0
    total = len(train_loader)
    global_step = int(global_step_start)
    log_to_wandb = "wandb" in globals() and wandb.run is not None

    for batch_idx, (inputs, labels, _) in enumerate(train_loader, 1):
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        raw_outputs = model(inputs)
        loss = criterion(raw_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step += 1

        if batch_idx % print_every == 0 or batch_idx == total:
            avg = running_loss / batch_idx
            print(
                f"[Train] e{epoch + 1} batch {batch_idx}/{total} "
                f"step={global_step} batch_loss={loss.item():.4f} "
                f"avg_loss={avg:.4f}"
            )

        if log_to_wandb and (batch_idx % wandb_log_every == 0 or batch_idx == total):
            avg = running_loss / batch_idx
            wandb.log(
                {
                    "train/batch_loss": float(loss.item()),
                    "train/running_avg_loss": float(avg),
                    "train/batch_idx": int(batch_idx),
                    "train/batches_in_epoch": int(total),
                    "train/epoch_progress": float(batch_idx) / float(total),
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

    return running_loss / total, global_step


def validate(model, val_loader, criterion, device, threshold=0.5):
    """
    Validate the model. Requires the dataset to be in debug=True mode so
    each batch yields (inputs, labels, paths). Path order is whatever the
    loader emits — no assumption about Subset indices or shuffle state.
    """
    model.eval()
    val_loss = 0.0

    all_val_labels = []      # will collect arrays shape [C]
    all_val_logits = []      # raw logits
    all_paths: list[str] = []

    with torch.inference_mode():
        for inputs, labels, paths in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)        # multi-label targets

            raw_outputs = model(inputs)               # [B, C]
            loss = criterion(raw_outputs, labels)
            val_loss += loss.item()

            all_val_labels.append(labels.cpu().numpy())
            all_val_logits.append(raw_outputs.cpu().numpy())
            all_paths.extend(paths)

    # Stack
    all_val_labels = np.vstack(all_val_labels)    # [N, C]
    all_val_logits = np.vstack(all_val_logits)    # [N, C]

    # Probabilities
    all_val_probs = 1 / (1 + np.exp(-all_val_logits))

    # Binarize with threshold per class
    val_preds = (all_val_probs >= threshold).astype(int)


    precision_macro = precision_score(all_val_labels, val_preds, average='macro', zero_division=0)
    recall_macro    = recall_score(all_val_labels, val_preds, average='macro', zero_division=0)
    f1_macro        = f1_score(all_val_labels, val_preds, average='macro', zero_division=0)

    # Also micro (global)
    precision_micro = precision_score(all_val_labels, val_preds, average='micro', zero_division=0)
    recall_micro    = recall_score(all_val_labels, val_preds, average='micro', zero_division=0)
    f1_micro        = f1_score(all_val_labels, val_preds, average='micro', zero_division=0)

    # Per-class precision/recall (array length C)
    precision_per_class = precision_score(all_val_labels, val_preds, average=None, zero_division=0)
    recall_per_class    = recall_score(all_val_labels, val_preds, average=None, zero_division=0)

    # Sample-level “misclassified” = any mismatch between label vector & prediction
    misclassified = []
    mism_mask = (val_preds != all_val_labels).any(axis=1)
    for i, bad in enumerate(mism_mask):
        if bad:
            misclassified.append(str(all_paths[i]))

    results = {
        'loss': val_loss / len(val_loader),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'labels': all_val_labels.tolist(),
        'probabilities': all_val_probs.tolist(),
        'predictions': val_preds.tolist(),
        'paths': all_paths,
        'misclassified': misclassified,
        'threshold': threshold
    }
    return results


def sample_fp_fn_indices(
    labels: np.ndarray,
    preds: np.ndarray,
    max_each: int = 20,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Sample-level FP/FN for multi-label validation.
    - Positive sample: any ground-truth class is 1.
    - Predicted positive: any predicted class is 1 (after threshold).
    FN: GT positive but model predicted all-negative.
    FP: GT negative but model predicted at least one positive.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    gt_pos = labels.sum(axis=1) > 0
    pred_pos = preds.sum(axis=1) > 0
    fn_idx = np.where(gt_pos & ~pred_pos)[0]
    fp_idx = np.where(~gt_pos & pred_pos)[0]
    rng = np.random.default_rng(seed)
    if len(fn_idx) > max_each:
        fn_idx = rng.choice(fn_idx, size=max_each, replace=False)
    if len(fp_idx) > max_each:
        fp_idx = rng.choice(fp_idx, size=max_each, replace=False)
    return fp_idx.tolist(), fn_idx.tolist()


def _array_to_inferno_hwc_rgb(arr_2d: np.ndarray) -> np.ndarray:
    """2D array -> resize to 224-like pipeline -> HWC uint8 inferno RGB."""
    x = np.stack([arr_2d.astype(np.float64)], axis=0)
    x = resize_to_224(x)[0]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (cm.inferno(x)[:, :, :3] * 255).astype(np.uint8)


def _load_patch_or_std_array(npy_patch_path: str, channel: str) -> np.ndarray:
    """Load 2D array from patch_normalized path or paired std_normalized.

    Raises FileNotFoundError if channel='std' but the paired std_normalized
    file is missing (no silent fallback to the patch — that produces
    misleading FP/FN panels in W&B).
    """
    p = Path(npy_patch_path)
    if channel == "patch":
        return np.load(p).astype(np.float64)
    if channel != "std":
        raise ValueError(f"Unknown channel {channel!r} (expected 'patch' or 'std')")
    std_path = Path(str(p).replace("patch_normalized", "std_normalized"))
    if not std_path.exists():
        raise FileNotFoundError(
            f"Missing std pair for {p}: expected {std_path}"
        )
    return np.load(std_path).astype(np.float64)


def npy_path_to_rgb_preview(npy_patch_path: str, channel: str = "std") -> np.ndarray:
    """
    Load patch or std .npy (path is under patch_normalized; std is sibling std_normalized).
    channel: \"patch\" = original patch array; \"std\" = paired std (dataset-style).
    """
    return _array_to_inferno_hwc_rgb(_load_patch_or_std_array(npy_patch_path, channel))


def npy_to_png_wandb_image(
    npy_patch_path: str,
    channel: str,
    caption: str,
    gt_vec: np.ndarray | None = None,
    pred_vec: np.ndarray | None = None,
    prob_vec: np.ndarray | None = None,
    threshold: float = 0.5,
) -> wandb.Image | None:
    """Render an .npy as a wandb.Image.

    If ``gt_vec``, ``pred_vec``, and ``prob_vec`` are all provided, the figure
    becomes a two-panel layout: the std/patch heatmap on the left, and a
    horizontal per-track probability bar chart on the right with each bar
    colored by its kind:

    * ``TP`` green   (gt=1, pred=1)
    * ``FN`` blue    (gt=1, pred=0)
    * ``FP`` red     (gt=0, pred=1)
    * ``TN`` gray    (gt=0, pred=0)

    A dashed vertical line marks the threshold. The same prediction info is
    burned into the figure title so it survives W&B's caption truncation.

    Returns ``None`` on any rendering failure (so the caller can skip).
    """
    try:
        arr = _load_patch_or_std_array(npy_patch_path, channel)
        x = np.stack([arr], axis=0)
        x = resize_to_224(x)[0]

        has_preds = (
            gt_vec is not None and pred_vec is not None and prob_vec is not None
        )

        if has_preds:
            gt = np.asarray(gt_vec).astype(int).ravel()
            pr = np.asarray(pred_vec).astype(int).ravel()
            pv = np.asarray(prob_vec, dtype=float).ravel()
            n = int(min(len(gt), len(pr), len(pv)))
            gt, pr, pv = gt[:n], pr[:n], pv[:n]

            fig, (ax_img, ax_bar) = plt.subplots(
                1, 2, figsize=(10, 5), dpi=120,
                gridspec_kw={"width_ratios": [1, 1]},
            )
            ax_img.imshow(
                x.T,
                cmap="inferno",
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            ax_img.axis("off")

            # Color per track by TP/FN/FP/TN
            colors = []
            for k in range(n):
                if gt[k] and pr[k]:
                    colors.append("#2ca02c")   # TP
                elif gt[k] and not pr[k]:
                    colors.append("#1f77b4")   # FN
                elif (not gt[k]) and pr[k]:
                    colors.append("#d62728")   # FP
                else:
                    colors.append("#888888")   # TN
            y = np.arange(n)
            ax_bar.barh(y, pv, color=colors, edgecolor="black", linewidth=0.3)
            ax_bar.axvline(threshold, color="black", linestyle="--", linewidth=0.7, alpha=0.7)
            ax_bar.set_xlim(0.0, 1.0)
            ax_bar.set_yticks(y)
            ax_bar.set_yticklabels([f"t{k:02d}" for k in range(n)], fontsize=6)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("prob (sigmoid)")
            gt_tracks = np.where(gt > 0)[0].tolist()
            pred_tracks = np.where(pr > 0)[0].tolist()
            ax_bar.set_title(
                f"GT={gt_tracks}  pred={pred_tracks}  thr={threshold:.2f}",
                fontsize=9,
            )
            # Annotate bars with their prob value if non-trivial
            for k in range(n):
                if pv[k] >= 0.05 or gt[k] or pr[k]:
                    ax_bar.text(
                        min(pv[k] + 0.02, 0.98),
                        k,
                        f"{pv[k]:.2f}",
                        va="center", fontsize=5, color="black",
                    )

            # Burn the caption into the figure title so it survives caption truncation.
            fig.suptitle(caption, fontsize=8, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
        else:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
            ax.imshow(
                x.T,
                cmap="inferno",
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            ax.axis("off")
            fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")
        return wandb.Image(pil_img, caption=caption)
    except Exception:
        plt.close("all")
        return None


def _fp_fn_pred_text(
    row_idx: int,
    labels_arr: np.ndarray,
    preds_arr: np.ndarray,
    probs_arr: np.ndarray,
) -> str:
    """One-line GT vs model (binary + probs at predicted tracks)."""
    gt = np.asarray(labels_arr[row_idx]).astype(int).ravel()
    pr = np.asarray(preds_arr[row_idx]).astype(int).ravel()
    pv = np.asarray(probs_arr[row_idx], dtype=float).ravel()
    gt_tr = np.where(gt > 0)[0].tolist()
    pr_tr = np.where(pr > 0)[0].tolist()
    if len(pr_tr):
        pr_p = [round(float(pv[t]), 4) for t in pr_tr]
        pred_s = f"pred_tracks={pr_tr} prob@pred={pr_p}"
    else:
        pred_s = "pred_tracks=[]"
    max_p = float(pv.max()) if pv.size else 0.0
    return f"GT_tracks={gt_tr} | {pred_s} | max_prob={max_p:.4f}"


def log_wandb_fp_fn_samples(
    epoch: int,
    val_results: dict,
    max_each: int = 20,
    threshold: float = 0.5,
    step: int | None = None,
) -> None:
    """Log FP/FN std images with the per-track prediction probabilities baked
    into a side panel of each image (no separate wandb.Table).

    Each W&B image is a two-panel figure: std heatmap + per-track probability
    bar chart with TP / FN / FP / TN color coding and a threshold line. The
    full GT/pred summary is also in the image caption.

    If ``step`` is None, falls back to ``epoch + 1``.
    """
    if "wandb" not in globals() or wandb.run is None:
        return
    labels = np.array(val_results["labels"])
    preds = np.array(val_results["predictions"])
    probs = np.array(val_results["probabilities"])
    paths = val_results["paths"]
    fp_idx, fn_idx = sample_fp_fn_indices(
        labels, preds, max_each=max_each, seed=42 + int(epoch)
    )

    ep = epoch + 1
    s = int(step) if step is not None else ep
    log_payload: dict = {
        "val/fp_fn_counts/false_positives_logged": len(fp_idx),
        "val/fp_fn_counts/false_negatives_logged": len(fn_idx),
    }

    # Render only the std-normalized view (matches the dataset's std channel).
    # GT / pred / prob vectors get drawn as a side bar chart inside the image.
    for i, idx in enumerate(fp_idx):
        p = paths[idx]
        name = Path(p).name
        cap = f"FP e{ep} std {name} | {_fp_fn_pred_text(idx, labels, preds, probs)}"
        wimg = npy_to_png_wandb_image(
            p, "std", caption=cap,
            gt_vec=labels[idx], pred_vec=preds[idx], prob_vec=probs[idx],
            threshold=threshold,
        )
        if wimg is not None:
            log_payload[f"val/fp/std_{i:02d}"] = wimg

    for i, idx in enumerate(fn_idx):
        p = paths[idx]
        name = Path(p).name
        cap = f"FN e{ep} std {name} | {_fp_fn_pred_text(idx, labels, preds, probs)}"
        wimg = npy_to_png_wandb_image(
            p, "std", caption=cap,
            gt_vec=labels[idx], pred_vec=preds[idx], prob_vec=probs[idx],
            threshold=threshold,
        )
        if wimg is not None:
            log_payload[f"val/fn/std_{i:02d}"] = wimg

    wandb.log(log_payload, step=s)


def visualize_gradcam(model, val_loader, device, full_ds, val_ds, 
                     gradcam_dir, num_positive_samples=20):
    """Generate GradCAM visualizations for positive samples"""
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Initialize Grad-CAM extractor
    cam_extractor = methods.GradCAM(model, target_layer="base_model.layer4")
    
    # Process a subset of validation data for visualization
    model.eval()
    count = 0
    
    # Create iterators for both loaders
    val_iter = iter(val_loader)

    
    # Process batches until we get enough positive samples
    while count < num_positive_samples:
        try:
            inputs, labels, file_names = next(val_iter)
            inputs = inputs.to(device)

            # Process each image in the batch, but only if it's a positive sample
            for i in range(inputs.size(0)):
                # Multi-hot label: positive sample == any track == 1
                if labels[i].sum() <= 0:
                    continue
                if count >= num_positive_samples:
                    break

                label = labels[i]
                gt_tracks = label.nonzero(as_tuple=True)[0].cpu().tolist()

                # No-grad forward for predictions (used in titles only)
                with torch.no_grad():
                    pred_output = model(inputs[i].unsqueeze(0))
                    pred_probs = torch.sigmoid(pred_output)[0].cpu().numpy()
                    pred_tracks = (pred_probs >= 0.5).nonzero()[0].tolist()

                # Per-class CAMs: GT-active tracks first, then any pred-only tracks
                # so we also see what the model is reacting to on FPs.
                cam_targets: list[int] = list(gt_tracks)
                for t in pred_tracks:
                    if t not in cam_targets:
                        cam_targets.append(int(t))
                if not cam_targets:
                    continue

                per_track_cams: dict[int, np.ndarray] = {}
                for t in cam_targets:
                    # Fresh forward per target so each backward gets its own graph.
                    inp_t = inputs[i].unsqueeze(0).clone().detach().requires_grad_()
                    out_t = model(inp_t)
                    cam_t = cam_extractor(int(t), out_t)
                    per_track_cams[int(t)] = cam_t[0].squeeze().cpu().numpy()

                # Combined CAM: per-pixel max across all per-class CAMs
                # (torchcam returns normalized maps in [0,1], so max-pool is comparable).
                combined_cam = np.maximum.reduce(list(per_track_cams.values()))

                # Layout: [original, *per-track, combined]
                plt.style.use('dark_background')
                n_panels = 1 + len(per_track_cams) + 1
                ncols = min(n_panels, 4)
                nrows = int(np.ceil(n_panels / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
                axes_flat = np.array(axes).reshape(-1)

                axes_flat[0].imshow(
                    inputs[i, 2, :, :].cpu().detach().numpy(),
                    cmap="inferno", aspect="auto", interpolation="nearest",
                )
                axes_flat[0].set_title(f"Original\nGT: {gt_tracks}")
                axes_flat[0].axis('off')

                for k, t in enumerate(cam_targets, start=1):
                    cam_2d = per_track_cams[t]
                    is_gt = t in gt_tracks
                    is_pred = t in pred_tracks
                    if is_gt and is_pred:
                        kind = "TP"
                    elif is_gt:
                        kind = "FN"
                    else:
                        kind = "FP"
                    axes_flat[k].imshow(
                        cam_2d, cmap='jet', aspect="auto", interpolation="bilinear",
                    )
                    axes_flat[k].set_title(
                        f"Track {t} ({kind})\np={float(pred_probs[t]):.3f}"
                    )
                    axes_flat[k].axis('off')

                combined_idx = 1 + len(per_track_cams)
                im_c = axes_flat[combined_idx].imshow(
                    combined_cam, cmap='jet', aspect="auto", interpolation="bilinear",
                )
                axes_flat[combined_idx].set_title(
                    f"Combined (max)\nPred: {pred_tracks}"
                )
                axes_flat[combined_idx].axis('off')
                plt.colorbar(im_c, ax=axes_flat[combined_idx], fraction=0.046, pad=0.04)

                for k in range(combined_idx + 1, len(axes_flat)):
                    axes_flat[k].axis('off')

                # file_names[i] is the full path string (dataset debug=True);
                # use the stem for filenames/titles to avoid path separators.
                sample_name = Path(file_names[i]).stem
                plt.suptitle(f"FHR4 multi-track GradCAM | {sample_name}")

                pred_tag = "_".join(str(t) for t in pred_tracks) if pred_tracks else "none"
                output_filename = (
                    f"positive_sample_{count}_{sample_name}_pred_{pred_tag}.png"
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(gradcam_dir, output_filename),
                    bbox_inches='tight', dpi=150,
                )
                plt.close(fig)

                if 'wandb' in globals():
                    wandb.log({
                        f"gradcam_positive_{count}": wandb.Image(
                            os.path.join(gradcam_dir, output_filename),
                            caption=(
                                f"File: {sample_name} | GT: {gt_tracks} | "
                                f"Pred: {pred_tracks}"
                            ),
                        )
                    })

                count += 1

        except StopIteration:
            break
            
    return count


def log_wandb_results(epoch, train_loss, val_results, step: int | None = None):
    """Log per-epoch scalar metrics to W&B at the given global step.

    If ``step`` is None, falls back to ``epoch + 1`` so existing call sites
    keep working.
    """
    if 'wandb' not in globals() or wandb.run is None:
        return

    s = int(step) if step is not None else int(epoch + 1)
    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_threshold": val_results["threshold"],
            "val_loss": val_results["loss"],
            "val_precision_macro": val_results["precision_macro"],
            "val_recall_macro": val_results["recall_macro"],
            "val_f1_macro": val_results["f1_macro"],
            "val_precision_micro": val_results["precision_micro"],
            "val_recall_micro": val_results["recall_micro"],
            "val_f1_micro": val_results["f1_micro"],
        },
        step=s,
    )

    
    
 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_num_tracks(args.num_tracks)

    # Model (ResNet head size must match dataset multi-hot length)
    model, weights = get_model(
        args.model, args.dense_units, args.dropout, num_tracks=args.num_tracks
    )
    model = model.to(device)

    # Transform pipeline - keep a separate copy for visualization
    # transform = transforms.Compose(
    #     [
    #         transforms.Lambda(resize_to_224),
    #         transforms.ToTensor(),
    #         transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    #         transforms.Normalize(
    #             mean=weights.transforms().mean, std=weights.transforms().std
    #         ),
    #     ]
    # )
    transform = transforms.Compose(
        [
            transforms.Lambda(resize_to_224), # should output (C, H, W)
            transforms.Lambda(lambda x: torch.from_numpy(x).float()),
            transforms.Normalize(
                mean=weights.transforms().mean, std=weights.transforms().std
            ),
        ]
    )
    

    # Dataset and DataLoader
    full_ds = NumpyImageFolder(
        root_dir=args.data_dir,
        transform=transform,
        debug=True,
        num_tracks=args.num_tracks,
    )

    train_ds, val_ds, split_mode, split_stats = make_train_val_subsets(
        full_ds,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        split_random=args.split_random,
    )
    if split_mode == "grouped":
        print(
            f"[Split] mode=grouped val_fraction={args.val_fraction} seed={args.split_seed} "
            f"groups={split_stats.get('num_groups', 0)} "
            f"(train_groups={split_stats.get('train_groups', '?')} "
            f"val_groups={split_stats.get('val_groups', '?')}) "
            f"largest_group={split_stats.get('largest_group', '?')} "
            f"files train={len(train_ds)} val={len(val_ds)}"
        )
    else:
        print(
            f"[Split] mode={split_mode} val_fraction={args.val_fraction} seed={args.split_seed} "
            f"files train={len(train_ds)} val={len(val_ds)}"
        )

    if args.balance_train_pos_neg:
        _pre_n = len(train_ds.indices)
        _pre_pos, _pre_neg = count_neg_pos_images(full_ds, train_ds.indices)
        if _pre_pos == 0 or _pre_neg == 0:
            print(
                f"[Balance] skipped: need both classes in train (pos={_pre_pos}, neg={_pre_neg})"
            )
        else:
            _bal_idx = undersample_train_balanced_pos_neg(
                full_ds, list(train_ds.indices), seed=args.balance_train_seed
            )
            train_ds = torch.utils.data.Subset(full_ds, _bal_idx)
            _post_pos, _post_neg = count_neg_pos_images(full_ds, train_ds.indices)
            print(
                f"[Balance] undersampled train pos/neg: {_pre_pos} pos, {_pre_neg} neg → "
                f"{_post_pos} pos, {_post_neg} neg | {_pre_n} → {len(train_ds)} samples "
                f"(seed={args.balance_train_seed})"
            )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    # shuffle=False on val is no longer required for path alignment (validate() now
    # collects paths directly from the loader via debug=True), but kept for stable
    # epoch-to-epoch FP/FN ordering in W&B panels.
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- Loss: exactly one of plain BCE | per-track pos_weight | image-level sample weights
    train_n_neg, train_n_pos = count_neg_pos_images(full_ds, train_ds.indices)
    _PW_CAP = 20.0  # internal cap for pos_weight mode only

    if args.loss_mode == "plain":
        criterion = nn.BCEWithLogitsLoss()
        print("[Loss] mode=plain — nn.BCEWithLogitsLoss()")
    elif args.loss_mode == "pos_weight":
        pos_weight_tensor = compute_per_track_pos_weight(
            full_ds,
            train_ds.indices,
            args.num_tracks,
            cap=_PW_CAP,
        )
        print(
            f"[Loss] mode=pos_weight — per-track pos_weight (cap={_PW_CAP}); "
            f"train images {train_n_neg} neg / {train_n_pos} pos (not weighted)"
        )
        print(
            f"[Loss] per-track pos_weight min/mean/max: "
            f"{pos_weight_tensor.min().item():.3f} / "
            f"{pos_weight_tensor.mean().item():.3f} / "
            f"{pos_weight_tensor.max().item():.3f}"
        )
        criterion = CombinedMultiLabelBCELoss(
            pos_weight_tensor,
            w_neg=1.0,
            w_pos=1.0,
            use_pos_weight=True,
        ).to(device)
    else:  # sample_weight
        w_pos = 1.0
        w_neg = 1.0 if train_n_neg == 0 else (train_n_pos / train_n_neg) * w_pos
        print(
            f"[Loss] mode=sample_weight — w_neg={w_neg:.4f} w_pos={w_pos:.4f} "
            f"(train images: {train_n_neg} neg, {train_n_pos} pos); no per-track pos_weight"
        )
        criterion = CombinedMultiLabelBCELoss(
            None,
            w_neg=w_neg,
            w_pos=w_pos,
            use_pos_weight=False,
        ).to(device)

    # Prepare misclassified samples json file
    misclassified_json_path = os.path.join(args.save_dir, 'misclassified_samples.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(misclassified_json_path, 'w') as f:
        json.dump({}, f)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr,
        momentum=0.9,  # Adding momentum helps SGD converge better
        weight_decay=1e-4  # L2 regularization to prevent overfitting
    )
    # ReduceLROnPlateau on val_f1_macro (mode='max'): val_loss can drift even
    # while F1 plateaus on imbalanced multi-label data, so loss is the wrong signal.
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.lr_patience
    )

    # Run tag for checkpoint and artifact naming. Falls back to wandb run id if any,
    # else a timestamp, so reruns never silently overwrite each other.
    if args.run_tag:
        run_tag = args.run_tag
    elif wandb.run is not None:
        run_tag = wandb.run.id
    else:
        from datetime import datetime
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[Run] tag={run_tag}")
    os.makedirs(args.save_dir, exist_ok=True)

    best_f1 = -1.0
    best_path = os.path.join(args.save_dir, f"{args.model}_{run_tag}_best.pth")
    last_path = os.path.join(args.save_dir, f"{args.model}_{run_tag}_last.pth")
    epochs_since_improve = 0
    global_step = 0  # monotonic across epochs; defines the W&B X-axis.

    if args.train:
        for epoch in range(args.epochs):
            ep = epoch + 1  # 1-indexed everywhere user-facing
            train_loss, global_step = train_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch=epoch,
                global_step_start=global_step,
                wandb_log_every=args.wandb_log_every,
            )
            print(f"[Epoch {ep}] Train Loss: {train_loss:.4f} (global_step={global_step})")

            val_results = validate(
                model, val_loader, criterion, device, threshold=args.val_threshold
            )

            # Save misclassified samples for this epoch
            with open(misclassified_json_path, 'r+') as f:
                data = json.load(f)
                data[f'epoch_{ep}'] = val_results['misclassified']
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

            log_wandb_results(epoch, train_loss, val_results, step=global_step)
            log_wandb_fp_fn_samples(
                epoch, val_results,
                max_each=args.fp_fn_log_k,
                threshold=args.val_threshold,
                step=global_step,
            )

            # Step LR on F1 (maximize)
            f1 = float(val_results['f1_macro'])
            scheduler.step(f1)

            print(
                f"[Epoch {ep}] "
                f"f1_macro={f1:.4f} "
                f"precision_macro={val_results['precision_macro']:.4f} "
                f"recall_macro={val_results['recall_macro']:.4f} | "
                f"precision_micro={val_results['precision_micro']:.4f} "
                f"recall_micro={val_results['recall_micro']:.4f} | "
                f"val_loss={val_results['loss']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            # Always overwrite a "last" checkpoint; keep "best" by F1.
            torch.save(model.state_dict(), last_path)
            improved = f1 > best_f1
            if improved:
                best_f1 = f1
                epochs_since_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f"[Checkpoint] new best f1_macro={best_f1:.4f} → {best_path}")
            else:
                epochs_since_improve += 1

            # Log a single artifact per checkpoint kind ("last" + "best"), not 1 per epoch.
            if 'wandb' in globals() and improved:
                artifact = wandb.Artifact(
                    name=f"{args.model}_{run_tag}_best",
                    type="model",
                    description=(
                        f"Best val f1_macro so far ({best_f1:.4f}) at epoch "
                        f"{ep}/{args.epochs}"
                    ),
                )
                artifact.add_file(best_path)
                artifact.metadata = {
                    "epoch": ep,
                    "run_tag": run_tag,
                    "architecture": args.model,
                    "dense_units": args.dense_units,
                    "dropout": args.dropout,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "precision_macro": val_results['precision_macro'],
                    "recall_macro": val_results['recall_macro'],
                    "f1_macro": f1,
                    "train_loss": train_loss,
                    "val_loss": val_results['loss'],
                }
                wandb.log_artifact(artifact)

            # Early stopping on f1_macro plateau.
            if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
                print(
                    f"[EarlyStop] no f1_macro improvement for {epochs_since_improve} "
                    f"epochs (patience={args.early_stop_patience}); stopping at epoch {ep}."
                )
                break

        print(f"[Train] done. best f1_macro={best_f1:.4f} ({best_path})")
    else:
        ckpt_path = args.load_checkpoint
        if ckpt_path is None:
            # Prefer this run's best.pth if it exists, then any *_best.pth,
            # then fall back to the most recently modified .pth.
            preferred = [Path(best_path)]
            preferred += sorted(Path(args.save_dir).glob("*_best.pth"))
            preferred += sorted(
                Path(args.save_dir).glob("*.pth"),
                key=lambda p: p.stat().st_mtime,
            )
            for cand in preferred:
                if cand.exists():
                    ckpt_path = str(cand)
                    break
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"--no-train set but no checkpoint found in {args.save_dir} "
                    "and --load-checkpoint not provided."
                )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Model loaded from {ckpt_path}")

    # GradCAM is off by default (slow, requires per-sample backward passes).
    # Pass --gradcam to enable; tune the number of samples with --gradcam-num.
    if args.gradcam:
        gradcam_dir = os.path.join(args.save_dir, "gradcam_dark")
        num_samples = visualize_gradcam(
            model, val_loader, device,
            full_ds, val_ds, gradcam_dir,
            num_positive_samples=args.gradcam_num,
        )
        print(f"GradCAM visualizations saved to: {gradcam_dir} ({num_samples} positive samples)")
    else:
        print("[GradCAM] skipped (pass --gradcam to enable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet", choices=["resnet", "mobilenet"]
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/ml_data_v4",
        help="Folder containing neg/ and pos/ (ml_data_v4 layout; see dataset_multi_channel_v4)",
    )
    parser.add_argument(
        "--split-random",
        action="store_true",
        help="Use legacy random index split (A/B vs default group-safe split via group_id_for_path)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="RNG seed for grouped split shuffling or random_split generator",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction: fraction of *groups* (grouped) or *samples* (random)",
    )
    parser.add_argument(
        "--balance-train-pos-neg",
        action="store_true",
        help=(
            "Undersample training only: equal count of negative (all-zero labels) vs "
            "positive (≥1 track) images; uses min(n_pos,n_neg) each"
        ),
    )
    parser.add_argument(
        "--balance-train-seed",
        type=int,
        default=42,
        help="RNG seed for which train positives/negatives to keep when balancing",
    )

    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the model (default). Use --no-train to skip training and only run GradCAM on a loaded checkpoint.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a .pth checkpoint to load when --no-train is set. "
            "If omitted, the most recently modified .pth in --save-dir is used."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--num-tracks",
        type=int,
        default=22,
        help="Circumferential tracks: multi-label size, ResNet output dim, and data_utils default (via set_num_tracks)",
    )
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="ReduceLROnPlateau patience (epochs without f1_macro improvement before LR decay).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop training after this many epochs without f1_macro improvement (0 to disable).",
    )
    parser.add_argument(
        "--val-threshold",
        type=float,
        default=0.5,
        help="Probability threshold to binarize multi-label predictions during validation.",
    )
    parser.add_argument(
        "--gradcam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run GradCAM visualization at end of training (off by default; slow).",
    )
    parser.add_argument(
        "--gradcam-num",
        type=int,
        default=100,
        help="Max number of positive validation samples to render with GradCAM when enabled.",
    )
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="models/dent_models")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Tag used in checkpoint and W&B artifact names. "
            "Defaults to wandb.run.id (or a timestamp if W&B is offline). "
            "Set this to avoid overwriting checkpoints from previous runs."
        ),
    )
    parser.add_argument(
        "--fp-fn-log-k",
        type=int,
        default=10,
        help="Max false positive and false negative validation samples to log to wandb per epoch",
    )
    parser.add_argument(
        "--wandb-log-every",
        type=int,
        default=1000,
        help=(
            "Log training loss to W&B every N batches (and at end of each epoch). "
            "Set very high to effectively log only per-epoch."
        ),
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        default="sample_weight",
        choices=("plain", "pos_weight", "sample_weight"),
        help=(
            "plain: BCEWithLogitsLoss; "
            "pos_weight: per-track pos_weight from train split (cap=20); "
            "sample_weight: image-level w_neg/w_pos only (w_neg auto n_pos/n_neg, w_pos=1)"
        ),
    )
    args = parser.parse_args()
    set_seed(42)
    
    # Initialize wandb
    wandb.init(project="dent_arm_encoder", config=vars(args))
    main(args)  