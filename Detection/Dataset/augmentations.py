"""
ILI-specific augmentations for tubeview object detection.

1. **Track shift (horizontal)**: Per-track horizontal shifts, matching CorrelationFields.
   Bboxes are adjusted by the min/max shift over the tracks they overlap.

2. **Circular track roll (vertical)**: Simulates tool rotation - top tracks move down
   or bottom tracks move up, wrapping. When a bbox crosses the wrap boundary, it is
   split into two bboxes (each visible part remains a valid detection target).
"""

import copy
import random

import numpy as np


def apply_track_shift(image, annotations, num_tracks, max_shift, height, width):
    """
    Apply per-track horizontal shifts (with horizontal wrap, like CorrelationFields).

    Bbox transformation: a box [x1,y1,x2,y2] overlaps tracks floor(y1/th)..floor(y2/th).
    The axis-aligned enclosing bbox after shift is [x1+min_s, y1, x2+max_s, y2]
    where min_s, max_s are the min/max shifts over those tracks.

    Args:
        image: (H, W, C) uint8 numpy array.
        annotations: List of dicts with "bbox" [x, y, w, h] (XYWH_ABS) and "category_id".
        num_tracks: Number of vertical tracks (e.g. 22).
        max_shift: Max absolute pixel shift per track.
        height: Image height.
        width: Image width.

    Returns:
        (shifted_image, transformed_annotations)
    """
    track_height = height // num_tracks
    if track_height <= 0:
        return image, annotations

    # Random per-track shifts (pixels)
    shift_vector = np.random.randint(-max_shift, max_shift + 1, size=num_tracks)

    # Shift each track horizontally (numpy roll wraps)
    shifted = np.copy(image)
    for i in range(num_tracks):
        start = i * track_height
        end = min(start + track_height, height)
        if shift_vector[i] != 0:
            shifted[start:end, :] = np.roll(image[start:end, :], shift_vector[i], axis=1)

    # Transform bboxes (annotations are XYWH)
    new_annos = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        y1, y2 = y, y + h
        track_start = int(y1) // track_height
        track_end = int(y2 - 1) // track_height if y2 > y1 else track_start
        track_start = max(0, min(track_start, num_tracks - 1))
        track_end = max(0, min(track_end, num_tracks - 1))
        min_s = int(shift_vector[track_start : track_end + 1].min())
        max_s = int(shift_vector[track_start : track_end + 1].max())

        # Shifted bbox: left edge moves by min_s, right by max_s (axis-aligned envelope)
        new_x = x + min_s
        new_x2 = x + w + max_s

        # Skip annotations whose content wraps around the horizontal edge
        # (np.roll wraps pixels but the bbox cannot represent a split region)
        if new_x < 0 or new_x2 > width:
            # Only skip if substantial wrap; minor sub-pixel clips are fine
            if new_x < -1 or new_x2 > width + 1:
                continue

        new_x = max(0, min(new_x, width - 1))
        new_x2 = max(new_x + 1, min(new_x2, width))
        new_w = new_x2 - new_x

        new_ann = copy.deepcopy(ann)
        new_ann["bbox"] = [new_x, y, new_w, h]
        new_annos.append(new_ann)

    return shifted, new_annos


def _merge_top_bottom_component_pairs(annotations, height, width, top_bottom_threshold=0.1):
    """
    Merge pairs of annotations that represent the same component split across top
    and bottom (human-labeled with imperfect width alignment). Uses max width when
    combining.

    Returns:
        List of annotations with top/bottom pairs merged into single full-height boxes.
    """
    if len(annotations) < 2:
        return annotations

    thresh = max(int(height * top_bottom_threshold), 20)
    used = [False] * len(annotations)
    merged = []

    for i, a1 in enumerate(annotations):
        if used[i]:
            continue
        x1, y1, w1, h1 = a1["bbox"]
        y1_end = y1 + h1

        # a1 at top: starts near 0
        at_top = y1 < thresh
        # a1 at bottom: ends near height
        at_bottom = y1_end > height - thresh
        if not (at_top or at_bottom):
            merged.append(copy.deepcopy(a1))
            used[i] = True
            continue

        best_j = None
        for j, a2 in enumerate(annotations):
            if i == j or used[j]:
                continue
            if a1.get("category_id") != a2.get("category_id"):
                continue
            x2, y2, w2, h2 = a2["bbox"]
            y2_end = y2 + h2

            a2_at_top = y2 < thresh
            a2_at_bottom = y2_end > height - thresh

            # One at top, one at bottom
            if not ((at_top and a2_at_bottom) or (at_bottom and a2_at_top)):
                continue

            # X overlap or close (human labels may not align perfectly)
            left1, right1 = x1, x1 + w1
            left2, right2 = x2, x2 + w2
            if right1 < left2 - width * 0.1 or right2 < left1 - width * 0.1:
                continue
            best_j = j
            break

        if best_j is not None:
            a2 = annotations[best_j]
            x2, y2, w2, h2 = a2["bbox"]
            x_min = min(x1, x2)
            x_max = max(x1 + w1, x2 + w2)
            w_merged = max(w1, w2, x_max - x_min)
            comb = copy.deepcopy(a1)
            comb["bbox"] = [
                max(0.0, min(x_min, width - 1)),
                0.0,
                float(min(w_merged, width - max(0, x_min))),
                float(height),
            ]
            merged.append(comb)
            used[i] = True
            used[best_j] = True
        else:
            merged.append(copy.deepcopy(a1))
            used[i] = True

    return merged


def apply_circular_roll(image, annotations, num_tracks, height, width, split_wrapped=True):
    """
    Circular roll in the vertical dimension (tracks move up or down, wrapping).

    Simulates the inspection tool rotating: e.g. track 0 moves to bottom, or
    last tracks move to top. When a bbox crosses the wrap boundary, it is
    split into two bboxes if split_wrapped=True.

    Edge cases:
    1. Full-height components: If a bbox spans the full image height, after roll
       it remains full height — output exactly 1 bbox [x, 0, w, height].
    2. Components at top/bottom: Human-labeled components that span top and bottom
       may be split into two boxes with imperfect width alignment. These are
       detected (same category, one at top / one at bottom, x overlap) and merged
       into one full-height box using max width.

    Args:
        image: (H, W, C) uint8 numpy array.
        annotations: List of dicts with "bbox" [x, y, w, h] (XYWH_ABS) and "category_id".
        num_tracks: Number of tracks (for choosing roll amount).
        height: Image height.
        width: Image width.
        split_wrapped: If True, split boxes that wrap into two boxes. If False,
            use the axis-aligned union (very loose bbox).

    Returns:
        (rolled_image, transformed_annotations) — annotations may have more items
        than input when split_wrapped and boxes wrap.
    """
    track_height = height // num_tracks
    if track_height <= 0:
        return image, annotations

    # Pre-merge top/bottom pairs (same component, human-labeled with width variance)
    annotations = _merge_top_bottom_component_pairs(annotations, height, width)

    # Roll by a random number of tracks (positive = top to bottom)
    k_tracks = random.randint(1, num_tracks - 1) if num_tracks > 1 else 0
    if random.random() < 0.5:
        k_tracks = -k_tracks
    k_pixels = k_tracks * track_height

    rolled = np.roll(image, k_pixels, axis=0)

    new_annos = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        y1, y2 = y, y + h

        # Edge case 1: Full-height component — remains one full-height bbox after roll
        if h >= height:
            new_ann = copy.deepcopy(ann)
            new_ann["bbox"] = [x, 0.0, w, float(height)]
            new_annos.append(new_ann)
            continue

        # Source region [y1, y2) maps to [y1+k, y2+k) mod H
        y1_new = (y1 + k_pixels) % height
        y2_new = (y2 + k_pixels) % height

        if y1_new < y2_new:
            # No wrap — single contiguous region
            new_ann = copy.deepcopy(ann)
            new_ann["bbox"] = [x, float(y1_new), w, float(y2_new - y1_new)]
            new_annos.append(new_ann)
        elif y1_new == y2_new:
            # Collapsed to zero (e.g. full-height edge case) — keep as 1 full-height
            new_ann = copy.deepcopy(ann)
            new_ann["bbox"] = [x, 0.0, w, float(height)]
            new_annos.append(new_ann)
        else:
            # Wraps: split into [y1_new, H) and [0, y2_new) — at most 2 boxes
            if split_wrapped:
                # Top part (wrapped from bottom)
                h1 = height - y1_new
                if h1 >= 1:
                    a1 = copy.deepcopy(ann)
                    a1["bbox"] = [x, float(y1_new), w, float(h1)]
                    new_annos.append(a1)
                # Bottom part (wrapped from top)
                h2 = y2_new
                if h2 >= 1:
                    a2 = copy.deepcopy(ann)
                    a2["bbox"] = [x, 0.0, w, float(h2)]
                    new_annos.append(a2)
            else:
                # Union bbox (loose)
                new_ann = copy.deepcopy(ann)
                new_ann["bbox"] = [x, 0.0, w, float(height)]
                new_annos.append(new_ann)

    return rolled, new_annos
