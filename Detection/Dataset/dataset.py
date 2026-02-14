"""
ILI Component Detection dataset registration for Detectron2.

Loads annotations from a single Labelbox-exported JSON (data.json) and registers
train/valid/test splits using sklearn train_test_split. Images live in data_dir/images/.

JSON format (per entry):
    {
        "data_row": {"external_id": "image_filename.png"},
        "media_attributes": {"height": H, "width": W},
        "projects": {
            "<project_id>": {
                "labels": [{
                    "annotations": {
                        "objects": [{
                            "name": "tap",
                            "bounding_box": {"left": x, "top": y, "width": w, "height": h}
                        }, ...]
                    }
                }]
            }
        }
    }
"""

import os
import json
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog


def build_category_from_config(class_names, class_merge=None):
    """
    Build category name → id mapping from config class_names + optional merges.

    Uses the ordered list from config as source of truth. Annotations whose
    class is not in the mapping (neither class_names nor class_merge keys)
    are ignored during data loading.

    Args:
        class_names: Ordered list of final output class names.
        class_merge: Optional dict mapping source label → target class name.
            e.g. {"tap": "Fitting", "tee": "stopple"} merges tap into Fitting
            and tee into stopple. Target names must exist in class_names.
    """
    if not class_names:
        raise ValueError("class_names must be a non-empty list")
    mapping = {name: i for i, name in enumerate(class_names)}
    if class_merge:
        for src, tgt in class_merge.items():
            if tgt not in mapping:
                raise ValueError(
                    f"class_merge target '{tgt}' is not in class_names {class_names}"
                )
            mapping[src] = mapping[tgt]
    return mapping


def get_detection_data(data_dir, json_file, images_folder, split_indices, category_mapping):
    """
    Load detection data from a Labelbox JSON file in Detectron2 format.

    Args:
        data_dir: Base directory (e.g. Data/).
        json_file: Name of the JSON annotation file (e.g. data.json).
        images_folder: Subfolder containing images (e.g. images).
        split_indices: Array of indices to include in this split.
        category_mapping: Dict mapping class name → category_id.

    Returns:
        List of dicts in Detectron2 standard dataset format.
    """
    if category_mapping is None:
        raise ValueError("category_mapping is required")

    json_path = os.path.join(data_dir, json_file)
    with open(json_path) as f:
        imgs_anns = json.load(f)

    dataset = []
    for idx, v in enumerate(imgs_anns):
        record = {}

        filename = os.path.join(data_dir, images_folder, v["data_row"]["external_id"])
        height = v["media_attributes"]["height"]
        width = v["media_attributes"]["width"]

        try:
            project_id = list(v["projects"])[0]
            annotations = v["projects"][project_id]["labels"][0]["annotations"]["objects"]
        except (KeyError, IndexError):
            annotations = []

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["annotations"] = [
            {
                "bbox": [
                    obj["bounding_box"]["left"],
                    obj["bounding_box"]["top"],
                    obj["bounding_box"]["width"],
                    obj["bounding_box"]["height"],
                ],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": category_mapping[obj["name"]],
            }
            for obj in annotations
            if obj["name"] in category_mapping
        ]
        dataset.append(record)

    if split_indices is not None:
        dataset = [dataset[i] for i in split_indices]

    return dataset


def _kept_indices(full_data, max_empty_ratio, random_state):
    """
    Return the original-data indices to keep after subsampling empty images.

    Indices refer to positions in *full_data* (and therefore the source JSON),
    so they can be passed directly to ``get_detection_data(split_indices=…)``.

    Args:
        full_data: List of dataset dicts (Detectron2 format).
        max_empty_ratio: Maximum fraction of the kept dataset that may be
            empty (no annotations).  ``None`` keeps everything.  ``0.0``
            removes all empty images.
        random_state: Seed for reproducible subsampling.

    Returns:
        Sorted numpy array of kept indices.
    """
    if max_empty_ratio is None:
        return np.arange(len(full_data))

    annotated_idx = [i for i, d in enumerate(full_data) if d["annotations"]]
    empty_idx = [i for i, d in enumerate(full_data) if not d["annotations"]]

    if not empty_idx:
        return np.arange(len(full_data))

    n_ann = len(annotated_idx)
    # Solve: n_keep / (n_ann + n_keep) <= max_empty_ratio
    if max_empty_ratio <= 0:
        n_keep = 0
    else:
        n_keep = int(max_empty_ratio * n_ann / (1 - max_empty_ratio))
    n_keep = min(n_keep, len(empty_idx))

    if n_keep < len(empty_idx):
        rs = np.random.RandomState(random_state)
        chosen = rs.choice(len(empty_idx), size=n_keep, replace=False)
        empty_idx = [empty_idx[i] for i in sorted(chosen)]

    return np.array(sorted(annotated_idx + empty_idx))


def register_detection_datasets(config):
    """
    Register train, valid, and test datasets with Detectron2.

    Uses stratified train_test_split on the full dataset for train/val/test.
    Ratios from config: train_split (e.g. 0.7), val_split (e.g. 0.15), remainder is test.

    Args:
        config: data_dir, data_json, images_folder, train_split, val_split,
            random_state, class_names, max_empty_ratio.

    Returns:
        (n_total, n_train, n_val, n_test, num_classes, class_names).
    """
    data_dir = config["data_dir"]
    json_file = config.get("data_json", config.get("train_json", "data.json"))
    images_folder = config.get("images_folder", "images")
    train_ratio = config.get("train_split", 0.7)
    val_ratio = config.get("val_split", 0.15)
    random_state = config.get("random_state", 111)
    class_names = config.get("class_names", [])
    if not class_names:
        raise ValueError("class_names must be defined in config")
    class_merge = config.get("class_merge", {})
    max_empty_ratio = config.get("max_empty_ratio", None)

    category_mapping = build_category_from_config(class_names, class_merge)

    # Load full dataset (no split filter)
    full_data = get_detection_data(
        data_dir, json_file, images_folder,
        split_indices=None,
        category_mapping=category_mapping,
    )

    # Filter empty images (those with no annotations after class filtering).
    # kept_inds are original JSON indices — safe to pass to get_detection_data.
    n_before = len(full_data)
    kept_inds = _kept_indices(full_data, max_empty_ratio, random_state)
    n_total = len(kept_inds)
    n_empty = sum(1 for i in kept_inds if not full_data[i]["annotations"])
    if n_before != n_total:
        print(
            f"Empty image filtering: {n_before} → {n_total} images "
            f"({n_empty} empty, {n_empty / n_total * 100:.1f}%)"
        )

    # Use a sentinel label (-1) for empty images so they form their own
    # stratum and don't collide with category_id 0.
    stratify_labels = []
    for i in kept_inds:
        annos = full_data[i].get("annotations", [])
        if not annos:
            stratify_labels.append(-1)
        else:
            cids = [a["category_id"] for a in annos]
            stratify_labels.append(Counter(cids).most_common(1)[0][0])

    # Split positions (0..n_total-1) within kept_inds, then map back to
    # original JSON indices for get_detection_data.
    positions = np.arange(n_total)
    # First split: train+val (train_ratio + val_ratio) vs test
    train_val_ratio = train_ratio + val_ratio
    try:
        tv_pos, test_pos = train_test_split(
            positions,
            train_size=train_val_ratio,
            stratify=stratify_labels,
            random_state=random_state,
        )
    except ValueError:
        rs = np.random.RandomState(random_state)
        perm = rs.permutation(n_total)
        n_tv = int(n_total * train_val_ratio)
        tv_pos, test_pos = perm[:n_tv], perm[n_tv:]

    # Second split: train vs val within train_val
    val_ratio_in_tv = val_ratio / train_val_ratio if train_val_ratio > 0 else 0.15
    try:
        train_pos, valid_pos = train_test_split(
            tv_pos,
            train_size=1 - val_ratio_in_tv,
            stratify=[stratify_labels[i] for i in tv_pos],
            random_state=random_state,
        )
    except ValueError:
        rs = np.random.RandomState(random_state)
        perm = rs.permutation(len(tv_pos))
        n_train = int(len(tv_pos) * (1 - val_ratio_in_tv))
        train_pos = tv_pos[perm[:n_train]]
        valid_pos = tv_pos[perm[n_train:]]

    # Map positions back to original JSON indices
    train_inds = kept_inds[train_pos]
    valid_inds = kept_inds[valid_pos]
    test_inds = kept_inds[test_pos]

    n_train = len(train_inds)
    n_val = len(valid_inds)
    n_test = len(test_inds)

    for name in ["data_detection_train", "data_detection_valid", "data_detection_test"]:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)

    cat = category_mapping
    dd, jf, imgf = data_dir, json_file, images_folder
    DatasetCatalog.register(
        "data_detection_train",
        lambda ti=train_inds: get_detection_data(dd, jf, imgf, ti, cat),
    )
    DatasetCatalog.register(
        "data_detection_valid",
        lambda vi=valid_inds: get_detection_data(dd, jf, imgf, vi, cat),
    )
    DatasetCatalog.register(
        "data_detection_test",
        lambda te=test_inds: get_detection_data(dd, jf, imgf, te, cat),
    )

    for name in ["data_detection_train", "data_detection_valid", "data_detection_test"]:
        MetadataCatalog.get(name).set(thing_classes=class_names)

    return n_total, n_train, n_val, n_test, len(class_names), class_names
