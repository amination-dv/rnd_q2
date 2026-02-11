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


def build_category_from_config(class_names):
    """
    Build category name → id mapping from config class_names.

    Uses the ordered list from config as source of truth. Annotations whose
    class is not in class_names are ignored during data loading.
    """
    if not class_names:
        raise ValueError("class_names must be a non-empty list")
    return {name: i for i, name in enumerate(class_names)}


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


def register_detection_datasets(config):
    """
    Register train, valid, and test datasets with Detectron2.

    Uses stratified train_test_split on the full dataset for train/val/test.
    Ratios from config: train_split (e.g. 0.7), val_split (e.g. 0.15), remainder is test.

    Args:
        config: data_dir, data_json, images_folder, train_split, val_split, random_state, class_names.

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

    category_mapping = build_category_from_config(class_names)

    # Load full dataset (no split filter)
    full_data = get_detection_data(
        data_dir, json_file, images_folder,
        split_indices=None,
        category_mapping=category_mapping,
    )
    n_total = len(full_data)

    stratify_labels = []
    for d in full_data:
        annos = d.get("annotations", [])
        if not annos:
            stratify_labels.append(0)
        else:
            cids = [a["category_id"] for a in annos]
            stratify_labels.append(Counter(cids).most_common(1)[0][0])

    inds = np.arange(n_total)
    # First split: train+val (train_ratio + val_ratio) vs test
    train_val_ratio = train_ratio + val_ratio
    try:
        train_val_inds, test_inds = train_test_split(
            inds,
            train_size=train_val_ratio,
            stratify=stratify_labels,
            random_state=random_state,
        )
    except ValueError:
        rs = np.random.RandomState(random_state)
        perm = rs.permutation(n_total)
        n_tv = int(n_total * train_val_ratio)
        train_val_inds, test_inds = perm[:n_tv], perm[n_tv:]

    # Second split: train vs val within train_val
    val_ratio_in_tv = val_ratio / train_val_ratio if train_val_ratio > 0 else 0.15
    try:
        train_inds, valid_inds = train_test_split(
            train_val_inds,
            train_size=1 - val_ratio_in_tv,
            stratify=[stratify_labels[i] for i in train_val_inds],
            random_state=random_state,
        )
    except ValueError:
        rs = np.random.RandomState(random_state)
        perm = rs.permutation(len(train_val_inds))
        n_train = int(len(train_val_inds) * (1 - val_ratio_in_tv))
        train_inds = train_val_inds[perm[:n_train]]
        valid_inds = train_val_inds[perm[n_train:]]

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
