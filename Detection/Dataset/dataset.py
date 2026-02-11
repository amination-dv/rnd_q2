"""
ILI Component Detection dataset registration for Detectron2.

Loads annotations from Labelbox-exported JSON files and registers
train/valid/test splits with Detectron2's DatasetCatalog.

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


def get_detection_data(data_dir, json_file, split_type, split_indices=None, category_mapping=None):
    """
    Load detection data from a Labelbox JSON file in Detectron2 format.

    Args:
        data_dir: Base directory containing train/ and test/ image folders.
        json_file: Name of the JSON annotation file.
        split_type: 'train', 'valid', or 'test'.
        split_indices: Optional array of indices for train/val splitting.
        category_mapping: Dict mapping class name → category_id (from build_category_from_config).

    Returns:
        List of dicts in Detectron2 standard dataset format.
    """
    if category_mapping is None:
        raise ValueError("category_mapping is required")
    # valid images live inside the train/ folder
    img_folder = "train" if split_type == "valid" else split_type
    json_path = os.path.join(data_dir, json_file)

    with open(json_path) as f:
        imgs_anns = json.load(f)

    dataset = []
    for idx, v in enumerate(imgs_anns):
        record = {}

        # Image metadata
        filename = os.path.join(data_dir, img_folder, v["data_row"]["external_id"])
        height = v["media_attributes"]["height"]
        width = v["media_attributes"]["width"]

        # Annotations
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

    # Apply split indices if provided (for train/val split)
    if split_indices is not None:
        dataset = [dataset[i] for i in split_indices]

    return dataset


def register_detection_datasets(config):
    """
    Register train, valid, and test datasets with Detectron2.

    Creates a 90/10 train/val split from the training JSON using a fixed
    random seed for reproducibility.

    Args:
        config: Configuration dict with data_dir, train_json, test_json, etc.

    Returns:
        (n_dataset, n_train, n_val): Dataset split sizes.
    """
    data_dir = config['data_dir']
    train_json = config['train_json']
    test_json = config['test_json']
    train_split = config.get('train_split', 0.9)
    random_state = config.get('random_state', 111)
    class_names = config.get('class_names', [])
    if not class_names:
        raise ValueError("class_names must be defined in config")

    # Build category mapping from config class_names; annotations not in this list are ignored
    category_mapping = build_category_from_config(class_names)

    # Load full dataset and compute stratified train/val split by dominant category per image
    full_train = get_detection_data(data_dir, train_json, "train", category_mapping=category_mapping)
    n_dataset = len(full_train)

    # Stratification label: dominant (most frequent) category per image; empty images use 0
    stratify_labels = []
    for d in full_train:
        annos = d.get("annotations", [])
        if not annos:
            stratify_labels.append(0)
        else:
            cids = [a["category_id"] for a in annos]
            stratify_labels.append(Counter(cids).most_common(1)[0][0])

    try:
        train_inds, valid_inds = train_test_split(
            np.arange(n_dataset),
            train_size=train_split,
            stratify=stratify_labels,
            random_state=random_state,
        )
    except ValueError:
        # Fallback: stratification fails if a class has <2 samples
        rs = np.random.RandomState(random_state)
        inds = rs.permutation(n_dataset)
        n_train = int(n_dataset * train_split)
        train_inds, valid_inds = inds[:n_train], inds[n_train:]
    n_train = len(train_inds)

    # Unregister if already registered (useful for re-runs)
    for name in ["data_detection_train", "data_detection_valid", "data_detection_test"]:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)

    # Register each split — use default args in lambdas to capture values
    cat = category_mapping  # capture for lambda
    DatasetCatalog.register(
        "data_detection_train",
        lambda dd=data_dir, tj=train_json, ti=train_inds, c=cat: get_detection_data(dd, tj, "train", ti, c),
    )
    DatasetCatalog.register(
        "data_detection_valid",
        lambda dd=data_dir, tj=train_json, vi=valid_inds, c=cat: get_detection_data(dd, tj, "valid", vi, c),
    )
    DatasetCatalog.register(
        "data_detection_test",
        lambda dd=data_dir, tj=test_json, c=cat: get_detection_data(dd, tj, "test", split_indices=None, category_mapping=c),
    )

    for name in ["data_detection_train", "data_detection_valid", "data_detection_test"]:
        MetadataCatalog.get(name).set(thing_classes=class_names)

    return n_dataset, n_train, len(valid_inds), len(class_names), class_names
