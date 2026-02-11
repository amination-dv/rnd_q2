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

import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog


def build_category_from_json(data_dir, json_file):
    """
    Build category name → id mapping from a Labelbox JSON file.

    Scans all annotations and collects unique object names. Returns a dict
    mapping each name to a stable index (sorted alphabetically for reproducibility).
    """
    json_path = os.path.join(data_dir, json_file)
    with open(json_path) as f:
        imgs_anns = json.load(f)

    names = set()
    for v in imgs_anns:
        try:
            project_id = list(v["projects"])[0]
            objs = v["projects"][project_id]["labels"][0]["annotations"]["objects"]
            for obj in objs:
                names.add(obj["name"])
        except (KeyError, IndexError):
            continue

    return {name: i for i, name in enumerate(sorted(names))}


def get_detection_data(data_dir, json_file, split_type, split_indices=None, category_mapping=None):
    """
    Load detection data from a Labelbox JSON file in Detectron2 format.

    Args:
        data_dir: Base directory containing train/ and test/ image folders.
        json_file: Name of the JSON annotation file.
        split_type: 'train', 'valid', or 'test'.
        split_indices: Optional array of indices for train/val splitting.
        category_mapping: Dict mapping class name → category_id (from build_category_from_json).

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

    # Build category mapping from train.json (source of truth)
    category_mapping = build_category_from_json(data_dir, train_json)
    class_names = [k for k, _ in sorted(category_mapping.items(), key=lambda x: x[1])]

    # Compute train/val split indices
    rs = np.random.RandomState(random_state)
    full_train = get_detection_data(data_dir, train_json, "train", category_mapping=category_mapping)
    n_dataset = len(full_train)
    n_train = int(n_dataset * train_split)
    inds = rs.permutation(n_dataset)
    train_inds = inds[:n_train]
    valid_inds = inds[n_train:]

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
