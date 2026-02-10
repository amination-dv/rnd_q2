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


# Category mapping — must match the annotation labels
CATEGORY = {"tap": 0, "flange": 1, "bend": 2, "tee": 3}


def get_detection_data(data_dir, json_file, split_type, split_indices=None):
    """
    Load detection data from a Labelbox JSON file in Detectron2 format.

    Args:
        data_dir: Base directory containing train/ and test/ image folders.
        json_file: Name of the JSON annotation file.
        split_type: 'train', 'valid', or 'test'.
        split_indices: Optional array of indices for train/val splitting.

    Returns:
        List of dicts in Detectron2 standard dataset format.
    """
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
                "category_id": CATEGORY[obj["name"]],
            }
            for obj in annotations
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
    class_names = config.get('class_names', list(CATEGORY.keys()))

    # Compute train/val split indices
    rs = np.random.RandomState(random_state)
    full_train = get_detection_data(data_dir, train_json, "train")
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
    DatasetCatalog.register(
        "data_detection_train",
        lambda dd=data_dir, tj=train_json, ti=train_inds: get_detection_data(dd, tj, "train", ti),
    )
    DatasetCatalog.register(
        "data_detection_valid",
        lambda dd=data_dir, tj=train_json, vi=valid_inds: get_detection_data(dd, tj, "valid", vi),
    )
    DatasetCatalog.register(
        "data_detection_test",
        lambda dd=data_dir, tj=test_json: get_detection_data(dd, tj, "test"),
    )

    for name in ["data_detection_train", "data_detection_valid", "data_detection_test"]:
        MetadataCatalog.get(name).set(thing_classes=class_names)

    return n_dataset, n_train, len(valid_inds)
