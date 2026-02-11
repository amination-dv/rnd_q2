"""
Custom DatasetMapper for ILI tubeview detection with track-based augmentations.

Applies track shift and circular roll before Detectron2's standard transforms.
"""

import copy
import random

import numpy as np
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from Dataset.augmentations import apply_track_shift, apply_circular_roll


class ILIDatasetMapper:
    """
    DatasetMapper that runs ILI-specific augmentations first, then standard augs.

    ILI augmentations (only during training):
        - Track shift: per-track horizontal shifts (bboxes adjusted by overlapping tracks).
        - Circular roll: vertical wrap simulating tool rotation (split wrapped bboxes into two).
    """

    @configurable
    def __init__(
        self,
        is_train,
        *,
        augmentations,
        image_format,
        num_tracks,
        max_track_shift,
        track_shift_prob,
        circular_roll_prob,
        split_wrapped_boxes,
        use_instance_mask=False,
        use_keypoint=False,
        instance_mask_format="polygon",
        keypoint_hflip_indices=None,
        precomputed_proposal_topk=None,
        recompute_boxes=False,
    ):
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.use_keypoint = use_keypoint
        self.instance_mask_format = instance_mask_format
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk = precomputed_proposal_topk
        self.recompute_boxes = recompute_boxes
        self.num_tracks = num_tracks
        self.max_track_shift = max_track_shift
        self.track_shift_prob = track_shift_prob
        self.circular_roll_prob = circular_roll_prob
        self.split_wrapped_boxes = split_wrapped_boxes

    @classmethod
    def from_config(cls, cfg, is_train=True, num_tracks=22, max_track_shift=15,
                    track_shift_prob=0.5, circular_roll_prob=0.3, split_wrapped_boxes=True):
        augs = utils.build_augmentation(cfg, is_train)
        if hasattr(cfg.INPUT, 'CROP') and getattr(cfg.INPUT.CROP, 'ENABLED', False) and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        recompute_boxes = getattr(cfg.MODEL, 'MASK_ON', False)

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": getattr(cfg.MODEL, 'MASK_ON', False),
            "instance_mask_format": getattr(cfg.INPUT, 'MASK_FORMAT', 'polygon'),
            "use_keypoint": getattr(cfg.MODEL, 'KEYPOINT_ON', False),
            "recompute_boxes": recompute_boxes,
            "num_tracks": num_tracks,
            "max_track_shift": max_track_shift,
            "track_shift_prob": track_shift_prob,
            "circular_roll_prob": circular_roll_prob,
            "split_wrapped_boxes": split_wrapped_boxes,
        }
        if ret["use_keypoint"] and getattr(cfg.DATASETS, 'TRAIN', None):
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        if getattr(cfg.MODEL, 'LOAD_PROPOSALS', False):
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        height, width = image.shape[:2]
        annotations = dataset_dict.get("annotations", [])
        annos = [a for a in annotations if a.get("iscrowd", 0) == 0]

        # ---- ILI-specific augmentations (training only) ----
        if self.is_train and annos:
            if random.random() < self.track_shift_prob:
                image, annos = apply_track_shift(
                    image, annos, self.num_tracks, self.max_track_shift, height, width,
                )
            if random.random() < self.circular_roll_prob:
                image, annos = apply_circular_roll(
                    image, annos, self.num_tracks, height, width,
                    split_wrapped=self.split_wrapped_boxes,
                )
            dataset_dict["annotations"] = annos

        # ---- Standard Detectron2 augmentations ----
        sem_seg_gt = None
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]

        dataset_dict["image"] = self._to_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = self._to_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms,
                proposal_topk=self.proposal_topk,
            )

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict

    @staticmethod
    def _to_tensor(x):
        import torch
        return torch.as_tensor(x)

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape,
                keypoint_hflip_indices=self.keypoint_hflip_indices,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format,
        )
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
