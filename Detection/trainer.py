"""
High-level trainer module for ILI component detection.

Orchestrates:
    - Detectron2 configuration from ``config.yaml``.
    - Dataset registration.
    - WandB experiment tracking.
    - Train / test / save entry points.
"""

import os

import wandb
import torch
from dotenv import load_dotenv
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog

from Dataset import register_detection_datasets
from training import DetectionTrainer
from utils import (
    load_config,
    createDirectory,
    current_timestamp,
    fix_random_seed,
    get_last_checkpoint,
)

load_dotenv()


class TrainerModule:
    """
    Entry-point class that wires config → Detectron2 cfg → trainer.

    Usage::

        module = TrainerModule('config.yaml')
        module.train()   # train + test best
        module.test()    # test only
        module.save_model('path/to/model.pth')
    """

    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.project_name = os.getenv('PROJECT_NAME', 'ili-detection')
        self.checkpoint_dir = "checkpoints"
        createDirectory(self.checkpoint_dir)
        self.setup()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Register datasets, build Detectron2 cfg, init WandB."""
        # Datasets (category mapping derived from train.json)
        n_total, n_train, n_val, num_classes, class_names = register_detection_datasets(self.config)
        print(f"Datasets registered: total={n_total}, train={n_train}, val={n_val}")
        print(f"Classes ({num_classes}): {class_names}")

        # Detectron2 config
        self.cfg = get_cfg()
        self.cfg.OUTPUT_DIR = self.checkpoint_dir

        model_config = self.config.get(
            'model_config', 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
        )
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))

        # Datasets
        self.cfg.DATASETS.TRAIN = ("data_detection_train",)
        self.cfg.DATASETS.VALID = ("data_detection_valid",)
        self.cfg.DATASETS.TEST = ("data_detection_test",)

        # Model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        self.cfg.MODEL.FREEZE_AT = self.config.get('freeze_at', 0)
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.config.get('roi_heads_batch_size', 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.get('score_thresh_test', 0.6)

        # Solver
        self.cfg.SOLVER.IMS_PER_BATCH = self.config.get('batch_size', 1)
        self.cfg.SOLVER.BASE_LR = self.config.get('lr', 0.00025)
        self.cfg.SOLVER.MAX_ITER = self.config.get('max_iter', 10000)
        self.cfg.SOLVER.STEPS = self.config.get('steps', [])
        self.cfg.SOLVER.LR_SCHEDULER_NAME = self.config.get('lr_scheduler', 'WarmupMultiStepLR')
        self.cfg.SOLVER.CHECKPOINT_PERIOD = self.config.get('checkpoint_period', 500)

        # Data loader
        self.cfg.DATALOADER.NUM_WORKERS = self.config.get('num_workers', 4)

        # Evaluation
        self.cfg.TEST.EVAL_PERIOD = self.config.get('eval_period', 100)

        # ILI augmentations
        self.cfg.NUM_TRACKS = self.config.get('num_tracks', 22)
        self.cfg.MAX_TRACK_SHIFT = self.config.get('max_track_shift', 15)
        self.cfg.TRACK_SHIFT_PROB = self.config.get('track_shift_prob', 0.5)
        self.cfg.CIRCULAR_ROLL_PROB = self.config.get('circular_roll_prob', 0.3)
        self.cfg.SPLIT_WRAPPED_BOXES = self.config.get('split_wrapped_boxes', True)

        # Device
        devices = self.config.get('device', [0])
        if isinstance(devices, list) and len(devices) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))

        # Reproducibility
        fix_random_seed(self.config.get('random_state', 111))

        # WandB
        wandb.login(key=os.getenv('WANDB_KEY'))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self):
        wandb.init(
            project=self.project_name,
            name=f"train-{self.project_name}-{current_timestamp()}",
            config=self.config,
        )

        # Log dataset sizes
        wandb.log({
            "dataset/train": len(DatasetCatalog.get("data_detection_train")),
            "dataset/valid": len(DatasetCatalog.get("data_detection_valid")),
            "dataset/test": len(DatasetCatalog.get("data_detection_test")),
        })

        trainer = DetectionTrainer(self.cfg, self.config)

        # Resume or start fresh
        last_ckpt = get_last_checkpoint(self.checkpoint_dir)
        if last_ckpt:
            print(f"Resuming from: {last_ckpt}")
            self.cfg.MODEL.WEIGHTS = last_ckpt
            trainer.resume_or_load(resume=True)
        else:
            print("Starting training from scratch")
            trainer.resume_or_load(resume=False)

        trainer.train()

        # Evaluate best model
        best_path = get_last_checkpoint(self.checkpoint_dir, return_best=True)
        if best_path:
            print(f"Testing best model: {best_path}")
            self.cfg.MODEL.WEIGHTS = best_path
            state = torch.load(best_path, map_location="cpu")
            trainer.model.load_state_dict(state["model"])
            trainer.test_with_visualization(self.cfg, trainer.model)

        wandb.finish()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        wandb.init(
            project=self.project_name,
            name=f"test-{self.project_name}-{current_timestamp()}",
            config=self.config,
        )

        best_path = get_last_checkpoint(self.checkpoint_dir, return_best=True)
        if not best_path:
            raise ValueError("No checkpoint found for testing")

        print(f"Testing: {best_path}")
        self.cfg.MODEL.WEIGHTS = best_path

        trainer = DetectionTrainer(self.cfg, self.config)
        trainer.resume_or_load(resume=False)
        trainer.test_with_visualization(self.cfg, trainer.model)

        wandb.finish()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_model(self, model_path):
        import shutil
        best_path = get_last_checkpoint(self.checkpoint_dir, return_best=True)
        if not best_path:
            raise ValueError("No checkpoint found for saving")

        print(f"Copying {best_path} → {model_path}")
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        shutil.copy(best_path, model_path)
        print(f"Model saved to {model_path}")
