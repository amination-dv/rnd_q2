"""
Custom Detectron2 Trainer with WandB logging and ILI track augmentations.

Extends DefaultTrainer to:
    - Use ILIDatasetMapper with track shift and circular roll augmentations.
    - Log metrics to WandB via a custom EventWriter.
    - Periodically log sample prediction visualisations.
    - Run test-time evaluation with visualisation.
"""

import os
import logging
import random

import cv2
import wandb
import torch
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import CallbackHook
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.events import EventWriter, get_event_storage

from Dataset import ILIDatasetMapper
from utils import log_predictions_to_wandb

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WandB event writer
# ---------------------------------------------------------------------------

class WandbWriter(EventWriter):
    """Streams Detectron2 training metrics to WandB. Eval metrics get val/ prefix."""

    def __init__(self):
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        metrics = {}
        for k, (v, _) in storage.latest_with_smoothing_hint().items():
            # Eval metrics (bbox/*, segm/*) -> val/ for clarity
            if k.startswith(("bbox/", "segm/")):
                metrics[f"val/{k}"] = v
            else:
                metrics[k] = v
        if metrics:
            wandb.log(metrics, step=storage.iter)
        self._last_write = storage.iter

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Detection trainer
# ---------------------------------------------------------------------------

class DetectionTrainer(DefaultTrainer):
    """Detectron2 DefaultTrainer extended with ILI augmentations and WandB logging."""

    def __init__(self, cfg, config_dict):
        self.config_dict = config_dict
        self.img_log_step = config_dict.get('img_log_step', 100)
        self.eval_period = cfg.TEST.EVAL_PERIOD
        super().__init__(cfg)

    # -- Train loader with ILI mapper ---------------------------------------

    @classmethod
    def build_train_loader(cls, cfg):
        # Use ILIDatasetMapper(cfg, ...) not .from_config() — from_config returns kwargs dict, not instance
        mapper = ILIDatasetMapper(
            cfg,
            is_train=True,
            num_tracks=getattr(cfg, 'NUM_TRACKS', 22),
            max_track_shift=getattr(cfg, 'MAX_TRACK_SHIFT', 15),
            track_shift_prob=getattr(cfg, 'TRACK_SHIFT_PROB', 0.5),
            circular_roll_prob=getattr(cfg, 'CIRCULAR_ROLL_PROB', 0.3),
            split_wrapped_boxes=getattr(cfg, 'SPLIT_WRAPPED_BOXES', True),
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    # -- Evaluator ----------------------------------------------------------

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([
            COCOEvaluator(dataset_name, output_dir=output_folder),
        ])

    # -- Writers ------------------------------------------------------------

    def build_writers(self):
        writers = super().build_writers()
        writers.append(WandbWriter())
        return writers

    def build_hooks(self):
        hooks = super().build_hooks()
        # Log validation images after each eval
        hooks.append(CallbackHook(after_step=self._maybe_log_valid_images))
        return hooks

    def _maybe_log_valid_images(self):
        """After eval steps, log validation prediction samples to WandB."""
        next_iter = self.iter + 1
        if self.eval_period <= 0 or next_iter % self.eval_period != 0:
            return
        if next_iter == self.max_iter:
            return  # Last eval handled in after_train
        self._log_valid_sample()

    # -- Training step with image logging -----------------------------------

    def run_step(self):
        super().run_step()
        if self.iter % self.img_log_step == 0:
            self._log_train_sample()

    def _log_train_sample(self):
        """Log a random training prediction to WandB."""
        try:
            dataset_name = self.cfg.DATASETS.TRAIN[0]
            dataset_dicts = DatasetCatalog.get(dataset_name)
            metadata = MetadataCatalog.get(dataset_name)
            sample = random.choice(dataset_dicts)

            self.model.eval()
            try:
                with torch.no_grad():
                    img = cv2.imread(sample["file_name"])
                    height, width = img.shape[:2]
                    image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                    inputs = [{"image": image, "height": height, "width": width}]
                    outputs = self.model(inputs)[0]
                wandb_dict = log_predictions_to_wandb(outputs, sample, metadata, phase="train")
                wandb.log(wandb_dict, step=self.iter)
            finally:
                self.model.train()
        except Exception as e:
            LOGGER.warning(f"Failed to log training sample: {e}")
            self.model.train()

    def _log_valid_sample(self):
        """Log a random validation prediction to WandB."""
        try:
            if "data_detection_valid" not in self.cfg.DATASETS.TEST:
                return
            dataset_dicts = DatasetCatalog.get("data_detection_valid")
            if not dataset_dicts:
                return
            metadata = MetadataCatalog.get("data_detection_valid")
            sample = random.choice(dataset_dicts)

            self.model.eval()
            try:
                with torch.no_grad():
                    img = cv2.imread(sample["file_name"])
                    height, width = img.shape[:2]
                    image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                    inputs = [{"image": image, "height": height, "width": width}]
                    outputs = self.model(inputs)[0]
                wandb_dict = log_predictions_to_wandb(outputs, sample, metadata, phase="valid")
                wandb.log(wandb_dict, step=self.iter)
            finally:
                self.model.train()
        except Exception as e:
            LOGGER.warning(f"Failed to log validation sample: {e}")
            self.model.train()

    # -- Test with visualisation --------------------------------------------

    def test_with_visualization(self, cfg, model, num_vis_samples=10, test_datasets=None):
        """
        Run COCO evaluation on the test set and log sample visualisations.

        Args:
            cfg: Detectron2 config.
            model: Model to evaluate.
            num_vis_samples: Number of sample images to log.
            test_datasets: Optional list of dataset names for eval. If None, uses cfg.DATASETS.TEST.

        Returns:
            dict: COCO evaluation results.
        """
        datasets = test_datasets if test_datasets is not None else list(cfg.DATASETS.TEST)
        results = {}
        for dataset_name in datasets:
            # Quantitative eval
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = self.build_evaluator(cfg, dataset_name)
            eval_results = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = eval_results

            # Log metrics with test/ prefix for final evaluation
            prefix = "test/" if dataset_name == "data_detection_test" else "valid/"
            if "bbox" in eval_results:
                for metric, value in eval_results["bbox"].items():
                    wandb.log({f"{prefix}{metric}": value})

            # Visual samples
            self._log_test_predictions(cfg, model, dataset_name, num_vis_samples)

        return results

    def _log_test_predictions(self, cfg, model, dataset_name, num_samples=10):
        """Log a handful of test predictions to WandB for visual inspection."""
        try:
            dataset_dicts = DatasetCatalog.get(dataset_name)
            metadata = MetadataCatalog.get(dataset_name)
            samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))

            model.eval()
            for idx, sample in enumerate(samples):
                with torch.no_grad():
                    img = cv2.imread(sample["file_name"])
                    height, width = img.shape[:2]
                    image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                    inputs = [{"image": image, "height": height, "width": width}]
                    outputs = model(inputs)[0]

                phase = "test" if dataset_name == "data_detection_test" else "valid"
                wandb_dict = log_predictions_to_wandb(
                    outputs, sample, metadata, phase=f"{phase}_sample_{idx}",
                )
                wandb.log(wandb_dict)
        except Exception as e:
            LOGGER.warning(f"Failed to log test predictions: {e}")
