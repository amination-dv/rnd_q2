"""
Custom Detectron2 Trainer with WandB logging.

Extends DefaultTrainer to:
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
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.utils.events import EventWriter, get_event_storage

from utils import log_predictions_to_wandb

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WandB event writer
# ---------------------------------------------------------------------------

class WandbWriter(EventWriter):
    """Streams Detectron2 training metrics to WandB."""

    def __init__(self):
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        metrics = {}
        for k, (v, _) in storage.latest_with_smoothing_hint().items():
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
    """Detectron2 DefaultTrainer extended with WandB image logging."""

    def __init__(self, cfg, config_dict):
        self.config_dict = config_dict
        self.img_log_step = config_dict.get('img_log_step', 100)
        super().__init__(cfg)

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
            with torch.no_grad():
                img = cv2.imread(sample["file_name"])
                height, width = img.shape[:2]
                image = self.aug.get_transform(img).apply_image(img)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs = [{"image": image, "height": height, "width": width}]
                outputs = self.model(inputs)[0]
            self.model.train()

            wandb_dict = log_predictions_to_wandb(outputs, sample, metadata, phase="train")
            wandb.log(wandb_dict, step=self.iter)
        except Exception as e:
            LOGGER.warning(f"Failed to log training sample: {e}")

    # -- Test with visualisation --------------------------------------------

    def test_with_visualization(self, cfg, model, num_vis_samples=10):
        """
        Run COCO evaluation on the test set and log sample visualisations.

        Returns:
            dict: COCO evaluation results.
        """
        results = {}
        for dataset_name in cfg.DATASETS.TEST:
            # Quantitative eval
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = self.build_evaluator(cfg, dataset_name)
            eval_results = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = eval_results

            # Log metrics
            if "bbox" in eval_results:
                for metric, value in eval_results["bbox"].items():
                    wandb.log({f"test/{metric}": value})

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
                    image = self.aug.get_transform(img).apply_image(img)
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    inputs = [{"image": image, "height": height, "width": width}]
                    outputs = model(inputs)[0]

                wandb_dict = log_predictions_to_wandb(
                    outputs, sample, metadata, phase=f"test_sample_{idx}",
                )
                wandb.log(wandb_dict)
        except Exception as e:
            LOGGER.warning(f"Failed to log test predictions: {e}")
