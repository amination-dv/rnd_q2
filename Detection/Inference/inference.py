"""
Inference module for ILI component detection.

Supports:
    - Single-image inference.
    - Batch inference over a directory.
    - Threshold sweep for optimal AP50.
    - False-negative analysis.
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

# Allow importing from the parent package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_config


class DetectionInference:
    """
    Standalone inference class for a trained Detectron2 detection model.

    Args:
        config_path: Path to ``config.yaml``.
        model_path: Path to the trained ``.pth`` checkpoint.
        score_threshold: Confidence threshold for predictions.
    """

    def __init__(self, config_path, model_path, score_threshold=None):
        self.config = load_config(config_path)
        self.model_path = model_path
        self.score_threshold = score_threshold or self.config.get('score_thresh_test', 0.6)
        self._setup()

    def _setup(self):
        self.cfg = get_cfg()
        model_config = self.config.get(
            'model_config', 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
        )
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))

        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.config.get('num_classes', 4)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold

        self.predictor = DefaultPredictor(self.cfg)

        class_names = self.config.get('class_names', ["tap", "flange", "bend", "tee"])
        # Ensure metadata is registered for visualisation
        if "inference_dataset" not in MetadataCatalog:
            MetadataCatalog.get("inference_dataset").set(thing_classes=class_names)
        self.metadata = MetadataCatalog.get("inference_dataset")

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    def predict_image(self, image_path):
        """
        Run inference on a single image.

        Returns:
            (outputs, visualised_image_bgr)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        outputs = self.predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis = out.get_image()[:, :, ::-1]

        return outputs, vis

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def predict_batch(self, input_dir, output_dir, save_vis=True):
        """
        Run inference on every image in *input_dir*.

        Returns:
            dict mapping image paths to their predictions.
        """
        os.makedirs(output_dir, exist_ok=True)
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for ext in exts:
            files.extend(list(Path(input_dir).glob(ext)))

        results = {}
        for fp in tqdm(files, desc="Inference"):
            try:
                outputs, vis = self.predict_image(fp)
                results[str(fp)] = {
                    'boxes': outputs["instances"].pred_boxes.tensor.cpu().numpy(),
                    'scores': outputs["instances"].scores.cpu().numpy(),
                    'classes': outputs["instances"].pred_classes.cpu().numpy(),
                }
                if save_vis:
                    cv2.imwrite(os.path.join(output_dir, f"pred_{fp.name}"), vis)
            except Exception as e:
                print(f"Error processing {fp.name}: {e}")
        return results

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_dataset(self, dataset_name, output_dir=None):
        """
        Run COCO evaluation on a registered dataset.

        Args:
            dataset_name: Registered name, e.g. ``data_detection_test``.
            output_dir: Where to store eval artefacts.

        Returns:
            COCO eval metrics dict.
        """
        if output_dir is None:
            output_dir = "./test_output"
        os.makedirs(output_dir, exist_ok=True)

        evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
        test_loader = build_detection_test_loader(self.cfg, dataset_name)
        results = inference_on_dataset(self.predictor.model, test_loader, evaluator)
        return results

    def find_best_threshold(self, dataset_name, output_dir=None,
                            low=0.1, high=1.0, step=0.05):
        """
        Sweep score thresholds and return the one that maximises AP50.
        """
        if output_dir is None:
            output_dir = "./test_output"
        os.makedirs(output_dir, exist_ok=True)

        best_ap50, best_thresh = -1, -1
        for thresh in np.arange(low, high, step):
            self.update_threshold(float(thresh))
            evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
            test_loader = build_detection_test_loader(self.cfg, dataset_name)
            metrics = inference_on_dataset(self.predictor.model, test_loader, evaluator)
            ap50 = metrics["bbox"]["AP50"]
            print(f"  threshold={thresh:.2f}  AP50={ap50:.3f}")
            if ap50 > best_ap50:
                best_ap50 = ap50
                best_thresh = thresh

        print(f"\nBest AP50={best_ap50:.3f} at threshold={best_thresh:.2f}")
        self.update_threshold(float(best_thresh))
        return best_thresh, best_ap50

    def compute_false_negatives(self, dataset_dicts, iou_threshold=0.5):
        """
        Compute false-negative statistics for a list of dataset dicts.

        Returns:
            (fn_count, total_count, category_fn_counts)
        """
        class_names = self.config.get('class_names', ["tap", "flange", "bend", "tee"])
        cat_fn = {name: 0 for name in class_names}
        fn_total, total = 0, 0

        for d in tqdm(dataset_dicts, desc="FN analysis"):
            img = cv2.imread(d["file_name"])
            outputs = self.predictor(img)
            pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

            for ann in d["annotations"]:
                total += 1
                gt = BoxMode.convert(ann["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                ious = [self._iou(gt, pb) for pb in pred_boxes]
                if not ious or max(ious) < iou_threshold:
                    fn_total += 1
                    cat_fn[class_names[ann["category_id"]]] += 1

        return fn_total, total, cat_fn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def update_threshold(self, new_threshold):
        self.score_threshold = new_threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = new_threshold
        self.predictor = DefaultPredictor(self.cfg)

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return inter / float(areaA + areaB - inter)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ILI Component Detection — Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--model_path', type=str, required=True, help='Trained .pth checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Image path or directory')
    parser.add_argument('--output_dir', type=str, default='./inference_results')
    parser.add_argument('--score_threshold', type=float, default=None)
    args = parser.parse_args()

    inference = DetectionInference(args.config, args.model_path, args.score_threshold)

    if os.path.isfile(args.input):
        outputs, vis = inference.predict_image(args.input)
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"pred_{Path(args.input).name}")
        cv2.imwrite(out_path, vis)
        print(f"Saved to {out_path}")
    elif os.path.isdir(args.input):
        results = inference.predict_batch(args.input, args.output_dir)
        print(f"Processed {len(results)} images → {args.output_dir}")
    else:
        raise ValueError(f"Invalid input: {args.input}")


if __name__ == '__main__':
    main()
