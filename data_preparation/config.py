"""Configuration for Semi-Automatic QC Pipeline."""

from pathlib import Path

# Base paths
BASE_DATA_PATH = Path("/home/zmirikha/Github/rnd_q2/data/0ABP0TFUSH1/Approved")

# PNG visualization paths (for UI display)
PNG_STD_QC_PATH = BASE_DATA_PATH / "std_normalized" / "qc"
PNG_PATCH_QC_PATH = BASE_DATA_PATH / "patch_normalized" / "qc"

# Numpy paths (for inference)
NPY_STD_PATH = BASE_DATA_PATH / "std_normalized"
NPY_PATCH_PATH = BASE_DATA_PATH / "patch_normalized"

# Model configuration
MODEL_S3_URI = "s3://dv-ml-models/dent_detection_ae/v3/resnet_v1_3_16.pth"
MODEL_CACHE_DIR = Path("/home/zmirikha/Github/rnd_q2/data_preparation/model_cache")
MODEL_LOCAL_PATH = MODEL_CACHE_DIR / "resnet_v1_3_16.pth"

MODEL_CONFIG = {
    "name": "resnet",
    "dense_units": 128,
    "dropout": 0.5,
    "num_classes": 20,
}

# Inference settings
INFERENCE_BATCH_SIZE = 32
IMAGE_SIZE = 224
TOTAL_TRACKS = 22  # Total tracks in data (model outputs 20)

# QC thresholds
PROBABILITY_THRESHOLD = 0.7  # Auto-approve if max prob > threshold
CONFIDENCE_THRESHOLD = 0.5  # Low confidence if below this

# Output paths
QC_RESULTS_DIR = Path("/home/zmirikha/Github/rnd_q2/data_preparation/qc_results")
QC_RESULTS_FILE = QC_RESULTS_DIR / "qc_results.json"

# Label mapping (tracks 0-21)
NUM_LABELS = 22
LABEL_NAMES = [f"{i:03d}" for i in range(NUM_LABELS)]
