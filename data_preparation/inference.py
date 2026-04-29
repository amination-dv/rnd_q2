"""Model inference for QC pipeline."""

from pathlib import Path

import boto3
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from statsmodels.tsa.seasonal import seasonal_decompose

from config import (
    MODEL_S3_URI,
    MODEL_LOCAL_PATH,
    MODEL_CACHE_DIR,
    MODEL_CONFIG,
    IMAGE_SIZE,
    TOTAL_TRACKS,
)


class ResNet18Classifier(nn.Module):
    """ResNet18-based classifier for track-level predictions."""

    def __init__(
        self,
        dense_units: int = 128,
        dropout_rate: float = 0.5,
        num_classes: int = 20,
        weights: ResNet18_Weights = None,
    ):
        super().__init__()
        self.base_model = models.resnet18(weights=weights)
        in_features = self.base_model.fc.in_features
        classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes),
        )
        self.base_model.fc = classifier

    def forward(self, x):
        return self.base_model(x)


def download_model_from_s3(s3_uri: str, local_path: Path) -> Path:
    """Download model from S3 if not already cached."""
    if local_path.exists():
        print(f"Model already cached at {local_path}")
        return local_path

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    print(f"Downloading model from s3://{bucket}/{key}...")

    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))
    print(f"Model saved to {local_path}")
    return local_path


def load_model(model_path: Path | None = None) -> tuple[ResNet18Classifier, transforms.Compose]:
    """Load model and return with preprocessing transform."""
    if model_path is None:
        model_path = download_model_from_s3(MODEL_S3_URI, MODEL_LOCAL_PATH)

    model = ResNet18Classifier(
        dense_units=MODEL_CONFIG["dense_units"],
        dropout_rate=MODEL_CONFIG["dropout"],
        num_classes=MODEL_CONFIG["num_classes"],
        weights=None,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    weights = ResNet18_Weights.DEFAULT
    transform = transforms.Compose([
        transforms.Lambda(lambda x: resize_to_224(x)),
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    return model, transform


def resize_to_224(x: np.ndarray) -> np.ndarray:
    """Resize image to 224x224."""
    if x.ndim == 2:
        return cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return np.stack(
        [cv2.resize(xi, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA) for xi in x],
        axis=0,
    )


def create_3channel_image(img_std: np.ndarray, img_patch: np.ndarray) -> np.ndarray:
    """Create 3-channel image from std and patch normalized arrays.
    
    Args:
        img_std: Standard normalized array, shape (tracks, time_steps)
        img_patch: Patch normalized array, shape (tracks, time_steps)
    
    Returns:
        3-channel image: (3, tracks, time_steps) with channels:
        - Channel 0: patch normalized
        - Channel 1: patch trend (seasonal decomposition)
        - Channel 2: std normalized
    """
    img_patch_trend = seasonal_decompose(
        img_patch.T, model="additive", period=30, extrapolate_trend=2
    ).trend.T

    return np.stack((img_patch, img_patch_trend, img_std), axis=0)


def run_inference(
    model: ResNet18Classifier,
    transform: transforms.Compose,
    img_std: np.ndarray,
    img_patch: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run inference on a single sample.
    
    Args:
        model: Loaded model
        transform: Preprocessing transform
        img_std: Standard normalized array, shape (tracks, time_steps)
        img_patch: Patch normalized array, shape (tracks, time_steps)
        device: Device to run inference on
    
    Returns:
        Probability array of shape (num_classes,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    img = create_3channel_image(img_std, img_patch)
    
    num_tracks = img.shape[1]
    model_tracks = MODEL_CONFIG["num_classes"]

    if num_tracks > model_tracks:
        img_left = img[:, :model_tracks, :]
        img_right = img[:, num_tracks - model_tracks:, :]

        img_left = transform(img_left).unsqueeze(0).to(device)
        img_right = transform(img_right).unsqueeze(0).to(device)

        with torch.inference_mode():
            out_left = model(img_left)
            out_right = model(img_right)

            overlap_start = num_tracks - model_tracks
            overlap_end = model_tracks

            raw_outputs = torch.zeros(num_tracks, device=device)
            raw_outputs[:overlap_start] = out_left[0, :overlap_start]
            raw_outputs[overlap_start:overlap_end] = torch.max(
                out_left[0, overlap_start:overlap_end],
                out_right[0, : (overlap_end - overlap_start)],
            )
            raw_outputs[overlap_end:] = out_right[0, (overlap_end - overlap_start):]

            probs = raw_outputs.sigmoid().cpu().numpy()
    else:
        padding_needed = model_tracks - num_tracks
        channel_mean = img.mean(axis=1, keepdims=True)
        padding = np.repeat(channel_mean, padding_needed, axis=1)
        img_padded = np.concatenate([img, padding], axis=1)

        img_tensor = transform(img_padded).unsqueeze(0).to(device)

        with torch.inference_mode():
            raw_outputs = model(img_tensor)
            probs = raw_outputs[0].sigmoid().cpu().numpy()

    return probs


def run_batch_inference(
    model: ResNet18Classifier,
    transform: transforms.Compose,
    samples: list[tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
    device: torch.device | None = None,
) -> list[np.ndarray]:
    """Run inference on a batch of samples.
    
    Args:
        model: Loaded model
        transform: Preprocessing transform
        samples: List of (img_std, img_patch) tuples
        batch_size: Batch size for inference
        device: Device to run inference on
    
    Returns:
        List of probability arrays
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    results = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        batch_probs = []

        for img_std, img_patch in batch:
            probs = run_inference(model, transform, img_std, img_patch, device)
            batch_probs.append(probs)

        results.extend(batch_probs)

    return results


if __name__ == "__main__":
    print("Loading model...")
    model, transform = load_model()
    print(f"Model loaded: {MODEL_CONFIG}")
    
    test_std = np.random.randn(TOTAL_TRACKS, 300).astype(np.float32)
    test_patch = np.random.randn(TOTAL_TRACKS, 300).astype(np.float32)
    
    print("Running inference on test data...")
    probs = run_inference(model, transform, test_std, test_patch)
    print(f"Output shape: {probs.shape}")
    print(f"Probabilities: {probs}")
