"""Data management for QC pipeline."""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np

from config import (
    PNG_STD_QC_PATH,
    PNG_PATCH_QC_PATH,
    NPY_STD_PATH,
    NPY_PATCH_PATH,
    QC_RESULTS_FILE,
    QC_RESULTS_DIR,
    NUM_LABELS,
)


class QCStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


@dataclass
class Sample:
    """Represents a single QC sample."""
    
    sample_id: str
    raw_labels: list[str]
    png_std_path: str | None = None
    png_patch_path: str | None = None
    npy_std_path: str | None = None
    npy_patch_path: str | None = None
    model_predictions: list[float] | None = None
    final_labels: list[str] | None = None
    status: str = QCStatus.PENDING
    reviewer_notes: str = ""
    timestamp: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Sample":
        return cls(**data)


def parse_filename(filename: str) -> tuple[str, list[str]] | None:
    """Parse filename to extract sample ID and raw labels.
    
    Filename formats:
    - PNG: 0000034579_['002'].png  or  0000922910_015.png (sampleid_track)
    - NPY: d0000034579_['002'].npy or  d0000922910_015.npy
    
    Returns:
        Tuple of (sample_id, labels) or None if parsing fails
    """
    name = Path(filename).stem

    if name.startswith("d"):
        name = name[1:]

    # Format: sampleid_['002'] or sampleid_["002", "003"]
    match = re.match(r"(\d+)_(\[.*\])", name)
    if match:
        sample_id = match.group(1)
        labels_str = match.group(2)
        try:
            labels_str = labels_str.replace("'", '"')
            labels = json.loads(labels_str)
            if isinstance(labels, list):
                labels = [str(l).zfill(3) for l in labels]
            else:
                labels = [str(labels).zfill(3)]
        except json.JSONDecodeError:
            labels_match = re.findall(r"'(\d+)'", match.group(2))
            labels = [l.zfill(3) for l in labels_match]
        return sample_id, labels

    # Format: sampleid_track (e.g. 0000922910_015.png)
    match_simple = re.match(r"(\d+)_(\d+)", name)
    if match_simple:
        sample_id = match_simple.group(1)
        track = match_simple.group(2).zfill(3)
        return sample_id, [track]

    return None


def find_matching_files(
    sample_id: str,
    raw_labels: list[str],
    png_std_dir: Path,
    png_patch_dir: Path,
    npy_std_dir: Path,
    npy_patch_dir: Path,
) -> dict[str, str | None]:
    """Find matching files across all directories for a sample."""
    
    labels_str = str(raw_labels).replace('"', "'")
    
    png_name = f"{sample_id}_{labels_str}.png"
    npy_name = f"d{sample_id}_{labels_str}.npy"
    # Also support simple format: sampleid_track (e.g. 0000922910_015.png)
    if len(raw_labels) == 1:
        png_name_simple = f"{sample_id}_{raw_labels[0]}.png"
        npy_name_simple = f"d{sample_id}_{raw_labels[0]}.npy"
    else:
        png_name_simple = npy_name_simple = None
    
    def find_file(directory: Path, patterns: list[str]) -> str | None:
        for pattern in patterns:
            if pattern is None:
                continue
            candidate = directory / pattern
            if candidate.exists():
                return str(candidate)
        for f in directory.iterdir():
            if sample_id in f.name:
                return str(f)
        return None
    
    png_patterns = [png_name, png_name_simple]
    npy_patterns = [npy_name, npy_name_simple]
    return {
        "png_std_path": find_file(png_std_dir, png_patterns),
        "png_patch_path": find_file(png_patch_dir, png_patterns),
        "npy_std_path": find_file(npy_std_dir, npy_patterns),
        "npy_patch_path": find_file(npy_patch_dir, npy_patterns),
    }


class DataManager:
    """Manages QC samples and results."""
    
    def __init__(
        self,
        png_std_qc_path: Path = PNG_STD_QC_PATH,
        png_patch_qc_path: Path = PNG_PATCH_QC_PATH,
        npy_std_path: Path = NPY_STD_PATH,
        npy_patch_path: Path = NPY_PATCH_PATH,
        results_file: Path = QC_RESULTS_FILE,
    ):
        self.png_std_qc_path = Path(png_std_qc_path)
        self.png_patch_qc_path = Path(png_patch_qc_path)
        self.npy_std_path = Path(npy_std_path)
        self.npy_patch_path = Path(npy_patch_path)
        self.results_file = Path(results_file)
        
        self.samples: dict[str, Sample] = {}
        self._load_results()
    
    def _load_results(self):
        """Load existing QC results from file."""
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                data = json.load(f)
                for sample_id, sample_data in data.items():
                    # Migrate old folder names so saved paths work with std_/patch_normalized
                    for key in ("png_std_path", "png_patch_path", "npy_std_path", "npy_patch_path"):
                        if key in sample_data and sample_data[key]:
                            sample_data[key] = (
                                sample_data[key]
                                .replace("grouped_std_normalized", "std_normalized")
                                .replace("grouped_patch_normalized", "patch_normalized")
                            )
                    self.samples[sample_id] = Sample.from_dict(sample_data)
            print(f"Loaded {len(self.samples)} samples from {self.results_file}")
    
    def save_results(self):
        """Save QC results to file."""
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: s.to_dict() for sid, s in self.samples.items()}
        with open(self.results_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.samples)} samples to {self.results_file}")
    
    def scan_samples(self) -> int:
        """Scan QC folder and build sample index.
        
        Uses PNG files in std_normalized/qc as reference.
        
        Returns:
            Number of new samples found
        """
        new_count = 0
        
        for png_file in self.png_std_qc_path.glob("*.png"):
            parsed = parse_filename(png_file.name)
            if parsed is None:
                print(f"Warning: Could not parse filename: {png_file.name}")
                continue
            
            sample_id, raw_labels = parsed
            
            if sample_id in self.samples:
                continue
            
            files = find_matching_files(
                sample_id,
                raw_labels,
                self.png_std_qc_path,
                self.png_patch_qc_path,
                self.npy_std_path,
                self.npy_patch_path,
            )
            
            sample = Sample(
                sample_id=sample_id,
                raw_labels=raw_labels,
                **files,
            )
            self.samples[sample_id] = sample
            new_count += 1
        
        print(f"Found {new_count} new samples, {len(self.samples)} total")
        return new_count
    
    def get_sample(self, sample_id: str) -> Sample | None:
        """Get a sample by ID."""
        return self.samples.get(sample_id)
    
    def get_all_samples(self) -> list[Sample]:
        """Get all samples."""
        return list(self.samples.values())
    
    def get_samples_by_status(self, status: QCStatus) -> list[Sample]:
        """Get samples filtered by status."""
        return [s for s in self.samples.values() if s.status == status]
    
    def get_pending_samples(self) -> list[Sample]:
        """Get all pending samples."""
        return self.get_samples_by_status(QCStatus.PENDING)
    
    def update_sample(
        self,
        sample_id: str,
        status: QCStatus | None = None,
        final_labels: list[str] | None = None,
        model_predictions: list[float] | None = None,
        reviewer_notes: str | None = None,
    ):
        """Update a sample's QC results."""
        sample = self.samples.get(sample_id)
        if sample is None:
            raise ValueError(f"Sample not found: {sample_id}")
        
        if status is not None:
            sample.status = status
        if final_labels is not None:
            sample.final_labels = final_labels
        if model_predictions is not None:
            sample.model_predictions = model_predictions
        if reviewer_notes is not None:
            sample.reviewer_notes = reviewer_notes
        
        sample.timestamp = datetime.now().isoformat()
    
    def load_sample_data(self, sample: Sample) -> tuple[np.ndarray, np.ndarray] | None:
        """Load numpy arrays for a sample.
        
        Returns:
            Tuple of (img_std, img_patch) or None if files not found
        """
        if sample.npy_std_path is None or sample.npy_patch_path is None:
            return None
        
        try:
            img_std = np.load(sample.npy_std_path)
            img_patch = np.load(sample.npy_patch_path)
            return img_std, img_patch
        except Exception as e:
            print(f"Error loading sample {sample.sample_id}: {e}")
            return None
    
    def get_statistics(self) -> dict:
        """Get QC statistics."""
        total = len(self.samples)
        by_status = {}
        for status in QCStatus:
            by_status[status] = len(self.get_samples_by_status(status))
        
        return {
            "total": total,
            "by_status": by_status,
            "progress_percent": (
                (by_status.get(QCStatus.APPROVED, 0) + 
                 by_status.get(QCStatus.REJECTED, 0) + 
                 by_status.get(QCStatus.EDITED, 0)) / total * 100
                if total > 0 else 0
            ),
        }
    
    def export_to_csv(self, output_path: Path | str):
        """Export QC results to CSV."""
        import csv
        
        output_path = Path(output_path)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            header = [
                "sample_id",
                "raw_labels",
                "final_labels",
                "status",
                "model_predictions",
                "reviewer_notes",
                "timestamp",
            ]
            writer.writerow(header)
            
            for sample in self.samples.values():
                row = [
                    sample.sample_id,
                    str(sample.raw_labels),
                    str(sample.final_labels) if sample.final_labels else "",
                    sample.status,
                    str(sample.model_predictions) if sample.model_predictions else "",
                    sample.reviewer_notes,
                    sample.timestamp or "",
                ]
                writer.writerow(row)
        
        print(f"Exported {len(self.samples)} samples to {output_path}")


if __name__ == "__main__":
    dm = DataManager()
    dm.scan_samples()
    
    stats = dm.get_statistics()
    print(f"\nStatistics: {stats}")
    
    samples = dm.get_all_samples()[:3]
    for s in samples:
        print(f"\nSample {s.sample_id}:")
        print(f"  Raw labels: {s.raw_labels}")
        print(f"  PNG std: {s.png_std_path}")
        print(f"  PNG patch: {s.png_patch_path}")
        print(f"  NPY std: {s.npy_std_path}")
        print(f"  NPY patch: {s.npy_patch_path}")
    
    dm.save_results()
