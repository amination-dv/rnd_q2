# Semi-Automatic QC Pipeline

A web-based tool for quality control of dent detection data using a pretrained ResNet18 model.

## Quick Start

```bash
# Navigate to the folder
cd /home/zmirikha/Github/rnd_q2/data_preparation

# Install dependencies (if not already installed)
uv pip install -r requirements.txt --python /home/zmirikha/ilipy/bin/python

# Run the app
./run.sh
```

Then open http://localhost:8501 in your browser.

## Features

| Page | Description |
|------|-------------|
| **Dashboard** | Overview of QC progress with status breakdown and label distribution |
| **Auto QC** | Run batch inference on pending samples, auto-approve high confidence matches |
| **Manual Review** | Review samples with images, probability chart, run model, approve/reject/edit |
| **Export** | Download QC results as CSV |

## Manual Review Features

- **Navigation**: Previous/Next buttons, dropdown selector, **Next Pending** button to jump to next unreviewed sample
- **Sample Viewer**: Side-by-side STD and Patch normalized images
- **Probability Display**: 
  - Raw labels and model predictions list
  - Interactive bar chart with color coding (green > 0.7, orange 0.3-0.7, gray < 0.3)
  - Top predictions summary
- **Label Editor**: Checkbox grid for all 22 tracks, threshold slider
- **Actions**: Approve, Reject, Save Edit, Reset to Pending
- **Model Inference**:
  - Run model on current sample (or re-run if predictions exist)
  - Run model on all samples without predictions (batch processing with progress bar)

## Workflow

1. **Dashboard**: Check overall progress
2. **Auto QC** (optional): Set probability threshold (default 0.7), click "Run Auto QC" to auto-approve matching predictions
3. **Manual Review**: 
   - Use "Run Model on All Samples" to generate predictions for all samples
   - Use "Next Pending" to navigate through unreviewed samples
   - Review predictions, edit labels if needed, approve/reject
4. **Export**: Download final results as CSV

## Data Paths

- **PNG Images (visualization)**: 
  - `data/0ABP0TFUSH1/Approved/std_normalized/qc/`
  - `data/0ABP0TFUSH1/Approved/patch_normalized/qc/`
- **NPY Files (inference)**:
  - `data/0ABP0TFUSH1/Approved/std_normalized/`
  - `data/0ABP0TFUSH1/Approved/patch_normalized/`

## Model

- **S3 Path**: `s3://dv-ml-models/dent_detection_ae/v3/resnet_v1_3_16.pth`
- **Architecture**: ResNet18 with custom classifier (128 dense units, 20 outputs for tracks 0-19, extended to 22 via overlapping inference)
- **Local Cache**: `model_cache/resnet_v1_3_16.pth`

## Configuration

Edit `config.py` to change:
- Data paths
- Probability thresholds (default: 0.7 for auto-approve, 0.5 for low confidence)
- Model parameters

## Output

- **QC Results**: `qc_results/qc_results.json` (auto-saved after each action)
- **Export**: CSV download with sample_id, raw_labels, final_labels, status, model_max_prob, notes, timestamp

## Auto-Approval for Pre-QC'd Distances

Samples with distance < 0007912876 or > 0124590222 can be auto-approved via script (already QC'd outside the API):

```python
# Already applied - see qc_results_backup.json for original state
```
