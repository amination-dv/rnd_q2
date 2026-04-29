import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt

from torchcam import methods
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rnd.utils.modeling import get_model
from rnd.utils.transform import resize_to_224
from rnd.utils.dataset_multi_channel_v3 import NumpyImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import random

#criterion = nn.BCEWithLogitsLoss()
#
def set_seed(seed):
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Ensure that CUDA operations are deterministic if using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        # # slower but more reproducibility
        # torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimizer, device, print_every=100):
    model.train()
    running_loss = 0.0
    total = len(train_loader)
    for batch_idx, (inputs, labels, _) in enumerate(train_loader, 1):
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        raw_outputs = model(inputs)
        loss = criterion(raw_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % print_every == 0 or batch_idx == total:
            avg = running_loss / batch_idx
            print(f"[Train] Batch {batch_idx}/{total}  BatchLoss: {loss.item():.4f}  AvgLoss: {avg:.4f}")

    return running_loss / total


def validate(model, val_loader, criterion, device, full_ds, val_ds, batch_size, threshold=0.5):
    model.eval()
    val_loss = 0.0

    all_val_labels = []      # will collect arrays shape [C]
    all_val_logits = []      # raw logits
    all_paths = []

    with torch.inference_mode():
        for batch_idx, (inputs, labels, _) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.float().to(device)        # multi-label targets

            raw_outputs = model(inputs)               # [B, C]
            loss = criterion(raw_outputs, labels)
            val_loss += loss.item()

            all_val_labels.append(labels.cpu().numpy())         # accumulate
            all_val_logits.append(raw_outputs.cpu().numpy())

            # Keep paths (approximate mapping)
            start = batch_idx * batch_size
            end = start + len(labels)
            if hasattr(val_ds, 'indices'):
                indices = val_ds.indices[start:end]
            else:
                indices = list(range(start, end))
            batch_paths = [full_ds.samples[i][0] for i in indices]
            all_paths.extend(batch_paths)

    # Stack
    all_val_labels = np.vstack(all_val_labels)    # [N, C]
    all_val_logits = np.vstack(all_val_logits)    # [N, C]

    # Probabilities
    all_val_probs = 1 / (1 + np.exp(-all_val_logits))

    # Binarize with threshold per class
    val_preds = (all_val_probs >= threshold).astype(int)


    precision_macro = precision_score(all_val_labels, val_preds, average='macro', zero_division=0)
    recall_macro    = recall_score(all_val_labels, val_preds, average='macro', zero_division=0)
    f1_macro        = f1_score(all_val_labels, val_preds, average='macro', zero_division=0)

    # Also micro (global)
    precision_micro = precision_score(all_val_labels, val_preds, average='micro', zero_division=0)
    recall_micro    = recall_score(all_val_labels, val_preds, average='micro', zero_division=0)
    f1_micro        = f1_score(all_val_labels, val_preds, average='micro', zero_division=0)

    # Per-class precision/recall (array length C)
    precision_per_class = precision_score(all_val_labels, val_preds, average=None, zero_division=0)
    recall_per_class    = recall_score(all_val_labels, val_preds, average=None, zero_division=0)

    # Sample-level “misclassified” = any mismatch between label vector & prediction
    misclassified = []
    mism_mask = (val_preds != all_val_labels).any(axis=1)
    for i, bad in enumerate(mism_mask):
        if bad:
            misclassified.append(str(all_paths[i]))

    results = {
        'loss': val_loss / len(val_loader),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'labels': all_val_labels.tolist(),
        'probabilities': all_val_probs.tolist(),
        'predictions': val_preds.tolist(),
        'paths': all_paths,
        'misclassified': misclassified,
        'threshold': threshold
    }
    return results



def validate_all_positive_binary(model, val_loader, device, threshold=0.5):
    """
    Validation for a special set where every sample is known to be POSITIVE (dent present),
    but we do NOT know which of the multi-label classes apply.

    Rule:
      - Model outputs logits [B, C].
      - Convert to probs with sigmoid.
      - A sample is predicted POSITIVE if any class prob >= threshold.
      - Otherwise predicted NEGATIVE.

    Ground-truth labels here = 1 for every sample.
    Precision will always be 1.0 (no negative GT), recall measures detection rate.
    """
    model.eval()
    all_max_probs = []
    all_preds = []
    all_paths = []

    with torch.inference_mode():
        for batch_idx, (inputs, _, _) in enumerate(val_loader):
            inputs = inputs.to(device)
            logits = model(inputs)                # [B, C]
            probs = torch.sigmoid(logits)         # [B, C]
            max_prob, _ = probs.max(dim=1)        # [B]
            preds = (max_prob >= threshold).int() # [B]

            all_max_probs.append(max_prob.cpu().numpy())
            all_preds.append(preds.cpu().numpy())


    all_max_probs = np.concatenate(all_max_probs)   # [N]
    all_preds = np.concatenate(all_preds).astype(int)
    N = all_preds.shape[0]

    # All ground-truth = 1
    tp = all_preds.sum()
    fn = N - tp
    precision = 1.0 if tp > 0 else 0.0          # by construction
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0

    missed_indices = np.where(all_preds == 0)[0]
    missed_paths = [all_paths[i] for i in missed_indices] if all_paths else []

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed_paths": missed_paths
    }



def visualize_gradcam(model, val_loader, device, full_ds, val_ds, 
                     gradcam_dir, num_positive_samples=20):
    """Generate GradCAM visualizations for positive samples"""
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Initialize Grad-CAM extractor
    cam_extractor = methods.GradCAM(model, target_layer="base_model.layer4")
    
    # Process a subset of validation data for visualization
    model.eval()
    count = 0
    
    # Create iterators for both loaders
    val_iter = iter(val_loader)

    
    # Process batches until we get enough positive samples
    while count < num_positive_samples:
        try:
            # Get normalized inputs for model and original inputs for visualization
            inputs, labels, file_names = next(val_iter)
            
            # Get model predictions
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(axis=1)

            # Process each image in the batch, but only if it's a positive sample
            for i in range(inputs.size(0)):
                # Only process positive samples (where label is 1)
                if 0 in labels[i]:
                    if count >= num_positive_samples:
                        break
                        
                    
                    # Get normalized image for model
                    input_tensor = inputs[i].unsqueeze(0)
                    label = labels[i]
                    pred = preds[i]
                        
                    # Grad-CAM requires gradients
                    input_tensor.requires_grad_()
                    output = model(input_tensor)
                    probs = torch.sigmoid(output)
                    predicted_labels = (probs >= 0.5).int()
                    class_idx = output.argmax().item()
                    
                    # Compute activation map
                    activation_map = cam_extractor(class_idx, output)
                    
                    # Get CAM as numpy array
                    cam = activation_map[0].squeeze().cpu().numpy()
                    #change plt styple to dark background
                    plt.style.use('dark_background')
                    # Create figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Plot original image
                    ax1.imshow(
                        input_tensor[0,2,:,:].cpu().detach().numpy(), cmap="inferno", aspect="auto",
                        interpolation="nearest",
                    )
                    ax1.set_title(f"Original (Label: {label})")
                    ax1.axis('off')
                    
                    # Plot heatmap
                    im = ax2.imshow(cam, cmap='jet', aspect="auto",
                        interpolation="bilinear", alpha=0.8)
                    ax2.set_title(f"GradCAM Heatmap (Pred: {(predicted_labels == 1).nonzero(as_tuple=True)[1].cpu().numpy().tolist()})")
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                    
                    # # Plot overlay
                    # ax3.imshow(
                    #     input_tensor[0,0,:,:].cpu().detach().numpy(), cmap="inferno", aspect="auto",
                    #     interpolation="nearest",
                    # )
                    # ax3.imshow(cam, cmap='jet', aspect="auto",
                    #     interpolation="bilinear", alpha=0.2)
                    # ax3.set_title("Overlay")
                    # ax3.axis('off')
                    
                    # Add title with prediction information and filename
                    # prediction_result = "Correct" if pred == 1 else "Incorrect"
                    # plt.suptitle(f"File: {file_names[i]}\nPositive Sample - {prediction_result} Prediction", 
                    #             fontsize=16)
                    plt.suptitle("FHR4 data qualitative analysis")

                    output_filename = f"positive_sample_{count}_{file_names[i]}_pred_{pred}.png"
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(gradcam_dir, output_filename), 
                                bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    
                    # Log to wandb if available
                    if 'wandb' in globals():
                        wandb.log({
                            f"gradcam_positive_{count}": wandb.Image(
                                os.path.join(gradcam_dir, output_filename),
                                caption=f"File: {file_names[i]} | Prediction: {pred}"
                            )
                        })
                    
                    count += 1
                
        except StopIteration:
            break
            
    return count


def log_wandb_results(epoch, train_loss, val_results, val_results2):
    """Log metrics to Weights & Biases"""

    if 'wandb' not in globals():
        return

    # Log metrics
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_threshold': val_results['threshold'],
        'val_loss': val_results['loss'],
        'val_precision_macro': val_results['precision_macro'],
        'val_recall_macro': val_results['recall_macro'],
        'val_f1_macro': val_results['f1_macro'],
        'val_precision_micro': val_results['precision_micro'],
        'val_recall_micro': val_results['recall_micro'],
        'val_f1_micro': val_results['f1_micro'],
        'val2_precision': val_results2['precision'],
        'val2_recall': val_results2['recall'],
        'val2_f1': val_results2['f1'],
    })

    
    
 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model, weights = get_model(args.model, args.dense_units, args.dropout)
    model = model.to(device)

    # Transform pipeline - keep a separate copy for visualization
    # transform = transforms.Compose(
    #     [
    #         transforms.Lambda(resize_to_224),
    #         transforms.ToTensor(),
    #         transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    #         transforms.Normalize(
    #             mean=weights.transforms().mean, std=weights.transforms().std
    #         ),
    #     ]
    # )
    transform = transforms.Compose(
        [
            transforms.Lambda(resize_to_224), # should output (C, H, W)
            transforms.Lambda(lambda x: torch.from_numpy(x).float()),
            transforms.Normalize(
                mean=weights.transforms().mean, std=weights.transforms().std
            ),
        ]
    )
    

    # Dataset and DataLoader
    full_ds = NumpyImageFolder(root_dir=args.data_dir, transform=transform, debug=True)
    val_ds2 = NumpyImageFolder(root_dir=args.val_data_dir, transform=transform, debug=True)
    
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    # Use same random split for both datasets
# Use same random split for both datasets
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=generator)

    
    # Get labels for all samples in your training set
    #train_labels = [full_ds.samples[i][1] for i in train_ds.indices]

    # # # Calculate weights: more weight for positive class
    # pos_weight = 0.3 / sum(np.array(train_labels) == 1)
    # neg_weight = 0.7 / sum(np.array(train_labels) == 0)
    # weights = [pos_weight if label == 1 else neg_weight for label in train_labels]

    # sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
    val_loader2 = DataLoader(val_ds2, batch_size=args.batch_size, shuffle=True)

    # Prepare misclassified samples json file
    misclassified_json_path = os.path.join(args.save_dir, 'misclassified_samples.json')
    if os.path.exists(misclassified_json_path):
        with open(misclassified_json_path, 'w') as f:
            json.dump({}, f)
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(misclassified_json_path, 'w') as f:
            json.dump({}, f)

    criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr,
        momentum=0.9,  # Adding momentum helps SGD converge better
        weight_decay=1e-4  # L2 regularization to prevent overfitting
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    if args.train:
        # Training
        for epoch in range(args.epochs):
            # Train for one epoch
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}")
            
            # Validate
            val_results = validate(model, val_loader, criterion, device, 
                                full_ds, val_ds, args.batch_size,threshold=0.9)
            val_results2 = validate_all_positive_binary(model, val_loader2, device,
                                 threshold=0.9)

            # Save misclassified samples for this epoch
            with open(misclassified_json_path, 'r+') as f:
                data = json.load(f)
                data[f'epoch_{epoch+1}'] = val_results['misclassified']
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
                
            # Log metrics to wandb
            log_wandb_results(epoch, train_loss, val_results, val_results2)

            # Update learning rate scheduler
            scheduler.step(val_results['loss'])
            print(f"Val precision_macro: {val_results['precision_macro']:.4f}, ",
                f"Val recall_macro: {val_results['recall_macro']:.4f}, ",
                f"Val precision_micro: {val_results['precision_micro']:.4f}, ",
                f"Val recall_micro: {val_results['recall_micro']:.4f}, ",
                f"Val precision_per_class: {val_results['precision_per_class']}, ",
                f"Val recall_per_class: {val_results['recall_per_class']}, ",
                f"Val_loss: {val_results['loss']:.4f}")
            print(f"LR: {scheduler.get_last_lr()}")

            # Save model
            os.makedirs(args.save_dir, exist_ok=True)
            model_filename = f"{args.model}_v1_33_{epoch}.pth"
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), model_path)

            # Save the model to wandb for this epoch
            if 'wandb' in globals():
                # Create a unique artifact for this epoch
                artifact = wandb.Artifact(
                    name=f"{args.model}_model_v1_33_epoch_{epoch+1}", 
                    type="model",
                    description=f"Dent detection {args.model} model at epoch {epoch+1}/{args.epochs}"
                )
                
                # Add the model file to the artifact
                artifact.add_file(model_path)
                
                # Log metadata specific to this epoch
                metadata = {
                    "epoch": epoch+1,
                    "architecture": args.model,
                    "dense_units": args.dense_units,
                    "dropout": args.dropout,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "precision_macro": val_results['precision_macro'],
                    "recall_macro": val_results['recall_macro'],
                    "f1_macro": val_results['f1_macro'],
                    "train_loss": train_loss,
                    "val_loss": val_results['loss']
                }
                artifact.metadata = metadata
                
                # Log the artifact to W&B
                wandb.log_artifact(artifact)
    else:

        # Load model if not training
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{args.model}_v1_33_{epoch}.pth")))
        print(f"Model loaded from {args.save_dir}/{args.model}_v1_33_{epoch}.pth")
        #v1, v1.0, v1.1, v1.2

    # Generate GradCAM visualizations
    gradcam_dir = os.path.join(args.save_dir, "gradcam_dark")
    num_samples = visualize_gradcam(model, val_loader, device, 
                                   full_ds, val_ds, gradcam_dir, num_positive_samples=100)
    
    print(f"✅ GradCAM visualizations saved to: {gradcam_dir} ({num_samples} positive samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet", choices=["resnet", "mobilenet"]
    )
    parser.add_argument("--data-dir", type=str, default="./data/fhr4/final_data_v1.3/training")
    parser.add_argument("--val-data-dir", type=str, default="./data/fhr4/final_data_v1.3/validation")

    parser.add_argument("--train", action="store_true", default=True, help="Set to False to load model and generate GradCAM")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=20, help="Number of classes for one-hot encoding")
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="models/dent_models")
    args = parser.parse_args()
    set_seed(42)
    
    # Initialize wandb
    wandb.init(project="dent_arm_encoder", config=vars(args))
    main(args)  