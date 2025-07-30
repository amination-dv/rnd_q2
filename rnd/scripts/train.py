import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve
import wandb
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt

from torchcam import methods
from torchcam.utils import overlay_mask
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rnd.utils.modeling import get_model
from rnd.utils.transform import resize_to_224
from rnd.utils.dataset import NumpyImageFolder
from rnd.utils.augmentation import (
    RandomHorizontalRoll,
    RandomRowSwap,
    RandomApplyNp,
    RandomJitter,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler


criterion = nn.CrossEntropyLoss()

# Then in train_epoch, modify how outputs and labels are handled
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    
    for inputs, labels, _ in train_loader:
        inputs = inputs.to(device)
        # For CrossEntropyLoss, use integer labels without unsqueezing
        labels = labels.long().to(device)  # Changed from float to long, removed unsqueeze

        # Get model outputs and reshape for CrossEntropyLoss 
        raw_outputs = model(inputs)
        loss = criterion(raw_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    train_loss = running_loss / len(train_loader)
    return train_loss


def validate(model, val_loader, criterion, device, full_ds, val_ds, batch_size):
    """Run validation and compute metrics"""
    model.eval()
    correct, total = 0, 0
    all_val_labels = []
    all_val_probs = []
    all_val_paths = []
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels, _) in enumerate(val_loader):
            inputs = inputs.to(device)
            # For CrossEntropyLoss, use integer labels without unsqueezing
            labels = labels.long().to(device)  # Match format used in train_epoch

            # Get model outputs and reshape for CrossEntropyLoss
            raw_outputs = model(inputs)

            loss = criterion(raw_outputs, labels)
            val_loss += loss.item()

            preds = raw_outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_val_labels.extend(labels.cpu().numpy().flatten())
            all_val_probs.extend(raw_outputs.softmax(dim=1).cpu().numpy())  # Store positive class probs

            # Track sample paths for this batch
            start = batch_idx * batch_size
            end = start + len(labels)
            if hasattr(val_ds, 'indices'):
                indices = val_ds.indices[start:end]
            else:
                indices = list(range(start, end))
            batch_paths = [full_ds.samples[i][0] for i in indices]
            all_val_paths.extend(batch_paths)
            
    val_loss = val_loss / len(val_loader)
    acc = correct / total
    
    # Calculate precision, recall, and other metrics
    val_preds = np.argmax(np.array(all_val_probs), axis=1).tolist()
    precision = precision_score(all_val_labels, val_preds, average='macro', zero_division=0)
    recall = recall_score(all_val_labels, val_preds, average='macro', zero_division=0)
    
    # Prepare results
    misclassified = [str(all_val_paths[i]) for i, (pred, label) in 
                     enumerate(zip(val_preds, all_val_labels)) if pred != label]
    
    results = {
        'loss': val_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'labels': all_val_labels,
        'predictions': val_preds,
        'probabilities': all_val_probs,
        'paths': all_val_paths,
        'misclassified': misclassified
    }
    
    return results


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
                if labels[i].item() != 0:
                    if count >= num_positive_samples:
                        break
                        
                    
                    # Get normalized image for model
                    input_tensor = inputs[i].unsqueeze(0)
                    label = labels[i].item()
                    pred = preds[i].item() if preds.dim() > 0 else preds.item()
                        
                    # Grad-CAM requires gradients
                    input_tensor.requires_grad_()
                    output = model(input_tensor)
                    
                    class_idx = output.argmax().item()
                    
                    # Compute activation map
                    activation_map = cam_extractor(class_idx, output)
                    
                    # Get CAM as numpy array
                    cam = activation_map[0].squeeze().cpu().numpy()
                    
                    # Create figure with three subplots
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Plot original image
                    ax1.imshow(
                        input_tensor[0,0,:,:].cpu().detach().numpy(), cmap="inferno", aspect="auto",
                        interpolation="nearest",
                    )
                    ax1.set_title(f"Original (Label: {label})")
                    ax1.axis('off')
                    
                    # Plot heatmap
                    im = ax2.imshow(cam, cmap='jet', aspect="auto",
                        interpolation="bilinear", alpha=0.8)
                    ax2.set_title(f"GradCAM Heatmap (Pred: {pred})")
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                    
                    # Plot overlay
                    ax3.imshow(
                        input_tensor[0,0,:,:].cpu().detach().numpy(), cmap="inferno", aspect="auto",
                        interpolation="nearest",
                    )
                    ax3.imshow(cam, cmap='jet', aspect="auto",
                        interpolation="bilinear", alpha=0.2)
                    ax3.set_title("Overlay")
                    ax3.axis('off')
                    
                    # Add title with prediction information and filename
                    prediction_result = "Correct" if pred == 1 else "Incorrect"
                    plt.suptitle(f"File: {file_names[i]}\nPositive Sample - {prediction_result} Prediction", 
                                fontsize=16)
                    
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


def log_wandb_results(epoch, train_loss, val_results):
    """Log metrics to Weights & Biases"""

    if 'wandb' not in globals():
        return

    # Log metrics
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_results['loss'],
        'val_accuracy': val_results['accuracy'],
        'val_precision': val_results['precision'],
        'val_recall': val_results['recall'],
    })
    
 

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model, weights = get_model(args.model, args.dense_units, args.dropout)
    model = model.to(device)

    # Transform pipeline - keep a separate copy for visualization
    transform = transforms.Compose(
        [
            transforms.Lambda(resize_to_224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(
                mean=weights.transforms().mean, std=weights.transforms().std
            ),
        ]
    )
    

    # Dataset and DataLoader
    full_ds = NumpyImageFolder(root_dir=args.data_dir, transform=transform, debug=True)

    
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    # Use same random split for both datasets
# Use same random split for both datasets
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=generator)

    
    # Get labels for all samples in your training set
    train_labels = [full_ds.samples[i][1] for i in train_ds.indices]

    # Calculate weights: more weight for positive class
    pos_weight = 0.3 / sum(np.array(train_labels) == 1)
    neg_weight = 0.7 / sum(np.array(train_labels) == 0)
    weights = [pos_weight if label == 1 else neg_weight for label in train_labels]

    sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)

    # Prepare misclassified samples json file
    misclassified_json_path = os.path.join(args.save_dir, 'misclassified_samples.json')
    if os.path.exists(misclassified_json_path):
        with open(misclassified_json_path, 'w') as f:
            json.dump({}, f)
    else:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(misclassified_json_path, 'w') as f:
            json.dump({}, f)

    criterion = torch.nn.CrossEntropyLoss()
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
                                full_ds, val_ds, args.batch_size)
            
            # Save misclassified samples for this epoch
            with open(misclassified_json_path, 'r+') as f:
                data = json.load(f)
                data[f'epoch_{epoch+1}'] = val_results['misclassified']
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
                
            # Log metrics to wandb
            log_wandb_results(epoch, train_loss, val_results)
            
            # Update learning rate scheduler
            scheduler.step(val_results['loss'])
            print(f"Val Accuracy: {val_results['accuracy']:.4f}, "
                f"Precision: {val_results['precision']:.4f}, "
                f"Recall: {val_results['recall']:.4f}")
            print(f"LR: {scheduler.get_last_lr()}")

        # Save model
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(
            model.state_dict(), os.path.join(args.save_dir, f"{args.model}_final.pth")
        )
    else:

        # Load model if not training
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{args.model}_final.pth")))
        print(f"Model loaded from {args.save_dir}/{args.model}_final.pth")
    
    # Generate GradCAM visualizations
    gradcam_dir = os.path.join(args.save_dir, "gradcam")
    num_samples = visualize_gradcam(model, val_loader, device, 
                                   full_ds, val_ds, gradcam_dir, num_positive_samples=300)
    
    print(f"✅ Model saved to: {args.save_dir}/{args.model}_final.pth")
    print(f"✅ GradCAM visualizations saved to: {gradcam_dir} ({num_samples} positive samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="resnet", choices=["resnet", "mobilenet"]
    )
    parser.add_argument("--data-dir", type=str, default="./data/fhr4/final_data_21_classifier")
    parser.add_argument("--train", action="store_true", default=False, help="Set to False to load model and generate GradCAM")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="models/dent_models")
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="dent_arm_encoder", config=vars(args))
    main(args)