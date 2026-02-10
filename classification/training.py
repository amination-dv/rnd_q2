"""
ILI Strip-ViT Training Module (PyTorch Lightning).

Multi-task training:
    1. Classification: Detect component type (Tee, Stopple, Weld, Background).
    2. Alignment: Regress per-track horizontal pixel shifts.

Loss:
    L_total = L_class + lambda_reg * (L_reg * M_component) + lambda_msg * (L_msg * M_component)

    - L_class: CrossEntropyLoss (all samples).
    - L_reg:   L1Loss on predicted vs. ground-truth shifts (masked for background).
    - L_msg:   MSGLoss between predicted aligned image and GT aligned image (optional).

Logging:
    - Class distribution at fit start.
    - Per-step: uncorrelated input, correlated GT, predicted correlated image.
"""

import logging

import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightning.pytorch as ptl
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from Dataset import ILIDataset
from Losses import MultiClassCrossEntropy, MSGLoss
from Models import StripViT
from utils import (
    apply_track_shifts_differentiable,
    log_ili_predictions_to_wandb,
)

LOGGER = logging.getLogger(__name__)


class TrainingModule(ptl.LightningModule):
    """
    PyTorch Lightning module for multi-task ILI Strip-ViT training.

    Handles:
        - Dataset creation and train/val/test splitting.
        - Model instantiation (StripViT).
        - Multi-task loss with background masking.
        - WandB image and metric logging.
    """

    def __init__(self, config):
        super().__init__()
        LOGGER.info("ILI Strip-ViT Training Module init")
        self.config = config
        self.num_tracks = config['num_tracks']

        # ---- Dataset ----
        self.dataset = ILIDataset(
            root=config['data_dir'],
            img_height=config['img_height'],
            img_width=config['img_width'],
            num_tracks=config['num_tracks'],
            max_shift=config['max_shift'],
            background_class=config.get('background_class', 'background'),
            augment=True,
        )

        self.train_size = int(config['train_size'] * len(self.dataset))
        self.val_size = int(config['val_size'] * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        generator = torch.Generator().manual_seed(config['random_state'])
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [self.train_size, self.val_size, self.test_size],
            generator=generator,
        )

        # ---- Model ----
        self.model = StripViT(
            img_height=config['img_height'],
            num_tracks=config['num_tracks'],
            num_classes=config['num_classes'],
            in_channels=config.get('in_channels', 1),
            patch_width=config.get('patch_width', 14),
            backbone_name=config['backbone'],
            train_backbone=config.get('train_backbone', True),
            unfreeze_last_n_blocks=config.get('unfreeze_last_n_blocks', 4),
            intermediate_block_idx=config.get('intermediate_block_idx', 7),
        )

        # ---- Losses ----
        self.ce_loss = MultiClassCrossEntropy()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.use_msg_loss = config.get('use_msg_loss', False)
        if self.use_msg_loss:
            self.msg_loss_fn = MSGLoss()

        # ---- Tracking ----
        self.N_epochs = 5
        self.val_epoch_count = 0
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
        self.all_paths = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.last_N_losses = []

        self.save_hyperparameters()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = optim.RAdam(
            self.model.parameters(),
            lr=self.config['lr'],
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999)),
            weight_decay=self.config.get('weight_decay', 1e-5),
        )
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.config['max_epochs']),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'run_avg_val_loss',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def on_fit_start(self):
        self.logger.log_metrics({
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
        })
        # Log class names
        class_names = self.dataset.classes
        LOGGER.info(f"Classes: {class_names}")
        wandb.log({"class_names": wandb.Table(
            columns=["idx", "name"],
            data=[[i, n] for i, n in enumerate(class_names)],
        )})

    def train_dataloader(self):
        # Balanced sampling: weight each sample by inverse class frequency
        train_labels = [self.dataset.targets[i] for i in self.train_dataset.indices]
        class_counts = np.bincount(train_labels, minlength=self.config['num_classes'])
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [class_weights[label] for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=self.config['num_workers'],
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            persistent_workers=True,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        results = self._do_step(batch)

        self.log('train_ce_loss', results['ce_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('train_reg_loss', results['reg_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('train_total_loss', results['total_loss'], on_epoch=True, prog_bar=True, logger=True)
        if self.use_msg_loss:
            self.log('train_msg_loss', results['msg_loss'], on_epoch=True, prog_bar=True, logger=True)

        # Periodically log sample images
        if batch_idx % self.config['img_log_step'] == 0:
            images_dict = log_ili_predictions_to_wandb(
                shifted_images=results['shifted_images'],
                original_images=results['original_images'],
                shift_preds=results['shift_preds'],
                labels=results['labels'],
                preds=results['preds'],
                target_shifts=results['target_shifts'],
                has_component=results['has_component'],
                paths=results['paths'],
                class_names=self.dataset.classes,
                num_tracks=self.num_tracks,
                phase='train',
            )
            wandb.log(images_dict)

        return {"loss": results['total_loss']}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        results = self._do_step(batch)

        self.all_labels.extend(results['labels_np'])
        self.all_preds.extend(results['preds_np'])
        self.validation_step_outputs.append(results['total_loss'])

        self.log('val_ce_loss', results['ce_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_reg_loss', results['reg_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_total_loss', results['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % self.config['img_log_step'] == 0:
            images_dict = log_ili_predictions_to_wandb(
                shifted_images=results['shifted_images'],
                original_images=results['original_images'],
                shift_preds=results['shift_preds'],
                labels=results['labels'],
                preds=results['preds'],
                target_shifts=results['target_shifts'],
                has_component=results['has_component'],
                paths=results['paths'],
                class_names=self.dataset.classes,
                num_tracks=self.num_tracks,
                phase='val',
            )
            wandb.log(images_dict)

        return {"loss": results['total_loss']}

    def on_validation_epoch_end(self):
        f1 = f1_score(self.all_labels, self.all_preds, average='macro')
        precision = precision_score(self.all_labels, self.all_preds, average='macro')
        recall = recall_score(self.all_labels, self.all_preds, average='macro')

        avg_epoch_loss = torch.stack(self.validation_step_outputs).mean()
        self.last_N_losses.append(avg_epoch_loss)
        self.last_N_losses = self.last_N_losses[-self.N_epochs:]
        run_avg_val_loss = sum(self.last_N_losses) / len(self.last_N_losses)

        if self.val_epoch_count % self.config['lr_log_step'] == 0:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log(
                f'lr_per_{self.config["lr_log_step"]}_epoch',
                current_lr, prog_bar=True, logger=True,
            )

        self.log('run_avg_val_loss', run_avg_val_loss, prog_bar=True, logger=True)
        self.log('val_f1', f1, prog_bar=True, logger=True)
        self.log('val_precision', precision, prog_bar=True, logger=True)
        self.log('val_recall', recall, prog_bar=True, logger=True)

        self.val_epoch_count += 1
        self.all_labels.clear()
        self.all_preds.clear()
        self.validation_step_outputs.clear()

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_step(self, batch):
        results = self._do_step(batch)

        self.all_labels.extend(results['labels_np'])
        self.all_preds.extend(results['preds_np'])
        self.all_probs.extend(results['probs_np'])
        self.all_paths.extend(results['paths'])
        self.test_step_outputs.append(results['total_loss'])

        self.log('test_ce_loss', results['ce_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_reg_loss', results['reg_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_total_loss', results['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": results['total_loss']}

    def on_test_epoch_end(self):
        f1 = f1_score(self.all_labels, self.all_preds, average='macro')
        precision = precision_score(self.all_labels, self.all_preds, average='macro')
        recall = recall_score(self.all_labels, self.all_preds, average='macro')

        avg_epoch_loss = torch.stack(self.test_step_outputs).mean()
        self.log('test_avg_loss', avg_epoch_loss, prog_bar=True, logger=True)
        self.log('test_f1', f1, prog_bar=True, logger=True)
        self.log('test_precision', precision, prog_bar=True, logger=True)
        self.log('test_recall', recall, prog_bar=True, logger=True)

        # Log misclassified images
        table = wandb.Table(
            columns=["Image", "True Label", "Predicted Label", "Probability"]
        )
        for i in range(len(self.all_labels)):
            if self.all_labels[i] != self.all_preds[i]:
                table.add_data(
                    wandb.Image(self.all_paths[i]),
                    self.dataset.classes[self.all_labels[i]],
                    self.dataset.classes[self.all_preds[i]],
                    self.all_probs[i],
                )
        wandb.log({"misclassified_test_images": table})

        # Confusion matrix
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.dataset.classes,
            yticklabels=self.dataset.classes,
        )
        plt.title('Test Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        wandb.log({"test_confusion_matrix": wandb.Image(plt)})
        plt.close()

        self.all_labels.clear()
        self.all_preds.clear()
        self.all_probs.clear()
        self.all_paths.clear()
        self.test_step_outputs.clear()

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _do_step(self, batch):
        """
        Shared forward + loss computation for train / val / test.

        Returns a dict with losses, predictions, and raw tensors for logging.
        """
        shifted_images, labels, target_shifts, has_component, original_images, paths = batch

        # Ensure correct types
        labels = labels.long()
        has_component = has_component.float()

        # ---- Forward pass ----
        class_logits, shift_preds = self.model(shifted_images)

        # ---- Classification loss (all samples) ----
        ce_loss = self.ce_loss(class_logits, labels) * self.config['ce_loss_weight']

        # ---- Regression loss (masked for background) ----
        # Per-sample, per-track L1
        reg_loss_raw = self.l1_loss(shift_preds, target_shifts)  # (B, num_tracks)
        reg_loss_per_sample = reg_loss_raw.mean(dim=1)           # (B,)
        mask = has_component                                      # (B,)
        reg_loss = (reg_loss_per_sample * mask).sum() / (mask.sum() + 1e-8)
        reg_loss = reg_loss * self.config['reg_loss_weight']

        # ---- Optional MSG loss (image-level alignment quality) ----
        msg_loss = torch.tensor(0.0, device=shifted_images.device)
        if self.use_msg_loss and mask.sum() > 0:
            # Reconstruct predicted aligned image (differentiable)
            predicted_aligned = apply_track_shifts_differentiable(
                shifted_images, shift_preds, self.num_tracks,
            )
            # Create spatial mask from per-sample mask
            B, C, H, W = shifted_images.shape
            spatial_mask = mask.view(B, 1, 1, 1).expand(B, 1, H, W)
            self.msg_loss_fn.to_device(shifted_images.device)
            msg_loss = self.msg_loss_fn(predicted_aligned, original_images, spatial_mask)
            msg_loss = msg_loss * self.config.get('msg_loss_weight', 0.1)

        # ---- Total loss ----
        total_loss = ce_loss + reg_loss + msg_loss

        # ---- Predictions ----
        probs = F.softmax(class_logits, dim=-1)
        probabilities, predicted = torch.max(probs, dim=1)

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'reg_loss': reg_loss,
            'msg_loss': msg_loss,
            'class_logits': class_logits,
            'shift_preds': shift_preds,
            'labels': labels,
            'preds': predicted,
            'labels_np': labels.int().detach().cpu().numpy(),
            'preds_np': predicted.detach().cpu().numpy(),
            'probs_np': probabilities.detach().cpu().numpy(),
            'shifted_images': shifted_images,
            'original_images': original_images,
            'target_shifts': target_shifts,
            'has_component': has_component,
            'paths': list(paths),
        }
