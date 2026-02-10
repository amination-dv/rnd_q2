"""
CorrelationFields Training Module (PyTorch Lightning).

Pure regression training for per-track shift prediction.

Loss:
    L_total = w_l1 * L1(pred_shifts, gt_shifts)
            + w_msg * MSGLoss(reconstructed_img, gt_img)   [optional]

Logging:
    - Uncorrelated input, correlated GT, predicted correlated image.
    - Per-track MAE (pixels).
"""

import logging

import wandb
import numpy as np
import lightning.pytorch as ptl
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from Dataset import ILIDataset
from Losses import MSGLoss
from Models import CorrelationNet
from utils import (
    apply_track_shifts_differentiable,
    log_correlation_predictions_to_wandb,
)

LOGGER = logging.getLogger(__name__)


class TrainingModule(ptl.LightningModule):
    """
    Lightning module for per-track shift regression.

    Handles:
        - Dataset creation and train/val/test splitting.
        - CorrelationNet model instantiation.
        - L1 + optional MSGLoss.
        - WandB image and metric logging.
    """

    def __init__(self, config):
        super().__init__()
        LOGGER.info("CorrelationFields Training Module init")
        self.config = config
        self.num_tracks = config['num_tracks']

        # ---- Dataset ----
        self.dataset = ILIDataset(
            root=config['data_dir'],
            img_height=config['img_height'],
            img_width=config['img_width'],
            num_tracks=config['num_tracks'],
            max_shift=config['max_shift'],
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
        self.model = CorrelationNet(
            in_channels=config.get('in_channels', 1),
            backbone_name=config['backbone'],
            num_tracks=config['num_tracks'],
            hidden_dim=config.get('hidden_dim', 256),
            freeze_backbone=config.get('freeze_backbone', False),
        )

        # ---- Losses ----
        self.l1_loss = nn.L1Loss()
        self.use_msg_loss = config.get('use_msg_loss', False)
        if self.use_msg_loss:
            self.msg_loss_fn = MSGLoss()

        # ---- Tracking ----
        self.N_epochs = 5
        self.val_epoch_count = 0
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.last_N_losses = []
        self.val_track_maes = []
        self.test_track_maes = []

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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
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

        self.log('train_l1_loss', results['l1_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', results['total_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('train_track_mae', results['track_mae'], on_epoch=True, prog_bar=True, logger=True)
        if self.use_msg_loss:
            self.log('train_msg_loss', results['msg_loss'], on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % self.config['img_log_step'] == 0:
            images_dict = log_correlation_predictions_to_wandb(
                shifted_images=results['shifted_images'],
                original_images=results['original_images'],
                pred_shifts=results['pred_shifts'],
                target_shifts=results['target_shifts'],
                num_tracks=self.num_tracks,
                paths=results['paths'],
                phase='train',
            )
            wandb.log(images_dict)

        return {"loss": results['total_loss']}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        results = self._do_step(batch)

        self.validation_step_outputs.append(results['total_loss'])
        self.val_track_maes.append(results['track_mae'].detach())

        self.log('val_l1_loss', results['l1_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', results['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_track_mae', results['track_mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % self.config['img_log_step'] == 0:
            images_dict = log_correlation_predictions_to_wandb(
                shifted_images=results['shifted_images'],
                original_images=results['original_images'],
                pred_shifts=results['pred_shifts'],
                target_shifts=results['target_shifts'],
                num_tracks=self.num_tracks,
                paths=results['paths'],
                phase='val',
            )
            wandb.log(images_dict)

        return {"loss": results['total_loss']}

    def on_validation_epoch_end(self):
        avg_epoch_loss = torch.stack(self.validation_step_outputs).mean()
        avg_track_mae = torch.stack(self.val_track_maes).mean()

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
        self.log('val_epoch_track_mae', avg_track_mae, prog_bar=True, logger=True)

        self.val_epoch_count += 1
        self.validation_step_outputs.clear()
        self.val_track_maes.clear()

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_step(self, batch):
        results = self._do_step(batch)

        self.test_step_outputs.append(results['total_loss'])
        self.test_track_maes.append(results['track_mae'].detach())

        self.log('test_l1_loss', results['l1_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_loss', results['total_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_track_mae', results['track_mae'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": results['total_loss']}

    def on_test_epoch_end(self):
        avg_epoch_loss = torch.stack(self.test_step_outputs).mean()
        avg_track_mae = torch.stack(self.test_track_maes).mean()

        self.log('test_avg_loss', avg_epoch_loss, prog_bar=True, logger=True)
        self.log('test_avg_track_mae', avg_track_mae, prog_bar=True, logger=True)

        self.test_step_outputs.clear()
        self.test_track_maes.clear()

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _do_step(self, batch):
        """
        Shared forward + loss computation for train / val / test.
        """
        shifted_images, target_shifts, original_images, paths = batch

        # ---- Forward pass ----
        pred_shifts = self.model(shifted_images)  # (B, num_tracks)

        # ---- L1 loss on per-track shifts ----
        l1_loss = self.l1_loss(pred_shifts, target_shifts)
        l1_loss = l1_loss * self.config['l1_loss_weight']

        # ---- Optional MSGLoss on reconstructed image ----
        msg_loss = torch.tensor(0.0, device=shifted_images.device)
        if self.use_msg_loss:
            # Reconstruct aligned image using predicted shifts (differentiable)
            predicted_aligned = apply_track_shifts_differentiable(
                shifted_images, pred_shifts, self.num_tracks,
            )
            B, C, H, W = shifted_images.shape
            spatial_mask = torch.ones(B, 1, H, W, device=shifted_images.device)
            self.msg_loss_fn.to_device(shifted_images.device)
            msg_loss = self.msg_loss_fn(predicted_aligned, original_images, spatial_mask)
            msg_loss = msg_loss * self.config.get('msg_loss_weight', 0.1)

        # ---- Total loss ----
        total_loss = l1_loss + msg_loss

        # ---- Per-track MAE (metric) ----
        track_mae = F.l1_loss(pred_shifts, target_shifts)

        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'msg_loss': msg_loss,
            'track_mae': track_mae,
            'pred_shifts': pred_shifts,
            'target_shifts': target_shifts,
            'shifted_images': shifted_images,
            'original_images': original_images,
            'paths': list(paths),
        }
