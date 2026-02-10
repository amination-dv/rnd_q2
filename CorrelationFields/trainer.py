import os
import wandb
import torch
from dotenv import load_dotenv
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from training import TrainingModule
from utils import *

load_dotenv()

class TrainerModule:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.project_name = os.getenv('PROJECT_NAME')
        self.checkpoint_dir = f"checkpoints"
        self.setup()

    def setup(self):
        self.checkpoint_callback = ModelCheckpoint(
            monitor='val_loss_epoch',
            dirpath=self.checkpoint_dir,
            filename='best-checkpoint-{epoch}-{val_loss_epoch:.4f}',
            save_top_k=3,
            save_last=True,
            mode='min',
        )
        last_checkpoint_path = get_last_checkpoint(self.checkpoint_dir)
        self.resume_from_checkpoint = last_checkpoint_path if (last_checkpoint_path is not None and os.path.exists(last_checkpoint_path)) else None

        fix_random_seed(32)
        torch.multiprocessing.set_start_method('spawn', force=True)
        torch.set_float32_matmul_precision('high')

        wandb.login(key=os.getenv('WANDB_KEY'))

    def train(self):
        wandb_logger = WandbLogger(project=self.project_name, name=f"train-{self.project_name}-{current_timestamp()}")

        model = TrainingModule(self.config)

        trainer = Trainer(
            accelerator='gpu',
            devices=self.config['device'],
            max_epochs=self.config['max_epochs'],
            logger=[wandb_logger],
            callbacks=[self.checkpoint_callback]
        )

        trainer.fit(
            model=model,
            ckpt_path=self.resume_from_checkpoint
        )

        best_model_path = get_last_checkpoint(self.checkpoint_dir, return_best=True)
        print(f"Best model path: {best_model_path}")
        best_model = TrainingModule.load_from_checkpoint(
            best_model_path,
            config=self.config
        )
        trainer.test(best_model)
        
        wandb.finish()

    def test(self):
        wandb_logger = WandbLogger(project=self.project_name, name=f"test-{self.project_name}-{current_timestamp()}")

        best_model_path = get_last_checkpoint(self.checkpoint_dir, return_best=True)
        print(f"Best model path: {best_model_path}")

        trainer = Trainer(
            accelerator='gpu',
            devices=self.config['device'],
            logger=[wandb_logger],
            callbacks=[self.checkpoint_callback]
        )

        best_model = TrainingModule.load_from_checkpoint(
            best_model_path,
            config=self.config
        )

        trainer.test(best_model)
        
        wandb.finish()

    def save_model(self, model_path):
        best_model_path = get_last_checkpoint(self.checkpoint_dir, return_best=True)
        print(f"Best model path: {best_model_path}")

        best_model = TrainingModule.load_from_checkpoint(
            best_model_path,
            config=self.config
        )
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
