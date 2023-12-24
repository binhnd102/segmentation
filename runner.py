import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, EarlyStopping
import torch
import torch.nn.functional as F
from lightning.pytorch.demos import Transformer, WikiText2
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from typing import Protocol, List
from dataclasses import dataclass, field
from model import AttentionUNet
from data import get_image_and_mask, SegmentationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import WandbLogger
import wandb
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
import os
from callbacks import VisualizeValDataCallback
from metrics import dice_coeff
from losses import FocalLoss


WANDB_KEY = "c80518dc0bbbe535960d500c6885d81a9ef26bcb"

wandb.login(key=WANDB_KEY)
wandb_logger = WandbLogger(log_model=True)


class WandbArtifactCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        run = trainer.logger.experiment
        print(f'Ending run: {run.id}')
        artifact = wandb.Artifact(f'{run.id}_model', type='model')
        for path, val_loss in trainer.checkpoint_callback.best_k_models.items():
            print(f'Adding artifact: {path}')
            artifact.add_file(path)
        run.log_artifact(artifact)

class SegmentationModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AttentionUNet()
        self.loss_function = FocalLoss(gamma=2)
    
    def shared_step(self, batch, batch_idx, step):
        input, mask = batch
        mask = mask.unsqueeze(1)
        output = self.model(input)
        loss = self.loss_function(output, mask)
        y_pred = output.data.cpu().numpy().ravel()
        y_true = mask.data.cpu().numpy().ravel()
        dice_score = dice_coeff(y_pred, y_true)
        self.log(f"{step}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{step}_dice", dice_score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "test")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)


if __name__ == "__main__":
    DATA_PATH = "data/annotations/annotations/list.txt"
    
    image_files, label_files = tuple(zip(*list(get_image_and_mask(DATA_PATH, prefix="./data"))))
    train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
        image_files, label_files, test_size=0.2, random_state=42)

    train_dataset = SegmentationDataset(train_image_files, train_mask_files)
    val_dataset = SegmentationDataset(val_image_files, val_mask_files)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    model = SegmentationModel()
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', monitor='val_loss', save_top_k=3)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    visualizer = VisualizeValDataCallback()

    trainer = L.Trainer(
        gradient_clip_val=0.25, 
        max_epochs=5, 
        callbacks=[checkpoint_callback, visualizer, early_stopping],
        logger=wandb_logger)

    trainer.fit(model, train_dataloader, val_dataloader)

    # Resume
    # trainer = L.Trainer(max_epochs=5, callbacks=[checkpoint_callback])
    # trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="checkpoints/epoch=0-step=80.ckpt")

