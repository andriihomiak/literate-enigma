from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, f1
from torch import nn
from torch.nn import functional as F


class ClassificationNet(LightningModule):
    def __init__(self, 
            input_channels: int = 1, 
            num_classes: int = 10, 
            lr: float = 1e-3, 
            betas: Tuple[float, float] = (0.9, 0.999),
            reduce_lr_patience: int = 10,
            reduce_lr_threshold: float = 0.001,
            reduce_lr_factor: float = 0.2,
            reduce_lr_cooldown: int = 10,
            ):
        super(ClassificationNet, self).__init__()
        self.lr = lr
        self.betas = betas
        self.num_classes = num_classes
        self.reduce_lr_cooldown = reduce_lr_cooldown
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_threshold = reduce_lr_threshold
        self.net = nn.Sequential(
            # 3 * 32 * 32
            self._make_block(in_channels=3, out_channels=32, dropout=0.2),
            # 32 * 16 * 16
            self._make_block(in_channels=32, out_channels=64, dropout=0.3),
            # 64 * 8 * 8
            self._make_block(in_channels=64, out_channels=128, dropout=0.4),
            # 128 * 4 * 4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def _make_block(self, in_channels: int, out_channels: int, dropout: float):
        return nn.Sequential(
            # in_channels * D * D
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            # out_channels * D * D
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d((2, 2)),
            # out_channels * D / 2 * D / 2
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)

    def _generic_step(self, batch, batch_idx, prefix):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        preds = F.softmax(out, dim=-1)
        self.log(f"{prefix}_loss", loss)
        self.log_metrics(preds=preds, y=y, prefix=prefix)
        return loss

    def training_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, prefix="test")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=self.betas)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            threshold=self.reduce_lr_threshold, 
            patience=self.reduce_lr_patience, 
            factor=self.reduce_lr_factor,
            mode="min", verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def log_metrics(self, preds, y, prefix: str):
        accuracy_score = accuracy(preds, y)
        f1_score = f1(preds, y, num_classes=self.num_classes)
        self.log(f"{prefix}_accuracy", accuracy_score)
        self.log(f"{prefix}_error", 1 - accuracy_score)
        self.log(f"{prefix}_f1", f1_score)
