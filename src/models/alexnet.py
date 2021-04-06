from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, f1
from torch import nn
from torch.nn import functional as F


class AlexNet(LightningModule):
    def __init__(self, input_channels: int = 1, n_classes: int = 10, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999)):
        super(AlexNet, self).__init__()
        self.lr = lr
        self.betas = betas
        self.n_classes = n_classes
        self.net = nn.Sequential(
            # input_channels * 28 * 28
            nn.Conv2d(in_channels=input_channels,
                      out_channels=20, kernel_size=(5, 5)),
            # 20 * 24 * 24
            nn.MaxPool2d(kernel_size=(2, 2)),
            # 20 * 12 * 12
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            # 50 * 8 * 8
            nn.MaxPool2d(kernel_size=(2, 2)),
            # 50 * 4 * 4
            nn.Flatten(),
            nn.Linear(in_features=50 * 4 * 4, out_features=500),
            nn.Linear(in_features=500, out_features=self.n_classes),
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
        return optimizer

    def log_metrics(self, preds, y, prefix: str):
        accuracy_score = accuracy(preds, y)
        f1_score = f1(preds, y, num_classes=self.n_classes)
        self.log(f"{prefix}_accuracy", accuracy_score)
        self.log(f"{prefix}_f1", f1_score)
