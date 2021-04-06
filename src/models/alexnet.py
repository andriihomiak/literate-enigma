from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class AlexNet(LightningModule):
    def __init__(self, input_channels: int = 1, lr: float=1e-3, betas: Tuple[float, float]=(0.9, 0.999)):
        super(AlexNet, self).__init__()
        self.lr = lr
        self.betas = betas
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
            nn.Linear(in_features=500, out_features=10),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log("train_loss", loss)
        results = {
            "loss": loss,
        }
        return results

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        self.log("val_loss", loss)
        results = {
            "loss": loss,
        }
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        return optimizer
