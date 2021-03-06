from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models import ClassificationNet
from util.config import TrainingConfig


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor()
    ])


def get_dataloaders(data_dir: Path, params: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    transform = get_transforms()
    train_dataset = CIFAR10(data_dir, train=True, transform=transform)
    val_dataset = CIFAR10(data_dir, train=False, transform=transform)
    dataloader_kwargs = {
        "num_workers": cpu_count(),
        "batch_size": params.batch_size,
    }
    return DataLoader(train_dataset, **dataloader_kwargs), DataLoader(val_dataset, **dataloader_kwargs)


def get_trainer(params: TrainingConfig) -> pl.Trainer:
    lr_logger = pl.callbacks.LearningRateMonitor("step")
    return pl.Trainer(
        gpus=params.gpus, 
        max_epochs=params.max_epochs, 
        deterministic=True, 
        auto_lr_find=params.auto_lr_find,
        callbacks=[lr_logger]
        )


def get_model(params: TrainingConfig) -> pl.LightningModule:
    return ClassificationNet(
        input_channels=3, 
        num_classes=10, 
        lr=params.learning_rate, 
        betas=params.betas,
        reduce_lr_cooldown=params.reduce_lr_cooldown,
        reduce_lr_patience=params.reduce_lr_patience,
        reduce_lr_threshold=params.reduce_lr_threshold,
        reduce_lr_factor=params.reduce_lr_factor)


def save_model(model: pl.LightningModule, folder: Path):
    folder.mkdir(exist_ok=True, parents=True)
    file = folder/"best.pt"
    torch.save(model.state_dict(), file)


def run_training(data_dir: Path, out_dir: Path, params: TrainingConfig):
    train_dataloader, val_dataloader = get_dataloaders(data_dir, params=params)
    trainer = get_trainer(params=params)
    model = get_model(params=params)
    pl.utilities.seed.seed_everything(params.seed)
    trainer.fit(model, train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader)
    save_model(model, folder=out_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_training(data_dir=args.input_dir, out_dir=args.output_dir,
                 params=TrainingConfig.load_file(key="train"))
