import json
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as pl_metrics
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


def get_dataloaders(data_dir: Path, params: TrainingConfig) -> DataLoader:
    transform = get_transforms()
    val_dataset = CIFAR10(data_dir, train=False, transform=transform)
    dataloader_kwargs = {
        "num_workers": cpu_count(),
        "batch_size": params.batch_size,
    }
    return DataLoader(val_dataset, **dataloader_kwargs)


def get_trainer(params: TrainingConfig) -> pl.Trainer:
    lr_logger = pl.callbacks.LearningRateMonitor("step")
    return pl.Trainer(
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        deterministic=True,
        auto_lr_find=params.auto_lr_find,
        callbacks=[lr_logger]
    )


def get_model(file: Path, params: TrainingConfig) -> pl.LightningModule:
    model = ClassificationNet(
        input_channels=3,
        num_classes=10,
        lr=params.learning_rate,
        betas=params.betas,
        reduce_lr_cooldown=params.reduce_lr_cooldown,
        reduce_lr_patience=params.reduce_lr_patience,
        reduce_lr_threshold=params.reduce_lr_threshold,
        reduce_lr_factor=params.reduce_lr_factor)
    model.load_state_dict(torch.load(file))
    model.eval()
    return model


def save_metrics(metrics: dict, metrics_file: Path):
    metrics_file.write_text(json.dumps(metrics, indent=2))


def test_model_performance(model_file: Path, data_dir: Path, metrics_file: Path, params: TrainingConfig):
    val_dataloader = get_dataloaders(data_dir, params=params)
    trainer = get_trainer(params=params)
    model = get_model(file=model_file, params=params)
    pl.utilities.seed.seed_everything(params.seed)
    results = trainer.test(model=model, test_dataloaders=val_dataloader)
    save_metrics(metrics=results[0], metrics_file=metrics_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    args = parser.parse_args()
    test_model_performance(
        model_file=args.model_file,
        data_dir=args.data_dir,
        metrics_file=args.metrics_file,
        params=TrainingConfig.load_file(key="train")
    )
