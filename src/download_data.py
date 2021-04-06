from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets import CIFAR10


def download_data_to_folder(folder: Path):
    CIFAR10(root=folder, download=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    args = parser.parse_args()
    download_data_to_folder(folder=args.target_dir)
