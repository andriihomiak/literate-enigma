from argparse import ArgumentParser
from pathlib import Path

from torchvision.datasets import MNIST


def download_mnist_to_folder(folder: Path):
    MNIST(root=folder, download=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    args = parser.parse_args()
    download_mnist_to_folder(folder=args.target_dir)
