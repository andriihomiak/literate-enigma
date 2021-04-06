from pathlib import Path
from typing import Tuple

import yaml
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    gpus: int
    max_epochs: int
    betas: Tuple[float, float]
    seed: int
    
    @staticmethod
    def load_file(key: str, file: Path = Path("params.yaml")):
        config = yaml.safe_load(file.read_text())[key]
        return TrainingConfig(**config)
    