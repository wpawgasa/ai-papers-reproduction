"""
Configuration module for RL's Razor experiments.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainConfig:
    """Training configuration."""
    lr: float = 1e-3
    batch_size: int = 128
    pretrain_steps: int = 800
    finetune_steps: int = 400
    seed: int = 0
    device: str = "cuda"  # or "cpu"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden1: int = 256
    hidden2: int = 256
    out_dim: int = 10


@dataclass
class DataConfig:
    """Data configuration."""
    root: str = "./data"
    pretrain_samples: int = 5000
    eval_batch_size: int = 512
    num_workers: int = 2


@dataclass
class ExperimentConfig:
    """Overall experiment configuration."""
    train: TrainConfig
    model: ModelConfig
    data: DataConfig

    @classmethod
    def default(cls):
        """Return default configuration."""
        return cls(
            train=TrainConfig(),
            model=ModelConfig(),
            data=DataConfig()
        )
