"""ResNet CIFAR-10 degradation experiment package."""

from .config import ExperimentConfig, TrainConfig, ModelConfig, DataConfig
from .data import get_dataloaders
from .models import ResNetCIFAR, PlainNetCIFAR
from .training import train_one_epoch, train_model
from .evaluation import evaluate, Metrics
from .utils import set_seed

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "TrainConfig",
    "ModelConfig",
    "DataConfig",
    "get_dataloaders",
    "ResNetCIFAR",
    "PlainNetCIFAR",
    "train_one_epoch",
    "train_model",
    "evaluate",
    "Metrics",
    "set_seed",
]
