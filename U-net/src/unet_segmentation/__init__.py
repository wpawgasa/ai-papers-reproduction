"""U-Net: Convolutional Networks for Biomedical Image Segmentation.

Reproduction of Ronneberger et al., MICCAI 2015.
Demonstrates elastic deformation augmentation for data-efficient segmentation.
"""

from .config import (
    AugmentationConfig,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)
from .data.oxford_pets import PetSegmentationDataset, get_dataloaders
from .evaluation.metrics import (
    SegmentationMetrics,
    compute_dice,
    compute_iou,
    compute_pixel_accuracy,
    evaluate_model,
)
from .models.unet import UNet
from .training.trainer import train_model, train_one_epoch
from .utils.loss import WeightedCrossEntropyLoss
from .utils.seed import set_seed

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "AugmentationConfig",
    "TrainConfig",
    "LossConfig",
    "UNet",
    "PetSegmentationDataset",
    "get_dataloaders",
    "train_one_epoch",
    "train_model",
    "evaluate_model",
    "SegmentationMetrics",
    "compute_iou",
    "compute_dice",
    "compute_pixel_accuracy",
    "WeightedCrossEntropyLoss",
    "set_seed",
]
