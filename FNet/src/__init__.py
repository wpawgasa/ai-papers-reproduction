"""
FNet: Mixing Tokens with Fourier Transforms
PyTorch reproduction of Lee-Thorp et al. (NAACL 2022)

Paper: https://arxiv.org/abs/2105.03824
"""

from .model import (
    FNetModel,
    FNetForSequenceClassification,
    FNetForMaskedLM,
    FNetEncoderBlock,
    FourierTransformLayer,
    FNET_CONFIGS,
)
from .data import GLUEDataset, TASK_KEYS, TASK_NUM_LABELS
from .evaluate import compute_metrics

__version__ = "0.1.0"
__all__ = [
    "FNetModel",
    "FNetForSequenceClassification",
    "FNetForMaskedLM",
    "FNetEncoderBlock",
    "FourierTransformLayer",
    "FNET_CONFIGS",
    "GLUEDataset",
    "TASK_KEYS",
    "TASK_NUM_LABELS",
    "compute_metrics",
]
