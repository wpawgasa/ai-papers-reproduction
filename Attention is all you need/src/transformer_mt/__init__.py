"""
Reproduction of "Attention Is All You Need" (Vaswani et al., NeurIPS 2017).
Transformer architecture for sequence-to-sequence machine translation.
"""

from .config import ExperimentConfig
from .data import (
    BPETokenizer,
    TranslationDataset,
    create_masks,
    get_dataloaders,
    train_tokenizer,
)
from .evaluation import average_checkpoints, beam_search, greedy_decode
from .models import Transformer, build_transformer_base, build_transformer_big
from .training import LabelSmoothingLoss, TransformerLRScheduler, train
from .utils import set_seed

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "BPETokenizer",
    "TranslationDataset",
    "Transformer",
    "build_transformer_base",
    "build_transformer_big",
    "create_masks",
    "get_dataloaders",
    "train_tokenizer",
    "beam_search",
    "greedy_decode",
    "average_checkpoints",
    "LabelSmoothingLoss",
    "TransformerLRScheduler",
    "train",
    "set_seed",
]
