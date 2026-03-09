"""Reproducibility utilities."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds and configure deterministic execution for reproducibility.

    Seeds Python ``random``, NumPy, and PyTorch RNGs.  When CUDA is available,
    also seeds all GPU RNGs and enables cuDNN deterministic mode (which may
    reduce throughput) and disables benchmark autotuning.

    Note: ``torch.use_deterministic_algorithms(True, warn_only=True)`` is set
    when supported; non-deterministic operations will emit a warning rather
    than raising an error.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass
