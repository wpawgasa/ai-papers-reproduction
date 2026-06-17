"""Reproducibility utilities."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42, env=None) -> None:
    """Set random seeds for reproducibility.

    Seeds Python ``random``, NumPy, and PyTorch RNGs.  When CUDA is available,
    also seeds all GPU RNGs.  cuDNN benchmark autotuning is left enabled because
    the DQN conv stack has a fixed input shape and benefits from it; strict
    determinism is not attempted here (RL training is stochastic through the
    environment regardless).

    Args:
        seed: Random seed value.
        env: Optional Gymnasium environment whose ``reset`` and action space are
            also seeded.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if env is not None:
        # Gymnasium seeds the RNG via reset(seed=...); also seed action sampling.
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        except (TypeError, AttributeError):
            pass
