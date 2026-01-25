"""Configuration dataclasses for ResNet CIFAR-10 experiment."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    depth: int = 56
    num_classes: int = 10

    def __post_init__(self):
        assert (self.depth - 2) % 6 == 0, "Depth must be 6n+2 (e.g., 20, 32, 44, 56, 110)"


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "./data"
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = False

    # Learning rate schedule
    lr_schedule: Literal["multistep"] = "multistep"
    lr_milestones: tuple[float, float] = (0.5, 0.75)  # Fractions of total epochs
    lr_gamma: float = 0.1

    # Device and reproducibility
    device: str = "cuda"
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    model: ModelConfig
    data: DataConfig
    train: TrainConfig

    @classmethod
    def default(cls, depth: int = 56) -> "ExperimentConfig":
        """Create default configuration for a specific depth."""
        return cls(
            model=ModelConfig(depth=depth),
            data=DataConfig(),
            train=TrainConfig(),
        )

    @classmethod
    def quick_test(cls, depth: int = 20) -> "ExperimentConfig":
        """Create configuration for quick testing."""
        return cls(
            model=ModelConfig(depth=depth),
            data=DataConfig(batch_size=64),
            train=TrainConfig(epochs=50, lr=0.1),
        )
