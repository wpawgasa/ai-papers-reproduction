"""Configuration dataclasses for U-Net segmentation experiment."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """U-Net model architecture configuration."""

    in_channels: int = 3
    num_classes: int = 2
    base_features: int = 64
    use_padding: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    elastic_deformation: bool = True
    elastic_sigma: float = 10.0
    elastic_grid_spacing: int = 3
    rotation: bool = True
    max_rotation_degrees: float = 30.0
    flip: bool = True
    gray_value_variation: bool = True
    gray_value_std: float = 0.1


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "./data"
    input_size: int = 256
    batch_size: int = 4
    eval_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    train_subset_size: int | None = None


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 50
    lr: float = 0.01
    momentum: float = 0.99
    weight_decay: float = 0.0

    # Learning rate schedule
    lr_schedule_factor: float = 0.1
    lr_schedule_patience: int = 10

    # Device and reproducibility
    device: str = "cuda"
    seed: int = 42


@dataclass
class LossConfig:
    """Loss function configuration."""

    w0: float = 10.0
    sigma: float = 5.0
    use_border_weights: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    model: ModelConfig
    data: DataConfig
    augmentation: AugmentationConfig
    train: TrainConfig
    loss: LossConfig

    @classmethod
    def default(cls, **kwargs) -> "ExperimentConfig":
        """Create default configuration for full training."""
        return cls(
            model=ModelConfig(**{k: v for k, v in kwargs.items() if k in ModelConfig.__dataclass_fields__}),
            data=DataConfig(**{k: v for k, v in kwargs.items() if k in DataConfig.__dataclass_fields__}),
            augmentation=AugmentationConfig(
                **{k: v for k, v in kwargs.items() if k in AugmentationConfig.__dataclass_fields__}
            ),
            train=TrainConfig(**{k: v for k, v in kwargs.items() if k in TrainConfig.__dataclass_fields__}),
            loss=LossConfig(**{k: v for k, v in kwargs.items() if k in LossConfig.__dataclass_fields__}),
        )

    @classmethod
    def quick_test(cls, **kwargs) -> "ExperimentConfig":
        """Create configuration for quick testing (fewer epochs, smaller data)."""
        defaults = {
            "epochs": 5,
            "batch_size": 2,
            "eval_batch_size": 4,
            "train_subset_size": 30,
            "input_size": 128,
        }
        defaults.update(kwargs)
        return cls(
            model=ModelConfig(**{k: v for k, v in defaults.items() if k in ModelConfig.__dataclass_fields__}),
            data=DataConfig(**{k: v for k, v in defaults.items() if k in DataConfig.__dataclass_fields__}),
            augmentation=AugmentationConfig(
                **{k: v for k, v in defaults.items() if k in AugmentationConfig.__dataclass_fields__}
            ),
            train=TrainConfig(**{k: v for k, v in defaults.items() if k in TrainConfig.__dataclass_fields__}),
            loss=LossConfig(**{k: v for k, v in defaults.items() if k in LossConfig.__dataclass_fields__}),
        )
