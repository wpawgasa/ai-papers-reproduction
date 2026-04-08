"""
Dataclass-based configuration for the Transformer reproduction.
Hyperparameters from "Attention Is All You Need" (Vaswani et al., 2017).
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Transformer model hyperparameters (Table 3, Section 3)."""
    d_model: int = 512
    d_ff: int = 2048        # 4x expansion in FFN
    n_layers: int = 6       # N = 6 for both encoder and decoder
    n_heads: int = 8        # h = 8
    dropout: float = 0.1    # P_drop
    max_seq_len: int = 512
    share_embeddings: bool = True  # Section 3.4: shared input/output/pre-softmax weights


@dataclass
class DataConfig:
    """Data pipeline configuration (Section 5.1)."""
    dataset: str = "wmt14_en_de"
    vocab_size: int = 37000          # Shared BPE vocabulary
    data_dir: str = "./data"
    batch_tokens: int = 25000        # ~25K source + ~25K target tokens per batch
    max_seq_len: int = 512
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters (Section 5)."""
    # Optimizer: Adam (Section 5.3)
    lr: float = 0.0                  # Set by scheduler
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-9
    # LR schedule (Equation 5)
    warmup_steps: int = 4000
    total_steps: int = 100000
    # Regularization (Section 5.4)
    label_smoothing: float = 0.1     # epsilon_ls
    # Training
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 100
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"


@dataclass
class InferenceConfig:
    """Inference settings (Section 6.1)."""
    beam_size: int = 4
    length_penalty_alpha: float = 0.6
    max_length_offset: int = 50      # max_output = input_length + 50
    checkpoint_averaging: int = 5    # Average last 5 checkpoints (base model)


@dataclass
class ExperimentConfig:
    """Composes all configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def default(cls) -> "ExperimentConfig":
        """Base Transformer configuration from the paper."""
        return cls()

    @classmethod
    def big(cls) -> "ExperimentConfig":
        """Big Transformer configuration from the paper."""
        return cls(
            model=ModelConfig(
                d_model=1024,
                d_ff=4096,
                n_heads=16,
                dropout=0.3,
            ),
            train=TrainConfig(
                total_steps=300000,
            ),
            inference=InferenceConfig(
                checkpoint_averaging=20,
            ),
        )

    @classmethod
    def quick_test(cls) -> "ExperimentConfig":
        """Small configuration for quick testing."""
        return cls(
            model=ModelConfig(
                d_model=64,
                d_ff=128,
                n_layers=2,
                n_heads=4,
                dropout=0.1,
            ),
            data=DataConfig(
                vocab_size=1000,
                batch_tokens=500,
                max_seq_len=32,
            ),
            train=TrainConfig(
                warmup_steps=100,
                total_steps=500,
                log_interval=10,
            ),
        )
