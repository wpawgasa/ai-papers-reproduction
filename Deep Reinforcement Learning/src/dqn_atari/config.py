"""Configuration dataclasses for the DQN Atari reproduction.

Hyperparameters follow Mnih et al. (2013), "Playing Atari with Deep
Reinforcement Learning" (arXiv:1312.5602).  Section references in comments
point to ``summary.md``.  Where the workshop paper leaves a value unspecified
(notably the RMSProp learning rate / epsilon), we adopt the values reported in
the 2015 Nature follow-up and flag them below.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """DQN CNN architecture (summary Section 4.1 / 2.4).

    Conv1: 16 filters 8x8 stride 4  -> 20x20x16
    Conv2: 32 filters 4x4 stride 2  ->  9x9x32  (flatten = 2592)
    FC:    256 units
    Head:  one linear output per action.
    """

    in_channels: int = 4  # stacked frames
    num_actions: int = 6  # overwritten from the environment's action space
    conv1_filters: int = 16
    conv1_kernel: int = 8
    conv1_stride: int = 4
    conv2_filters: int = 32
    conv2_kernel: int = 4
    conv2_stride: int = 2
    fc_hidden: int = 256


@dataclass
class EnvConfig:
    """Atari environment / preprocessing configuration (summary Section 2.2.5, 2.2.7)."""

    env_id: str = "ALE/Pong-v5"
    frame_skip: int = 4  # k=4 frame-skip; action repeats on skipped frames
    frame_stack: int = 4  # last 4 frames stacked -> Markov-ish state
    screen_size: int = 84  # 84x84 playing-area crop
    grayscale: bool = True
    noop_max: int = 30  # random no-ops on reset for start-state diversity
    terminal_on_life_loss: bool = False  # Pong has no lives; keep episodes whole
    clip_rewards: bool = True  # clip rewards to {-1, 0, +1}
    max_episode_steps: int = 108_000  # ALE default cap (~30 min at 60Hz / skip)


@dataclass
class ReplayConfig:
    """Experience replay buffer (summary Section 2.2.3)."""

    capacity: int = 250_000  # paper uses 1e6; 250k fits comfortably and suffices for Pong
    min_size_to_train: int = 25_000  # warm-up: fill buffer before learning starts
    batch_size: int = 32


@dataclass
class TrainConfig:
    """Training loop / optimizer / exploration (summary Section 2.2.7)."""

    total_frames: int = 2_000_000  # ~enough for Pong to reach strong play
    gamma: float = 0.99  # discount factor

    # Optimizer. Paper uses RMSProp; lr/eps unspecified in workshop version,
    # so we adopt the Nature 2015 values (documented in summary Section 2.5).
    optimizer: Literal["rmsprop", "adam"] = "rmsprop"
    lr: float = 2.5e-4
    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 1e-2
    adam_eps: float = 1.5e-4
    grad_clip: float = 10.0  # gradient-norm clip for stability
    loss: Literal["mse", "huber"] = "huber"  # paper uses squared error; Huber is robust default

    train_freq: int = 4  # one SGD update per 4 env frames

    # Epsilon-greedy schedule: anneal 1.0 -> 0.1 over the first 1M frames.
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_frames: int = 1_000_000

    # Target network. OFF by default for fidelity to the 2013 workshop paper
    # (it has no target network; summary Section 2.5 weakness #4). Toggle ON for
    # the more stable 2015-style update.
    use_target_network: bool = False
    target_update_freq: int = 10_000  # frames between target-net syncs (when enabled)

    device: str = "cuda"
    seed: int = 42

    log_freq_frames: int = 10_000  # console logging cadence


@dataclass
class EvalConfig:
    """Evaluation protocol (summary Section 2.3 + Figure 2)."""

    eval_epsilon: float = 0.05  # epsilon-greedy eval as in the paper
    eval_episodes: int = 10
    eval_freq_frames: int = 100_000  # evaluate every N training frames
    n_holdout_states: int = 500  # fixed state set for the avg-maxQ curve (Figure 2 right)
    record_video: bool = False


@dataclass
class ExperimentConfig:
    """Top-level configuration composing all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def default(cls, **kwargs) -> "ExperimentConfig":
        """Full Pong training configuration (~1-2M frames, CUDA)."""
        cfg = cls()
        cfg._apply_overrides(kwargs)
        return cfg

    @classmethod
    def quick_test(cls, **kwargs) -> "ExperimentConfig":
        """Tiny smoke-test configuration.

        Runs the full pipeline end-to-end in minutes.  It will NOT reach
        paper-level performance -- it only verifies that the training loop,
        replay buffer, agent update, and evaluation all work together.
        """
        cfg = cls()
        cfg.replay.capacity = 10_000
        cfg.replay.min_size_to_train = 1_000
        cfg.train.total_frames = 20_000
        cfg.train.eps_decay_frames = 10_000
        cfg.train.log_freq_frames = 2_000
        cfg.eval.eval_episodes = 2
        cfg.eval.eval_freq_frames = 10_000
        cfg.eval.n_holdout_states = 100
        cfg._apply_overrides(kwargs)
        return cfg

    def _apply_overrides(self, kwargs: dict) -> None:
        """Route flat keyword overrides to the matching sub-config field."""
        subconfigs = [self.model, self.env, self.replay, self.train, self.eval]
        for key, value in kwargs.items():
            for sub in subconfigs:
                if key in sub.__dataclass_fields__:
                    setattr(sub, key, value)
                    break
            else:
                raise KeyError(f"Unknown config override: {key!r}")
