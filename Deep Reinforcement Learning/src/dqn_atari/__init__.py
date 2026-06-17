"""Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013).

A faithful, runnable reproduction of the 2013 DQN targeting Pong: a CNN trained
end-to-end from raw pixels with Q-learning + experience replay. See ``summary.md``
for the paper analysis and ``README.md`` for usage.
"""

from .agent import DQNAgent
from .config import (
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    ReplayConfig,
    TrainConfig,
)
from .evaluation.evaluate import avg_max_q, collect_holdout_states, evaluate_agent, record_episode
from .models.dqn import DQN
from .preprocessing import env_spec, make_env, obs_to_array, preprocess_frame
from .replay_buffer import Batch, ReplayBuffer
from .training.trainer import train
from .utils.seed import set_seed

__version__ = "0.1.0"

__all__ = [
    # config
    "ExperimentConfig",
    "ModelConfig",
    "EnvConfig",
    "ReplayConfig",
    "TrainConfig",
    "EvalConfig",
    # model
    "DQN",
    # data / env
    "make_env",
    "preprocess_frame",
    "obs_to_array",
    "env_spec",
    "ReplayBuffer",
    "Batch",
    # agent / training / eval
    "DQNAgent",
    "train",
    "evaluate_agent",
    "collect_holdout_states",
    "avg_max_q",
    "record_episode",
    # utils
    "set_seed",
]
