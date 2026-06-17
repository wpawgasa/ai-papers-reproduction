"""Verification tests for the DQN reproduction.

Covers the preprocessing pipeline, model shapes/param count, replay buffer
semantics, and agent action selection / update.  The environment smoke test is
skipped automatically if Gymnasium / ALE / ROMs are unavailable.

Run with:  pytest test_dqn.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dqn_atari import (  # noqa: E402
    DQN,
    DQNAgent,
    ExperimentConfig,
    ReplayBuffer,
    preprocess_frame,
    set_seed,
)


# ----------------------------------------------------------------- preprocessing
def test_preprocess_shape():
    """phi maps a raw 210x160x3 RGB frame to an 84x84 grayscale array."""
    raw = np.random.randint(0, 256, size=(210, 160, 3), dtype=np.uint8)
    out = preprocess_frame(raw, screen_size=84)
    assert out.shape == (84, 84)
    assert out.dtype == np.uint8


def test_stacked_obs_shape():
    """A stack of 4 preprocessed frames forms an (4, 84, 84) observation."""
    frames = [preprocess_frame(np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)) for _ in range(4)]
    stacked = np.stack(frames)
    assert stacked.shape == (4, 84, 84)


# ------------------------------------------------------------------------ model
def test_model_forward():
    """DQN maps (B, 4, 84, 84) -> (B, num_actions)."""
    cfg = ExperimentConfig.default()
    cfg.model.num_actions = 6
    model = DQN(cfg.model)
    x = torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8)
    out = model(x)
    assert out.shape == (8, 6)


def test_conv_feature_size():
    """Conv stack flattens to 9*9*32 = 2592 features (summary Section 2.4)."""
    cfg = ExperimentConfig.default()
    model = DQN(cfg.model)
    assert model._conv_output_size() == 9 * 9 * 32


def test_param_count():
    """Total trainable parameters for the exact 2013 dims (16/32 filters, FC 256).

    The FC1 layer (2592 -> 256) dominates: 2592*256 + 256 = 663,808 weights, so
    the whole network is ~677k parameters. (summary Section 2.4's "~1.7M" figure
    is an overestimate for these exact dimensions; the value below is verified.)
    """
    cfg = ExperimentConfig.default()
    cfg.model.num_actions = 4
    model = DQN(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 677_172


# ---------------------------------------------------------------- replay buffer
def test_replay_push_sample():
    """Pushed transitions can be sampled with correct shapes."""
    device = torch.device("cpu")
    buf = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84), device=device)
    for _ in range(50):
        s = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        s2 = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        buf.push(s, action=2, reward=1.0, next_obs=s2, done=False)
    assert len(buf) == 50
    batch = buf.sample(16)
    assert batch.obs.shape == (16, 4, 84, 84)
    assert batch.actions.shape == (16,)
    assert batch.rewards.shape == (16,)
    assert batch.dones.shape == (16,)


def test_replay_fifo_eviction():
    """Buffer length is capped at capacity (FIFO overwrite)."""
    buf = ReplayBuffer(capacity=10, obs_shape=(4, 84, 84), device=torch.device("cpu"))
    for _ in range(25):
        s = np.zeros((4, 84, 84), dtype=np.uint8)
        buf.push(s, 0, 0.0, s, False)
    assert len(buf) == 10


# ------------------------------------------------------------------------ agent
def _make_agent(use_target=False):
    set_seed(0)
    cfg = ExperimentConfig.default(use_target_network=use_target)
    cfg.model.num_actions = 6
    cfg.train.device = "cpu"
    model = DQN(cfg.model)
    return DQNAgent(model, num_actions=6, config=cfg.train, device=torch.device("cpu"))


def test_epsilon_random_action():
    """With epsilon=1 the agent returns a valid random action."""
    agent = _make_agent()
    obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
    a = agent.select_action(obs, epsilon=1.0)
    assert 0 <= a < 6


def test_epsilon_greedy_deterministic():
    """With epsilon=0 the agent is a deterministic argmax over Q-values."""
    agent = _make_agent()
    obs = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
    a1 = agent.select_action(obs, epsilon=0.0)
    a2 = agent.select_action(obs, epsilon=0.0)
    assert a1 == a2


def test_epsilon_schedule():
    """Epsilon anneals from start to end and then holds."""
    agent = _make_agent()
    c = agent.config
    assert agent.current_epsilon(0) == pytest.approx(c.eps_start)
    assert agent.current_epsilon(c.eps_decay_frames) == pytest.approx(c.eps_end)
    assert agent.current_epsilon(c.eps_decay_frames * 2) == pytest.approx(c.eps_end)


@pytest.mark.parametrize("use_target", [False, True])
def test_td_loss_and_update(use_target):
    """TD loss is a finite scalar and one update step runs for both variants."""
    agent = _make_agent(use_target=use_target)
    buf = ReplayBuffer(capacity=64, obs_shape=(4, 84, 84), device=torch.device("cpu"))
    for _ in range(40):
        s = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        s2 = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        buf.push(s, np.random.randint(6), float(np.random.choice([-1, 0, 1])), s2, bool(np.random.random() < 0.1))
    batch = buf.sample(32)
    loss = agent.compute_td_loss(batch)
    assert torch.isfinite(loss)
    optimizer = agent.build_optimizer()
    loss_val = agent.update(batch, optimizer)
    assert np.isfinite(loss_val)


def test_target_network_presence():
    """Target net exists only when enabled."""
    assert _make_agent(use_target=False).target is None
    assert _make_agent(use_target=True).target is not None


# -------------------------------------------------------------------- env smoke
def test_env_smoke():
    """make_env builds, resets, and steps; obs shape is (4, 84, 84)."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("ale_py")
    from dqn_atari import env_spec, make_env, obs_to_array

    cfg = ExperimentConfig.quick_test()
    try:
        env = make_env(cfg.env, seed=0)
    except Exception as exc:  # ROM/registration issues -> skip rather than fail
        pytest.skip(f"Atari env unavailable: {exc}")

    obs_shape, num_actions = env_spec(env)
    assert obs_shape == (4, 84, 84)
    assert num_actions >= 2

    obs, _ = env.reset(seed=0)
    obs = obs_to_array(obs)
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8

    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert obs_to_array(obs).shape == (4, 84, 84)
    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
