"""Evaluation: epsilon-greedy rollouts, held-out avg-maxQ, and video recording.

These implement the paper's evaluation protocol (summary Section 2.3) and the
two complementary progress signals from Figure 2: noisy episode reward and the
smooth average max-Q over a fixed held-out state set.
"""

from typing import List

import numpy as np
import torch

from ..preprocessing import obs_to_array


@torch.no_grad()
def evaluate_agent(agent, env, n_episodes: int = 10, epsilon: float = 0.05) -> dict:
    """Run greedy-ish rollouts and return reward statistics.

    Args:
        agent: A :class:`DQNAgent`.
        env: A preprocessed Atari environment.
        n_episodes: Number of episodes to average over.
        epsilon: Evaluation exploration rate (paper uses 0.05).

    Returns:
        ``{"mean_reward", "std_reward", "max_reward", "min_reward", "rewards"}``.
    """
    rewards: List[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.select_action(obs_to_array(obs), epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)

    arr = np.array(rewards, dtype=np.float32)
    return {
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "max_reward": float(arr.max()),
        "min_reward": float(arr.min()),
        "rewards": rewards,
    }


@torch.no_grad()
def collect_holdout_states(env, n: int, policy_epsilon: float = 1.0) -> np.ndarray:
    """Collect a fixed set of states by acting (semi-)randomly in the env.

    The set is gathered once before training and reused for every avg-maxQ
    measurement, mirroring the paper's "fixed set of states" (Figure 2 right).

    Returns:
        ``(n, C, H, W)`` uint8 array of stacked-frame states.
    """
    states = []
    obs, _ = env.reset()
    while len(states) < n:
        states.append(obs_to_array(obs))
        action = env.action_space.sample() if np.random.random() < policy_epsilon else 0
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return np.stack(states[:n]).astype(np.uint8)


@torch.no_grad()
def avg_max_q(agent, states: np.ndarray, batch_size: int = 256) -> float:
    """Average of max_a Q(s, a) over a fixed held-out state set (Figure 2 right)."""
    agent.model.eval()
    total, n = 0.0, 0
    for start in range(0, len(states), batch_size):
        chunk = states[start : start + batch_size]
        x = torch.from_numpy(chunk).to(agent.device).float()
        q = agent.model(x)
        total += float(q.max(dim=1).values.sum().item())
        n += len(chunk)
    agent.model.train()
    return total / max(n, 1)


@torch.no_grad()
def record_episode(agent, env, path: str, epsilon: float = 0.05, fps: int = 30) -> str:
    """Record one episode to an mp4 file.

    The env must be built with ``render_mode="rgb_array"``.  Returns the path.
    """
    import imageio

    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        action = agent.select_action(obs_to_array(obs), epsilon)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    imageio.mimsave(path, frames, fps=fps)
    return path
