"""FIFO experience-replay buffer with uniform sampling (summary Section 2.2.3).

Transitions ``(s, a, r, s', done)`` are appended to a fixed-capacity ring
buffer.  Each gradient step samples a minibatch uniformly at random, breaking
the strong serial correlation between consecutive frames and giving SGD
i.i.d.-like batches.

Observations are stored as uint8 (the model normalizes to [0, 1] on the fly),
which keeps memory roughly 4x smaller than float32.
"""

from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    """A sampled minibatch, with tensors already on the target device."""

    obs: torch.Tensor  # (B, C, H, W) float in [0, 255]
    actions: torch.Tensor  # (B,) long
    rewards: torch.Tensor  # (B,) float
    next_obs: torch.Tensor  # (B, C, H, W) float in [0, 255]
    dones: torch.Tensor  # (B,) float in {0, 1}


class ReplayBuffer:
    """Fixed-capacity FIFO buffer storing stacked-frame transitions."""

    def __init__(self, capacity: int, obs_shape, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs_shape = tuple(obs_shape)

        self._obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self._next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._pos = 0
        self._size = 0

    def push(self, obs, action, reward, next_obs, done) -> None:
        """Append one transition, overwriting the oldest when full (FIFO)."""
        i = self._pos
        self._obs[i] = np.asarray(obs, dtype=np.uint8)
        self._next_obs[i] = np.asarray(next_obs, dtype=np.uint8)
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Uniformly sample a minibatch and move it to the device."""
        idx = np.random.randint(0, self._size, size=batch_size)
        obs = torch.from_numpy(self._obs[idx]).to(self.device).float()
        next_obs = torch.from_numpy(self._next_obs[idx]).to(self.device).float()
        actions = torch.from_numpy(self._actions[idx]).to(self.device)
        rewards = torch.from_numpy(self._rewards[idx]).to(self.device)
        dones = torch.from_numpy(self._dones[idx]).to(self.device)
        return Batch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)

    def __len__(self) -> int:
        return self._size
