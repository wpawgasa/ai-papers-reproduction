"""DQN agent: epsilon-greedy action selection and the Q-learning update.

Implements the loss of Eq. (2)/(3) and the epsilon-greedy behaviour policy of
Algorithm 1 (summary Sections 2.2.2-2.2.4).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig
from .models.dqn import DQN
from .replay_buffer import Batch


class DQNAgent:
    """Holds the online Q-network (and an optional target network) plus the
    epsilon-greedy policy and TD-error update."""

    def __init__(self, model: DQN, num_actions: int, config: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.num_actions = num_actions
        self.config = config
        self.device = device

        if config.use_target_network:
            # 2015-style: a periodically-cloned target network.
            self.target = copy.deepcopy(model).to(device)
            self.target.eval()
            for p in self.target.parameters():
                p.requires_grad_(False)
        else:
            # 2013 fidelity: no separate target net. The target uses a detached
            # forward of the online net (approximating "theta_{i-1} held fixed").
            self.target = None

    # ------------------------------------------------------------------ policy
    def current_epsilon(self, frame: int) -> float:
        """Linearly anneal epsilon from eps_start to eps_end over
        eps_decay_frames, then hold constant (summary Section 2.2.7)."""
        c = self.config
        if frame >= c.eps_decay_frames:
            return c.eps_end
        frac = frame / c.eps_decay_frames
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection (single forward pass over K actions)."""
        if np.random.random() < epsilon:
            return int(np.random.randint(self.num_actions))
        state = torch.from_numpy(np.asarray(obs, dtype=np.uint8)).to(self.device).float().unsqueeze(0)
        q_values = self.model(state)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------ learning
    def compute_td_loss(self, batch: Batch) -> torch.Tensor:
        """TD loss for a minibatch: y = r + gamma * max_a' Q(s', a') * (1 - done).

        When the target network is disabled, the bootstrap target is a detached
        forward of the online network (2013 behaviour); when enabled it uses the
        cloned target network (2015 behaviour)."""
        c = self.config

        # Q(s, a) for the actions actually taken.
        q = self.model(batch.obs)
        q_taken = q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            net = self.target if self.target is not None else self.model
            next_q = net(batch.next_obs).max(dim=1).values
            target = batch.rewards + c.gamma * next_q * (1.0 - batch.dones)

        if c.loss == "huber":
            return F.smooth_l1_loss(q_taken, target)
        return F.mse_loss(q_taken, target)

    def update(self, batch: Batch, optimizer: torch.optim.Optimizer) -> float:
        """One SGD step on the TD loss. Returns the scalar loss."""
        loss = self.compute_td_loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip is not None and self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        """Copy online weights into the target network (no-op if disabled)."""
        if self.target is not None:
            self.target.load_state_dict(self.model.state_dict())

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Construct the optimizer specified by the config."""
        c = self.config
        if c.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=c.lr, eps=c.adam_eps)
        return torch.optim.RMSprop(
            self.model.parameters(),
            lr=c.lr,
            alpha=c.rmsprop_alpha,
            eps=c.rmsprop_eps,
        )
