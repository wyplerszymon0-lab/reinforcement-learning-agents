"""
Experience replay and rollout buffers.

Two fundamentally different data collection strategies:
  - ReplayBuffer (off-policy): stores all past transitions, samples randomly.
    Breaks temporal correlation; enables data reuse. Used by DQN.
  - RolloutBuffer (on-policy): stores one batch of fresh transitions only.
    Discarded after each policy update. Used by PPO.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Circular replay buffer for off-policy learning.

    Mnih et al. (2015) showed that uniform random sampling from a large
    replay buffer was sufficient to achieve human-level Atari performance.
    The buffer size is a critical hyperparameter: too small -> correlated
    data; too large -> stale data dominates early training.
    """

    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self._pos = 0
        self._size = 0

        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._next_states[self._pos] = next_state
        self._dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {self._size}")
        indices = np.random.choice(self._size, batch_size, replace=False)
        return (
            torch.FloatTensor(self._states[indices]).to(self.device),
            torch.LongTensor(self._actions[indices]).to(self.device),
            torch.FloatTensor(self._rewards[indices]).to(self.device),
            torch.FloatTensor(self._next_states[indices]).to(self.device),
            torch.FloatTensor(self._dones[indices]).to(self.device),
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= self.capacity // 10


class RolloutBuffer:
    """
    Fixed-size on-policy rollout buffer with GAE computation.

    GAE (Schulman et al., 2015b) trades off bias and variance via λ:
        A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)

    λ=1 recovers Monte Carlo returns (low bias, high variance).
    λ=0 recovers 1-step TD (high bias, low variance).
    λ=0.95 is the standard empirical sweet spot.
    """

    def __init__(self, num_steps: int, obs_dim: int, device: torch.device) -> None:
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.device = device
        self._reset()

    def _reset(self) -> None:
        self._states = np.zeros((self.num_steps, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros(self.num_steps, dtype=np.int64)
        self._rewards = np.zeros(self.num_steps, dtype=np.float32)
        self._values = np.zeros(self.num_steps, dtype=np.float32)
        self._log_probs = np.zeros(self.num_steps, dtype=np.float32)
        self._dones = np.zeros(self.num_steps, dtype=np.float32)
        self._advantages = np.zeros(self.num_steps, dtype=np.float32)
        self._returns = np.zeros(self.num_steps, dtype=np.float32)
        self._pos = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        assert self._pos < self.num_steps, "Buffer is full — call compute_gae and reset first"
        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._values[self._pos] = value
        self._log_probs[self._pos] = log_prob
        self._dones[self._pos] = float(done)
        self._pos += 1

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Backward pass through time to compute GAE advantages."""
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            next_non_terminal = 1.0 - self._dones[t]
            next_value = last_value if t == self.num_steps - 1 else self._values[t + 1]

            delta = (
                self._rewards[t]
                + gamma * next_value * next_non_terminal
                - self._values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self._advantages[t] = last_gae

        self._returns = self._advantages + self._values

    def get_tensors(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.FloatTensor(self._states).to(self.device),
            torch.LongTensor(self._actions).to(self.device),
            torch.FloatTensor(self._log_probs).to(self.device),
            torch.FloatTensor(self._advantages).to(self.device),
            torch.FloatTensor(self._returns).to(self.device),
            torch.FloatTensor(self._values).to(self.device),
        )

    def reset(self) -> None:
        self._reset()

    @property
    def is_full(self) -> bool:
        return self._pos >= self.num_steps
