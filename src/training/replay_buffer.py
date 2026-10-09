"""
Experience replay and rollout buffers.

Two fundamentally different data collection strategies:
  - ReplayBuffer (off-policy): stores all past transitions, samples randomly.
    Breaks temporal correlation; enables data reuse. Used by DQN.
  - RolloutBuffer (on-policy): stores one batch of fresh transitions only.
    Discarded after each policy update. Used by PPO.
"""
from __future__ import annotations

from typing import Any, Tuple

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

    def __init__(self, num_steps: int, obs_dim: int, device: torch.device, num_envs: int = 1) -> None:
        # num_steps is per environment: a full buffer holds num_steps * num_envs transitions.
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        self._reset()

    def _reset(self) -> None:
        shape = (self.num_steps, self.num_envs)
        self._states = np.zeros((*shape, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros(shape, dtype=np.int64)
        self._rewards = np.zeros(shape, dtype=np.float32)
        self._values = np.zeros(shape, dtype=np.float32)
        self._log_probs = np.zeros(shape, dtype=np.float32)
        self._dones = np.zeros(shape, dtype=np.float32)
        self._advantages = np.zeros(shape, dtype=np.float32)
        self._returns = np.zeros(shape, dtype=np.float32)
        self._pos = 0

    def push(
        self,
        state: np.ndarray,
        action: Any,
        reward: Any,
        value: Any,
        log_prob: Any,
        done: Any,
    ) -> None:
        """One time step: scalars for a single environment, arrays of length num_envs otherwise."""
        assert self._pos < self.num_steps, "Buffer is full — call compute_gae and reset first"
        n = self.num_envs
        self._states[self._pos] = np.asarray(state, dtype=np.float32).reshape(n, self.obs_dim)
        self._actions[self._pos] = np.asarray(action).reshape(n)
        self._rewards[self._pos] = np.asarray(reward, dtype=np.float32).reshape(n)
        self._values[self._pos] = np.asarray(value, dtype=np.float32).reshape(n)
        self._log_probs[self._pos] = np.asarray(log_prob, dtype=np.float32).reshape(n)
        self._dones[self._pos] = np.asarray(done, dtype=np.float32).reshape(n)
        self._pos += 1

    def compute_gae(
        self,
        last_value: Any,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Backward pass through time to compute GAE advantages, for every environment at once."""
        last_value_arr = np.asarray(last_value, dtype=np.float32).reshape(self.num_envs)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.num_steps)):
            next_non_terminal = 1.0 - self._dones[t]
            next_value = last_value_arr if t == self.num_steps - 1 else self._values[t + 1]

            delta = (
                self._rewards[t]
                + gamma * next_value * next_non_terminal
                - self._values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self._advantages[t] = last_gae

        self._returns = self._advantages + self._values

    def get_tensors(self) -> Tuple[torch.Tensor, ...]:
        """All transitions, flattened to (num_steps * num_envs, ...)."""
        def flat(a: np.ndarray) -> np.ndarray:
            return a.reshape(self.num_steps * self.num_envs, *a.shape[2:])

        return (
            torch.FloatTensor(flat(self._states)).to(self.device),
            torch.LongTensor(flat(self._actions)).to(self.device),
            torch.FloatTensor(flat(self._log_probs)).to(self.device),
            torch.FloatTensor(flat(self._advantages)).to(self.device),
            torch.FloatTensor(flat(self._returns)).to(self.device),
            torch.FloatTensor(flat(self._values)).to(self.device),
        )

    def reset(self) -> None:
        self._reset()

    @property
    def is_full(self) -> bool:
        return self._pos >= self.num_steps
