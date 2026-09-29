"""
Gymnasium environment wrappers.

Wrappers follow the decorator pattern: each wrapper modifies exactly one
aspect of the environment interface. Stacking wrappers is composable.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np


class RewardScalingWrapper(gym.RewardWrapper):
    """
    Scale rewards by a constant factor.

    Useful when raw rewards are on a very different scale than the
    value function's expected output range. For LunarLander the raw
    rewards range from ~-300 to +300; scaling by 1/100 keeps values
    near [-3, 3] which is more compatible with small networks.
    """

    def __init__(self, env: gym.Env, scale: float) -> None:
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: SupportsFloat) -> float:
        return float(reward) * self.scale


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Online running mean/std normalization of observations.

    Uses Welford's online algorithm for numerically stable variance
    estimation. Normalization is critical for environments where
    observation components have very different scales (e.g., LunarLander
    mixes positions, velocities, and boolean leg contact flags).
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8) -> None:
        super().__init__(env)
        self.epsilon = epsilon
        obs_shape = env.observation_space.shape
        if obs_shape is None:
            raise ValueError("NormalizeObservationWrapper needs an observation space with a fixed shape (e.g. Box)")
        self._mean = np.zeros(obs_shape, dtype=np.float64)
        self._var = np.ones(obs_shape, dtype=np.float64)
        self._count = 0.0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update_stats(obs)
        return ((obs - self._mean) / np.sqrt(self._var + self.epsilon)).astype(np.float32)

    def _update_stats(self, obs: np.ndarray) -> None:
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._var += (delta * delta2 - self._var) / self._count


class EpisodeMonitor(gym.Wrapper):
    """
    Records episode statistics without modifying the MDP.

    Attaches `episode` info dict with keys: `r` (return), `l` (length)
    to the info returned on episode termination. This mirrors the
    RecordEpisodeStatistics wrapper from stable-baselines3.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._episode_return = 0.0
        self._episode_length = 0
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        self._episode_length += 1
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_return,
                "l": self._episode_length,
            }
        return obs, reward, terminated, truncated, info


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip rewards to [-1, 1] — Atari-style training signal."""

    def __init__(self, env: gym.Env, low: float = -1.0, high: float = 1.0) -> None:
        super().__init__(env)
        self.low = low
        self.high = high

    def reward(self, reward: SupportsFloat) -> float:
        return float(np.clip(float(reward), self.low, self.high))


class GaussianObservationNoise(gym.ObservationWrapper):
    """
    Add zero-mean Gaussian noise to every observation the agent sees.

    Models imperfect sensors: the environment's true state is unchanged, only
    the agent's view of it is corrupted. `sigma` is a scalar or one standard
    deviation per observation dimension.
    """

    def __init__(self, env: gym.Env, sigma: float | np.ndarray, seed: Optional[int] = None) -> None:
        super().__init__(env)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0.0, 1.0, size=np.shape(obs)) * self.sigma
        return (obs + noise).astype(np.asarray(obs).dtype)


def make_env(
    env_id: str,
    seed: int = 0,
    monitor: bool = True,
    normalize_obs: bool = False,
    reward_scale: Optional[float] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Factory function for reproducible environment creation."""
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if monitor:
        env = EpisodeMonitor(env)
    if reward_scale is not None:
        env = RewardScalingWrapper(env, reward_scale)
    if normalize_obs:
        env = NormalizeObservationWrapper(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
