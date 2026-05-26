"""Tests for environment wrappers and utilities."""
import numpy as np
import pytest
import gymnasium as gym

from src.environments.wrappers import (
    ClipRewardWrapper,
    EpisodeMonitor,
    NormalizeObservationWrapper,
    RewardScalingWrapper,
    make_env,
)
from src.environments.utils import get_env_dims, set_global_seed


# ---------------------------------------------------------------------------
# RewardScalingWrapper
# ---------------------------------------------------------------------------


def _make_cartpole():
    return gym.make("CartPole-v1")


def test_reward_scaling_halves_reward():
    env = RewardScalingWrapper(_make_cartpole(), scale=0.5)
    obs, _ = env.reset(seed=0)
    _, reward, *_ = env.step(env.action_space.sample())
    env.close()
    assert abs(reward - 0.5) < 1e-6


def test_reward_scaling_negative():
    env = RewardScalingWrapper(_make_cartpole(), scale=-1.0)
    obs, _ = env.reset(seed=0)
    _, reward, *_ = env.step(env.action_space.sample())
    env.close()
    assert reward < 0.0


# ---------------------------------------------------------------------------
# ClipRewardWrapper
# ---------------------------------------------------------------------------


def test_clip_reward_within_bounds():
    env = ClipRewardWrapper(RewardScalingWrapper(_make_cartpole(), scale=10.0))
    obs, _ = env.reset(seed=0)
    _, reward, *_ = env.step(0)
    env.close()
    assert -1.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# NormalizeObservationWrapper
# ---------------------------------------------------------------------------


def test_normalize_obs_returns_float32():
    env = NormalizeObservationWrapper(_make_cartpole())
    obs, _ = env.reset(seed=0)
    env.close()
    assert obs.dtype == np.float32


def test_normalize_obs_shape_preserved():
    base = _make_cartpole()
    env = NormalizeObservationWrapper(base)
    obs, _ = env.reset(seed=0)
    env.close()
    assert obs.shape == base.observation_space.shape


# ---------------------------------------------------------------------------
# EpisodeMonitor
# ---------------------------------------------------------------------------


def test_episode_monitor_attaches_episode_info():
    env = EpisodeMonitor(_make_cartpole())
    obs, _ = env.reset(seed=0)
    done = False
    info = {}
    while not done:
        obs, _, terminated, truncated, info = env.step(0)
        done = terminated or truncated
    env.close()
    assert "episode" in info
    assert "r" in info["episode"]
    assert "l" in info["episode"]
    assert info["episode"]["r"] > 0


def test_episode_monitor_length_matches_steps():
    env = EpisodeMonitor(_make_cartpole())
    obs, _ = env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        steps += 1
    env.close()
    assert info["episode"]["l"] == steps


# ---------------------------------------------------------------------------
# make_env factory
# ---------------------------------------------------------------------------


def test_make_env_cartpole():
    env = make_env("CartPole-v1", seed=42)
    obs, _ = env.reset()
    assert obs.shape == (4,)
    env.close()


def test_make_env_with_reward_scale():
    env = make_env("CartPole-v1", seed=0, reward_scale=0.01)
    obs, _ = env.reset()
    _, reward, *_ = env.step(0)
    env.close()
    assert abs(reward) <= 0.02


# ---------------------------------------------------------------------------
# Environment utils
# ---------------------------------------------------------------------------


def test_get_env_dims_cartpole():
    env = gym.make("CartPole-v1")
    obs_dim, action_dim = get_env_dims(env)
    env.close()
    assert obs_dim == 4
    assert action_dim == 2


def test_get_env_dims_lunarlander():
    pytest.importorskip("Box2D", reason="Box2D not installed — run: pip install swig gymnasium[box2d]")
    env = gym.make("LunarLander-v3")
    obs_dim, action_dim = get_env_dims(env)
    env.close()
    assert obs_dim == 8
    assert action_dim == 4


def test_set_global_seed_reproducibility():
    set_global_seed(42)
    a = np.random.randn(5)
    set_global_seed(42)
    b = np.random.randn(5)
    np.testing.assert_array_equal(a, b)
