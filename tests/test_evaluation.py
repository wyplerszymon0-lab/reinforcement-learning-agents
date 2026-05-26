"""Tests for evaluation and plotting utilities."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import gymnasium as gym

from src.evaluation.evaluator import Evaluator
from src.evaluation.plotting import (
    plot_comparison,
    plot_loss_curve,
    plot_reward_distribution,
    plot_training_curve,
    smooth,
)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class _RandomAgent:
    """Minimal agent that always picks a random action."""

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        return np.random.randint(self.action_dim)


@pytest.fixture
def cartpole_env():
    env = gym.make("CartPole-v1")
    yield env
    env.close()


def test_evaluator_returns_expected_keys(cartpole_env):
    agent = _RandomAgent(action_dim=2)
    evaluator = Evaluator(cartpole_env, n_episodes=3)
    stats = evaluator.evaluate(agent)
    expected = {"mean_reward", "std_reward", "min_reward", "max_reward", "mean_length"}
    assert set(stats.keys()) == expected


def test_evaluator_mean_reward_positive(cartpole_env):
    agent = _RandomAgent(action_dim=2)
    evaluator = Evaluator(cartpole_env, n_episodes=5)
    stats = evaluator.evaluate(agent)
    assert stats["mean_reward"] > 0.0


def test_evaluator_min_leq_mean_leq_max(cartpole_env):
    agent = _RandomAgent(action_dim=2)
    evaluator = Evaluator(cartpole_env, n_episodes=10)
    stats = evaluator.evaluate(agent)
    assert stats["min_reward"] <= stats["mean_reward"] <= stats["max_reward"]


def test_evaluator_with_trajectories_returns_episodes(cartpole_env):
    agent = _RandomAgent(action_dim=2)
    evaluator = Evaluator(cartpole_env, n_episodes=3)
    stats, trajectories = evaluator.evaluate_with_trajectories(agent)
    assert len(trajectories) == 3
    assert all(len(t) > 0 for t in trajectories)


# ---------------------------------------------------------------------------
# smooth()
# ---------------------------------------------------------------------------


def test_smooth_shorter_than_window():
    arr = [1.0, 2.0, 3.0]
    result = smooth(arr, window=10)
    np.testing.assert_array_equal(result, arr)


def test_smooth_output_length():
    arr = list(range(100))
    result = smooth(arr, window=20)
    assert len(result) == 100 - 20 + 1


def test_smooth_constant_input():
    arr = [5.0] * 50
    result = smooth(arr, window=10)
    np.testing.assert_allclose(result, 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Plotting functions (just verify they save without error)
# ---------------------------------------------------------------------------


def test_plot_training_curve_saves_png():
    rewards = list(np.random.randn(100))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "curve.png"
        plot_training_curve(rewards, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_comparison_saves_png():
    data = {"DQN": list(np.random.randn(80)), "PPO": list(np.random.randn(80))}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "comparison.png"
        plot_comparison(data, save_path=path)
        assert path.exists()


def test_plot_reward_distribution_saves_png():
    data = {"DQN": list(np.random.randn(50) * 10 + 200)}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dist.png"
        plot_reward_distribution(data, save_path=path)
        assert path.exists()


def test_plot_loss_curve_saves_png():
    losses = list(np.abs(np.random.randn(200)))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "loss.png"
        plot_loss_curve(losses, save_path=path)
        assert path.exists()
