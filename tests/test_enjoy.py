"""Tests for scripts/enjoy.py, using the models committed in results/models."""
import pytest

from scripts import enjoy
from scripts.train import load_trained_agent


def test_find_run_picks_the_requested_seed():
    run = enjoy.find_run("cartpole", "dqn", seed=1)
    assert (run["algo"], run["env"], run["seed"]) == ("dqn", "cartpole", 1)


def test_find_run_without_seed_picks_the_best_evaluated_saved_model():
    run = enjoy.find_run("lunarlander", "ppo", seed=None)
    assert enjoy.model_path(run).exists()


def test_find_run_explains_what_is_available():
    with pytest.raises(SystemExit, match="no saved model for dqn_cartpole_seed99; available: .*dqn_cartpole_seed1"):
        enjoy.find_run("cartpole", "dqn", seed=99)


def test_play_without_a_window_returns_one_return_per_episode(capsys):
    run = enjoy.find_run("cartpole", "ppo", seed=1)
    returns = enjoy.play(load_trained_agent(run), run["env_id"], episodes=2, sample=False, render_mode=None)
    assert len(returns) == 2
    assert all(r == 500 for r in returns)  # this model balances perfectly when greedy
    assert "episode 2: return 500.0" in capsys.readouterr().out
