"""Tests for the optional scalar logging (TensorBoard) in the trainers."""
from collections import defaultdict

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts import train
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.environments.wrappers import make_env
from src.training.loggers import TensorBoardLogger
from src.training.trainer import DQNTrainer, PPOTrainer

DEVICE = torch.device("cpu")


class RecordingLogger:
    def __init__(self):
        self.points = defaultdict(list)  # tag -> [(step, value)]
        self.closed = False

    def scalar(self, tag, value, step):
        self.points[tag].append((step, float(value)))

    def close(self):
        self.closed = True


def run_dqn(logger, steps=3_000):
    agent = DQNAgent(obs_dim=4, action_dim=2, hidden_dims=(16,), buffer_capacity=1_000, batch_size=16,
                     epsilon_decay_steps=2_000, device=DEVICE)
    trainer = DQNTrainer(agent=agent, env=make_env("CartPole-v1", seed=0), total_timesteps=steps,
                         checkpoint_interval=10**9, verbose=False, logger=logger)
    return trainer.train()


def run_ppo(logger, steps=1_024):
    agent = PPOAgent(obs_dim=4, action_dim=2, hidden_dims=(16,), num_steps=128, num_minibatches=2,
                     update_epochs=1, total_timesteps=steps, device=DEVICE)
    trainer = PPOTrainer(agent=agent, env=make_env("CartPole-v1", seed=0), total_timesteps=steps,
                         checkpoint_interval=10**9, verbose=False, logger=logger)
    return trainer.train()


def test_dqn_logs_returns_loss_and_epsilon():
    log = RecordingLogger()
    metrics = run_dqn(log)

    returns = log.points["episode/return"]
    assert [v for _, v in returns] == pytest.approx(metrics.episode_rewards)
    assert [s for s, _ in returns] == metrics.steps  # x-axis = environment step
    assert [s for s, _ in log.points["train/epsilon"]] == [1_000, 2_000, 3_000]
    eps = [v for _, v in log.points["train/epsilon"]]
    assert eps == sorted(eps, reverse=True)  # epsilon decays
    assert log.points["train/loss"], "loss is logged once updates have started"
    assert set(log.points) == {"episode/return", "episode/length", "train/epsilon", "train/loss"}


def test_ppo_logs_losses_entropy_and_kl_once_per_update():
    log = RecordingLogger()
    run_ppo(log)
    expected = {"train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl", "train/clip_fraction"}
    assert expected <= set(log.points)
    for tag in expected:
        assert [s for s, _ in log.points[tag]] == [128 * k for k in range(1, 9)]  # 8 rollouts of 128 steps
    entropy = [v for _, v in log.points["train/entropy"]]
    assert all(0 < h <= 0.6932 for h in entropy)  # a 2-action policy: 0 < H <= ln 2
    assert all(kl >= -1e-6 for _, kl in log.points["train/approx_kl"])


def test_no_logger_means_no_logging():
    # The default stays as before: nothing but the in-memory metrics.
    assert run_dqn(None, steps=500).episode_rewards
    assert run_ppo(None, steps=256).episode_rewards


def test_tensorboard_logger_writes_readable_events(tmp_path):
    logger = TensorBoardLogger(tmp_path / "run")
    for step, value in [(10, 1.0), (20, 2.5), (30, 4.0)]:
        logger.scalar("episode/return", value, step)
    logger.close()

    events = EventAccumulator(str(tmp_path / "run"))
    events.Reload()
    points = [(e.step, e.value) for e in events.Scalars("episode/return")]
    assert points == [(10, 1.0), (20, 2.5), (30, 4.0)]


def test_train_script_tensorboard_flag_writes_a_run_directory(tmp_path):
    train.run("cartpole", "ppo", seed=0, timesteps=512, out_dir=tmp_path / "results", tensorboard_dir=tmp_path / "tb")
    events = EventAccumulator(str(tmp_path / "tb" / "ppo_cartpole_seed0"))
    events.Reload()
    assert {"episode/return", "train/entropy", "train/approx_kl"} <= set(events.Tags()["scalars"])
