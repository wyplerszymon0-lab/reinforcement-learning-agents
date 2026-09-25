"""Tests for trainers, early stopping, the train script and result artifacts."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import train
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.environments.wrappers import make_env
from src.evaluation.plotting import plot_seeds_by_steps
from src.evaluation.recording import record_episode
from src.training.callbacks import BaseCallback, CallbackList, EarlyStoppingCallback
from src.training.trainer import DQNTrainer, PPOTrainer

DEVICE = torch.device("cpu")


class StopAfter(BaseCallback):
    def __init__(self, episodes: int) -> None:
        self.episodes = episodes
        self.seen = 0

    def on_episode_end(self, trainer, reward, length):
        self.seen += 1
        if self.seen >= self.episodes:
            trainer.stop_training = True


def small_dqn():
    return DQNAgent(obs_dim=4, action_dim=2, hidden_dims=(16,), buffer_capacity=500, batch_size=8, device=DEVICE)


def small_ppo(total_timesteps=10_000):
    return PPOAgent(
        obs_dim=4, action_dim=2, hidden_dims=(16,), num_steps=64, num_minibatches=2,
        update_epochs=1, total_timesteps=total_timesteps, device=DEVICE,
    )


def test_early_stopping_triggers_only_on_full_window():
    class Trainer:
        stop_training = False

    trainer = Trainer()
    cb = EarlyStoppingCallback(reward_threshold=10.0, n_episodes=3)
    cb.on_episode_end(trainer, 50.0, 1)
    cb.on_episode_end(trainer, 50.0, 1)
    assert not trainer.stop_training
    cb.on_episode_end(trainer, 50.0, 1)
    assert trainer.stop_training


@pytest.mark.parametrize("trainer_cls,agent_fn", [(DQNTrainer, small_dqn), (PPOTrainer, small_ppo)])
def test_trainer_stops_when_callback_asks(tmp_path, trainer_cls, agent_fn):
    env = make_env("CartPole-v1", seed=0)
    trainer = trainer_cls(
        agent=agent_fn(), env=env, total_timesteps=10_000, checkpoint_dir=tmp_path,
        callbacks=CallbackList([StopAfter(3)]), verbose=False,
    )
    metrics = trainer.train()
    env.close()
    # DQN stops right after the 3rd episode; PPO finishes the rollout it is in.
    assert 3 <= len(metrics.episode_rewards) <= 3 + 64
    assert metrics.steps[-1] < 10_000


def test_ppo_greedy_action_is_deterministic():
    agent = small_ppo()
    obs = np.array([0.01, -0.02, 0.03, 0.04], dtype=np.float32)
    assert len({agent.act(obs, training=False) for _ in range(50)}) == 1


def test_ppo_greedy_action_is_argmax_of_logits():
    agent = small_ppo()
    obs = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    with torch.no_grad():
        logits, _ = agent.network(torch.as_tensor(obs).unsqueeze(0))
    assert agent.act(obs, training=False) == int(logits.argmax())


def test_solved_at_step():
    rewards = [0] * 5 + [10] * 5
    steps = list(range(100, 1100, 100))
    assert train.solved_at_step(rewards, steps, threshold=10, window=5) == 1000
    assert train.solved_at_step(rewards, steps, threshold=11, window=5) is None


@pytest.mark.parametrize("algo,env", [("dqn", "cartpole"), ("ppo", "lunarlander")])
def test_config_builds_every_agent(algo, env):
    hp = train.load_hyperparams(algo, env)
    obs_dim, action_dim = (4, 2) if env == "cartpole" else (8, 4)
    agent = train.build_agent(algo, obs_dim, action_dim, hp)
    assert agent.act(np.zeros(obs_dim, dtype=np.float32), training=False) in range(action_dim)


def test_train_script_writes_run_and_model(tmp_path):
    result = train.run("cartpole", "ppo", seed=0, timesteps=1_024, out_dir=tmp_path)
    saved = json.loads((tmp_path / "runs" / "ppo_cartpole_seed0.json").read_text())
    assert saved["total_steps"] == result["total_steps"] > 0
    assert len(saved["episode_rewards"]) == len(saved["episode_steps"])
    assert set(saved["evaluation"]) >= {"mean_reward", "std_reward"}
    assert (tmp_path / "models" / "ppo_cartpole_seed0.pt").is_file()


def test_plot_seeds_by_steps_saves_png(tmp_path):
    seeds = [([10, 20, 30, 40], [1, 2, 3, 4]), ([15, 25], [2, 5])]
    path = tmp_path / "cmp.png"
    plot_seeds_by_steps({"A": seeds}, threshold=3, save_path=path)
    assert path.stat().st_size > 0


def test_record_episode_writes_gif(tmp_path):
    path = tmp_path / "ep.gif"
    ret = record_episode(small_dqn(), "CartPole-v1", path, seed=0)
    assert ret > 0
    assert path.read_bytes()[:3] == b"GIF"


def test_value_loss_does_not_touch_actor_parameters():
    """Regression: a shared trunk let the (large) value loss drown the policy gradient."""
    from src.agents.networks import ActorCriticNetwork

    net = ActorCriticNetwork(obs_dim=8, action_dim=4, hidden_dims=(16, 16))
    value = net.get_value(torch.randn(32, 8))
    (value ** 2).mean().backward()
    actor_params = list(net.actor_trunk.parameters()) + list(net.actor_head.parameters())
    assert all(p.grad is None for p in actor_params)
    assert all(p.grad is not None for p in net.critic_trunk.parameters())


def test_dqn_train_freq_skips_updates_between_steps():
    agent = DQNAgent(obs_dim=4, action_dim=2, hidden_dims=(16,), buffer_capacity=500,
                     batch_size=8, train_freq=4, device=DEVICE)
    obs = np.zeros(4, dtype=np.float32)
    updated = []
    for _ in range(600):
        agent.observe(obs, 0, 1.0, obs, False)
        updated.append(bool(agent.update()))
    first = updated.index(True)
    assert all(u == ((i + 1) % 4 == 0) for i, u in enumerate(updated) if i >= first)
