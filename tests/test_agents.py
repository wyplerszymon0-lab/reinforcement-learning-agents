"""Integration tests for DQN and PPO agents."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent

DEVICE = torch.device("cpu")
OBS_DIM = 4
ACTION_DIM = 2


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------


@pytest.fixture
def dqn_agent():
    return DQNAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=(64, 64),
        lr=1e-3,
        buffer_capacity=1000,
        batch_size=32,
        epsilon_decay_steps=10_000,
        device=DEVICE,
    )


def test_dqn_act_returns_valid_action(dqn_agent):
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = dqn_agent.act(obs, training=False)
    assert 0 <= action < ACTION_DIM


def test_dqn_act_greedy_when_not_training(dqn_agent):
    """With epsilon=0, action should always be greedy (same for identical obs)."""
    dqn_agent.epsilon = 0.0
    obs = np.ones(OBS_DIM, dtype=np.float32)
    actions = {dqn_agent.act(obs, training=False) for _ in range(20)}
    assert len(actions) == 1  # deterministic


def test_dqn_epsilon_decays_on_observe(dqn_agent):
    initial_epsilon = dqn_agent.epsilon
    for _ in range(100):
        dqn_agent.observe(
            np.zeros(OBS_DIM), 0, 1.0, np.ones(OBS_DIM), False
        )
    assert dqn_agent.epsilon < initial_epsilon


def test_dqn_update_returns_empty_before_buffer_ready(dqn_agent):
    result = dqn_agent.update()
    assert result == {}


def test_dqn_update_returns_metrics_after_warmup(dqn_agent):
    for _ in range(200):
        dqn_agent.observe(
            np.random.randn(OBS_DIM).astype(np.float32),
            np.random.randint(ACTION_DIM),
            np.random.randn(),
            np.random.randn(OBS_DIM).astype(np.float32),
            False,
        )
    metrics = dqn_agent.update()
    assert "loss" in metrics
    assert "epsilon" in metrics
    assert metrics["loss"] >= 0.0


def test_dqn_save_load_roundtrip(dqn_agent):
    for _ in range(50):
        dqn_agent.observe(np.zeros(OBS_DIM), 0, 1.0, np.ones(OBS_DIM), False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "dqn.pt"
        dqn_agent.save(ckpt)

        new_agent = DQNAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dims=(64, 64), device=DEVICE)
        new_agent.load(ckpt)

        obs = np.ones(OBS_DIM, dtype=np.float32)
        assert dqn_agent.act(obs, training=False) == new_agent.act(obs, training=False)
        assert dqn_agent.total_steps == new_agent.total_steps
        assert abs(dqn_agent.epsilon - new_agent.epsilon) < 1e-6


def test_dqn_target_network_updates(dqn_agent):
    """After target_update_freq updates, target should match online net."""
    dqn_agent.target_update_freq = 5
    for _ in range(500):
        dqn_agent.observe(
            np.random.randn(OBS_DIM).astype(np.float32),
            0,
            1.0,
            np.random.randn(OBS_DIM).astype(np.float32),
            False,
        )

    for _ in range(50):
        dqn_agent.update()

    # Force update
    dqn_agent._update_target()
    for p_online, p_target in zip(
        dqn_agent.online_net.parameters(), dqn_agent.target_net.parameters()
    ):
        assert torch.allclose(p_online, p_target)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------


@pytest.fixture
def ppo_agent():
    return PPOAgent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dims=(64, 64),
        lr=2.5e-4,
        num_steps=64,
        num_minibatches=4,
        update_epochs=2,
        total_timesteps=10_000,
        device=DEVICE,
    )


def test_ppo_act_returns_valid_action(ppo_agent):
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = ppo_agent.act(obs)
    assert 0 <= action < ACTION_DIM


def test_ppo_act_with_extras_shapes(ppo_agent):
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action, log_prob, value = ppo_agent.act_with_extras(obs)
    assert 0 <= action < ACTION_DIM
    assert log_prob <= 0.0
    assert isinstance(value, float)


def test_ppo_update_returns_metrics(ppo_agent):
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    for _ in range(ppo_agent.num_steps):
        action, log_prob, value = ppo_agent.act_with_extras(obs)
        ppo_agent.observe(obs, action, 1.0, value, log_prob, False)
        obs = np.random.randn(OBS_DIM).astype(np.float32)

    ppo_agent.finish_rollout(obs, False)
    metrics = ppo_agent.update()

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy_loss" in metrics
    assert "approx_kl" in metrics


def test_ppo_save_load_roundtrip(ppo_agent):
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "ppo.pt"
        ppo_agent.save(ckpt)

        new_agent = PPOAgent(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, num_steps=64, num_minibatches=4,
            total_timesteps=10_000, device=DEVICE,
        )
        new_agent.load(ckpt)

        obs = np.ones(OBS_DIM, dtype=np.float32)
        # Same weights -> same greedy action
        torch.manual_seed(0)
        a1 = ppo_agent.act(obs)
        torch.manual_seed(0)
        a2 = new_agent.act(obs)
        assert a1 == a2


def test_ppo_rollout_buffer_resets_after_update(ppo_agent):
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    for _ in range(ppo_agent.num_steps):
        action, log_prob, value = ppo_agent.act_with_extras(obs)
        ppo_agent.observe(obs, action, 1.0, value, log_prob, False)
    ppo_agent.finish_rollout(obs, False)
    ppo_agent.update()
    assert not ppo_agent.rollout_buffer.is_full
