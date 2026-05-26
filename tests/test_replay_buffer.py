"""Tests for ReplayBuffer and RolloutBuffer."""
import numpy as np
import pytest
import torch

from src.training.replay_buffer import ReplayBuffer, RolloutBuffer

DEVICE = torch.device("cpu")
OBS_DIM = 4


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(capacity=1000, obs_dim=OBS_DIM, device=DEVICE)


def test_replay_buffer_push_increments_size(replay_buffer):
    assert len(replay_buffer) == 0
    replay_buffer.push(np.zeros(OBS_DIM), 0, 1.0, np.ones(OBS_DIM), False)
    assert len(replay_buffer) == 1


def test_replay_buffer_overflow_wraps(replay_buffer):
    buf = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, device=DEVICE)
    for i in range(20):
        buf.push(np.zeros(OBS_DIM), 0, float(i), np.ones(OBS_DIM), False)
    assert len(buf) == 10


def test_replay_buffer_sample_shape(replay_buffer):
    for _ in range(200):
        replay_buffer.push(
            np.random.randn(OBS_DIM), 0, 1.0, np.random.randn(OBS_DIM), False
        )
    states, actions, rewards, next_states, dones = replay_buffer.sample(32)
    assert states.shape == (32, OBS_DIM)
    assert actions.shape == (32,)
    assert rewards.shape == (32,)
    assert next_states.shape == (32, OBS_DIM)
    assert dones.shape == (32,)


def test_replay_buffer_sample_no_duplicates(replay_buffer):
    for i in range(100):
        replay_buffer.push(np.full(OBS_DIM, float(i)), i % 2, 1.0, np.zeros(OBS_DIM), False)
    states, _, _, _, _ = replay_buffer.sample(50)
    # All 50 drawn states should be distinct (since we store unique observations)
    unique_rows = torch.unique(states, dim=0)
    assert len(unique_rows) == 50


def test_replay_buffer_sample_raises_when_undersized():
    buf = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, device=DEVICE)
    buf.push(np.zeros(OBS_DIM), 0, 0.0, np.zeros(OBS_DIM), False)
    with pytest.raises(ValueError):
        buf.sample(10)


def test_replay_buffer_done_stored_correctly():
    buf = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, device=DEVICE)
    buf.push(np.zeros(OBS_DIM), 0, 1.0, np.zeros(OBS_DIM), True)
    buf.push(np.zeros(OBS_DIM), 0, 1.0, np.zeros(OBS_DIM), False)
    # Ensure done flags come back as floats 1.0 and 0.0
    _, _, _, _, dones = buf.sample(2)
    assert set(dones.tolist()) == {0.0, 1.0}


def test_replay_buffer_is_ready():
    buf = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, device=DEVICE)
    assert not buf.is_ready
    for _ in range(10):  # 10 = 100 // 10
        buf.push(np.zeros(OBS_DIM), 0, 0.0, np.zeros(OBS_DIM), False)
    assert buf.is_ready


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


NUM_STEPS = 128


@pytest.fixture
def rollout_buffer():
    return RolloutBuffer(num_steps=NUM_STEPS, obs_dim=OBS_DIM, device=DEVICE)


def _fill_rollout_buffer(buf: RolloutBuffer) -> None:
    for _ in range(NUM_STEPS):
        buf.push(
            np.random.randn(OBS_DIM),
            action=0,
            reward=1.0,
            value=0.5,
            log_prob=-1.0,
            done=False,
        )


def test_rollout_buffer_is_full_after_n_steps(rollout_buffer):
    assert not rollout_buffer.is_full
    _fill_rollout_buffer(rollout_buffer)
    assert rollout_buffer.is_full


def test_rollout_buffer_push_overflow_raises(rollout_buffer):
    _fill_rollout_buffer(rollout_buffer)
    with pytest.raises(AssertionError):
        rollout_buffer.push(np.zeros(OBS_DIM), 0, 0.0, 0.0, 0.0, False)


def test_rollout_buffer_gae_shapes(rollout_buffer):
    _fill_rollout_buffer(rollout_buffer)
    rollout_buffer.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    states, actions, log_probs, advantages, returns, values = rollout_buffer.get_tensors()
    assert states.shape == (NUM_STEPS, OBS_DIM)
    assert advantages.shape == (NUM_STEPS,)
    assert returns.shape == (NUM_STEPS,)


def test_rollout_buffer_gae_terminal_bootstrap():
    """With done=True on the last step, bootstrap value must be zero."""
    buf = RolloutBuffer(num_steps=4, obs_dim=2, device=DEVICE)
    for i in range(3):
        buf.push(np.zeros(2), 0, 1.0, 0.5, -1.0, False)
    buf.push(np.zeros(2), 0, 1.0, 0.5, -1.0, True)  # terminal
    buf.compute_gae(last_value=10.0, gamma=0.99, gae_lambda=1.0)
    _, _, _, advantages, returns, _ = buf.get_tensors()
    # returns at last terminal step should not include last_value=10.0
    assert returns[-1].item() < 5.0


def test_rollout_buffer_reset(rollout_buffer):
    _fill_rollout_buffer(rollout_buffer)
    rollout_buffer.reset()
    assert not rollout_buffer.is_full
    assert rollout_buffer._pos == 0
