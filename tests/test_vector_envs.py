"""PPO with gymnasium vector environments (N copies stepped together)."""
import numpy as np
import pytest
import torch

from scripts import train
from src.agents.ppo import PPOAgent
from src.environments.utils import set_global_seed
from src.environments.wrappers import make_env, make_vec_env
from src.training.replay_buffer import RolloutBuffer
from src.training.trainer import PPOTrainer

DEVICE = torch.device("cpu")


def agent(num_envs, num_steps=64, total=2_048):
    return PPOAgent(obs_dim=4, action_dim=2, hidden_dims=(16,), num_steps=num_steps, num_minibatches=2,
                    update_epochs=1, total_timesteps=total, device=DEVICE, num_envs=num_envs)


def test_vector_gae_equals_gae_of_each_env_on_its_own():
    rng = np.random.default_rng(0)
    T, N = 16, 4
    rewards, values, dones = rng.normal(size=(T, N)), rng.normal(size=(T, N)), rng.random((T, N)) < 0.2
    last = rng.normal(size=N)

    vec = RolloutBuffer(T, 3, DEVICE, num_envs=N)
    for t in range(T):
        vec.push(np.zeros((N, 3)), np.zeros(N, dtype=int), rewards[t], values[t], np.zeros(N), dones[t])
    vec.compute_gae(last, gamma=0.99, gae_lambda=0.95)

    for i in range(N):
        one = RolloutBuffer(T, 3, DEVICE)
        for t in range(T):
            one.push(np.zeros(3), 0, rewards[t, i], values[t, i], 0.0, dones[t, i])
        one.compute_gae(last[i], gamma=0.99, gae_lambda=0.95)
        np.testing.assert_allclose(vec._advantages[:, i], one._advantages[:, 0], rtol=1e-5, atol=1e-6)

    states, *_ = vec.get_tensors()
    assert states.shape == (T * N, 3)


def test_a_vector_of_one_reproduces_the_single_env_run_exactly():
    def run(env):
        set_global_seed(3)
        trainer = PPOTrainer(agent(1), env, total_timesteps=1_024, checkpoint_interval=10**9, verbose=False)
        return trainer.train()

    single = run(make_env("CartPole-v1", seed=3))
    vector = run(make_vec_env("CartPole-v1", 1, seed=3))
    assert single.episode_rewards == vector.episode_rewards
    assert single.steps == vector.steps


def test_four_envs_collect_four_times_the_steps_per_rollout():
    ag = agent(4, num_steps=64, total=2_048)  # 8 updates of 64 x 4 transitions
    metrics = PPOTrainer(ag, make_vec_env("CartPole-v1", 4, seed=0), total_timesteps=2_048,
                         checkpoint_interval=10**9, verbose=False).train()
    assert ag.batch_size == 256 and ag.total_steps == 2_048
    # CartPole pays 1 per step, so every logged episode's return equals its length:
    # no step was lost or counted twice when copies reset at different times.
    assert metrics.episode_rewards == [float(n) for n in metrics.episode_lengths]
    assert sum(metrics.episode_lengths) <= 2_048 and len(metrics.episode_lengths) > 20
    assert metrics.steps == sorted(metrics.steps)


def test_act_batch_returns_one_action_per_env():
    actions, log_probs, values = agent(4).act_batch(np.zeros((4, 4), dtype=np.float32))
    assert actions.shape == log_probs.shape == values.shape == (4,)
    assert set(actions.tolist()) <= {0, 1}


def test_mismatched_env_count_is_refused():
    with pytest.raises(ValueError, match="3 copies"):
        PPOTrainer(agent(4), make_vec_env("CartPole-v1", 3), total_timesteps=512, verbose=False).train()
    with pytest.raises(ValueError, match="vector env"):
        PPOTrainer(agent(4), make_env("CartPole-v1"), total_timesteps=512, verbose=False).train()


def test_train_script_num_envs(tmp_path):
    result = train.run("cartpole", "ppo", seed=0, timesteps=1_024, out_dir=tmp_path, num_envs=4)
    assert result["hyperparams"]["num_envs"] == 4
    assert result["hyperparams"]["num_steps"] * 4 == train.load_hyperparams("ppo", "cartpole")["num_steps"]
    with pytest.raises(ValueError, match="PPO"):
        train.run("cartpole", "dqn", seed=0, timesteps=1_000, out_dir=tmp_path, num_envs=2)
