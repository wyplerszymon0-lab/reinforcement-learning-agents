"""
Train DQN and PPO on LunarLander-v3.

LunarLander-v3 is considered solved at mean return >= 200 over 100 episodes.
It is substantially harder than CartPole:
  - 8-dimensional continuous observation (position, velocity, angle, angular
    velocity, leg contact booleans)
  - Sparse-ish shaped reward with a large penalty for crashing (-100)
  - Reward for using main engine (-0.3/frame) encourages fuel efficiency

Key tuning differences vs CartPole:
  - DQN: larger replay buffer (100k), slower epsilon decay (100k steps)
  - PPO: larger rollout buffer (2048), more timesteps needed (~1M)

Usage:
    python scripts/train_lunarlander.py --algo dqn
    python scripts/train_lunarlander.py --algo ppo
    python scripts/train_lunarlander.py --algo both
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym

from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.environments.utils import get_env_dims, set_global_seed
from src.environments.wrappers import make_env
from src.evaluation.evaluator import Evaluator
from src.evaluation.plotting import plot_comparison, plot_training_curve
from src.training.callbacks import CallbackList, EarlyStoppingCallback
from src.training.trainer import DQNTrainer, PPOTrainer


def train_dqn(seed: int = 42, total_timesteps: int = 500_000):
    set_global_seed(seed)
    env = make_env("LunarLander-v3", seed=seed)
    obs_dim, action_dim = get_env_dims(env)

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100_000,
        buffer_capacity=100_000,
        batch_size=64,
        target_update_freq=1_000,
        dueling=True,
        double_dqn=True,
    )

    callbacks = CallbackList([EarlyStoppingCallback(reward_threshold=200.0)])
    trainer = DQNTrainer(
        agent=agent,
        env=env,
        total_timesteps=total_timesteps,
        log_interval=10_000,
        checkpoint_dir=Path("checkpoints/lunarlander/dqn"),
        callbacks=callbacks,
    )

    print(f"\n{'='*60}")
    print("Training DQN on LunarLander-v3")
    print(f"{'='*60}\n")
    metrics = trainer.train()
    env.close()

    plot_training_curve(
        metrics.episode_rewards,
        title="DQN — LunarLander-v3 Training Curve",
        save_path=Path("results/plots/dqn_lunarlander_training.png"),
    )

    eval_env = gym.make("LunarLander-v3")
    evaluator = Evaluator(eval_env, n_episodes=20)
    stats = evaluator.evaluate(agent)
    eval_env.close()

    print(f"\nDQN Evaluation (20 episodes):")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}")

    return metrics


def train_ppo(seed: int = 42, total_timesteps: int = 1_000_000):
    set_global_seed(seed)
    env = make_env("LunarLander-v3", seed=seed)
    obs_dim, action_dim = get_env_dims(env)

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(64, 64),
        lr=3e-4,
        gamma=0.999,
        gae_lambda=0.98,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        num_steps=1024,
        num_minibatches=4,
        update_epochs=4,
        normalize_advantages=True,
        anneal_lr=True,
        total_timesteps=total_timesteps,
    )

    trainer = PPOTrainer(
        agent=agent,
        env=env,
        total_timesteps=total_timesteps,
        checkpoint_dir=Path("checkpoints/lunarlander/ppo"),
    )

    print(f"\n{'='*60}")
    print("Training PPO on LunarLander-v3")
    print(f"{'='*60}\n")
    metrics = trainer.train()
    env.close()

    plot_training_curve(
        metrics.episode_rewards,
        title="PPO — LunarLander-v3 Training Curve",
        save_path=Path("results/plots/ppo_lunarlander_training.png"),
    )

    eval_env = gym.make("LunarLander-v3")
    evaluator = Evaluator(eval_env, n_episodes=20)
    stats = evaluator.evaluate(agent)
    eval_env.close()

    print(f"\nPPO Evaluation (20 episodes):")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dqn_metrics = ppo_metrics = None

    if args.algo in ("dqn", "both"):
        dqn_metrics = train_dqn(seed=args.seed)

    if args.algo in ("ppo", "both"):
        ppo_metrics = train_ppo(seed=args.seed)

    if dqn_metrics and ppo_metrics:
        plot_comparison(
            {"DQN": dqn_metrics.episode_rewards, "PPO": ppo_metrics.episode_rewards},
            title="DQN vs PPO — LunarLander-v3",
            save_path=Path("results/plots/dqn_vs_ppo_lunarlander.png"),
        )
        print("\nComparison plot saved to results/plots/dqn_vs_ppo_lunarlander.png")


if __name__ == "__main__":
    main()
