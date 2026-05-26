"""
Train DQN and PPO on CartPole-v1.

CartPole-v1 is considered solved when the agent achieves a mean return
of >= 475 over 100 consecutive episodes (OpenAI's original benchmark).

Usage:
    python scripts/train_cartpole.py --algo dqn
    python scripts/train_cartpole.py --algo ppo
    python scripts/train_cartpole.py --algo both
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


def train_dqn(seed: int = 42, total_timesteps: int = 200_000):
    set_global_seed(seed)
    env = make_env("CartPole-v1", seed=seed)
    obs_dim, action_dim = get_env_dims(env)

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,
        buffer_capacity=50_000,
        batch_size=64,
        target_update_freq=500,
        dueling=True,
        double_dqn=True,
    )

    callbacks = CallbackList([EarlyStoppingCallback(reward_threshold=475.0)])
    trainer = DQNTrainer(
        agent=agent,
        env=env,
        total_timesteps=total_timesteps,
        log_interval=5_000,
        checkpoint_dir=Path("checkpoints/cartpole/dqn"),
        callbacks=callbacks,
    )

    print(f"\n{'='*60}")
    print("Training DQN on CartPole-v1")
    print(f"{'='*60}\n")
    metrics = trainer.train()
    env.close()

    # Save plots
    plot_training_curve(
        metrics.episode_rewards,
        title="DQN — CartPole-v1 Training Curve",
        save_path=Path("results/plots/dqn_cartpole_training.png"),
    )

    # Evaluate
    eval_env = gym.make("CartPole-v1")
    evaluator = Evaluator(eval_env, n_episodes=20)
    stats = evaluator.evaluate(agent)
    eval_env.close()

    print(f"\nDQN Evaluation (20 episodes):")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}")

    return metrics


def train_ppo(seed: int = 42, total_timesteps: int = 500_000):
    set_global_seed(seed)
    env = make_env("CartPole-v1", seed=seed)
    obs_dim, action_dim = get_env_dims(env)

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(64, 64),
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        num_steps=512,
        num_minibatches=8,
        update_epochs=10,
        total_timesteps=total_timesteps,
    )

    trainer = PPOTrainer(
        agent=agent,
        env=env,
        total_timesteps=total_timesteps,
        checkpoint_dir=Path("checkpoints/cartpole/ppo"),
    )

    print(f"\n{'='*60}")
    print("Training PPO on CartPole-v1")
    print(f"{'='*60}\n")
    metrics = trainer.train()
    env.close()

    plot_training_curve(
        metrics.episode_rewards,
        title="PPO — CartPole-v1 Training Curve",
        save_path=Path("results/plots/ppo_cartpole_training.png"),
    )

    eval_env = gym.make("CartPole-v1")
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
            title="DQN vs PPO — CartPole-v1",
            save_path=Path("results/plots/dqn_vs_ppo_cartpole.png"),
        )
        print("\nComparison plot saved to results/plots/dqn_vs_ppo_cartpole.png")


if __name__ == "__main__":
    main()
