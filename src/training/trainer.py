"""
Training loops for DQN and PPO.

Both trainers share a common metrics-logging interface but differ
fundamentally in their data collection strategy:
  - DQNTrainer: step-by-step off-policy loop with replay buffer
  - PPOTrainer: rollout-based on-policy loop (collect N steps, update, repeat)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.training.callbacks import CallbackList


@dataclass
class TrainingMetrics:
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def log_episode(self, reward: float, length: int, step: int) -> None:
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.steps.append(step)
        self.timestamps.append(time.time())

    def log_loss(self, loss: float) -> None:
        self.losses.append(loss)

    @property
    def mean_reward_last_n(self) -> float:
        if not self.episode_rewards:
            return float("-inf")
        return float(np.mean(self.episode_rewards[-100:]))


class DQNTrainer:
    def __init__(
        self,
        agent: DQNAgent,
        env: gym.Env,
        total_timesteps: int = 500_000,
        log_interval: int = 1_000,
        checkpoint_interval: int = 50_000,
        checkpoint_dir: Optional[Path] = None,
        callbacks: Optional[CallbackList] = None,
        verbose: bool = True,
    ) -> None:
        self.agent = agent
        self.env = env
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.callbacks = callbacks or CallbackList([])
        self.verbose = verbose
        self.metrics = TrainingMetrics()

    def train(self) -> TrainingMetrics:
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        best_mean_reward = float("-inf")

        self.callbacks.on_training_start(self)
        start_time = time.time()

        for step in range(1, self.total_timesteps + 1):
            action = self.agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # For replay, store non-truncated done signal.
            # Truncation (TimeLimit) does NOT mean terminal state, so we mask it
            # to avoid teaching the agent to expect low value at time horizon.
            replay_done = terminated and not truncated
            self.agent.observe(obs, action, float(reward), next_obs, replay_done)

            episode_reward += float(reward)
            episode_length += 1
            obs = next_obs

            if done:
                self.metrics.log_episode(episode_reward, episode_length, step)
                self.callbacks.on_episode_end(self, episode_reward, episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

            update_metrics = self.agent.update()
            if update_metrics and "loss" in update_metrics:
                self.metrics.log_loss(update_metrics["loss"])

            if self.verbose and step % self.log_interval == 0:
                mean_r = self.metrics.mean_reward_last_n
                elapsed = time.time() - start_time
                fps = step / elapsed
                eps = self.agent.epsilon
                print(
                    f"[DQN] step={step:>7} | "
                    f"mean_reward(100)={mean_r:>8.2f} | "
                    f"epsilon={eps:.3f} | "
                    f"fps={fps:.0f}"
                )

            if step % self.checkpoint_interval == 0:
                ckpt_path = self.checkpoint_dir / f"dqn_step_{step}.pt"
                self.agent.save(ckpt_path)

                mean_r = self.metrics.mean_reward_last_n
                if mean_r > best_mean_reward:
                    best_mean_reward = mean_r
                    self.agent.save(self.checkpoint_dir / "dqn_best.pt")

        self.callbacks.on_training_end(self)
        return self.metrics


class PPOTrainer:
    def __init__(
        self,
        agent: PPOAgent,
        env: gym.Env,
        total_timesteps: int = 1_000_000,
        log_interval: int = 1,
        checkpoint_interval: int = 10,
        checkpoint_dir: Optional[Path] = None,
        callbacks: Optional[CallbackList] = None,
        verbose: bool = True,
    ) -> None:
        self.agent = agent
        self.env = env
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.callbacks = callbacks or CallbackList([])
        self.verbose = verbose
        self.metrics = TrainingMetrics()

    def train(self) -> TrainingMetrics:
        obs, _ = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        best_mean_reward = float("-inf")
        num_updates = self.total_timesteps // self.agent.num_steps

        self.callbacks.on_training_start(self)
        start_time = time.time()

        for update in range(1, num_updates + 1):
            # Collect rollout
            for _ in range(self.agent.num_steps):
                action, log_prob, value = self.agent.act_with_extras(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                step_done = terminated or truncated

                self.agent.observe(obs, action, float(reward), value, log_prob, step_done)

                episode_reward += float(reward)
                episode_length += 1
                obs = next_obs
                done = step_done

                if step_done:
                    self.metrics.log_episode(episode_reward, episode_length, self.agent.total_steps)
                    self.callbacks.on_episode_end(self, episode_reward, episode_length)
                    obs, _ = self.env.reset()
                    done = False
                    episode_reward = 0.0
                    episode_length = 0

            self.agent.finish_rollout(obs, done)
            update_metrics = self.agent.update()

            if "policy_loss" in update_metrics:
                self.metrics.log_loss(update_metrics["policy_loss"])

            if self.verbose and update % self.log_interval == 0:
                mean_r = self.metrics.mean_reward_last_n
                elapsed = time.time() - start_time
                fps = self.agent.total_steps / elapsed
                kl = update_metrics.get("approx_kl", 0.0)
                print(
                    f"[PPO] update={update:>5} | "
                    f"steps={self.agent.total_steps:>8} | "
                    f"mean_reward(100)={mean_r:>8.2f} | "
                    f"approx_kl={kl:.4f} | "
                    f"fps={fps:.0f}"
                )

            if update % self.checkpoint_interval == 0:
                ckpt_path = self.checkpoint_dir / f"ppo_update_{update}.pt"
                self.agent.save(ckpt_path)

                mean_r = self.metrics.mean_reward_last_n
                if mean_r > best_mean_reward:
                    best_mean_reward = mean_r
                    self.agent.save(self.checkpoint_dir / "ppo_best.pt")

        self.callbacks.on_training_end(self)
        return self.metrics
