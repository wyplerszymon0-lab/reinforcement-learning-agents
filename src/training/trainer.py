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
from src.training.loggers import ScalarLogger

# DQN updates every few steps; its loss and epsilon are averaged over this many
# environment steps before they are logged, so the event file stays small.
DQN_SCALAR_EVERY = 1_000


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
        logger: Optional[ScalarLogger] = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.callbacks = callbacks or CallbackList([])
        self.verbose = verbose
        self.logger = logger
        self.metrics = TrainingMetrics()
        self.stop_training = False

    def train(self) -> TrainingMetrics:
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        best_mean_reward = float("-inf")
        recent_losses: List[float] = []

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
                if self.logger:
                    self.logger.scalar("episode/return", episode_reward, step)
                    self.logger.scalar("episode/length", episode_length, step)
                self.callbacks.on_episode_end(self, episode_reward, episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                if self.stop_training:
                    break

            update_metrics = self.agent.update()
            if update_metrics and "loss" in update_metrics:
                self.metrics.log_loss(update_metrics["loss"])
                recent_losses.append(update_metrics["loss"])

            if self.logger and step % DQN_SCALAR_EVERY == 0:
                self.logger.scalar("train/epsilon", self.agent.epsilon, step)
                if recent_losses:
                    self.logger.scalar("train/loss", float(np.mean(recent_losses)), step)
                    recent_losses.clear()

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
        env: gym.Env | gym.vector.VectorEnv,
        total_timesteps: int = 1_000_000,
        log_interval: int = 1,
        checkpoint_interval: int = 10,
        checkpoint_dir: Optional[Path] = None,
        callbacks: Optional[CallbackList] = None,
        verbose: bool = True,
        logger: Optional[ScalarLogger] = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.callbacks = callbacks or CallbackList([])
        self.verbose = verbose
        self.logger = logger
        self.metrics = TrainingMetrics()
        self.stop_training = False

    def train(self) -> TrainingMetrics:
        # A gymnasium vector env runs agent.num_envs copies in lockstep; a plain env is one copy.
        env = self.env
        self._vector = isinstance(env, gym.vector.VectorEnv)
        n = self.agent.num_envs
        if isinstance(env, gym.vector.VectorEnv) and env.num_envs != n:
            raise ValueError(f"env has {env.num_envs} copies but the agent was built for {n}")
        if not self._vector and n != 1:
            raise ValueError(f"agent was built for {n} environments; pass a gymnasium vector env")

        self._obs, _ = self.env.reset()
        self._done: Any = np.zeros(n, dtype=bool) if self._vector else False
        self._ep_return = np.zeros(n)
        self._ep_length = np.zeros(n, dtype=np.int64)
        best_mean_reward = float("-inf")
        num_updates = self.total_timesteps // self.agent.batch_size

        self.callbacks.on_training_start(self)
        start_time = time.time()

        for update in range(1, num_updates + 1):
            if self.stop_training:
                break
            for _ in range(self.agent.num_steps):
                if self._vector:
                    self._step_vector()
                else:
                    self._step_single()

            self.agent.finish_rollout(self._obs, self._done)
            update_metrics = self.agent.update()

            if "policy_loss" in update_metrics:
                self.metrics.log_loss(update_metrics["policy_loss"])

            if self.logger and update_metrics:
                # entropy_loss is the mean policy entropy (subtracted in the loss), not its negative
                for tag, key in (
                    ("train/policy_loss", "policy_loss"),
                    ("train/value_loss", "value_loss"),
                    ("train/entropy", "entropy_loss"),
                    ("train/approx_kl", "approx_kl"),
                    ("train/clip_fraction", "clip_fraction"),
                ):
                    if key in update_metrics:
                        self.logger.scalar(tag, update_metrics[key], self.agent.total_steps)

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

    def _end_episode(self, reward: float, length: int) -> None:
        self.metrics.log_episode(reward, length, self.agent.total_steps)
        if self.logger:
            self.logger.scalar("episode/return", reward, self.agent.total_steps)
            self.logger.scalar("episode/length", length, self.agent.total_steps)
        self.callbacks.on_episode_end(self, reward, length)

    def _step_single(self) -> None:
        action, log_prob, value = self.agent.act_with_extras(self._obs)
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = bool(terminated or truncated)
        self.agent.observe(self._obs, action, float(reward), value, log_prob, done)
        self._ep_return[0] += float(reward)
        self._ep_length[0] += 1
        self._obs, self._done = next_obs, done
        if done:
            self._end_episode(float(self._ep_return[0]), int(self._ep_length[0]))
            self._obs, _ = self.env.reset()
            self._done = False
            self._ep_return[0], self._ep_length[0] = 0.0, 0

    def _step_vector(self) -> None:
        # The vector env resets finished copies within the same step (AutoresetMode.SAME_STEP,
        # see make_vec_env), so next_obs already starts the next episode where done is set.
        actions, log_probs, values = self.agent.act_batch(self._obs)
        next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
        dones = np.logical_or(terminated, truncated)
        self.agent.observe(self._obs, actions, rewards, values, log_probs, dones)
        self._ep_return += np.asarray(rewards, dtype=np.float64)
        self._ep_length += 1
        for i in np.flatnonzero(dones):
            self._end_episode(float(self._ep_return[i]), int(self._ep_length[i]))
            self._ep_return[i], self._ep_length[i] = 0.0, 0
        self._obs, self._done = next_obs, dones
