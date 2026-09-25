"""
Agent evaluation utilities.

Evaluation differs from training in three ways:
  1. No exploration: greedy policy (epsilon=0 for DQN, mode action for PPO)
  2. No learning: parameters are frozen
  3. Separate environment instance: avoids contaminating training statistics
"""
from __future__ import annotations

from typing import Dict, List

import gymnasium as gym
import numpy as np

from src.agents.base_agent import BaseAgent


class Evaluator:
    def __init__(self, env: gym.Env, n_episodes: int = 10) -> None:
        self.env = env
        self.n_episodes = n_episodes

    def evaluate(self, agent: BaseAgent, deterministic: bool = True) -> Dict[str, float]:
        """Run agent for n_episodes. Returns summary stats.

        deterministic=True takes the greedy action; False samples from the policy
        (only meaningful for stochastic policies such as PPO).
        """
        rewards: List[float] = []
        lengths: List[int] = []

        for _ in range(self.n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done:
                action = agent.act(obs, training=not deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                done = terminated or truncated

            rewards.append(ep_reward)
            lengths.append(ep_length)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
        }

    def evaluate_with_trajectories(
        self, agent: BaseAgent
    ) -> tuple[Dict[str, float], List[List[float]]]:
        """Returns summary stats and per-episode reward sequences for plotting."""
        rewards_per_step: List[List[float]] = []
        summary_rewards: List[float] = []

        for _ in range(self.n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_rewards: List[float] = []

            while not done:
                action = agent.act(obs, training=False)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_rewards.append(float(reward))
                done = terminated or truncated

            rewards_per_step.append(ep_rewards)
            summary_rewards.append(sum(ep_rewards))

        stats = {
            "mean_reward": float(np.mean(summary_rewards)),
            "std_reward": float(np.std(summary_rewards)),
            "min_reward": float(np.min(summary_rewards)),
            "max_reward": float(np.max(summary_rewards)),
        }
        return stats, rewards_per_step
