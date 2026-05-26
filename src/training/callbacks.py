"""Training callbacks for logging, early stopping, and custom hooks."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from src.training.trainer import DQNTrainer, PPOTrainer


class BaseCallback:
    def on_training_start(self, trainer: Any) -> None:
        pass

    def on_episode_end(self, trainer: Any, reward: float, length: int) -> None:
        pass

    def on_training_end(self, trainer: Any) -> None:
        pass


class CallbackList(BaseCallback):
    def __init__(self, callbacks: List[BaseCallback]) -> None:
        self.callbacks = callbacks

    def on_training_start(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_training_start(trainer)

    def on_episode_end(self, trainer: Any, reward: float, length: int) -> None:
        for cb in self.callbacks:
            cb.on_episode_end(trainer, reward, length)

    def on_training_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_training_end(trainer)


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training once a reward threshold is sustained for N consecutive episodes.

    CartPole-v1 is considered "solved" at mean reward >= 475 over 100 episodes
    (OpenAI's official threshold). LunarLander-v3 at >= 200.
    """

    def __init__(self, reward_threshold: float, n_episodes: int = 100) -> None:
        self.reward_threshold = reward_threshold
        self.n_episodes = n_episodes
        self._recent_rewards: list[float] = []

    def on_episode_end(self, trainer: Any, reward: float, length: int) -> None:
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > self.n_episodes:
            self._recent_rewards.pop(0)

        if len(self._recent_rewards) == self.n_episodes:
            mean_r = sum(self._recent_rewards) / self.n_episodes
            if mean_r >= self.reward_threshold:
                print(
                    f"\nSolved! Mean reward {mean_r:.2f} >= threshold {self.reward_threshold}"
                )
                trainer.total_timesteps = 0  # signal trainer to stop


class LoggingCallback(BaseCallback):
    """Accumulates episode stats for offline analysis."""

    def __init__(self) -> None:
        self.rewards: list[float] = []
        self.lengths: list[int] = []

    def on_episode_end(self, trainer: Any, reward: float, length: int) -> None:
        self.rewards.append(reward)
        self.lengths.append(length)
