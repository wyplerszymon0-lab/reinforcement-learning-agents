"""Abstract base class for all RL agents."""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class BaseAgent(abc.ABC):
    """
    Common interface for DQN and PPO agents.

    All agents share:
    - act(): select an action given an observation
    - update(): consume a batch of experience and return a dict of scalar metrics
    - save() / load(): checkpoint management
    """

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.total_steps = 0

    @abc.abstractmethod
    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select an action. Should be greedy / deterministic when training=False."""

    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Update agent parameters and return a dict of diagnostic metrics."""

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Serialize agent state to disk."""

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Restore agent state from disk."""

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(obs).unsqueeze(0).to(self.device)
