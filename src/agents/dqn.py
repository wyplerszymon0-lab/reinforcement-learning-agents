"""
Deep Q-Network agent.

Implements:
  - DQN (Mnih et al., 2015): neural Q-function + experience replay + target network
  - Double DQN (van Hasselt et al., 2016): decouple action selection from evaluation
    to eliminate the maximization bias inherent in standard DQN
  - Dueling architecture (Wang et al., 2016): optional, enabled by default

Key design decisions:
  - Huber loss (smooth L1) instead of MSE: bounded gradient for large TD errors
    prevents the destabilizing spikes that cause divergence in early training
  - Hard target updates every C steps (not soft τ-blend): more conservative,
    easier to tune, still works well for discrete control
  - Gradient clipping at 10.0: a safety net against occasional large gradients
  - Linear epsilon decay: simple and effective for discrete action spaces
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.agents.networks import DQNNetwork
from src.training.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1_000,
        dueling: bool = True,
        double_dqn: bool = True,
        grad_clip: float = 10.0,
        train_freq: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(obs_dim, action_dim, device)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.grad_clip = grad_clip
        self.train_freq = train_freq
        self._update_count = 0

        self.online_net = DQNNetwork(obs_dim, action_dim, hidden_dims, dueling).to(device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, obs_dim, device)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values = self.online_net(self._obs_to_tensor(obs))
        return int(q_values.argmax(dim=-1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self._decay_epsilon()

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        if not self.replay_buffer.is_ready or len(self.replay_buffer) < self.batch_size:
            return {}
        # One gradient step every `train_freq` environment steps. Updating on every
        # step overfits the replay buffer and makes CartPole oscillate and collapse.
        if self.total_steps % self.train_freq != 0:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: online net selects action, target net evaluates it.
                # This removes the upward bias: E[max Q] >= max E[Q].
                next_actions = self.online_net(next_states).argmax(dim=-1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(-1)
            else:
                next_q = self.target_net(next_states).max(dim=-1).values

            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._update_target()

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
            "epsilon": self.epsilon,
        }

    def _update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "epsilon": self.epsilon,
                "_update_count": self._update_count,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.epsilon = checkpoint["epsilon"]
        self._update_count = checkpoint["_update_count"]
