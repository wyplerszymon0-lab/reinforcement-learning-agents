"""
Neural network architectures for RL agents.

Design choices:
- Orthogonal initialization (Saxe et al., 2013): empirically outperforms Xavier/Kaiming in RL
- Dueling DQN architecture separates V(s) and A(s,a) for better credit assignment
- Tanh activations for actor-critic (PPO), ReLU for value-based (DQN)
- Shared-trunk actor-critic reduces computation and can improve sample efficiency
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        output_std: float = 0.01,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            layers.append(activation)
            in_dim = h_dim
        layers.append(layer_init(nn.Linear(in_dim, output_dim), std=output_std))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with optional dueling architecture (Wang et al., 2016).

    Dueling decomposition:
        Q(s, a) = V(s) + A(s, a) - (1/|A|) * sum_a' A(s, a')

    The mean-centering of advantages ensures identifiability: without it,
    V and A cannot be uniquely recovered from Q, making learning unstable.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (256, 256),
        dueling: bool = True,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.action_dim = action_dim

        trunk_layers: list[nn.Module] = []
        in_dim = obs_dim
        for h_dim in list(hidden_dims)[:-1]:
            trunk_layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            trunk_layers.append(nn.ReLU())
            in_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)
        final_hidden = list(hidden_dims)[-1]

        if dueling:
            self.value_stream = nn.Sequential(
                layer_init(nn.Linear(in_dim, final_hidden)),
                nn.ReLU(),
                layer_init(nn.Linear(final_hidden, 1), std=1.0),
            )
            self.advantage_stream = nn.Sequential(
                layer_init(nn.Linear(in_dim, final_hidden)),
                nn.ReLU(),
                layer_init(nn.Linear(final_hidden, action_dim), std=1.0),
            )
        else:
            self.q_head = nn.Sequential(
                layer_init(nn.Linear(in_dim, final_hidden)),
                nn.ReLU(),
                layer_init(nn.Linear(final_hidden, action_dim), std=1.0),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        if self.dueling:
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            return value + advantages - advantages.mean(dim=-1, keepdim=True)
        return self.q_head(features)


class ActorCriticNetwork(nn.Module):
    """
    Actor-critic for PPO with separate actor and critic networks.

    The two do not share a trunk on purpose. The value loss is measured in squared
    returns (thousands on LunarLander), while the policy loss is O(1): with a shared
    trunk the critic's gradient dominates the shared weights, and after global
    gradient-norm clipping the policy receives a tiny fraction of each step (measured
    at ~0.03% on LunarLander), so the agent barely learns.

    Actor head initialized with small std (0.01) so the initial policy is
    near-uniform — the agent explores before exploiting at the start.
    Critic head initialized with std=1.0 for better initial value estimates.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.actor_trunk = self._make_trunk(obs_dim, hidden_dims)
        self.critic_trunk = self._make_trunk(obs_dim, hidden_dims)
        self.actor_head = layer_init(nn.Linear(hidden_dims[-1], action_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(hidden_dims[-1], 1), std=1.0)

    @staticmethod
    def _make_trunk(obs_dim: int, hidden_dims) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            # Tanh is preferred over ReLU in on-policy methods: bounded activations
            # reduce variance in policy gradients and work well with GAE.
            layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            layers.append(nn.Tanh())
            in_dim = h_dim
        return nn.Sequential(*layers)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_trunk(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_head(self.actor_trunk(x))
        value = self.get_value(x)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor_head(self.actor_trunk(x)), self.get_value(x)
