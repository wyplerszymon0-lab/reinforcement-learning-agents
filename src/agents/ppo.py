"""
Proximal Policy Optimization agent (Schulman et al., 2017).

PPO constrains the policy update to stay close to the old policy,
avoiding the catastrophic performance collapse that plagues vanilla
policy gradient methods. The clipped objective is simpler and often
more robust than the KL-penalty variant:

    L_CLIP(θ) = E_t[ min(r_t(θ) · Â_t, clip(r_t(θ), 1−ε, 1+ε) · Â_t) ]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t).

The clip acts as a first-order barrier: it prevents large increases in
probability for positive-advantage actions and large decreases for
negative-advantage actions, without computing second-order curvature.

Advantage normalization (batch-level) reduces variance further and is
nearly universally used in practice, though it introduces a mild bias.

Value function clipping mirrors the policy clip to prevent the critic
from moving too far in a single update — less critical than policy clip
but helps stability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.agents.networks import ActorCriticNetwork
from src.training.replay_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_steps: int = 2048,
        num_minibatches: int = 32,
        update_epochs: int = 10,
        normalize_advantages: bool = True,
        clip_vloss: bool = True,
        target_kl: Optional[float] = None,
        anneal_lr: bool = True,
        total_timesteps: int = 1_000_000,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(obs_dim, action_dim, device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.normalize_advantages = normalize_advantages
        self.clip_vloss = clip_vloss
        self.target_kl = target_kl
        self.anneal_lr = anneal_lr
        self.total_timesteps = total_timesteps
        self.batch_size = num_steps
        self.minibatch_size = num_steps // num_minibatches
        self._num_updates = total_timesteps // num_steps

        self.network = ActorCriticNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self._actor_params = [*self.network.actor_trunk.parameters(), *self.network.actor_head.parameters()]
        self._critic_params = [*self.network.critic_trunk.parameters(), *self.network.critic_head.parameters()]
        self._initial_lr = lr
        self.rollout_buffer = RolloutBuffer(num_steps, obs_dim, device)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Sample from the policy while training; take its most likely action otherwise."""
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs)
            if not training:
                logits, _ = self.network(obs_t)
                return int(logits.argmax(dim=-1).item())
            action, _, _, _ = self.network.get_action_and_value(obs_t)
        return int(action.item())

    def act_with_extras(
        self, obs: np.ndarray
    ) -> tuple[int, float, float]:
        """Returns (action, log_prob, value) for rollout collection."""
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs)
            action, log_prob, _, value = self.network.get_action_and_value(obs_t)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.rollout_buffer.push(state, action, reward, value, log_prob, done)
        self.total_steps += 1

    def finish_rollout(self, last_obs: np.ndarray, last_done: bool) -> None:
        """Bootstrap value for the last state and compute GAE."""
        with torch.no_grad():
            last_value = float(
                self.network.get_value(self._obs_to_tensor(last_obs)).item()
            ) * (1.0 - float(last_done))
        self.rollout_buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        states, actions, old_log_probs, advantages, returns, old_values = (
            self.rollout_buffer.get_tensors()
        )

        if self.anneal_lr:
            frac = 1.0 - (self.total_steps / self.total_timesteps)
            lr_now = frac * self._initial_lr
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_now

        metrics: Dict[str, list] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        indices = np.arange(self.batch_size)
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, self.batch_size, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = old_values[mb_idx]

                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    mb_states, mb_actions
                )

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                # Approximate KL for early stopping diagnostic
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Clipped surrogate objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (with optional clipping to match policy constraint)
                new_values = new_values.view(-1)
                if self.clip_vloss:
                    v_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values, -self.clip_coef, self.clip_coef
                    )
                    vf_loss = torch.max(
                        nn.functional.mse_loss(new_values, mb_returns),
                        nn.functional.mse_loss(v_clipped, mb_returns),
                    )
                else:
                    vf_loss = nn.functional.mse_loss(new_values, mb_returns)

                entropy_loss = entropy.mean()

                total_loss = policy_loss + self.vf_coef * vf_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                # Clip actor and critic separately: the value gradient is orders of
                # magnitude larger, and one global norm would shrink the policy step
                # until Adam's epsilon swallows it.
                nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(vf_loss.item())
                metrics["entropy_loss"].append(entropy_loss.item())
                metrics["approx_kl"].append(approx_kl.item())
                metrics["clip_fraction"].append(clip_fraction.item())

            # KL early stopping: prevents too-large policy updates per rollout
            if self.target_kl is not None:
                mean_kl = np.mean(metrics["approx_kl"])
                if mean_kl > 1.5 * self.target_kl:
                    break

        self.rollout_buffer.reset()

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
