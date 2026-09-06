"""Soft Actor-Critic with twin Q critics and optional automatic temperature."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from torch import nn

from aprenderl.algorithms.base import TrainingMetrics
from aprenderl.algorithms.off_policy_actor_critic import (
    OffPolicyActorCritic,
    OffPolicyActorCriticConfig,
    frozen_parameters,
    polyak_update,
)
from aprenderl.buffers import ReplayBatch
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger


@dataclass(frozen=True)
class SACConfig(OffPolicyActorCriticConfig):
    """SAC settings; ent_coef='auto' learns alpha, a float holds it fixed."""

    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    ent_coef: float | str = "auto"
    initial_ent_coef: float = 0.2
    ent_coef_learning_rate: float = 3e-4
    target_entropy: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.ent_coef, str):
            if self.ent_coef != "auto":
                raise ValueError("ent_coef must be 'auto' or a nonnegative number")
        elif not math.isfinite(self.ent_coef) or self.ent_coef < 0:
            raise ValueError("ent_coef must be finite and nonnegative")
        for name in ("initial_ent_coef", "ent_coef_learning_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.target_entropy is not None and not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")


class SAC(OffPolicyActorCritic):
    """Reparameterized Gaussian SAC for finite Box actions.

    The actor returns (means, log_stds), each shaped (batch, action_dim).
    Two independent critics accept observations and flattened environment actions.
    Log densities and target_entropy are measured in environment action units.
    """

    config_type = SACConfig
    stochastic = True
    n_critics = 2

    def __init__(
        self,
        env: gym.Env[Any, Any],
        policy_network: nn.Module | None = None,
        *,
        critic_networks: Sequence[nn.Module] | None = None,
        config: SACConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            policy_network,
            critic_networks=critic_networks,
            config=config,
            device=device,
            callback=callback,
            logger=logger,
        )
        if self.config.target_entropy is not None:
            self.target_entropy = self.config.target_entropy
        elif self.continuous:
            scale = (self.policy_action_space.high - self.policy_action_space.low) / 2
            # Account for affine scaling because our log density includes it.
            self.target_entropy = -float(self.action_dim) + scale.log().sum().item()
        else:
            self.target_entropy = 0.98 * math.log(self.action_dim)
        if not self.continuous and not 0 <= self.target_entropy <= math.log(
            self.action_dim
        ):
            raise ValueError("discrete target_entropy must be in [0, log(action_dim)]")
        self.log_ent_coef = None
        self.ent_coef_optimizer = None
        if self.config.ent_coef == "auto":
            self.log_ent_coef = torch.tensor(
                math.log(self.config.initial_ent_coef),
                device=self.device,
                requires_grad=True,
            )
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef], lr=self.config.ent_coef_learning_rate
            )

    @property
    def ent_coef(self) -> torch.Tensor:
        """Return the detached temperature used by the actor and critic."""
        if self.log_ent_coef is not None:
            return self.log_ent_coef.detach().exp()
        return torch.tensor(float(self.config.ent_coef), device=self.device)

    def _sample_policy(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.policy_action_space.distribution(
            self.policy_network(observations)
        )
        actions, log_prob = distribution.rsample_with_log_prob()
        return actions, log_prob.unsqueeze(-1)

    # ------------------------------------------------------------------
    # SAC learning rule
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _td_target(self, batch: ReplayBatch) -> torch.Tensor:
        actions, log_prob = self._sample_policy(batch.next_observations)
        next_q = torch.minimum(
            self.target_critics[0](batch.next_observations, actions),
            self.target_critics[1](batch.next_observations, actions),
        )
        soft_value = next_q - self.ent_coef * log_prob
        return batch.rewards + self.config.gamma * (1 - batch.terminated) * soft_value

    def _train_step(self) -> TrainingMetrics:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        targets = self._td_target(batch)
        critic_loss = sum(
            nn.functional.mse_loss(critic(batch.observations, batch.actions), targets)
            for critic in self.critic_networks
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic_networks.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        with frozen_parameters(self.critic_networks):
            actions, log_prob = self._sample_policy(batch.observations)
            q = torch.minimum(
                self.critic_networks[0](batch.observations, actions),
                self.critic_networks[1](batch.observations, actions),
            )
            actor_loss = (self.ent_coef * log_prob - q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), self.config.max_grad_norm
            )
            self.actor_optimizer.step()

        metrics = {}
        if self.log_ent_coef is not None:
            # Gradient descent increases alpha when entropy falls below target.
            temperature_loss = -(
                self.log_ent_coef * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.ent_coef_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.ent_coef_optimizer.step()
            metrics["train/temperature_loss"] = temperature_loss.item()
        polyak_update(self.critic_networks, self.target_critics, self.config.tau)
        self.num_updates += 1
        metrics.update(
            {
                "train/loss": critic_loss.item(),
                "train/actor_loss": actor_loss.item(),
                "train/entropy": -log_prob.detach().mean().item(),
                "train/ent_coef": self.ent_coef.item(),
                "train/updates": self.num_updates,
            }
        )
        return metrics

    def _checkpoint_extra(self) -> dict[str, Any]:
        return {
            "log_ent_coef": (
                self.log_ent_coef.detach() if self.log_ent_coef is not None else None
            ),
            "ent_coef_optimizer": (
                self.ent_coef_optimizer.state_dict()
                if self.ent_coef_optimizer is not None
                else None
            ),
        }

    def _restore_extra(self, state: dict[str, Any]) -> None:
        if self.log_ent_coef is not None:
            with torch.no_grad():
                self.log_ent_coef.copy_(state["log_ent_coef"])
            self.ent_coef_optimizer.load_state_dict(state["ent_coef_optimizer"])
