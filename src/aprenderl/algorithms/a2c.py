"""Advantage Actor-Critic with multi-step GAE targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import torch
from torch import nn

from aprenderl.algorithms.actor_critic import ActorCritic, ActorCriticConfig
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger


@dataclass(frozen=True)
class A2CConfig(ActorCriticConfig):
    """Hyperparameters for :class:`A2C`."""

    learning_rate: float = 7e-4
    value_learning_rate: float = 7e-4
    n_steps: int = 5
    max_grad_norm: float = 0.5
    gae_lambda: float = 1.0
    normalize_advantage: bool = False
    value_loss_coefficient: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if self.value_loss_coefficient < 0:
            raise ValueError("value_loss_coefficient cannot be negative")


class A2C(ActorCritic):
    """Synchronous Advantage Actor-Critic for discrete or continuous actions.

    This compact implementation uses one environment worker. It collects a
    fixed-length on-policy rollout, estimates advantages with GAE, then takes
    one gradient step for the stochastic actor and state-value critic.
    """

    config_class: ClassVar[type[A2CConfig]] = A2CConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        policy_network: nn.Module | None = None,
        *,
        value_network: nn.Module | None = None,
        config: A2CConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            policy_network,
            value_network=value_network,
            config=config or A2CConfig(),
            device=device,
            callback=callback,
            logger=logger,
        )

    def _gae_lambda(self) -> float:
        return self.config.gae_lambda

    def _prepare_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_advantage or advantages.numel() <= 1:
            return advantages
        return (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

    def _value_objective_loss(self, value_loss: torch.Tensor) -> torch.Tensor:
        return self.config.value_loss_coefficient * value_loss
