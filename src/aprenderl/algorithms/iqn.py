"""Implicit Quantile Network (IQN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from aprenderl.algorithms.dqn import DQN, DQNConfig, _environment_dimensions
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import ImplicitQuantileNetwork


@dataclass(frozen=True)
class IQNConfig(DQNConfig):
    """DQN settings for implicit quantile sampling."""

    quantiles: int = 32
    target_quantiles: int = 32
    embedding_dim: int = 64
    huber_threshold: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantiles <= 0 or self.target_quantiles <= 0:
            raise ValueError("quantile counts must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.huber_threshold <= 0:
            raise ValueError("huber_threshold must be positive")


class IQN(DQN):
    """Learn a continuous return quantile function from sampled fractions."""

    config_type = IQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: IQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        actual_config = config or IQNConfig()
        provided_network = network is not None
        if network is None:
            observation_shape, action_dim, _ = _environment_dimensions(env, "IQN")
            network = ImplicitQuantileNetwork(
                int(np.prod(observation_shape)),
                action_dim,
                embedding_dim=actual_config.embedding_dim,
            )
        super().__init__(
            env,
            network,
            config=actual_config,
            device=device,
            callback=callback,
            logger=logger,
        )
        self._uses_default_network = not provided_network

    def _train_step(self) -> dict[str, float | int]:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        predicted, taus = self.q_network(batch.observations, self.config.quantiles)
        action_indices = batch.actions.unsqueeze(-1).expand(
            -1, 1, self.config.quantiles
        )
        predicted = predicted.gather(1, action_indices).squeeze(1)

        with torch.no_grad():
            action_network = (
                self.q_network if self.config.double_dqn else self.target_network
            )
            next_values = self._q_values(action_network, batch.next_observations)
            next_actions = next_values.argmax(dim=1)
            target_quantiles, _ = self.target_network(
                batch.next_observations, self.config.target_quantiles
            )
            rows = torch.arange(batch.rewards.shape[0], device=self.device)
            target_quantiles = target_quantiles[rows, next_actions]
            target_quantiles = (
                batch.rewards
                + self.config.gamma * (1 - batch.terminated) * target_quantiles
            )

        loss = quantile_huber_loss(
            predicted,
            target_quantiles,
            taus,
            self.config.huber_threshold,
        )
        return self._optimize(loss)

    def _q_values(self, network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        try:
            return network.q_values(observations, self.config.quantiles)
        except AttributeError as error:
            raise TypeError("an IQN network must implement q_values()") from error

    def _validate_network(self, network: nn.Module) -> None:
        observations = torch.zeros((1, *self.observation_shape), device=self.device)
        try:
            with torch.no_grad():
                output, taus = network(observations, self.config.quantiles)
                q_values = network.q_values(observations, self.config.quantiles)
        except Exception as error:
            raise ValueError("network must implement the IQN network API") from error
        expected = (1, self.action_dim, self.config.quantiles)
        if output.shape != expected or taus.shape != (1, self.config.quantiles):
            raise ValueError("network returned invalid IQN tensor shapes")
        if q_values.shape != (1, self.action_dim):
            raise ValueError("network returned invalid IQN action-value shape")


def quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return IQN's pairwise quantile Huber loss."""

    errors = target.unsqueeze(1) - predicted.unsqueeze(2)
    absolute_errors = errors.abs()
    huber = torch.where(
        absolute_errors <= threshold,
        0.5 * errors.square(),
        threshold * (absolute_errors - 0.5 * threshold),
    )
    weights = (taus.unsqueeze(2) - (errors.detach() < 0).float()).abs()
    return (weights * huber / threshold).mean()
