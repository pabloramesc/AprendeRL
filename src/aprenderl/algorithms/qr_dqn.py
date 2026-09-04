"""Quantile Regression DQN (QR-DQN)."""

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
from aprenderl.networks import QuantileQNetwork


@dataclass(frozen=True)
class QRDQNConfig(DQNConfig):
    """DQN settings for fixed quantile-regression learning."""

    quantiles: int = 51
    huber_threshold: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantiles <= 1:
            raise ValueError("quantiles must be greater than one")
        if self.huber_threshold <= 0:
            raise ValueError("huber_threshold must be positive")


class QRDQN(DQN):
    """Approximate the return distribution on a uniform quantile grid."""

    config_type = QRDQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: QRDQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        actual_config = config or QRDQNConfig()
        provided_network = network is not None
        if network is None:
            observation_shape, action_dim, _ = _environment_dimensions(env, "QRDQN")
            network = QuantileQNetwork(
                int(np.prod(observation_shape)), action_dim, actual_config.quantiles
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
        with torch.no_grad():
            if self.config.double_dqn:
                next_values = self._q_values(self.q_network, batch.next_observations)
            else:
                next_values = self._q_values(
                    self.target_network, batch.next_observations
                )
            next_actions = next_values.argmax(dim=1)
            target_quantiles = self.target_network(batch.next_observations)
            rows = torch.arange(batch.rewards.shape[0], device=self.device)
            target_quantiles = target_quantiles[rows, next_actions]
            target_quantiles = (
                batch.rewards
                + self.config.gamma * (1 - batch.terminated) * target_quantiles
            )

        predicted_quantiles = self.q_network(batch.observations)
        action_indices = batch.actions.unsqueeze(-1).expand(
            -1, 1, self.config.quantiles
        )
        predicted_quantiles = predicted_quantiles.gather(1, action_indices).squeeze(1)
        taus = (
            torch.arange(self.config.quantiles, device=self.device) + 0.5
        ) / self.config.quantiles
        loss = quantile_huber_loss(
            predicted_quantiles,
            target_quantiles,
            taus,
            self.config.huber_threshold,
        )
        return self._optimize(loss)

    def _q_values(self, network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        return network(observations).mean(dim=-1)

    def _validate_network(self, network: nn.Module) -> None:
        try:
            with torch.no_grad():
                output = network(
                    torch.zeros((1, *self.observation_shape), device=self.device)
                )
        except Exception as error:
            raise ValueError(
                "network could not process one environment observation"
            ) from error
        expected = (1, self.action_dim, self.config.quantiles)
        if not isinstance(output, torch.Tensor) or output.shape != expected:
            actual = getattr(output, "shape", None)
            raise ValueError(f"network must return shape {expected}, got {actual}")


def quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return QR-DQN's pairwise quantile Huber loss."""

    errors = target.unsqueeze(1) - predicted.unsqueeze(2)
    absolute_errors = errors.abs()
    huber = torch.where(
        absolute_errors <= threshold,
        0.5 * errors.square(),
        threshold * (absolute_errors - 0.5 * threshold),
    )
    if taus.ndim == 1:
        taus = taus.unsqueeze(0).expand(predicted.shape[0], -1)
    weights = (taus.unsqueeze(2) - (errors.detach() < 0).float()).abs()
    return (weights * huber / threshold).mean()
