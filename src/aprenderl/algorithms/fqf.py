"""Fully Parameterized Quantile Function (FQF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from torch import nn

from aprenderl.algorithms.dqn import DQN, DQNConfig
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import FQFNetwork


@dataclass(frozen=True)
class FQFConfig(DQNConfig):
    """DQN settings for learned quantile fractions and values."""

    quantiles: int = 32
    embedding_dim: int = 64
    huber_threshold: float = 1.0
    fraction_learning_rate: float = 2.5e-9
    entropy_coefficient: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantiles <= 1:
            raise ValueError("quantiles must be greater than one")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.huber_threshold <= 0:
            raise ValueError("huber_threshold must be positive")
        if self.fraction_learning_rate <= 0:
            raise ValueError("fraction_learning_rate must be positive")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient cannot be negative")


class FQF(DQN):
    """Learn both the quantile fractions and their return values."""

    config_type = FQFConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: FQFConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            network,
            config=config or FQFConfig(),
            device=device,
            callback=callback,
            logger=logger,
        )
        self.optimizer = torch.optim.Adam(
            self.q_network.quantile_parameters(),
            lr=self.config.learning_rate,
        )
        self.fraction_optimizer = torch.optim.RMSprop(
            self.q_network.fraction_parameters(),
            lr=self.config.fraction_learning_rate,
            alpha=0.95,
            eps=1e-5,
        )

    def _make_default_network(self) -> nn.Module:
        return FQFNetwork(
            self.observation_dim,
            self.action_dim,
            self.config.quantiles,
            embedding_dim=self.config.embedding_dim,
        )

    # ------------------------------------------------------------------
    # FQF learning rule
    # ------------------------------------------------------------------

    def _train_step(self) -> dict[str, float | int]:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        state_embeddings = self.q_network.state_embeddings(batch.observations)
        taus, tau_hats, entropy = self.q_network.fractions(state_embeddings.detach())
        quantile_values = self.q_network.quantile_values(state_embeddings, tau_hats)
        action_indices = batch.actions.unsqueeze(-1).expand(
            -1, 1, self.config.quantiles
        )
        predicted = quantile_values.gather(1, action_indices).squeeze(1)

        with torch.no_grad():
            boundary_values = self.q_network.quantile_values(
                state_embeddings.detach(), taus[:, 1:-1]
            )
            boundary_indices = batch.actions.unsqueeze(-1).expand(
                -1, 1, self.config.quantiles - 1
            )
            boundary_values = boundary_values.gather(1, boundary_indices).squeeze(1)
        fraction_loss = fqf_fraction_loss(taus, boundary_values, predicted.detach())
        fraction_objective = (
            fraction_loss - self.config.entropy_coefficient * entropy.mean()
        )
        self.fraction_optimizer.zero_grad(set_to_none=True)
        fraction_objective.backward()
        fraction_gradient_norm = nn.utils.clip_grad_norm_(
            self.q_network.fraction_parameters(), self.config.max_grad_norm
        )
        self.fraction_optimizer.step()

        with torch.no_grad():
            action_network = (
                self.q_network if self.config.double_dqn else self.target_network
            )
            next_actions = self._q_values(
                action_network, batch.next_observations
            ).argmax(dim=1)
            next_embeddings = self.target_network.state_embeddings(
                batch.next_observations
            )
            target_values = self.target_network.quantile_values(
                next_embeddings, tau_hats
            )
            rows = torch.arange(batch.rewards.shape[0], device=self.device)
            target_values = target_values[rows, next_actions]
            target_values = (
                batch.rewards
                + self.config.gamma * (1 - batch.terminated) * target_values
            )

        quantile_loss = quantile_huber_loss(
            predicted,
            target_values,
            tau_hats,
            self.config.huber_threshold,
        )
        self.optimizer.zero_grad(set_to_none=True)
        quantile_loss.backward()
        quantile_gradient_norm = nn.utils.clip_grad_norm_(
            self.q_network.quantile_parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.num_updates += 1
        return {
            "train/loss": float(quantile_loss.item()),
            "train/fraction_loss": float(fraction_loss.item()),
            "train/fraction_entropy": float(entropy.mean().item()),
            "train/gradient_norm": float(quantile_gradient_norm),
            "train/fraction_gradient_norm": float(fraction_gradient_norm),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }

    def _q_values(self, network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        try:
            return network.q_values(observations)
        except AttributeError as error:
            raise TypeError("an FQF network must implement q_values()") from error

    def _validate_network(self, network: nn.Module) -> None:
        observations = torch.zeros((1, *self.observation_shape), device=self.device)
        try:
            with torch.no_grad():
                values, taus, tau_hats, entropy = network(observations)
                q_values = network.q_values(observations)
                list(network.fraction_parameters())
                list(network.quantile_parameters())
        except Exception as error:
            raise ValueError("network must implement the FQF network API") from error
        expected_values = (1, self.action_dim, self.config.quantiles)
        if values.shape != expected_values:
            raise ValueError("network returned invalid FQF quantile-value shape")
        if taus.shape != (1, self.config.quantiles + 1):
            raise ValueError("network returned invalid FQF fraction-boundary shape")
        if tau_hats.shape != (1, self.config.quantiles):
            raise ValueError("network returned invalid FQF fraction-midpoint shape")
        if entropy.shape != (1,) or q_values.shape != (1, self.action_dim):
            raise ValueError("network returned invalid FQF summary shapes")

    def _checkpoint_state(self) -> dict[str, Any]:
        return {"fraction_optimizer": self.fraction_optimizer.state_dict()}

    def _restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if "fraction_optimizer" in state:
            self.fraction_optimizer.load_state_dict(state["fraction_optimizer"])


def quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return FQF's pairwise quantile Huber loss."""

    errors = target.unsqueeze(1) - predicted.unsqueeze(2)
    absolute_errors = errors.abs()
    huber = torch.where(
        absolute_errors <= threshold,
        0.5 * errors.square(),
        threshold * (absolute_errors - 0.5 * threshold),
    )
    weights = (taus.unsqueeze(2) - (errors.detach() < 0).float()).abs()
    return (weights * huber / threshold).mean()


def fqf_fraction_loss(
    taus: torch.Tensor,
    boundary_values: torch.Tensor,
    midpoint_values: torch.Tensor,
) -> torch.Tensor:
    """Optimize FQF's internal fraction boundaries using Proposition 1."""

    if boundary_values.shape != midpoint_values[:, :-1].shape:
        raise ValueError("boundary and midpoint quantiles have incompatible shapes")
    gradients = (
        2 * boundary_values - midpoint_values[:, :-1] - midpoint_values[:, 1:]
    )
    return (gradients.detach() * taus[:, 1:-1]).sum(dim=1).mean()
