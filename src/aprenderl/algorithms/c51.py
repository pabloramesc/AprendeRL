"""Categorical DQN (C51)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch
from torch import nn

from aprenderl.algorithms.dqn import DQN, DQNConfig
from aprenderl.buffers import ReplayBatch
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import CategoricalQNetwork


@dataclass(frozen=True)
class C51Config(DQNConfig):
    """DQN settings plus the fixed categorical value support."""

    atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.atoms <= 1:
            raise ValueError("atoms must be greater than one")
        if self.v_min >= self.v_max:
            raise ValueError("v_min must be less than v_max")


class C51(DQN):
    """Learn a categorical return distribution on 51 fixed atoms."""

    config_type = C51Config

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: C51Config | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            network,
            config=config or C51Config(),
            device=device,
            callback=callback,
            logger=logger,
        )
        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atoms,
            device=self.device,
        )

    def _make_default_network(self) -> nn.Module:
        return CategoricalQNetwork(
            self.observation_dim, self.action_dim, self.config.atoms
        )

    # ------------------------------------------------------------------
    # C51 learning rule
    # ------------------------------------------------------------------

    def _train_step(self) -> dict[str, float | int]:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        target_distribution = self._target_distribution(batch)
        logits = self.q_network(batch.observations)
        action_indices = batch.actions.unsqueeze(-1).expand(-1, 1, self.config.atoms)
        chosen_logits = logits.gather(1, action_indices).squeeze(1)
        loss = (
            -(target_distribution * chosen_logits.log_softmax(dim=-1))
            .sum(dim=-1)
            .mean()
        )
        return self._optimize(loss)

    @torch.no_grad()
    def _target_distribution(self, batch: ReplayBatch) -> torch.Tensor:
        next_probabilities = self._probabilities(
            self.target_network, batch.next_observations
        )
        if self.config.double_dqn:
            next_values = self._q_values(self.q_network, batch.next_observations)
        else:
            next_values = (next_probabilities * self.support).sum(dim=-1)
        next_actions = next_values.argmax(dim=1)
        rows = torch.arange(batch.rewards.shape[0], device=self.device)
        chosen_probabilities = next_probabilities[rows, next_actions]
        target_atoms = batch.rewards + self.config.gamma * (
            1 - batch.terminated
        ) * self.support.unsqueeze(0)
        return categorical_projection(
            target_atoms,
            chosen_probabilities,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
        )

    def _q_values(self, network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        probabilities = self._probabilities(network, observations)
        return (probabilities * self.support).sum(dim=-1)

    @staticmethod
    def _probabilities(network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        return network(observations).softmax(dim=-1)

    def _validate_network(self, network: nn.Module) -> None:
        sample = torch.zeros((1, *self.observation_shape), device=self.device)
        try:
            with torch.no_grad():
                output = network(sample)
        except Exception as error:
            raise ValueError(
                "network could not process one environment observation"
            ) from error
        expected_shape = (1, self.action_dim, self.config.atoms)
        actual_shape = getattr(output, "shape", None)
        if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
            raise ValueError(
                f"network must return shape {expected_shape}, got {actual_shape}"
            )


def categorical_projection(
    target_atoms: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    """Project categorical Bellman targets onto their fixed support."""

    atoms = probabilities.shape[1]
    clipped_atoms = target_atoms.clamp(v_min, v_max)
    atom_width = (v_max - v_min) / (atoms - 1)
    locations = (clipped_atoms - v_min) / atom_width
    lower = locations.floor().long()
    upper = locations.ceil().long()
    same_bin = lower == upper

    projection = torch.zeros_like(probabilities)
    offset = (
        torch.arange(target_atoms.shape[0], device=target_atoms.device) * atoms
    ).unsqueeze(1)
    lower_mass = probabilities * (
        upper.to(locations.dtype) - locations + same_bin.to(locations.dtype)
    )
    upper_mass = probabilities * (locations - lower.to(locations.dtype))
    projection.view(-1).index_add_(
        0, (lower + offset).view(-1), lower_mass.view(-1)
    )
    projection.view(-1).index_add_(
        0, (upper + offset).view(-1), upper_mass.view(-1)
    )
    return projection
