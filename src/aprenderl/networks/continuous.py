"""Actors and action-conditioned critics for off-policy continuous control."""

from collections.abc import Sequence

import torch
from torch import nn

from aprenderl.networks.mlp import build_mlp


class DeterministicPolicyNetwork(nn.Module):
    """Produce normalized actions in [-1, 1]; the algorithm rescales them."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.features, width = build_mlp(observation_dim, hidden_sizes)
        self.head = nn.Linear(width, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(observations.flatten(start_dim=1))).tanh()


class ContinuousQNetwork(nn.Module):
    """Map observations and flattened environment actions to Q(s, a)."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.features, width = build_mlp(observation_dim + action_dim, hidden_sizes)
        self.head = nn.Linear(width, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat(
            (observations.flatten(start_dim=1), actions.flatten(start_dim=1)), dim=-1
        )
        return self.head(self.features(inputs))


class SACPolicyNetwork(nn.Module):
    """Gaussian actor with state-dependent means and log standard deviations."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.features, width = build_mlp(observation_dim, hidden_sizes)
        self.mean_head = nn.Linear(width, action_dim)
        self.log_std_head = nn.Linear(width, action_dim)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observations.flatten(start_dim=1))
        return self.mean_head(features), self.log_std_head(features).clamp(-20, 2)
