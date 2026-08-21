"""Diagonal-Gaussian policy network for continuous actions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from aprenderl.networks.mlp import build_mlp


class GaussianPolicyNetwork(nn.Module):
    """An MLP mean head with one learned log standard deviation per action."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        initial_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.feature_extractor, feature_dim = build_mlp(observation_dim, hidden_sizes)
        self.mean_head = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian means and expanded log standard deviations."""

        features = self.feature_extractor(observations.flatten(start_dim=1))
        means = self.mean_head(features)
        return means, self.log_std.expand_as(means)
