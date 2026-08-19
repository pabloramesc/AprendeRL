"""Vanilla Q-network and architecture factory."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from aprenderl.networks.mlp import build_mlp


class QNetwork(nn.Module):
    """A vanilla MLP with one output per discrete action."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.feature_extractor, feature_dim = build_mlp(observation_dim, hidden_sizes)
        self.q_head = nn.Linear(feature_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one Q-value per action for each observation."""

        features = self.feature_extractor(observations.flatten(start_dim=1))
        return self.q_head(features)
