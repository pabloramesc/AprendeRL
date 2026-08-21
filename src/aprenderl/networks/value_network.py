"""State-value network for policy-gradient algorithms."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from aprenderl.networks.mlp import build_mlp


class ValueNetwork(nn.Module):
    """An MLP that estimates one scalar value per observation."""

    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        self.feature_extractor, feature_dim = build_mlp(observation_dim, hidden_sizes)
        self.value_head = nn.Linear(feature_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one state-value estimate for each observation."""

        features = self.feature_extractor(observations.flatten(start_dim=1))
        return self.value_head(features).squeeze(-1)
