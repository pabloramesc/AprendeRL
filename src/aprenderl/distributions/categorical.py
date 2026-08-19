"""Categorical actions for discrete policies."""

from __future__ import annotations

import torch
from torch.distributions import Categorical


class CategoricalDistribution:
    """Small adapter around :class:`torch.distributions.Categorical`."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.distribution = Categorical(logits=logits)

    def sample(self) -> torch.Tensor:
        """Draw actions from the policy."""

        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        """Return the most likely action."""

        return self.distribution.logits.argmax(dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return action log probabilities."""

        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        """Return categorical entropy."""

        return self.distribution.entropy()
