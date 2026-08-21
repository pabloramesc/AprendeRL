"""Plain and bounded diagonal-Gaussian continuous-action policies."""

from __future__ import annotations

import torch
from torch.distributions import Normal


class DiagonalGaussianDistribution:
    """An unsquashed Gaussian with independent action dimensions."""

    def __init__(self, means: torch.Tensor, log_stds: torch.Tensor) -> None:
        if means.shape != log_stds.shape:
            raise ValueError("means and log_stds must have the same shape")
        if means.ndim != 2:
            raise ValueError("means and log_stds must have shape (batch, actions)")

        self.means = means
        self.log_stds = log_stds.clamp(-20.0, 2.0)
        self.distribution = Normal(self.means, self.log_stds.exp())

    def sample(self) -> torch.Tensor:
        """Draw actions from the policy."""

        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        """Return the Gaussian mean."""

        return self.means

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return log probabilities, summed over action dimensions."""

        if actions.shape != self.means.shape:
            raise ValueError("actions must have shape (batch, actions)")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Return Gaussian entropy, summed over action dimensions."""

        return self.distribution.entropy().sum(dim=-1)


class SquashedGaussianDistribution(DiagonalGaussianDistribution):
    """A diagonal Gaussian transformed into finite action bounds.

    Samples are passed through ``tanh`` and then affinely rescaled from
    ``[-1, 1]`` to the environment bounds. Log probabilities include both
    transformations, while entropy uses the simple pre-squash Gaussian value.
    """

    def __init__(
        self,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> None:
        super().__init__(means, log_stds)
        if low.shape != (means.shape[-1],) or high.shape != low.shape:
            raise ValueError("action bounds must match the final action dimension")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("action bounds must be finite")
        if not torch.all(high > low):
            raise ValueError("every upper action bound must exceed its lower bound")

        self.low = low.to(device=means.device, dtype=means.dtype)
        self.high = high.to(device=means.device, dtype=means.dtype)
        self.scale = (self.high - self.low) / 2.0
        self.bias = (self.high + self.low) / 2.0
    def sample(self) -> torch.Tensor:
        """Draw bounded actions from the policy."""

        return self._squash(self.distribution.sample())

    def mode(self) -> torch.Tensor:
        """Return the bounded action produced by the Gaussian mean."""

        return self._squash(self.means)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return transformed log probabilities, summed over action dimensions."""

        if actions.shape != self.means.shape:
            raise ValueError("actions must have shape (batch, actions)")
        normalized_actions = (actions - self.bias) / self.scale
        normalized_actions = normalized_actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_actions = torch.atanh(normalized_actions)
        log_probabilities = self.distribution.log_prob(raw_actions)
        log_probabilities -= torch.log(self.scale)
        log_probabilities -= torch.log(1.0 - normalized_actions.square() + 1e-6)
        return log_probabilities.sum(dim=-1)

    def _squash(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return self.bias + self.scale * torch.tanh(raw_actions)
