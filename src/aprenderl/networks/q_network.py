"""Networks and layers shared by the DQN family."""

from __future__ import annotations

import math
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


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy layer used by Rainbow."""

    def __init__(
        self, in_features: int, out_features: int, *, initial_sigma: float = 0.5
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if initial_sigma <= 0:
            raise ValueError("initial_sigma must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, initial_sigma / math.sqrt(in_features))
        nn.init.constant_(self.bias_sigma, initial_sigma / math.sqrt(out_features))
        self.reset_noise()

    def reset_noise(self) -> None:
        """Draw one factorized noise sample for subsequent forward passes."""

        input_noise = self._scaled_noise(self.in_features, self.weight_mu.device)
        output_noise = self._scaled_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(output_noise.outer(input_noise))
        self.bias_epsilon.copy_(output_noise)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(inputs, weight, bias)

    @staticmethod
    def _scaled_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()


def build_noisy_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    initial_sigma: float,
) -> tuple[nn.Module, int]:
    """Build the noisy feature extractor used by Rainbow."""

    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        if size <= 0:
            raise ValueError("hidden sizes must be positive")
        layers.extend(
            [NoisyLinear(previous, size, initial_sigma=initial_sigma), nn.ReLU()]
        )
        previous = size
    return nn.Sequential(*layers), previous


def reset_noise(module: nn.Module) -> None:
    """Resample every :class:`NoisyLinear` below ``module``."""

    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.reset_noise()
