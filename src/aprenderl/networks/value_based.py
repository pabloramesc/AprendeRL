"""Network architectures used by the DQN family."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from aprenderl.networks.mlp import build_mlp


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer from NoisyNet."""

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
        """Draw one factorized noise sample for the next forwards."""

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


def reset_noise(module: nn.Module) -> None:
    """Resample every :class:`NoisyLinear` below ``module``."""

    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.reset_noise()


def _build_noisy_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    initial_sigma: float,
) -> tuple[nn.Module, int]:
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


class DuelingQNetwork(nn.Module):
    """Q-network that combines scalar value and centered advantages."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.feature_extractor, feature_dim = build_mlp(
            observation_dim, hidden_sizes
        )
        self.value_head = nn.Linear(feature_dim, 1)
        self.advantage_head = nn.Linear(feature_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(observations.flatten(start_dim=1))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class NoisyQNetwork(nn.Module):
    """MLP Q-network whose trainable layers use factorized noise."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        *,
        initial_sigma: float = 0.5,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.feature_extractor, feature_dim = _build_noisy_mlp(
            observation_dim, hidden_sizes, initial_sigma
        )
        self.q_head = NoisyLinear(
            feature_dim, action_dim, initial_sigma=initial_sigma
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(observations.flatten(start_dim=1))
        return self.q_head(features)

    def reset_noise(self) -> None:
        reset_noise(self)


class CategoricalQNetwork(nn.Module):
    """Categorical action-value distribution with optional Rainbow heads."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        atoms: int = 51,
        hidden_sizes: Sequence[int] = (128, 128),
        *,
        dueling: bool = False,
        noisy: bool = False,
        initial_sigma: float = 0.5,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if atoms <= 1:
            raise ValueError("atoms must be greater than one")
        self.action_dim = action_dim
        self.atoms = atoms
        builder = _build_noisy_mlp if noisy else None
        if builder is None:
            self.feature_extractor, feature_dim = build_mlp(
                observation_dim, hidden_sizes
            )
            linear = nn.Linear
        else:
            self.feature_extractor, feature_dim = builder(
                observation_dim, hidden_sizes, initial_sigma
            )
            linear = lambda inputs, outputs: NoisyLinear(  # noqa: E731
                inputs, outputs, initial_sigma=initial_sigma
            )
        self.dueling = dueling
        if dueling:
            self.value_head = linear(feature_dim, atoms)
            self.advantage_head = linear(feature_dim, action_dim * atoms)
        else:
            self.distribution_head = linear(feature_dim, action_dim * atoms)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(observations.flatten(start_dim=1))
        if self.dueling:
            value = self.value_head(features).view(-1, 1, self.atoms)
            advantage = self.advantage_head(features).view(
                -1, self.action_dim, self.atoms
            )
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.distribution_head(features).view(
            -1, self.action_dim, self.atoms
        )

    def probabilities(self, observations: torch.Tensor) -> torch.Tensor:
        return self(observations).softmax(dim=-1)

    def q_values(
        self, observations: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        return (self.probabilities(observations) * support).sum(dim=-1)

    def reset_noise(self) -> None:
        reset_noise(self)


class QuantileQNetwork(nn.Module):
    """Fixed-quantile action-value network used by QR-DQN."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        quantiles: int = 51,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if quantiles <= 1:
            raise ValueError("quantiles must be greater than one")
        self.action_dim = action_dim
        self.quantiles = quantiles
        self.feature_extractor, feature_dim = build_mlp(
            observation_dim, hidden_sizes
        )
        self.quantile_head = nn.Linear(feature_dim, action_dim * quantiles)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(observations.flatten(start_dim=1))
        return self.quantile_head(features).view(
            -1, self.action_dim, self.quantiles
        )

    def q_values(self, observations: torch.Tensor) -> torch.Tensor:
        return self(observations).mean(dim=-1)


class ImplicitQuantileNetwork(nn.Module):
    """IQN with cosine quantile embeddings and sampled quantile fractions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        *,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.feature_extractor, feature_dim = build_mlp(
            observation_dim, hidden_sizes
        )
        self.quantile_embedding = nn.Linear(embedding_dim, feature_dim)
        self.q_head = nn.Linear(feature_dim, action_dim)

    def forward(
        self,
        observations: torch.Tensor,
        num_quantiles: int,
        taus: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_quantiles <= 0:
            raise ValueError("num_quantiles must be positive")
        batch_size = observations.shape[0]
        features = self.feature_extractor(observations.flatten(start_dim=1))
        if taus is None:
            taus = torch.rand(
                batch_size, num_quantiles, device=observations.device
            )
        if taus.shape != (batch_size, num_quantiles):
            raise ValueError("taus must have shape (batch, num_quantiles)")
        frequencies = torch.arange(
            self.embedding_dim, device=observations.device
        )
        cosine = torch.cos(math.pi * taus.unsqueeze(-1) * frequencies)
        embedding = nn.functional.relu(self.quantile_embedding(cosine))
        combined = features.unsqueeze(1) * embedding
        quantile_values = self.q_head(combined).transpose(1, 2)
        return quantile_values, taus

    def q_values(
        self, observations: torch.Tensor, num_quantiles: int = 32
    ) -> torch.Tensor:
        taus = (
            torch.arange(num_quantiles, device=observations.device) + 0.5
        ) / num_quantiles
        taus = taus.expand(observations.shape[0], -1)
        quantile_values, _ = self(observations, num_quantiles, taus)
        return quantile_values.mean(dim=-1)
