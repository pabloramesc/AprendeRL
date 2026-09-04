"""Network architectures for distributional DQN agents."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch import nn

from aprenderl.networks.mlp import build_mlp
from aprenderl.networks.q_network import NoisyLinear, build_noisy_mlp, reset_noise


class CategoricalQNetwork(nn.Module):
    """Categorical value distribution used by C51 and Rainbow."""

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
        self.dueling = dueling
        if noisy:
            self.feature_extractor, feature_dim = build_noisy_mlp(
                observation_dim, hidden_sizes, initial_sigma
            )

            def linear(inputs: int, outputs: int) -> nn.Module:
                return NoisyLinear(inputs, outputs, initial_sigma=initial_sigma)

        else:
            self.feature_extractor, feature_dim = build_mlp(
                observation_dim, hidden_sizes
            )
            linear = nn.Linear

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
        return self.distribution_head(features).view(-1, self.action_dim, self.atoms)

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
        self.feature_extractor, feature_dim = build_mlp(observation_dim, hidden_sizes)
        self.quantile_head = nn.Linear(feature_dim, action_dim * quantiles)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(observations.flatten(start_dim=1))
        return self.quantile_head(features).view(-1, self.action_dim, self.quantiles)

    def q_values(self, observations: torch.Tensor) -> torch.Tensor:
        return self(observations).mean(dim=-1)


class ImplicitQuantileNetwork(nn.Module):
    """IQN with cosine embeddings of sampled quantile fractions."""

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
        self.feature_extractor, feature_dim = build_mlp(observation_dim, hidden_sizes)
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
            taus = torch.rand(batch_size, num_quantiles, device=observations.device)
        if taus.shape != (batch_size, num_quantiles):
            raise ValueError("taus must have shape (batch, num_quantiles)")
        frequencies = torch.arange(
            1, self.embedding_dim + 1, device=observations.device
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


class FQFNetwork(nn.Module):
    """FQF network with learned quantile fractions and quantile values."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        quantiles: int = 32,
        hidden_sizes: Sequence[int] = (128, 128),
        *,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if quantiles <= 1:
            raise ValueError("quantiles must be greater than one")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.action_dim = action_dim
        self.quantiles_count = quantiles
        self.embedding_dim = embedding_dim
        self.feature_extractor, feature_dim = build_mlp(observation_dim, hidden_sizes)
        self.fraction_net = nn.Linear(feature_dim, quantiles)
        nn.init.xavier_uniform_(self.fraction_net.weight, gain=0.01)
        nn.init.zeros_(self.fraction_net.bias)
        self.quantile_embedding = nn.Linear(embedding_dim, feature_dim)
        self.q_head = nn.Linear(feature_dim, action_dim)

    def state_embeddings(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations once for both FQF sub-networks."""

        return self.feature_extractor(observations.flatten(start_dim=1))

    def fractions(
        self, state_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return boundaries, detached midpoints, and proposal entropy."""

        log_probabilities = self.fraction_net(state_embeddings).log_softmax(dim=1)
        probabilities = log_probabilities.exp()
        zeros = torch.zeros(
            (state_embeddings.shape[0], 1),
            dtype=state_embeddings.dtype,
            device=state_embeddings.device,
        )
        taus = torch.cat((zeros, probabilities.cumsum(dim=1)), dim=1)
        tau_hats = ((taus[:, :-1] + taus[:, 1:]) / 2).detach()
        entropy = -(probabilities * log_probabilities).sum(dim=1)
        return taus, tau_hats, entropy

    def quantile_values(
        self, state_embeddings: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate every action's return quantile at each supplied fraction."""

        frequencies = torch.arange(
            1, self.embedding_dim + 1, device=state_embeddings.device
        )
        cosine = torch.cos(math.pi * taus.unsqueeze(-1) * frequencies)
        embedding = nn.functional.relu(self.quantile_embedding(cosine))
        combined = state_embeddings.unsqueeze(1) * embedding
        return self.q_head(combined).transpose(1, 2)

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_embeddings = self.state_embeddings(observations)
        taus, tau_hats, entropy = self.fractions(state_embeddings.detach())
        values = self.quantile_values(state_embeddings, tau_hats)
        return values, taus, tau_hats, entropy

    def q_values(self, observations: torch.Tensor) -> torch.Tensor:
        values, taus, _, _ = self(observations)
        probabilities = taus[:, 1:] - taus[:, :-1]
        return (values * probabilities.unsqueeze(1)).sum(dim=-1)

    def fraction_parameters(self) -> Iterator[nn.Parameter]:
        return self.fraction_net.parameters()

    def quantile_parameters(self) -> Iterator[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("fraction_net."):
                yield parameter
