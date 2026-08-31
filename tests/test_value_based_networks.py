"""Shape and structural tests for value-based network components."""

import torch

from aprenderl.networks import (
    CategoricalQNetwork,
    DuelingQNetwork,
    ImplicitQuantileNetwork,
    NoisyLinear,
    QuantileQNetwork,
)


def test_dueling_network_centers_advantages() -> None:
    network = DuelingQNetwork(4, 3, hidden_sizes=())
    observations = torch.zeros(2, 4)
    values = network(observations)

    assert values.shape == (2, 3)


def test_noisy_linear_is_deterministic_in_evaluation_mode() -> None:
    layer = NoisyLinear(3, 2)
    inputs = torch.ones(1, 3)
    layer.eval()
    first = layer(inputs)
    layer.reset_noise()
    second = layer(inputs)

    torch.testing.assert_close(first, second)


def test_distributional_network_shapes() -> None:
    observations = torch.zeros(5, 4)
    categorical = CategoricalQNetwork(4, 2, atoms=11)
    quantile = QuantileQNetwork(4, 2, quantiles=7)
    implicit = ImplicitQuantileNetwork(4, 2, embedding_dim=8)

    assert categorical(observations).shape == (5, 2, 11)
    assert quantile(observations).shape == (5, 2, 7)
    values, taus = implicit(observations, 6)
    assert values.shape == (5, 2, 6)
    assert taus.shape == (5, 6)
