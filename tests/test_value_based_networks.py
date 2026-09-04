"""Shape and structural tests for value-based network components."""

import torch

from aprenderl.networks import (
    CategoricalQNetwork,
    FQFNetwork,
    ImplicitQuantileNetwork,
    QuantileQNetwork,
)


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


def test_rainbow_categorical_network_combines_dueling_and_noisy_layers() -> None:
    network = CategoricalQNetwork(4, 2, atoms=11, dueling=True, noisy=True)
    observations = torch.zeros(3, 4)
    assert network(observations).shape == (3, 2, 11)

    network.eval()
    first = network(observations)
    network.reset_noise()
    second = network(observations)
    torch.testing.assert_close(first, second)


def test_fqf_network_learns_valid_fraction_partitions() -> None:
    network = FQFNetwork(4, 2, quantiles=8, embedding_dim=16)
    values, taus, tau_hats, entropy = network(torch.zeros(3, 4))

    assert values.shape == (3, 2, 8)
    assert taus.shape == (3, 9)
    assert tau_hats.shape == (3, 8)
    assert entropy.shape == (3,)
    torch.testing.assert_close(taus[:, 0], torch.zeros(3))
    torch.testing.assert_close(taus[:, -1], torch.ones(3))
    assert torch.all(taus[:, 1:] >= taus[:, :-1])
