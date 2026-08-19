"""Tests for the default DQN Q-network."""

import torch

from aprenderl.networks import QNetwork


def test_vanilla_q_network_output_shape() -> None:
    network = QNetwork(observation_dim=4, action_dim=2, hidden_sizes=(16,))

    assert network(torch.randn(7, 4)).shape == (7, 2)
