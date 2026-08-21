"""Tests for the default DQN Q-network."""

import torch

from aprenderl.networks import GaussianPolicyNetwork, QNetwork


def test_vanilla_q_network_output_shape() -> None:
    network = QNetwork(observation_dim=4, action_dim=2, hidden_sizes=(16,))

    assert network(torch.randn(7, 4)).shape == (7, 2)


def test_gaussian_policy_network_output_shapes() -> None:
    network = GaussianPolicyNetwork(
        observation_dim=3,
        action_dim=2,
        hidden_sizes=(16,),
        initial_log_std=-0.5,
    )

    means, log_stds = network(torch.randn(7, 3))

    assert means.shape == (7, 2)
    assert log_stds.shape == (7, 2)
    torch.testing.assert_close(log_stds, torch.full((7, 2), -0.5))
    assert network.log_std.requires_grad
