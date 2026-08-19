"""Shared multilayer-perceptron construction."""

from collections.abc import Sequence

from torch import nn


def build_mlp(input_dim: int, hidden_sizes: Sequence[int]) -> tuple[nn.Sequential, int]:
    """Build ReLU hidden layers and return their output dimension."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_sizes:
        if hidden_dim <= 0:
            raise ValueError("hidden layer sizes must be positive")
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
        previous_dim = hidden_dim
    return nn.Sequential(*layers), previous_dim
