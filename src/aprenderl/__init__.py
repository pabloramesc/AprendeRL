"""AprendeRL public API."""

from aprenderl.algorithms import (
    DQN,
    BaseAlgorithm,
    DQNConfig,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
)

__all__ = [
    "BaseAlgorithm",
    "DQN",
    "DQNConfig",
    "OffPolicyAlgorithm",
    "OnPolicyAlgorithm",
]

__version__ = "0.1.0"
