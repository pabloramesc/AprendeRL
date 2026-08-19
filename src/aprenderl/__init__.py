"""AprendeRL public API."""

from aprenderl.algorithms import (
    BaseAlgorithm,
    DoubleDQN,
    DoubleDQNConfig,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
)

__all__ = [
    "BaseAlgorithm",
    "DoubleDQN",
    "DoubleDQNConfig",
    "OffPolicyAlgorithm",
    "OnPolicyAlgorithm",
]

__version__ = "0.1.0"
