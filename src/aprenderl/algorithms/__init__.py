"""Learning algorithms and their common interfaces."""

from aprenderl.algorithms.base import (
    BaseAlgorithm,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
)
from aprenderl.algorithms.dqn import DoubleDQN, DoubleDQNConfig

__all__ = [
    "BaseAlgorithm",
    "DoubleDQN",
    "DoubleDQNConfig",
    "OffPolicyAlgorithm",
    "OnPolicyAlgorithm",
]
