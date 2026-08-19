"""Learning algorithms and their common interfaces."""

from aprenderl.algorithms.base import (
    BaseAlgorithm,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
)
from aprenderl.algorithms.dqn import DQN, DQNConfig

__all__ = [
    "BaseAlgorithm",
    "DQN",
    "DQNConfig",
    "OffPolicyAlgorithm",
    "OnPolicyAlgorithm",
]
