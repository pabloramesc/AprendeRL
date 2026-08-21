"""Learning algorithms and their common interfaces."""

from aprenderl.algorithms.base import (
    BaseAlgorithm,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
)
from aprenderl.algorithms.dqn import DQN, DQNConfig
from aprenderl.algorithms.q_learning import QLearning, QLearningConfig
from aprenderl.algorithms.reinforce import REINFORCE, REINFORCEConfig
from aprenderl.algorithms.sarsa import SARSA, SARSAConfig

__all__ = [
    "BaseAlgorithm",
    "DQN",
    "DQNConfig",
    "OffPolicyAlgorithm",
    "OnPolicyAlgorithm",
    "QLearning",
    "QLearningConfig",
    "REINFORCE",
    "REINFORCEConfig",
    "SARSA",
    "SARSAConfig",
]
