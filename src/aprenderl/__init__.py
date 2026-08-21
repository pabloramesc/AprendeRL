"""AprendeRL public API."""

from aprenderl.algorithms import (
    DQN,
    REINFORCE,
    SARSA,
    BaseAlgorithm,
    DQNConfig,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
    QLearning,
    QLearningConfig,
    REINFORCEConfig,
    SARSAConfig,
)

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

__version__ = "0.1.0"
