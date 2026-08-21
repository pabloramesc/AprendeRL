"""AprendeRL public API."""

from aprenderl.algorithms import (
    DQN,
    REINFORCE,
    SARSA,
    ActorCritic,
    ActorCriticConfig,
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
    "ActorCritic",
    "ActorCriticConfig",
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
