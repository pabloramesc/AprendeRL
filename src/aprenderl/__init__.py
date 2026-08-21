"""AprendeRL public API."""

from aprenderl.algorithms import (
    A2C,
    DQN,
    REINFORCE,
    SARSA,
    A2CConfig,
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
    "A2C",
    "A2CConfig",
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
