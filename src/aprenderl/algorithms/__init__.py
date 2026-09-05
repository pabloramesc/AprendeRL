"""Learning algorithms and their common interfaces."""

from aprenderl.algorithms.a2c import A2C, A2CConfig
from aprenderl.algorithms.actor_critic import ActorCritic, ActorCriticConfig
from aprenderl.algorithms.base import (
    BaseAlgorithm,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
)
from aprenderl.algorithms.c51 import C51, C51Config
from aprenderl.algorithms.dqn import DQN, DQNConfig
from aprenderl.algorithms.fqf import FQF, FQFConfig
from aprenderl.algorithms.iqn import IQN, IQNConfig
from aprenderl.algorithms.policy_gradient import PolicyGradientAlgorithm
from aprenderl.algorithms.ppo import PPO, PPOConfig
from aprenderl.algorithms.q_learning import QLearning, QLearningConfig
from aprenderl.algorithms.qr_dqn import QRDQN, QRDQNConfig
from aprenderl.algorithms.rainbow import RainbowDQN, RainbowDQNConfig
from aprenderl.algorithms.reinforce import REINFORCE, REINFORCEConfig
from aprenderl.algorithms.sarsa import SARSA, SARSAConfig
from aprenderl.algorithms.trpo import TRPO, TRPOConfig

__all__ = [
    "A2C",
    "A2CConfig",
    "ActorCritic",
    "ActorCriticConfig",
    "BaseAlgorithm",
    "C51",
    "C51Config",
    "DQN",
    "DQNConfig",
    "FQF",
    "FQFConfig",
    "IQN",
    "IQNConfig",
    "OffPolicyAlgorithm",
    "OnPolicyAlgorithm",
    "PolicyGradientAlgorithm",
    "PPO",
    "PPOConfig",
    "QLearning",
    "QLearningConfig",
    "REINFORCE",
    "REINFORCEConfig",
    "SARSA",
    "SARSAConfig",
    "TRPO",
    "TRPOConfig",
    "QRDQN",
    "QRDQNConfig",
    "RainbowDQN",
    "RainbowDQNConfig",
]
