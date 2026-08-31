"""Learning algorithms and their common interfaces."""

from aprenderl.algorithms.a2c import A2C, A2CConfig
from aprenderl.algorithms.actor_critic import ActorCritic, ActorCriticConfig
from aprenderl.algorithms.bandits import MultiArmedBandit, MultiArmedBanditConfig
from aprenderl.algorithms.base import (
    BaseAlgorithm,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
)
from aprenderl.algorithms.distributional_dqn import (
    C51,
    IQN,
    QRDQN,
    C51Config,
    IQNConfig,
    QRDQNConfig,
    RainbowDQN,
    RainbowDQNConfig,
)
from aprenderl.algorithms.dqn import DQN, DQNConfig
from aprenderl.algorithms.dqn_variants import (
    DoubleDQN,
    DoubleDQNConfig,
    DuelingDQN,
    DuelingDQNConfig,
    NoisyDQN,
    NoisyDQNConfig,
    NStepDQN,
    NStepDQNConfig,
    PrioritizedDQN,
    PrioritizedDQNConfig,
)
from aprenderl.algorithms.dynamic_programming import (
    DynamicProgrammingConfig,
    PolicyIteration,
    ValueIteration,
)
from aprenderl.algorithms.monte_carlo import (
    MonteCarloControl,
    MonteCarloControlConfig,
    MonteCarloPrediction,
    MonteCarloPredictionConfig,
)
from aprenderl.algorithms.q_learning import QLearning, QLearningConfig
from aprenderl.algorithms.reinforce import REINFORCE, REINFORCEConfig
from aprenderl.algorithms.sarsa import SARSA, SARSAConfig
from aprenderl.algorithms.td_control import (
    DynaQ,
    DynaQConfig,
    ExpectedSARSA,
    ExpectedSARSAConfig,
    NStepSARSA,
    NStepSARSAConfig,
    SARSALambda,
    SARSALambdaConfig,
)

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
    "DoubleDQN",
    "DoubleDQNConfig",
    "DuelingDQN",
    "DuelingDQNConfig",
    "DynaQ",
    "DynaQConfig",
    "DynamicProgrammingConfig",
    "ExpectedSARSA",
    "ExpectedSARSAConfig",
    "IQN",
    "IQNConfig",
    "MonteCarloControl",
    "MonteCarloControlConfig",
    "MonteCarloPrediction",
    "MonteCarloPredictionConfig",
    "MultiArmedBandit",
    "MultiArmedBanditConfig",
    "NStepSARSA",
    "NStepSARSAConfig",
    "NStepDQN",
    "NStepDQNConfig",
    "NoisyDQN",
    "NoisyDQNConfig",
    "OffPolicyAlgorithm",
    "OnPolicyAlgorithm",
    "QLearning",
    "QLearningConfig",
    "REINFORCE",
    "REINFORCEConfig",
    "SARSA",
    "SARSAConfig",
    "SARSALambda",
    "SARSALambdaConfig",
    "PolicyIteration",
    "PrioritizedDQN",
    "PrioritizedDQNConfig",
    "QRDQN",
    "QRDQNConfig",
    "RainbowDQN",
    "RainbowDQNConfig",
    "ValueIteration",
]
