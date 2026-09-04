"""Tabular one-step Q-learning for finite discrete environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from aprenderl.algorithms.base import OffPolicyAlgorithm
from aprenderl.algorithms.tabular import TabularTDConfig, TabularValueMixin
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.types import Transition


@dataclass(frozen=True)
class QLearningConfig(TabularTDConfig):
    """Hyperparameters for :class:`QLearning`."""


class QLearning(TabularValueMixin, OffPolicyAlgorithm[int, int]):
    """One-step tabular Q-learning with epsilon-greedy exploration.

    Time-limit truncations bootstrap from the final observation; true terminal
    transitions do not.
    """

    config_type: ClassVar[type[QLearningConfig]] = QLearningConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: QLearningConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            config=config,
            callback=callback,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Q-learning rule
    # ------------------------------------------------------------------

    def _update_from_transition(
        self, transition: Transition[int, int]
    ) -> dict[str, float | int]:
        return self._update(transition)

    def _update(self, transition: Transition[int, int]) -> dict[str, float | int]:
        """Update toward the largest next-state action value."""

        state = self._state_index(transition.observation)
        action = self._action_index(transition.action)
        next_value = 0.0
        if not transition.terminated:
            next_state = self._state_index(transition.next_observation)
            next_value = float(np.max(self.q_table[next_state]))

        td_target = transition.reward + self.config.gamma * next_value
        td_error = td_target - float(self.q_table[state, action])
        self.q_table[state, action] += self.config.learning_rate * td_error
        self.num_updates += 1
        return {
            "train/td_error": td_error,
            "train/q_value": float(self.q_table[state, action]),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }
