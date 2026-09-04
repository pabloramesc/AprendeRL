"""Tabular one-step SARSA for finite discrete environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym

from aprenderl.algorithms.base import OnPolicyAlgorithm
from aprenderl.algorithms.tabular import TabularTDConfig, TabularValueMixin
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.types import Transition


@dataclass(frozen=True)
class SARSAConfig(TabularTDConfig):
    """Hyperparameters for :class:`SARSA`."""


class SARSA(TabularValueMixin, OnPolicyAlgorithm[int, int]):
    """One-step tabular SARSA with an epsilon-greedy behavior policy.

    The action used in the TD target is retained for the next environment
    interaction. Truncations bootstrap; true terminations do not.
    """

    config_type: ClassVar[type[SARSAConfig]] = SARSAConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: SARSAConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self._action: int | None = None
        super().__init__(
            env,
            config=config,
            callback=callback,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # SARSA rule
    # ------------------------------------------------------------------

    def _sample_action(self, observation: int) -> int:
        if self._action is None:
            return self.predict(observation, deterministic=False)
        return self._action

    def _update_from_transition(
        self, transition: Transition[int, int]
    ) -> dict[str, float | int]:
        next_action = None
        if not transition.terminated:
            next_action = self._select_action(
                transition.next_observation,
                deterministic=False,
                step=self.num_timesteps + 1,
            )

        metrics = self._update(transition, next_action)
        self._action = None if transition.done else next_action
        return metrics

    def _update(
        self,
        transition: Transition[int, int],
        next_action: int | None,
    ) -> dict[str, float | int]:
        """Update toward the value of the selected next behavior action."""

        state = self._state_index(transition.observation)
        action = self._action_index(transition.action)
        next_value = 0.0
        if not transition.terminated:
            if next_action is None:
                raise ValueError("next_action is required for non-terminal transitions")
            next_state = self._state_index(transition.next_observation)
            next_value = float(
                self.q_table[next_state, self._action_index(next_action)]
            )

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
