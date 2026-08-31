"""Tabular one-step SARSA for finite discrete environments."""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from typing_extensions import Self

from aprenderl.algorithms.base import OnPolicyAlgorithm
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.policies import EpsilonGreedyPolicy, LinearSchedule
from aprenderl.types import Transition


@dataclass(frozen=True)
class SARSAConfig:
    """Hyperparameters for :class:`SARSA`."""

    learning_rate: float = 0.1
    gamma: float = 0.99
    exploration_initial_epsilon: float = 1.0
    exploration_final_epsilon: float = 0.05
    exploration_steps: int = 10_000
    initial_q_value: float = 0.0
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= self.exploration_final_epsilon <= 1:
            raise ValueError("exploration_final_epsilon must be in [0, 1]")
        if not (
            self.exploration_final_epsilon <= self.exploration_initial_epsilon <= 1
        ):
            raise ValueError(
                "exploration_initial_epsilon must be between final epsilon and 1"
            )
        if self.exploration_steps <= 0:
            raise ValueError("exploration_steps must be positive")
        if not np.isfinite(self.initial_q_value):
            raise ValueError("initial_q_value must be finite")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


class SARSA(OnPolicyAlgorithm[int, int]):
    """One-step tabular SARSA with an epsilon-greedy behavior policy.

    SARSA is on-policy: the action sampled for the temporal-difference target
    is retained and used for the next environment interaction. Time-limit
    truncations bootstrap from their final observation; true terminals do not.
    """

    config_type = SARSAConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: SARSAConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config or SARSAConfig()
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("SARSA requires a Discrete observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("SARSA requires a Discrete action space")

        self.observation_dim = int(env.observation_space.n)
        self.action_dim = int(env.action_space.n)
        self.observation_start = int(env.observation_space.start)
        self.action_start = int(env.action_space.start)
        self.q_table = np.full(
            (self.observation_dim, self.action_dim),
            self.config.initial_q_value,
            dtype=np.float32,
        )
        self.exploration = EpsilonGreedyPolicy(
            LinearSchedule(
                self.config.exploration_initial_epsilon,
                self.config.exploration_final_epsilon,
                self.config.exploration_steps,
            ),
            seed=self.config.seed,
        )
        self._action: int | None = None
        super().__init__(
            env,
            seed=self.config.seed,
            callback=callback,
            logger=logger,
            log_interval=self.config.log_interval,
        )

    @property
    def epsilon(self) -> float:
        """Return the current epsilon used for behavior actions."""

        return self.exploration.schedule.value(self.num_timesteps)

    def predict(self, observation: int, *, deterministic: bool = False) -> int:
        """Choose a greedy action, or an epsilon-greedy action when requested."""

        return self._select_action(
            observation,
            deterministic=deterministic,
            step=self.num_timesteps,
        )

    def save(self, path: str | Path) -> None:
        """Save the Q-table, configuration, counters, and exploration state."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": 1,
            "config": asdict(self.config),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "observation_start": self.observation_start,
            "action_start": self.action_start,
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,
            "exploration_rng_state": self.exploration.rng_state,
        }
        with checkpoint_path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                q_table=self.q_table,
                episode_returns=np.asarray(self.episode_returns, dtype=np.float64),
                episode_lengths=np.asarray(self.episode_lengths, dtype=np.int64),
                metadata=np.asarray(json.dumps(metadata)),
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env[Any, Any],
        **kwargs: Any,
    ) -> Self:
        """Restore a checkpoint and validate it against ``env``."""

        callback = kwargs.pop("callback", None)
        logger = kwargs.pop("logger", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {names}")

        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            q_table = checkpoint["q_table"].astype(np.float32, copy=True)
            episode_returns = checkpoint["episode_returns"].tolist()
            episode_lengths = checkpoint["episode_lengths"].tolist()

        algorithm = cls(
            env,
            config=cls.config_type(**metadata["config"]),
            callback=callback,
            logger=logger,
        )
        expected_space = (
            algorithm.observation_dim,
            algorithm.action_dim,
            algorithm.observation_start,
            algorithm.action_start,
        )
        checkpoint_space = (
            int(metadata["observation_dim"]),
            int(metadata["action_dim"]),
            int(metadata["observation_start"]),
            int(metadata["action_start"]),
        )
        if checkpoint_space != expected_space or q_table.shape != (
            algorithm.observation_dim,
            algorithm.action_dim,
        ):
            raise ValueError("checkpoint spaces do not match environment")

        algorithm.q_table[...] = q_table
        algorithm.num_timesteps = int(metadata["num_timesteps"])
        algorithm.num_updates = int(metadata["num_updates"])
        algorithm.episode_returns = [float(value) for value in episode_returns]
        algorithm.episode_lengths = [int(value) for value in episode_lengths]
        rng_state = metadata.get("exploration_rng_state", metadata.get("rng_state"))
        if rng_state is not None:
            algorithm.exploration.rng_state = rng_state
        return algorithm

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
        target = transition.reward + self.config.gamma * next_value
        td_error = target - float(self.q_table[state, action])
        self.q_table[state, action] += self.config.learning_rate * td_error
        self.num_updates += 1
        return {
            "train/td_error": td_error,
            "train/q_value": float(self.q_table[state, action]),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }

    def _select_action(
        self,
        observation: int,
        *,
        deterministic: bool,
        step: int,
    ) -> int:
        state = self._state_index(observation)
        action = self.exploration.select(
            self.q_table[state],
            step=step,
            deterministic=deterministic,
        )
        return action + self.action_start

    def _progress_metrics(
        self,
        training_metrics: Mapping[str, float | int] | None = None,
    ) -> dict[str, str | int]:
        metrics = super()._progress_metrics(training_metrics)
        metrics["epsilon"] = f"{self.epsilon:.3f}"
        if training_metrics:
            metrics["td_error"] = f"{float(training_metrics['train/td_error']):.3f}"
        return metrics

    def _state_index(self, observation: Any) -> int:
        try:
            state = operator.index(observation) - self.observation_start
        except TypeError as error:
            raise TypeError("observation must be an integer discrete state") from error
        if not 0 <= state < self.observation_dim:
            raise ValueError("observation is outside the environment space")
        return state

    def _action_index(self, action: Any) -> int:
        try:
            index = operator.index(action) - self.action_start
        except TypeError as error:
            raise TypeError("action must be an integer discrete action") from error
        if not 0 <= index < self.action_dim:
            raise ValueError("action is outside the environment space")
        return index
