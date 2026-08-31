"""Expected, multi-step, eligibility-trace, and model-based TD control."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from typing_extensions import Self

from aprenderl.algorithms.q_learning import QLearning, QLearningConfig
from aprenderl.algorithms.sarsa import SARSA, SARSAConfig
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.types import Transition


@dataclass(frozen=True)
class ExpectedSARSAConfig(SARSAConfig):
    """Hyperparameters for :class:`ExpectedSARSA`."""


class ExpectedSARSA(SARSA):
    """One-step SARSA using the exact epsilon-greedy expectation."""

    config_type = ExpectedSARSAConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: ExpectedSARSAConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            config=config or ExpectedSARSAConfig(),
            callback=callback,
            logger=logger,
        )

    def _update_from_transition(
        self, transition: Transition[int, int]
    ) -> dict[str, float | int]:
        state = self._state_index(transition.observation)
        action = self._action_index(transition.action)
        next_value = 0.0
        if not transition.terminated:
            next_state = self._state_index(transition.next_observation)
            epsilon = self.exploration.schedule.value(self.num_timesteps + 1)
            values = self.q_table[next_state]
            next_value = float(
                (1.0 - epsilon) * np.max(values) + epsilon * np.mean(values)
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


@dataclass(frozen=True)
class NStepSARSAConfig(SARSAConfig):
    """Hyperparameters for :class:`NStepSARSA`."""

    n_steps: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")


class NStepSARSA(SARSA):
    """On-policy SARSA with an explicit queue of n-step returns."""

    config_type = NStepSARSAConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: NStepSARSAConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            config=config or NStepSARSAConfig(),
            callback=callback,
            logger=logger,
        )
        self._pending: deque[tuple[Transition[int, int], int | None]] = deque()

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
        self._pending.append((transition, next_action))
        latest: dict[str, float | int] = {}
        if len(self._pending) >= self.config.n_steps:
            latest = self._update_oldest(self.config.n_steps)
            self._pending.popleft()
        if transition.done:
            while self._pending:
                latest = self._update_oldest(len(self._pending))
                self._pending.popleft()
            self._action = None
        else:
            self._action = next_action
        return latest

    def _update_oldest(self, horizon: int) -> dict[str, float | int]:
        items = list(self._pending)[:horizon]
        first = items[0][0]
        sampled_return = 0.0
        discount = 1.0
        for transition, _ in items:
            sampled_return += discount * transition.reward
            discount *= self.config.gamma
        last_transition, last_action = items[-1]
        if not last_transition.terminated:
            if last_action is None:
                raise ValueError("next action is required for a bootstrap target")
            next_state = self._state_index(last_transition.next_observation)
            sampled_return += discount * float(
                self.q_table[next_state, self._action_index(last_action)]
            )
        state = self._state_index(first.observation)
        action = self._action_index(first.action)
        td_error = sampled_return - float(self.q_table[state, action])
        self.q_table[state, action] += self.config.learning_rate * td_error
        self.num_updates += 1
        return {
            "train/td_error": td_error,
            "train/q_value": float(self.q_table[state, action]),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class SARSALambdaConfig(SARSAConfig):
    """Hyperparameters for accumulating-trace SARSA(lambda)."""

    trace_decay: float = 0.9

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.trace_decay <= 1:
            raise ValueError("trace_decay must be between 0 and 1")


class SARSALambda(SARSA):
    """SARSA(lambda) with accumulating eligibility traces."""

    config_type = SARSALambdaConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: SARSALambdaConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            config=config or SARSALambdaConfig(),
            callback=callback,
            logger=logger,
        )
        self.eligibility = np.zeros_like(self.q_table)

    def _update_from_transition(
        self, transition: Transition[int, int]
    ) -> dict[str, float | int]:
        next_action = None
        next_value = 0.0
        if not transition.terminated:
            next_action = self._select_action(
                transition.next_observation,
                deterministic=False,
                step=self.num_timesteps + 1,
            )
            next_state = self._state_index(transition.next_observation)
            next_value = float(
                self.q_table[next_state, self._action_index(next_action)]
            )
        state = self._state_index(transition.observation)
        action = self._action_index(transition.action)
        target = transition.reward + self.config.gamma * next_value
        td_error = target - float(self.q_table[state, action])
        self.eligibility[state, action] += 1.0
        self.q_table += self.config.learning_rate * td_error * self.eligibility
        self.eligibility *= self.config.gamma * self.config.trace_decay
        self.num_updates += 1
        if transition.done:
            self.eligibility.fill(0.0)
            self._action = None
        else:
            self._action = next_action
        return {
            "train/td_error": td_error,
            "train/q_value": float(self.q_table[state, action]),
            "train/trace_sum": float(self.eligibility.sum()),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class DynaQConfig(QLearningConfig):
    """Hyperparameters for tabular Dyna-Q."""

    planning_steps: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.planning_steps < 0:
            raise ValueError("planning_steps cannot be negative")


class DynaQ(QLearning):
    """Q-learning plus random planning updates from a learned one-step model."""

    config_type = DynaQConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: DynaQConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            config=config or DynaQConfig(),
            callback=callback,
            logger=logger,
        )
        self.model: dict[tuple[int, int], tuple[float, int, bool]] = {}
        self._planning_rng = np.random.default_rng(self.config.seed)

    def _update_from_transition(
        self, transition: Transition[int, int]
    ) -> dict[str, float | int]:
        metrics = self._update(transition)
        key = (
            self._state_index(transition.observation),
            self._action_index(transition.action),
        )
        self.model[key] = (
            transition.reward,
            self._state_index(transition.next_observation),
            transition.terminated,
        )
        keys = tuple(self.model)
        for _ in range(self.config.planning_steps):
            model_key = keys[int(self._planning_rng.integers(len(keys)))]
            reward, next_state, terminated = self.model[model_key]
            state, action = model_key
            next_value = 0.0 if terminated else float(np.max(self.q_table[next_state]))
            target = reward + self.config.gamma * next_value
            error = target - float(self.q_table[state, action])
            self.q_table[state, action] += self.config.learning_rate * error
            self.num_updates += 1
        metrics["train/updates"] = self.num_updates
        metrics["train/model_size"] = len(self.model)
        return metrics

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_model = [
            [state, action, reward, next_state, terminated]
            for (state, action), (reward, next_state, terminated) in self.model.items()
        ]
        metadata = {
            "version": 1,
            "config": asdict(self.config),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "observation_start": self.observation_start,
            "action_start": self.action_start,
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,
            "model": serialized_model,
            "exploration_rng_state": self.exploration.rng_state,
            "planning_rng_state": self._planning_rng.bit_generator.state,
        }
        with checkpoint_path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                q_table=self.q_table,
                episode_returns=np.asarray(self.episode_returns),
                episode_lengths=np.asarray(self.episode_lengths),
                metadata=np.asarray(json.dumps(metadata)),
            )

    @classmethod
    def load(
        cls, path: str | Path, env: gym.Env[Any, Any], **kwargs: Any
    ) -> Self:
        callback = kwargs.pop("callback", None)
        logger = kwargs.pop("logger", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {names}")
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            q_table = checkpoint["q_table"].copy()
            episode_returns = checkpoint["episode_returns"].tolist()
            episode_lengths = checkpoint["episode_lengths"].tolist()
        algorithm = cls(
            env,
            config=DynaQConfig(**metadata["config"]),
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
        if (
            checkpoint_space != expected_space
            or q_table.shape != algorithm.q_table.shape
        ):
            raise ValueError("checkpoint spaces do not match environment")
        algorithm.q_table[...] = q_table
        algorithm.model = {
            (int(state), int(action)): (float(reward), int(next_state), bool(done))
            for state, action, reward, next_state, done in metadata["model"]
        }
        algorithm.num_timesteps = int(metadata["num_timesteps"])
        algorithm.num_updates = int(metadata["num_updates"])
        algorithm.episode_returns = [float(value) for value in episode_returns]
        algorithm.episode_lengths = [int(value) for value in episode_lengths]
        algorithm.exploration.rng_state = metadata["exploration_rng_state"]
        algorithm._planning_rng.bit_generator.state = metadata["planning_rng_state"]
        return algorithm
