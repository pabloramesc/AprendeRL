"""First/every-visit Monte Carlo prediction and epsilon-greedy control."""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from typing_extensions import Self

from aprenderl.algorithms.base import OnPolicyAlgorithm
from aprenderl.algorithms.q_learning import QLearning
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.types import Transition

Policy = Callable[[int], int] | np.ndarray


@dataclass(frozen=True)
class MonteCarloPredictionConfig:
    """Settings for state-value estimation from complete episodes."""

    gamma: float = 0.99
    first_visit: bool = True
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


class MonteCarloPrediction(OnPolicyAlgorithm[int, int]):
    """Estimate ``V^pi`` by averaging sampled episode returns.

    ``policy`` can be a callable, a vector of deterministic actions, or a
    ``(states, actions)`` array of action probabilities.
    """

    def __init__(
        self,
        policy: Policy,
        env: gym.Env[Any, Any],
        *,
        config: MonteCarloPredictionConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config or MonteCarloPredictionConfig()
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("MonteCarloPrediction requires Discrete observations")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("MonteCarloPrediction requires Discrete actions")
        self.observation_dim = int(env.observation_space.n)
        self.action_dim = int(env.action_space.n)
        self.observation_start = int(env.observation_space.start)
        self.action_start = int(env.action_space.start)
        self.policy = policy
        self._validate_policy()
        self.values = np.zeros(self.observation_dim, dtype=np.float64)
        self.visit_counts = np.zeros(self.observation_dim, dtype=np.int64)
        self._episode: list[Transition[int, int]] = []
        self._rng = np.random.default_rng(self.config.seed)
        super().__init__(
            env,
            seed=self.config.seed,
            callback=callback,
            logger=logger,
            log_interval=self.config.log_interval,
        )

    def predict(self, observation: int, *, deterministic: bool = False) -> int:
        state = self._state_index(observation)
        if callable(self.policy):
            return int(self.policy(observation))
        policy = np.asarray(self.policy)
        if policy.ndim == 1:
            return int(policy[state])
        probabilities = policy[state]
        if deterministic:
            action = int(np.argmax(probabilities))
        else:
            action = int(self._rng.choice(self.action_dim, p=probabilities))
        return action + self.action_start

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        stored_policy = (
            np.asarray(self.policy) if not callable(self.policy) else np.asarray([])
        )
        metadata = {
            "version": 1,
            "config": asdict(self.config),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "observation_start": self.observation_start,
            "action_start": self.action_start,
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,
            "callable_policy": callable(self.policy),
            "rng_state": self._rng.bit_generator.state,
        }
        with checkpoint_path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                values=self.values,
                visit_counts=self.visit_counts,
                policy=stored_policy,
                episode_returns=np.asarray(self.episode_returns),
                episode_lengths=np.asarray(self.episode_lengths),
                metadata=np.asarray(json.dumps(metadata)),
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env[Any, Any],
        **kwargs: Any,
    ) -> Self:
        supplied_policy = kwargs.pop("policy", None)
        callback = kwargs.pop("callback", None)
        logger = kwargs.pop("logger", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {names}")
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint["metadata"].item()))
            values = checkpoint["values"].copy()
            visit_counts = checkpoint["visit_counts"].copy()
            stored_policy = checkpoint["policy"].copy()
            episode_returns = checkpoint["episode_returns"].tolist()
            episode_lengths = checkpoint["episode_lengths"].tolist()
        if metadata["callable_policy"] and supplied_policy is None:
            raise ValueError("loading a callable-policy checkpoint requires policy=")
        policy = supplied_policy if supplied_policy is not None else stored_policy
        algorithm = cls(
            policy,
            env,
            config=MonteCarloPredictionConfig(**metadata["config"]),
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
        if checkpoint_space != expected_space or values.shape != algorithm.values.shape:
            raise ValueError("checkpoint spaces do not match environment")
        algorithm.values[...] = values
        algorithm.visit_counts[...] = visit_counts
        algorithm.num_timesteps = int(metadata["num_timesteps"])
        algorithm.num_updates = int(metadata["num_updates"])
        algorithm.episode_returns = [float(value) for value in episode_returns]
        algorithm.episode_lengths = [int(value) for value in episode_lengths]
        algorithm._rng.bit_generator.state = metadata["rng_state"]
        return algorithm

    def _sample_action(self, observation: int) -> int:
        return self.predict(observation, deterministic=False)

    def _update_from_transition(
        self, transition: Transition[int, int]
    ) -> dict[str, float | int]:
        self._episode.append(transition)
        if not transition.done:
            return {}
        bootstrap = 0.0
        if transition.truncated and not transition.terminated:
            next_state = self._state_index(transition.next_observation)
            bootstrap = float(self.values[next_state])
        returns = self._discounted_returns(bootstrap)
        seen: set[int] = set()
        latest_error = 0.0
        for item, sampled_return in zip(self._episode, returns, strict=True):
            state = self._state_index(item.observation)
            if self.config.first_visit and state in seen:
                continue
            seen.add(state)
            self.visit_counts[state] += 1
            latest_error = sampled_return - self.values[state]
            self.values[state] += latest_error / self.visit_counts[state]
            self.num_updates += 1
        self._episode.clear()
        return {
            "train/value_error": latest_error,
            "train/updates": self.num_updates,
        }

    def _discounted_returns(self, bootstrap: float) -> list[float]:
        returns = [0.0] * len(self._episode)
        sampled_return = bootstrap
        for index in range(len(self._episode) - 1, -1, -1):
            sampled_return = (
                self._episode[index].reward + self.config.gamma * sampled_return
            )
            returns[index] = sampled_return
        return returns

    def _validate_policy(self) -> None:
        if callable(self.policy):
            return
        policy = np.asarray(self.policy)
        if policy.shape == (self.observation_dim,):
            return
        expected = (self.observation_dim, self.action_dim)
        if policy.shape != expected:
            message = f"policy must have shape ({self.observation_dim},) or {expected}"
            raise ValueError(message)
        if np.any(policy < 0) or not np.allclose(policy.sum(axis=1), 1.0):
            raise ValueError("policy rows must be probability distributions")

    def _state_index(self, observation: Any) -> int:
        try:
            state = operator.index(observation) - self.observation_start
        except TypeError as error:
            raise TypeError("observation must be an integer discrete state") from error
        if not 0 <= state < self.observation_dim:
            raise ValueError("observation is outside the environment space")
        return state


@dataclass(frozen=True)
class MonteCarloControlConfig:
    """Hyperparameters for epsilon-greedy Monte Carlo control.

    ``learning_rate=None`` uses sample averages. A constant rate gives more
    weight to recent episodes in a non-stationary environment.
    """

    learning_rate: float | None = None
    gamma: float = 0.99
    exploration_initial_epsilon: float = 1.0
    exploration_final_epsilon: float = 0.05
    exploration_steps: int = 10_000
    initial_q_value: float = 0.0
    first_visit: bool = True
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.learning_rate is not None and not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1] or None")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= self.exploration_final_epsilon <= 1:
            raise ValueError("exploration_final_epsilon must be in [0, 1]")
        if not (
            self.exploration_final_epsilon
            <= self.exploration_initial_epsilon
            <= 1
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


class MonteCarloControl(QLearning):
    """Learn action values by averaging complete epsilon-greedy returns."""

    config_type = MonteCarloControlConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: MonteCarloControlConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            config=config or MonteCarloControlConfig(),
            callback=callback,
            logger=logger,
        )
        self.visit_counts = np.zeros_like(self.q_table, dtype=np.int64)
        self._episode: list[Transition[int, int]] = []

    def save(self, path: str | Path) -> None:
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
                visit_counts=self.visit_counts,
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
            visit_counts = checkpoint["visit_counts"].copy()
            episode_returns = checkpoint["episode_returns"].tolist()
            episode_lengths = checkpoint["episode_lengths"].tolist()
        algorithm = cls(
            env,
            config=MonteCarloControlConfig(**metadata["config"]),
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
        algorithm.visit_counts[...] = visit_counts
        algorithm.num_timesteps = int(metadata["num_timesteps"])
        algorithm.num_updates = int(metadata["num_updates"])
        algorithm.episode_returns = [float(value) for value in episode_returns]
        algorithm.episode_lengths = [int(value) for value in episode_lengths]
        algorithm.exploration.rng_state = metadata["exploration_rng_state"]
        return algorithm

    def _update_from_transition(
        self, transition: Transition[int, int]
    ) -> dict[str, float | int]:
        self._episode.append(transition)
        if not transition.done:
            return {}
        sampled_return = 0.0
        if transition.truncated and not transition.terminated:
            next_state = self._state_index(transition.next_observation)
            sampled_return = float(np.max(self.q_table[next_state]))
        returns = [0.0] * len(self._episode)
        for index in range(len(self._episode) - 1, -1, -1):
            sampled_return = (
                self._episode[index].reward + self.config.gamma * sampled_return
            )
            returns[index] = sampled_return
        seen: set[tuple[int, int]] = set()
        latest_error = 0.0
        for item, sampled_return in zip(self._episode, returns, strict=True):
            state = self._state_index(item.observation)
            action = self._action_index(item.action)
            pair = (state, action)
            if self.config.first_visit and pair in seen:
                continue
            seen.add(pair)
            self.visit_counts[pair] += 1
            latest_error = sampled_return - float(self.q_table[pair])
            step_size = self.config.learning_rate
            if step_size is None:
                step_size = 1.0 / self.visit_counts[pair]
            self.q_table[pair] += step_size * latest_error
            self.num_updates += 1
        self._episode.clear()
        return {
            "train/td_error": latest_error,
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }
