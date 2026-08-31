"""Value and policy iteration for finite Gymnasium transition models."""

from __future__ import annotations

import json
import operator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm
from typing_extensions import Self

from aprenderl.algorithms.base import OffPolicyAlgorithm
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.types import Transition


@dataclass(frozen=True)
class DynamicProgrammingConfig:
    """Shared settings for exact finite-MDP planning."""

    gamma: float = 0.99
    tolerance: float = 1e-8
    evaluation_steps: int = 10_000
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.evaluation_steps <= 0:
            raise ValueError("evaluation_steps must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


class _DynamicProgramming(OffPolicyAlgorithm[int, int]):
    """Common transition-model validation and persistence."""

    config_type = DynamicProgrammingConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: DynamicProgrammingConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config or self.config_type()
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError("dynamic programming requires a Discrete observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("dynamic programming requires a Discrete action space")
        model = getattr(env.unwrapped, "P", None)
        if model is None:
            raise TypeError("environment must expose a finite transition model as P")
        self.model = model
        self.observation_dim = int(env.observation_space.n)
        self.action_dim = int(env.action_space.n)
        self.observation_start = int(env.observation_space.start)
        self.action_start = int(env.action_space.start)
        self.values = np.zeros(self.observation_dim, dtype=np.float64)
        self.policy = np.zeros(self.observation_dim, dtype=np.int64)
        self.converged = False
        super().__init__(
            env,
            seed=self.config.seed,
            callback=callback,
            logger=logger,
            log_interval=self.config.log_interval,
        )

    def predict(self, observation: int, *, deterministic: bool = False) -> int:
        del deterministic
        state = self._state_index(observation)
        return int(self.policy[state]) + self.action_start

    def q_values(self, state: int, values: np.ndarray | None = None) -> np.ndarray:
        """Return Bellman action values for a zero-based state index."""

        source = self.values if values is None else values
        result = np.zeros(self.action_dim, dtype=np.float64)
        model_state = state + self.observation_start
        for action in range(self.action_dim):
            model_action = action + self.action_start
            for probability, next_state, reward, terminated in self.model[model_state][
                model_action
            ]:
                next_index = self._state_index(next_state)
                bootstrap = 0.0 if terminated else source[next_index]
                result[action] += probability * (
                    float(reward) + self.config.gamma * bootstrap
                )
        return result

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
            "converged": self.converged,
        }
        with checkpoint_path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                values=self.values,
                policy=self.policy,
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
            values = checkpoint["values"].copy()
            policy = checkpoint["policy"].copy()
        algorithm = cls(
            env,
            config=cls.config_type(**metadata["config"]),
            callback=callback,
            logger=logger,
        )
        expected = (
            algorithm.observation_dim,
            algorithm.action_dim,
            algorithm.observation_start,
            algorithm.action_start,
        )
        actual = (
            int(metadata["observation_dim"]),
            int(metadata["action_dim"]),
            int(metadata["observation_start"]),
            int(metadata["action_start"]),
        )
        if actual != expected or values.shape != (algorithm.observation_dim,):
            raise ValueError("checkpoint spaces do not match environment")
        algorithm.values[...] = values
        algorithm.policy[...] = policy
        algorithm.num_timesteps = int(metadata["num_timesteps"])
        algorithm.num_updates = int(metadata["num_updates"])
        algorithm.converged = bool(metadata["converged"])
        return algorithm

    def _sample_action(self, observation: int) -> int:
        return self.predict(observation, deterministic=True)

    def _update_from_transition(
        self, transition: Transition[int, int]
    ) -> dict[str, float | int]:
        raise RuntimeError("planning algorithms do not sample environment transitions")

    def _state_index(self, observation: Any) -> int:
        try:
            index = operator.index(observation) - self.observation_start
        except TypeError as error:
            raise TypeError("observation must be an integer discrete state") from error
        if not 0 <= index < self.observation_dim:
            raise ValueError("observation is outside the environment space")
        return index


class ValueIteration(_DynamicProgramming):
    """Bellman optimality sweeps over an exact finite transition model."""

    def learn(self, total_timesteps: int, *, progress_bar: bool = True) -> Self:
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        self.callback.on_training_start(self)
        with tqdm(
            total=total_timesteps,
            desc=self.__class__.__name__,
            unit="sweep",
            disable=not progress_bar,
        ) as progress:
            for _ in range(total_timesteps):
                old_values = self.values.copy()
                for state in range(self.observation_dim):
                    self.values[state] = np.max(self.q_values(state, old_values))
                residual = float(np.max(np.abs(self.values - old_values)))
                self.num_timesteps += 1
                self.num_updates += self.observation_dim
                self.logger.record({"train/bellman_residual": residual})
                progress.update(1)
                if residual < self.config.tolerance:
                    self.converged = True
                    break
        for state in range(self.observation_dim):
            self.policy[state] = int(np.argmax(self.q_values(state)))
        self.callback.on_training_end(self)
        return self


class PolicyIteration(_DynamicProgramming):
    """Iterative policy evaluation followed by greedy improvement."""

    def learn(self, total_timesteps: int, *, progress_bar: bool = True) -> Self:
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        self.callback.on_training_start(self)
        with tqdm(
            total=total_timesteps,
            desc=self.__class__.__name__,
            unit="improvement",
            disable=not progress_bar,
        ) as progress:
            for _ in range(total_timesteps):
                residual = self._evaluate_policy()
                stable = True
                for state in range(self.observation_dim):
                    action = int(np.argmax(self.q_values(state)))
                    stable &= action == int(self.policy[state])
                    self.policy[state] = action
                self.num_timesteps += 1
                self.logger.record({"train/policy_residual": residual})
                progress.update(1)
                if stable:
                    self.converged = True
                    break
        self.callback.on_training_end(self)
        return self

    def _evaluate_policy(self) -> float:
        residual = float("inf")
        for _ in range(self.config.evaluation_steps):
            old_values = self.values.copy()
            for state in range(self.observation_dim):
                action_values = self.q_values(state, old_values)
                self.values[state] = action_values[self.policy[state]]
            self.num_updates += self.observation_dim
            residual = float(np.max(np.abs(self.values - old_values)))
            if residual < self.config.tolerance:
                break
        return residual
