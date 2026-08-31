"""Incremental action-value methods for stationary multi-armed bandits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from typing_extensions import Self

from aprenderl.algorithms.base import OffPolicyAlgorithm
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.policies import EpsilonGreedyPolicy, LinearSchedule
from aprenderl.types import Transition


@dataclass(frozen=True)
class MultiArmedBanditConfig:
    """Hyperparameters for :class:`MultiArmedBandit`.

    ``learning_rate=None`` uses the unbiased sample-average update. A fixed
    step size is useful for non-stationary bandits.
    """

    learning_rate: float | None = None
    exploration_initial_epsilon: float = 0.1
    exploration_final_epsilon: float = 0.1
    exploration_steps: int = 1
    initial_q_value: float = 0.0
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.learning_rate is not None and not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1] or None")
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


class MultiArmedBandit(OffPolicyAlgorithm[Any, int]):
    """Epsilon-greedy action-value learning for a Gymnasium bandit.

    The observation is deliberately ignored: every discrete action represents
    one arm and each environment step is one pull.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        config: MultiArmedBanditConfig | None = None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config or MultiArmedBanditConfig()
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("MultiArmedBandit requires a Discrete action space")
        self.action_dim = int(env.action_space.n)
        self.action_start = int(env.action_space.start)
        self.q_values = np.full(
            self.action_dim, self.config.initial_q_value, dtype=np.float32
        )
        self.action_counts = np.zeros(self.action_dim, dtype=np.int64)
        self.exploration = EpsilonGreedyPolicy(
            LinearSchedule(
                self.config.exploration_initial_epsilon,
                self.config.exploration_final_epsilon,
                self.config.exploration_steps,
            ),
            seed=self.config.seed,
        )
        super().__init__(
            env,
            seed=self.config.seed,
            callback=callback,
            logger=logger,
            log_interval=self.config.log_interval,
        )

    @property
    def epsilon(self) -> float:
        return self.exploration.schedule.value(self.num_timesteps)

    def predict(self, observation: Any, *, deterministic: bool = False) -> int:
        del observation
        index = self.exploration.select(
            self.q_values,
            step=self.num_timesteps,
            deterministic=deterministic,
        )
        return index + self.action_start

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": 1,
            "config": asdict(self.config),
            "action_dim": self.action_dim,
            "action_start": self.action_start,
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,
            "exploration_rng_state": self.exploration.rng_state,
        }
        with checkpoint_path.open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                q_values=self.q_values,
                action_counts=self.action_counts,
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
            q_values = checkpoint["q_values"].copy()
            action_counts = checkpoint["action_counts"].copy()
            episode_returns = checkpoint["episode_returns"].tolist()
            episode_lengths = checkpoint["episode_lengths"].tolist()
        algorithm = cls(
            env,
            config=MultiArmedBanditConfig(**metadata["config"]),
            callback=callback,
            logger=logger,
        )
        if (
            int(metadata["action_dim"]) != algorithm.action_dim
            or int(metadata["action_start"]) != algorithm.action_start
        ):
            raise ValueError("checkpoint action space does not match environment")
        algorithm.q_values[...] = q_values
        algorithm.action_counts[...] = action_counts
        algorithm.num_timesteps = int(metadata["num_timesteps"])
        algorithm.num_updates = int(metadata["num_updates"])
        algorithm.episode_returns = [float(value) for value in episode_returns]
        algorithm.episode_lengths = [int(value) for value in episode_lengths]
        algorithm.exploration.rng_state = metadata["exploration_rng_state"]
        return algorithm

    def _sample_action(self, observation: Any) -> int:
        return self.predict(observation, deterministic=False)

    def _update_from_transition(
        self, transition: Transition[Any, int]
    ) -> dict[str, float | int]:
        index = int(transition.action) - self.action_start
        if not 0 <= index < self.action_dim:
            raise ValueError("action is outside the environment space")
        self.action_counts[index] += 1
        step_size = self.config.learning_rate
        if step_size is None:
            step_size = 1.0 / float(self.action_counts[index])
        error = transition.reward - float(self.q_values[index])
        self.q_values[index] += step_size * error
        self.num_updates += 1
        return {
            "train/value_error": error,
            "train/q_value": float(self.q_values[index]),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }
