"""Algorithm interfaces and reusable environment-interaction loops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from typing_extensions import Self

from aprenderl.buffers import ReplayBuffer, RolloutBuffer
from aprenderl.callbacks import BaseCallback, CallbackList
from aprenderl.logging import TrainingLogger
from aprenderl.types import EpisodeMetrics, Transition
from aprenderl.utils import seed_everything

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class BaseAlgorithm(ABC, Generic[ObservationT, ActionT]):
    """Interface and shared state implemented by every algorithm."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        seed: int | None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
        log_interval: int = 10,
    ) -> None:
        if log_interval <= 0:
            raise ValueError("log_interval must be positive")
        self.env = env
        self.seed = seed
        self.callback = self._make_callback(callback)
        self.logger = logger or TrainingLogger()
        self.log_interval = log_interval
        self.num_timesteps = 0
        self.num_updates = 0
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self._observation: ObservationT | None = None
        self._episode_return = 0.0
        self._episode_length = 0

        seed_everything(seed)
        env.action_space.seed(seed)

    @abstractmethod
    def learn(self, total_timesteps: int) -> Self:
        """Train for ``total_timesteps`` interactions with the environment."""

    @abstractmethod
    def predict(
        self, observation: ObservationT, *, deterministic: bool = True
    ) -> ActionT:
        """Choose an action for one observation."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save enough state to resume training or run inference."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, env: Any, **kwargs: Any) -> Self:
        """Restore an algorithm checkpoint for ``env``."""

    def _current_observation(self) -> ObservationT:
        if self._observation is None:
            reset_seed = self.seed if self.num_timesteps == 0 else None
            observation, _ = self.env.reset(seed=reset_seed)
            self._observation = observation
        return self._observation

    def _record_step(self, transition: Transition[ObservationT, ActionT]) -> None:
        self.num_timesteps += 1
        self._episode_return += transition.reward
        self._episode_length += 1
        if transition.done:
            metrics = EpisodeMetrics(
                episode_return=self._episode_return,
                episode_length=self._episode_length,
                num_timesteps=self.num_timesteps,
            )
            self.episode_returns.append(metrics.episode_return)
            self.episode_lengths.append(metrics.episode_length)
            self.callback.on_episode_end(self, metrics)
            if len(self.episode_returns) % self.log_interval == 0:
                recent_returns = self.episode_returns[-self.log_interval :]
                self.logger.record(
                    {
                        "rollout/episode_return_mean": sum(recent_returns)
                        / len(recent_returns),
                        "rollout/episodes": len(self.episode_returns),
                    }
                )
                self.logger.dump(self.num_timesteps)
            observation, _ = self.env.reset()
            self._observation = observation
            self._episode_return = 0.0
            self._episode_length = 0
        else:
            self._observation = transition.next_observation

    @staticmethod
    def _make_callback(
        callback: BaseCallback | list[BaseCallback] | None,
    ) -> BaseCallback:
        if callback is None:
            return BaseCallback()
        if isinstance(callback, list):
            return CallbackList(callback)
        return callback


class OffPolicyAlgorithm(BaseAlgorithm[ObservationT, ActionT], ABC):
    """Reusable replay-based training loop for off-policy algorithms."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        replay_buffer: ReplayBuffer,
        *,
        learning_starts: int,
        train_frequency: int,
        gradient_steps: int,
        seed: int | None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
        log_interval: int = 10,
    ) -> None:
        super().__init__(
            env,
            seed=seed,
            callback=callback,
            logger=logger,
            log_interval=log_interval,
        )
        self.replay_buffer = replay_buffer
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.gradient_steps = gradient_steps

    def learn(self, total_timesteps: int) -> Self:
        """Collect transitions and perform scheduled replay updates."""

        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        self._before_learn(total_timesteps)
        self.callback.on_training_start(self)

        for _ in range(total_timesteps):
            observation = self._current_observation()
            action = self._sample_action(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            transition = Transition(
                observation=observation,
                action=action,
                reward=float(reward),
                next_observation=next_observation,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )
            self.replay_buffer.add(
                np.asarray(observation, dtype=np.float32),
                action,
                transition.reward,
                np.asarray(next_observation, dtype=np.float32),
                transition.terminated,
                transition.truncated,
            )
            self._record_step(transition)

            if self._ready_to_train():
                for _ in range(self.gradient_steps):
                    self.logger.record(self._train_step())
            self._after_step()
            if not self.callback.on_step(self, transition):
                break

        self.callback.on_training_end(self)
        return self

    def _ready_to_train(self) -> bool:
        return (
            self.num_timesteps >= self.learning_starts
            and self.num_timesteps % self.train_frequency == 0
            and self._has_enough_replay()
        )

    @abstractmethod
    def _sample_action(self, observation: ObservationT) -> ActionT:
        """Select the behavior action used to collect a transition."""

    @abstractmethod
    def _has_enough_replay(self) -> bool:
        """Whether the replay buffer contains one complete training batch."""

    @abstractmethod
    def _train_step(self) -> Mapping[str, float | int]:
        """Perform one optimizer update and return scalar metrics."""

    def _before_learn(self, total_timesteps: int) -> None:
        """Prepare algorithm-specific schedules before collection."""

    def _after_step(self) -> None:
        """Run algorithm-specific bookkeeping after an environment step."""


class OnPolicyAlgorithm(BaseAlgorithm[ObservationT, ActionT], ABC):
    """Reusable collect-then-update loop for future on-policy algorithms."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        rollout_buffer: RolloutBuffer,
        *,
        seed: int | None,
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
        log_interval: int = 10,
    ) -> None:
        super().__init__(
            env,
            seed=seed,
            callback=callback,
            logger=logger,
            log_interval=log_interval,
        )
        self.rollout_buffer = rollout_buffer

    def learn(self, total_timesteps: int) -> Self:
        """Alternate ordered rollout collection and policy updates."""

        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        remaining = total_timesteps
        self.callback.on_training_start(self)
        continue_training = True
        while remaining > 0 and continue_training:
            rollout_steps = min(remaining, self.rollout_buffer.capacity)
            continue_training = self._collect_rollout(rollout_steps)
            self.rollout_buffer.compute_returns_and_advantages()
            self.logger.record(self._update_from_rollout())
            self.num_updates += 1
            remaining -= rollout_steps
        self.callback.on_training_end(self)
        return self

    def _collect_rollout(self, steps: int) -> bool:
        self.rollout_buffer.reset()
        for _ in range(steps):
            observation = self._current_observation()
            action, value, log_probability = self._sample_rollout_action(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            next_value = 0.0 if terminated else self._estimate_value(next_observation)
            transition = Transition(
                observation=observation,
                action=action,
                reward=float(reward),
                next_observation=next_observation,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )
            self.rollout_buffer.add(
                np.asarray(observation, dtype=np.float32),
                np.asarray(action),
                transition.reward,
                transition.terminated,
                transition.truncated,
                value,
                next_value,
                log_probability,
            )
            self._record_step(transition)
            if not self.callback.on_step(self, transition):
                return False
        return True

    @abstractmethod
    def _sample_rollout_action(
        self, observation: ObservationT
    ) -> tuple[ActionT, float, float]:
        """Return an action, value estimate, and action log probability."""

    @abstractmethod
    def _estimate_value(self, observation: ObservationT) -> float:
        """Estimate the value of an observation for bootstrapping."""

    @abstractmethod
    def _update_from_rollout(self) -> Mapping[str, float | int]:
        """Update the current policy from its freshly collected rollout."""
