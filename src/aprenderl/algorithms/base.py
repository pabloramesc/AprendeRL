"""Algorithm interfaces and the shared environment-interaction loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
from tqdm.auto import tqdm
from typing_extensions import Self

from aprenderl.callbacks import BaseCallback, CallbackList
from aprenderl.logging import TrainingLogger
from aprenderl.types import EpisodeMetrics, Transition
from aprenderl.utils import seed_everything

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
TrainingMetrics = dict[str, float | int]


class BaseAlgorithm(ABC, Generic[ObservationT, ActionT]):
    """Common interface and state for every AprendeRL algorithm."""

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
    def learn(self, total_timesteps: int, *, progress_bar: bool = True) -> Self:
        """Train for ``total_timesteps`` interactions with the environment."""

    @abstractmethod
    def predict(
        self, observation: ObservationT, *, deterministic: bool = False
    ) -> ActionT:
        """Choose an action for one observation."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save enough state to resume training or run inference."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, env: Any, **kwargs: Any) -> Self:
        """Restore an algorithm checkpoint for ``env``."""

    def _learn_online(self, total_timesteps: int, *, progress_bar: bool) -> Self:
        """Run the explicit one-environment loop shared by current algorithms."""

        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        self.callback.on_training_start(self)
        latest_training_metrics: dict[str, float | int] = {}

        with tqdm(
            total=total_timesteps,
            desc=self.__class__.__name__,
            unit="step",
            disable=not progress_bar,
        ) as progress:
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

                training_metrics = dict(self._update_from_transition(transition))
                if training_metrics:
                    latest_training_metrics = training_metrics
                    self.logger.record(training_metrics)
                self._record_step(transition)
                self._after_step(transition)

                progress.update(1)
                progress.set_postfix(
                    self._progress_metrics(latest_training_metrics),
                    refresh=False,
                )
                if not self.callback.on_step(self, transition):
                    break

        self.callback.on_training_end(self)
        return self

    def _sample_action(self, observation: ObservationT) -> ActionT:
        """Select a behavior action; algorithms may override to cache actions."""

        return self.predict(observation, deterministic=False)

    @abstractmethod
    def _update_from_transition(
        self, transition: Transition[ObservationT, ActionT]
    ) -> Mapping[str, float | int]:
        """Process one transition and return any new training metrics."""

    def _after_step(self, transition: Transition[ObservationT, ActionT]) -> None:
        """Run optional bookkeeping after the transition has been recorded."""

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

    def _progress_metrics(
        self,
        training_metrics: Mapping[str, float | int] | None = None,
    ) -> dict[str, str | int]:
        """Format compact metrics for a training progress bar."""

        metrics: dict[str, str | int] = {
            "episodes": len(self.episode_returns),
            "updates": self.num_updates,
        }
        if self.episode_returns:
            metrics["return"] = f"{self.episode_returns[-1]:.1f}"
        if training_metrics and "train/loss" in training_metrics:
            metrics["loss"] = f"{float(training_metrics['train/loss']):.4f}"
        return metrics

    def _training_state(self) -> dict[str, Any]:
        """Return training counters and completed-episode history."""

        return {
            "num_timesteps": self.num_timesteps,
            "num_updates": self.num_updates,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
        }

    def _restore_training_state(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`_training_state`."""

        self.num_timesteps = int(state["num_timesteps"])
        self.num_updates = int(state["num_updates"])
        self.episode_returns = [float(value) for value in state["episode_returns"]]
        self.episode_lengths = [int(value) for value in state["episode_lengths"]]

    @staticmethod
    def _reject_unknown_arguments(arguments: Mapping[str, Any]) -> None:
        """Raise a consistent error for unsupported keyword arguments."""

        if arguments:
            names = ", ".join(sorted(arguments))
            raise TypeError(f"unexpected keyword arguments: {names}")

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
    """Base class for algorithms that learn a policy other than behavior."""

    def learn(self, total_timesteps: int, *, progress_bar: bool = True) -> Self:
        """Learn online while allowing algorithm-specific experience reuse."""

        return self._learn_online(total_timesteps, progress_bar=progress_bar)


class OnPolicyAlgorithm(BaseAlgorithm[ObservationT, ActionT], ABC):
    """Base class for algorithms updated from their current behavior policy."""

    def learn(self, total_timesteps: int, *, progress_bar: bool = True) -> Self:
        """Learn online from actions sampled from the current policy."""

        return self._learn_online(total_timesteps, progress_bar=progress_bar)
