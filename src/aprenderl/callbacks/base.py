"""Minimal callbacks for observing or stopping training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aprenderl.types import EpisodeMetrics, Transition

if TYPE_CHECKING:
    from aprenderl.algorithms.base import BaseAlgorithm


class BaseCallback:
    """No-op callback whose methods can be selectively overridden."""

    def on_training_start(self, algorithm: BaseAlgorithm[Any, Any]) -> None:
        """Run immediately before environment interaction starts."""

    def on_step(
        self,
        algorithm: BaseAlgorithm[Any, Any],
        transition: Transition[Any, Any],
    ) -> bool:
        """Return ``False`` to stop training after the current step."""

        return True

    def on_episode_end(
        self,
        algorithm: BaseAlgorithm[Any, Any],
        metrics: EpisodeMetrics,
    ) -> None:
        """Run after a terminated or truncated episode."""

    def on_training_end(self, algorithm: BaseAlgorithm[Any, Any]) -> None:
        """Run after training finishes or a callback requests a stop."""


class CallbackList(BaseCallback):
    """Compose callbacks without adding inheritance to algorithms."""

    def __init__(self, callbacks: list[BaseCallback]) -> None:
        self.callbacks = callbacks

    def on_training_start(self, algorithm: BaseAlgorithm[Any, Any]) -> None:
        for callback in self.callbacks:
            callback.on_training_start(algorithm)

    def on_step(
        self,
        algorithm: BaseAlgorithm[Any, Any],
        transition: Transition[Any, Any],
    ) -> bool:
        return all(
            callback.on_step(algorithm, transition) for callback in self.callbacks
        )

    def on_episode_end(
        self,
        algorithm: BaseAlgorithm[Any, Any],
        metrics: EpisodeMetrics,
    ) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(algorithm, metrics)

    def on_training_end(self, algorithm: BaseAlgorithm[Any, Any]) -> None:
        for callback in self.callbacks:
            callback.on_training_end(algorithm)
