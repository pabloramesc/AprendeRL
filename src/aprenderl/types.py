"""Shared data types used by training loops and callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class Transition(Generic[ObservationT, ActionT]):
    """One Gymnasium environment transition.

    ``terminated`` means an MDP terminal state was reached. ``truncated`` means
    an external limit ended the episode, so value targets may still bootstrap.
    """

    observation: ObservationT
    action: ActionT
    reward: float
    next_observation: ObservationT
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        """Whether Gymnasium requires the environment to be reset."""

        return self.terminated or self.truncated


@dataclass(frozen=True)
class EpisodeMetrics:
    """Summary emitted when an episode finishes."""

    episode_return: float
    episode_length: int
    num_timesteps: int
