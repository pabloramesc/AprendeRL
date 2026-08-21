"""Simple action-selection policies and exploration schedules."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class LinearSchedule:
    """Linearly interpolate from ``start`` to ``end`` over ``duration`` steps."""

    start: float
    end: float
    duration: int

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")

    def value(self, step: int) -> float:
        """Return the scheduled value, clamped after the final step."""

        progress = min(max(step, 0) / self.duration, 1.0)
        return self.start + progress * (self.end - self.start)


class EpsilonGreedyPolicy:
    """Select zero-based actions from tabular or neural Q-values."""

    def __init__(self, schedule: LinearSchedule, *, seed: int | None = None) -> None:
        self.schedule = schedule
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        q_values: np.ndarray | torch.Tensor,
        *,
        step: int,
        deterministic: bool,
    ) -> int:
        """Choose a greedy action unless the epsilon coin flip explores."""

        if q_values.ndim != 1 or q_values.shape[0] == 0:
            raise ValueError("q_values must contain one value per action")
        epsilon = self.schedule.value(step)
        if not deterministic and self._rng.random() < epsilon:
            return int(self._rng.integers(q_values.shape[0]))
        if isinstance(q_values, torch.Tensor):
            return int(q_values.argmax().item())
        return int(np.argmax(q_values))

    @property
    def rng_state(self) -> dict[str, Any]:
        """Return a copy of the random state for checkpointing."""

        return copy.deepcopy(self._rng.bit_generator.state)

    @rng_state.setter
    def rng_state(self, state: dict[str, Any]) -> None:
        self._rng.bit_generator.state = copy.deepcopy(state)
