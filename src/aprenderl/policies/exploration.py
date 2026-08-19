"""Simple exploration schedules."""

from collections.abc import Callable
from dataclasses import dataclass

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
    """Compose a Q-network's greedy action with random exploration."""

    def __init__(self, schedule: LinearSchedule, *, seed: int | None = None) -> None:
        self.schedule = schedule
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        q_values: torch.Tensor,
        random_action: Callable[[], int],
        *,
        step: int,
        deterministic: bool,
    ) -> int:
        """Choose a greedy action unless the epsilon coin flip explores."""

        epsilon = self.schedule.value(step)
        if not deterministic and self._rng.random() < epsilon:
            return int(random_action())
        return int(q_values.argmax(dim=-1).item())
