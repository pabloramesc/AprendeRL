"""A dependency-free training metric logger."""

from __future__ import annotations

import logging
from collections.abc import Mapping


class TrainingLogger:
    """Collect scalar metrics and optionally print periodic snapshots."""

    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose
        self.history: list[dict[str, float | int]] = []
        self._pending: dict[str, float | int] = {}
        self._logger = logging.getLogger("aprenderl")

    def record(self, metrics: Mapping[str, float | int]) -> None:
        """Add scalar values to the next snapshot."""

        self._pending.update(metrics)

    def dump(self, step: int) -> dict[str, float | int]:
        """Store and optionally print one metric snapshot."""

        snapshot: dict[str, float | int] = {"time/timesteps": step}
        snapshot.update(self._pending)
        self.history.append(snapshot)
        self._pending.clear()
        if self.verbose:
            rendered = " | ".join(
                f"{key}: {value:.4g}" for key, value in snapshot.items()
            )
            self._logger.info(rendered)
        return snapshot
