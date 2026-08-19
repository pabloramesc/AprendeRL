"""Configuration shared by examples and experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Small set of settings commonly needed to run an experiment."""

    total_timesteps: int = 20_000
    evaluation_episodes: int = 10
    seed: int = 42
    device: str = "auto"
    checkpoint_path: Path = Path("artifacts/dqn_cartpole.pt")

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")
