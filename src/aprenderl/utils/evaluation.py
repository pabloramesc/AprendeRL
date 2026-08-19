"""Deterministic policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np


class Predictable(Protocol):
    """Structural type accepted by :func:`evaluate_policy`."""

    def predict(self, observation: Any, *, deterministic: bool = True) -> Any:
        """Choose an action for one observation."""


@dataclass(frozen=True)
class EvaluationResult:
    """Returns and lengths from a policy evaluation."""

    returns: tuple[float, ...]
    lengths: tuple[int, ...]

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns))

    @property
    def return_std(self) -> float:
        return float(np.std(self.returns))


def evaluate_policy(
    algorithm: Predictable,
    env: gym.Env[Any, Any],
    *,
    episodes: int = 10,
    deterministic: bool = True,
    seed: int | None = None,
) -> EvaluationResult:
    """Evaluate without mutating the algorithm's training environment."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        observation, _ = env.reset(seed=episode_seed)
        total_reward = 0.0
        length = 0
        done = False
        while not done:
            action = algorithm.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            length += 1
            done = terminated or truncated
        episode_returns.append(total_reward)
        episode_lengths.append(length)

    return EvaluationResult(tuple(episode_returns), tuple(episode_lengths))
