"""Educational Gymnasium wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class EpisodeStatsWrapper(gym.Wrapper):
    """Add episode return and length to ``info`` when an episode ends."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._episode_return = 0.0
        self._episode_length = 0
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        self._episode_length += 1
        if terminated or truncated:
            info = dict(info)
            info["episode"] = {
                "return": self._episode_return,
                "length": self._episode_length,
            }
        return observation, float(reward), terminated, truncated, info
