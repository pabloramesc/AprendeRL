"""A compact NumPy replay buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayBatch:
    """A batch of transitions ready for a PyTorch update."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    """Fixed-size circular buffer for single-environment transitions.

    Termination and truncation are stored separately. Value-based algorithms
    should stop bootstrapping only for true MDP terminations.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        *,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not observation_shape:
            raise ValueError("observation_shape must contain at least one dimension")

        self.capacity = capacity
        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.truncated = np.empty((capacity, 1), dtype=np.float32)
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of transitions currently available."""

        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray | float | int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Append one transition, overwriting the oldest when full."""

        self.observations[self._position] = observation
        self.actions[self._position, 0] = action
        self.rewards[self._position, 0] = reward
        self.next_observations[self._position] = next_observation
        self.terminated[self._position, 0] = terminated
        self.truncated[self._position, 0] = truncated

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        """Sample transitions uniformly with replacement."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")

        indices = self._rng.integers(0, self._size, size=batch_size)
        return ReplayBatch(
            observations=torch.as_tensor(self.observations[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_observations=torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            terminated=torch.as_tensor(self.terminated[indices], device=device),
            truncated=torch.as_tensor(self.truncated[indices], device=device),
        )
