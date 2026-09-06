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
    By default actions are scalar integers. Supplying ``action_shape`` enables
    float32 continuous actions with that shape, preserving fractional values.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        *,
        seed: int | None = None,
        action_shape: tuple[int, ...] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not observation_shape:
            raise ValueError("observation_shape must contain at least one dimension")

        self.capacity = capacity
        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        if action_shape is not None and (
            not action_shape or any(dimension <= 0 for dimension in action_shape)
        ):
            raise ValueError("action_shape must contain positive dimensions")
        self.actions = np.empty(
            (capacity, *(action_shape if action_shape is not None else (1,))),
            dtype=np.float32 if action_shape is not None else np.int64,
        )
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
        self.actions[self._position] = action
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


@dataclass(frozen=True)
class NStepReplayBatch(ReplayBatch):
    """Replay batch with the discount exponent already accumulated."""

    discounts: torch.Tensor


class NStepReplayBuffer(ReplayBuffer):
    """Uniform replay storing an individual bootstrap discount per item."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(capacity, observation_shape, seed=seed)
        self.discounts = np.empty((capacity, 1), dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray | float | int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool = False,
        *,
        discount: float,
    ) -> None:
        position = self._position
        super().add(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        )
        self.discounts[position, 0] = discount

    def sample(self, batch_size: int, device: torch.device) -> NStepReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return NStepReplayBatch(
            observations=torch.as_tensor(self.observations[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_observations=torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            terminated=torch.as_tensor(self.terminated[indices], device=device),
            truncated=torch.as_tensor(self.truncated[indices], device=device),
            discounts=torch.as_tensor(self.discounts[indices], device=device),
        )


@dataclass(frozen=True)
class PrioritizedReplayBatch(NStepReplayBatch):
    """Prioritized replay tensors, sampling weights, and source indices."""

    weights: torch.Tensor
    indices: np.ndarray


class PrioritizedReplayBuffer(NStepReplayBuffer):
    """Proportional prioritized replay implemented with explicit NumPy arrays."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        *,
        alpha: float = 0.6,
        seed: int | None = None,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        super().__init__(capacity, observation_shape, seed=seed)
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float64)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray | float | int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool = False,
        *,
        discount: float = 1.0,
        priority: float | None = None,
    ) -> None:
        position = self._position
        super().add(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
            discount=discount,
        )
        maximum = float(self.priorities[: self._size].max(initial=1.0))
        self.priorities[position] = maximum if priority is None else priority

    def sample(
        self, batch_size: int, device: torch.device, *, beta: float = 0.4
    ) -> PrioritizedReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if not 0 <= beta <= 1:
            raise ValueError("beta must be between 0 and 1")
        scaled = self.priorities[: self._size] ** self.alpha
        probabilities = scaled / scaled.sum()
        indices = self._rng.choice(
            self._size, size=batch_size, replace=True, p=probabilities
        )
        weights = (self._size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return PrioritizedReplayBatch(
            observations=torch.as_tensor(self.observations[indices], device=device),
            actions=torch.as_tensor(self.actions[indices], device=device),
            rewards=torch.as_tensor(self.rewards[indices], device=device),
            next_observations=torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            terminated=torch.as_tensor(self.terminated[indices], device=device),
            truncated=torch.as_tensor(self.truncated[indices], device=device),
            discounts=torch.as_tensor(self.discounts[indices], device=device),
            weights=torch.as_tensor(
                weights[:, None], device=device, dtype=torch.float32
            ),
            indices=indices,
        )

    def update_priorities(
        self, indices: np.ndarray, priorities: np.ndarray
    ) -> None:
        if indices.shape != priorities.shape:
            raise ValueError("indices and priorities must have matching shapes")
        if np.any(priorities <= 0) or not np.all(np.isfinite(priorities)):
            raise ValueError("priorities must be finite and positive")
        self.priorities[indices] = priorities
