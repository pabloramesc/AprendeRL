"""Sequential rollout storage for on-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutBatch:
    """A complete rollout with targets computed by generalized advantage estimation."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Store one fixed-length, ordered rollout for an on-policy update.

    Unlike replay storage, this buffer is cleared after each policy update and
    preserves transition order so that returns and advantages can be computed.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        *,
        action_shape: tuple[int, ...] = (),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not observation_shape:
            raise ValueError("observation_shape must contain at least one dimension")
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be between 0 and 1")

        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.truncated = np.empty(capacity, dtype=np.float32)
        self.values = np.empty(capacity, dtype=np.float32)
        self.next_values = np.empty(capacity, dtype=np.float32)
        self.log_probabilities = np.empty(capacity, dtype=np.float32)
        self.advantages = np.empty(capacity, dtype=np.float32)
        self.returns = np.empty(capacity, dtype=np.float32)
        self._size = 0
        self._targets_ready = False

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        """Whether the rollout has reached its configured capacity."""

        return self._size == self.capacity

    def reset(self) -> None:
        """Discard the current rollout before collecting fresh policy data."""

        self._size = 0
        self._targets_ready = False

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray | float | int,
        reward: float,
        terminated: bool,
        truncated: bool,
        value: float,
        next_value: float,
        log_probability: float,
    ) -> None:
        """Append one transition in collection order."""

        if self.full:
            raise ValueError("rollout buffer is full; update the policy and reset it")

        self.observations[self._size] = observation
        self.actions[self._size] = action
        self.rewards[self._size] = reward
        self.terminated[self._size] = terminated
        self.truncated[self._size] = truncated
        self.values[self._size] = value
        self.next_values[self._size] = next_value
        self.log_probabilities[self._size] = log_probability
        self._size += 1
        self._targets_ready = False

    def compute_returns_and_advantages(self) -> None:
        """Compute GAE and targets while respecting both Gymnasium end flags.

        Targets bootstrap through truncations but advantage recursion stops at
        every episode boundary, preventing data from a reset episode leaking
        into the preceding one.
        """

        if self._size == 0:
            raise ValueError("cannot compute targets for an empty rollout")

        advantage = 0.0
        for step in reversed(range(self._size)):
            bootstrap = 1.0 - self.terminated[step]
            episode_continues = 1.0 - max(self.terminated[step], self.truncated[step])
            delta = (
                self.rewards[step]
                + self.gamma * self.next_values[step] * bootstrap
                - self.values[step]
            )
            advantage = (
                delta + self.gamma * self.gae_lambda * episode_continues * advantage
            )
            self.advantages[step] = advantage

        self.returns[: self._size] = (
            self.advantages[: self._size] + self.values[: self._size]
        )
        self._targets_ready = True

    def batch(self, device: torch.device) -> RolloutBatch:
        """Return the collected rollout as tensors for one policy update."""

        if not self._targets_ready:
            raise ValueError("compute returns and advantages before requesting a batch")
        rollout = slice(0, self._size)
        return RolloutBatch(
            observations=torch.as_tensor(self.observations[rollout], device=device),
            actions=torch.as_tensor(self.actions[rollout], device=device),
            rewards=torch.as_tensor(self.rewards[rollout], device=device),
            terminated=torch.as_tensor(self.terminated[rollout], device=device),
            truncated=torch.as_tensor(self.truncated[rollout], device=device),
            values=torch.as_tensor(self.values[rollout], device=device),
            next_values=torch.as_tensor(self.next_values[rollout], device=device),
            log_probabilities=torch.as_tensor(
                self.log_probabilities[rollout], device=device
            ),
            advantages=torch.as_tensor(self.advantages[rollout], device=device),
            returns=torch.as_tensor(self.returns[rollout], device=device),
        )
