"""Shared stochastic-policy handling for discrete and continuous actions."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from aprenderl.distributions import (
    CategoricalDistribution,
    DiagonalGaussianDistribution,
    SquashedGaussianDistribution,
)
from aprenderl.networks import GaussianPolicyNetwork, PolicyNetwork

PolicyAction = int | np.ndarray
PolicyDistribution = (
    CategoricalDistribution
    | DiagonalGaussianDistribution
    | SquashedGaussianDistribution
)


class PolicyActionSpace:
    """Adapt policy networks and distributions to one Gymnasium action space."""

    def __init__(
        self,
        action_space: gym.Space[Any],
        device: torch.device,
        algorithm_name: str,
    ) -> None:
        self.device = device
        self.low: torch.Tensor | None = None
        self.high: torch.Tensor | None = None
        self.low_array: np.ndarray | None = None
        self.high_array: np.ndarray | None = None

        if isinstance(action_space, gym.spaces.Discrete):
            self.kind = "discrete"
            self.squashed = False
            self.shape: tuple[int, ...] = ()
            self.dim = int(action_space.n)
            self.start = int(action_space.start)
            self.dtype = np.dtype(np.int64)
        elif isinstance(action_space, gym.spaces.Box):
            if action_space.shape is None or not action_space.shape:
                raise ValueError(
                    "continuous action space must have a non-empty shape"
                )
            if not np.issubdtype(action_space.dtype, np.floating):
                raise ValueError("Box action space must use a floating dtype")
            finite_bounds = np.isfinite(action_space.low).all() and np.isfinite(
                action_space.high
            ).all()
            unbounded = np.isneginf(action_space.low).all() and np.isposinf(
                action_space.high
            ).all()
            if not finite_bounds and not unbounded:
                raise ValueError(
                    f"{algorithm_name} requires Box action bounds to be finite "
                    "or fully unbounded"
                )
            if finite_bounds and not np.all(action_space.high > action_space.low):
                raise ValueError(
                    "every upper action bound must exceed its lower bound"
                )
            self.kind = "continuous"
            self.squashed = bool(finite_bounds)
            self.shape = tuple(action_space.shape)
            self.dim = int(np.prod(self.shape))
            self.start = 0
            self.dtype = np.dtype(action_space.dtype)
            self.low_array = np.asarray(action_space.low).copy()
            self.high_array = np.asarray(action_space.high).copy()
            if self.squashed:
                self.low = torch.as_tensor(
                    self.low_array.reshape(-1), dtype=torch.float32, device=device
                )
                self.high = torch.as_tensor(
                    self.high_array.reshape(-1), dtype=torch.float32, device=device
                )
        else:
            raise TypeError(
                f"{algorithm_name} requires a Discrete or Box action space"
            )

    @property
    def continuous(self) -> bool:
        return self.kind == "continuous"

    def default_network(self, observation_dim: int) -> nn.Module:
        """Build the default policy network for this action space."""

        if self.continuous:
            return GaussianPolicyNetwork(observation_dim, self.dim)
        return PolicyNetwork(observation_dim, self.dim)

    def distribution(self, output: Any) -> PolicyDistribution:
        """Construct the matching distribution from policy-network output."""

        if not self.continuous:
            if not isinstance(output, torch.Tensor):
                raise ValueError("discrete policy output must be a tensor")
            return CategoricalDistribution(output)

        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError(
                "continuous policy output must be a (means, log_stds) tuple"
            )
        means, log_stds = output
        if not isinstance(means, torch.Tensor) or not isinstance(
            log_stds, torch.Tensor
        ):
            raise ValueError("continuous policy outputs must be tensors")
        if self.squashed:
            assert self.low is not None and self.high is not None
            return SquashedGaussianDistribution(means, log_stds, self.low, self.high)
        return DiagonalGaussianDistribution(means, log_stds)

    def action_from_tensor(self, action: torch.Tensor) -> PolicyAction:
        """Convert one sampled tensor into the environment's action format."""

        if not self.continuous:
            return int(action.item()) + self.start
        array = action.detach().cpu().numpy().reshape(self.shape)
        if self.squashed:
            assert self.low_array is not None and self.high_array is not None
            array = np.clip(array, self.low_array, self.high_array)
        return array.astype(self.dtype, copy=False)

    def action_for_storage(self, action: PolicyAction) -> int | np.ndarray:
        """Convert one environment action into its policy-relative form."""

        if not self.continuous:
            return int(action) - self.start
        array = np.asarray(action, dtype=np.float32)
        if array.shape != self.shape:
            raise ValueError(f"expected action shape {self.shape}, got {array.shape}")
        return array

    def action_batch_tensor(self, actions: Any) -> torch.Tensor:
        """Convert stored actions into the shape expected by ``log_prob``."""

        if isinstance(actions, torch.Tensor):
            dtype = torch.float32 if self.continuous else torch.int64
            tensor = actions.to(device=self.device, dtype=dtype)
            if self.continuous:
                return tensor.reshape(tensor.shape[0], self.dim)
            return tensor
        if not self.continuous:
            return torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        tensor = torch.as_tensor(
            np.asarray(actions, dtype=np.float32), device=self.device
        )
        return tensor.reshape(tensor.shape[0], self.dim)

    def validate_network(
        self,
        network: nn.Module,
        observation_shape: tuple[int, ...],
        parameter_name: str,
    ) -> None:
        """Validate a custom or default policy against one observation batch."""

        try:
            with torch.no_grad():
                output = network(
                    torch.zeros((1, *observation_shape), device=self.device)
                )
        except Exception as error:
            raise ValueError(
                f"{parameter_name} could not process one environment observation"
            ) from error

        expected_shape = (1, self.dim)
        if not self.continuous:
            if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
                actual_shape = getattr(output, "shape", None)
                raise ValueError(
                    f"{parameter_name} must return shape {expected_shape}, "
                    f"got {actual_shape}"
                )
            return

        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError(
                f"{parameter_name} must return (means, log_stds) for Box actions"
            )
        means, log_stds = output
        if (
            not isinstance(means, torch.Tensor)
            or not isinstance(log_stds, torch.Tensor)
            or means.shape != expected_shape
            or log_stds.shape != expected_shape
        ):
            actual_shapes = tuple(getattr(item, "shape", None) for item in output)
            raise ValueError(
                f"{parameter_name} must return two tensors with shape "
                f"{expected_shape}, got {actual_shapes}"
            )

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a serialization-safe description of the action space."""

        return {
            "kind": self.kind,
            "shape": self.shape,
            "dim": self.dim,
            "start": self.start,
            "squashed": self.squashed if self.continuous else None,
            "low": (
                self.low_array.reshape(-1).tolist()
                if self.low_array is not None
                else None
            ),
            "high": (
                self.high_array.reshape(-1).tolist()
                if self.high_array is not None
                else None
            ),
        }

    def matches_checkpoint(self, state: dict[str, Any]) -> bool:
        """Whether a saved action-space description matches this adapter."""

        expected = self.checkpoint_state()
        saved_squashed = state.get("squashed")
        if saved_squashed is None and state.get("kind") == "continuous":
            # Continuous checkpoints before plain Gaussian support were finite.
            saved_squashed = True
        return (
            state.get("kind") == expected["kind"]
            and tuple(state.get("shape", ())) == expected["shape"]
            and int(state.get("dim", -1)) == expected["dim"]
            and int(state.get("start", 0)) == expected["start"]
            and saved_squashed == expected["squashed"]
            and state.get("low") == expected["low"]
            and state.get("high") == expected["high"]
        )
