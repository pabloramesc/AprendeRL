"""Shared neural-policy infrastructure for policy-gradient algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from typing_extensions import Self

from aprenderl.algorithms.base import BaseAlgorithm
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.policies.action_space import PolicyAction, PolicyActionSpace
from aprenderl.utils import resolve_device


class PolicyGradientConfig(Protocol):
    """Configuration fields needed by the shared policy infrastructure."""

    learning_rate: float
    log_interval: int
    seed: int | None


class PolicyGradientAlgorithm(BaseAlgorithm[np.ndarray, PolicyAction]):
    """Neural-policy plumbing independent of on-policy/off-policy classification."""

    def __init__(
        self,
        env: gym.Env[Any, Any],
        policy_network: nn.Module | None,
        *,
        config: PolicyGradientConfig,
        device: str | torch.device,
        callback: BaseCallback | list[BaseCallback] | None,
        logger: TrainingLogger | None,
        policy_network_name: str,
    ) -> None:
        self.config = config
        self.device = resolve_device(device)
        algorithm_name = type(self).__name__
        observation_space = env.observation_space
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError(f"{algorithm_name} requires a Box observation space")
        if not observation_space.shape:
            raise ValueError("the observation space must have a non-empty shape")

        self.observation_shape = tuple(observation_space.shape)
        self.observation_dim = int(np.prod(self.observation_shape))
        self.policy_action_space = PolicyActionSpace(
            env.action_space, self.device, algorithm_name
        )
        self.action_dim = self.policy_action_space.dim
        self.action_shape = self.policy_action_space.shape
        self.action_start = self.policy_action_space.start
        super().__init__(
            env,
            seed=config.seed,
            callback=callback,
            logger=logger,
            log_interval=config.log_interval,
        )

        self._uses_default_policy_network = policy_network is None
        self.policy_network = (
            policy_network
            if policy_network is not None
            else self._default_policy_network()
        ).to(self.device)
        self._validate_policy_network(policy_network_name)

    def learn(self, total_timesteps: int, *, progress_bar: bool = True) -> Self:
        """Run the shared interaction loop with the concrete learning rule."""

        return self._learn_online(total_timesteps, progress_bar=progress_bar)

    def _default_policy_network(self) -> nn.Module:
        return self.policy_action_space.default_network(self.observation_dim)

    def _validate_policy_network(self, name: str) -> None:
        self.policy_action_space.validate_network(
            self.policy_network, self.observation_shape, name
        )

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = False
    ) -> PolicyAction:
        """Return the policy mode/mean or sample a stochastic action."""

        observation_tensor = torch.as_tensor(
            self._as_observation(observation), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            distribution = self.policy_action_space.distribution(
                self.policy_network(observation_tensor)
            )
            action = distribution.mode() if deterministic else distribution.sample()
        return self.policy_action_space.action_from_tensor(action)

    def _as_observation(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != self.observation_shape:
            raise ValueError(
                f"expected observation shape {self.observation_shape}, "
                f"got {array.shape}"
            )
        return array

    def _validate_value_network(self, network: nn.Module) -> None:
        sample = torch.zeros((1, *self.observation_shape), device=self.device)
        try:
            with torch.no_grad():
                output = network(sample)
        except Exception as error:
            raise ValueError(
                "value_network could not process one environment observation"
            ) from error

        expected_shape = (1,)
        actual_shape = getattr(output, "shape", None)
        if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
            raise ValueError(
                f"value_network must return shape {expected_shape}, got {actual_shape}"
            )

    def _checkpoint_spaces_match(self, checkpoint: dict[str, Any]) -> bool:
        if tuple(checkpoint["observation_shape"]) != self.observation_shape:
            return False
        action_state = checkpoint.get("action_space")
        if action_state is not None:
            return self.policy_action_space.matches_checkpoint(action_state)
        return not self.policy_action_space.continuous and (
            int(checkpoint["action_dim"]),
            int(checkpoint["action_start"]),
        ) == (self.action_dim, self.action_start)

    def _progress_metrics(
        self,
        training_metrics: Mapping[str, float | int] | None = None,
    ) -> dict[str, str | int]:
        metrics = super()._progress_metrics(training_metrics)
        if training_metrics and "train/entropy" in training_metrics:
            metrics["entropy"] = f"{float(training_metrics['train/entropy']):.3f}"
        return metrics
