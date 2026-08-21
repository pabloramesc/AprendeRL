"""Monte Carlo REINFORCE for discrete and continuous action spaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from typing_extensions import Self

from aprenderl.algorithms.base import OnPolicyAlgorithm
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import ValueNetwork
from aprenderl.policies.action_space import PolicyAction, PolicyActionSpace
from aprenderl.types import Transition
from aprenderl.utils import resolve_device


@dataclass(frozen=True)
class REINFORCEConfig:
    """Hyperparameters for :class:`REINFORCE`."""

    learning_rate: float = 1e-2
    gamma: float = 0.99
    episodes_per_update: int = 5
    use_baseline: bool = False
    value_learning_rate: float = 1e-2
    normalize_returns: bool = True
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 1.0
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.episodes_per_update <= 0:
            raise ValueError("episodes_per_update must be positive")
        if self.value_learning_rate <= 0:
            raise ValueError("value_learning_rate must be positive")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


class REINFORCE(OnPolicyAlgorithm[np.ndarray, PolicyAction]):
    """Episodic policy gradient for discrete or continuous ``Box`` actions.

    The policy is updated only from complete episodes. Each action is weighted
    by its discounted reward-to-go. When ``use_baseline`` is enabled, a learned
    state-value baseline is subtracted from that return. Policy weights can be
    normalized across an update batch to reduce gradient variance.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        value_network: nn.Module | None = None,
        config: REINFORCEConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config or REINFORCEConfig()
        self.device = resolve_device(device)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("REINFORCE requires a Box observation space")
        if env.observation_space.shape is None or not env.observation_space.shape:
            raise ValueError("the observation space must have a non-empty shape")

        self.observation_shape = tuple(env.observation_space.shape)
        self.observation_dim = int(np.prod(self.observation_shape))
        self.policy_action_space = PolicyActionSpace(
            env.action_space, self.device, "REINFORCE"
        )
        self.action_dim = self.policy_action_space.dim
        self.action_shape = self.policy_action_space.shape
        self.action_start = self.policy_action_space.start
        super().__init__(
            env,
            seed=self.config.seed,
            callback=callback,
            logger=logger,
            log_interval=self.config.log_interval,
        )

        self._uses_default_network = network is None
        self.policy_network = (
            network
            if network is not None
            else self.policy_action_space.default_network(self.observation_dim)
        ).to(self.device)
        self._validate_network(self.policy_network)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.config.learning_rate
        )
        if value_network is not None and not self.config.use_baseline:
            raise ValueError("value_network requires use_baseline=True")
        self._uses_default_value_network = self.config.use_baseline and (
            value_network is None
        )
        self.value_network = (
            value_network
            if value_network is not None
            else ValueNetwork(self.observation_dim)
            if self.config.use_baseline
            else None
        )
        if self.value_network is not None:
            self.value_network.to(self.device)
            self._validate_value_network(self.value_network)
            self.value_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
                self.value_network.parameters(), lr=self.config.value_learning_rate
            )
        else:
            self.value_optimizer = None

        self._episode_observations: list[np.ndarray] = []
        self._episode_actions: list[int | np.ndarray] = []
        self._episode_rewards: list[float] = []
        self._batch_observations: list[np.ndarray] = []
        self._batch_actions: list[int | np.ndarray] = []
        self._batch_returns: list[float] = []
        self._batch_episodes = 0

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = False
    ) -> PolicyAction:
        """Choose the modal/mean action or sample from the policy."""

        array = self._as_observation(observation)
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.policy_action_space.distribution(
                self.policy_network(tensor)
            )
            action = distribution.mode() if deterministic else distribution.sample()
        return self.policy_action_space.action_from_tensor(action)

    def save(self, path: str | Path) -> None:
        """Save policy, optimizer, configuration, and training counters."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 3,
                "config": asdict(self.config),
                "uses_default_network": self._uses_default_network,
                "uses_default_value_network": self._uses_default_value_network,
                "observation_shape": self.observation_shape,
                "action_dim": self.action_dim,
                "action_start": self.action_start,
                "action_space": self.policy_action_space.checkpoint_state(),
                "policy_network": self.policy_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "value_network": (
                    self.value_network.state_dict()
                    if self.value_network is not None
                    else None
                ),
                "value_optimizer": (
                    self.value_optimizer.state_dict()
                    if self.value_optimizer is not None
                    else None
                ),
                "num_timesteps": self.num_timesteps,
                "num_updates": self.num_updates,
                "episode_returns": self.episode_returns,
                "episode_lengths": self.episode_lengths,
            },
            checkpoint_path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env[Any, Any],
        **kwargs: Any,
    ) -> Self:
        """Load a checkpoint and validate that it matches ``env``."""

        device = resolve_device(kwargs.pop("device", "auto"))
        network = kwargs.pop("network", None)
        value_network = kwargs.pop("value_network", None)
        callback = kwargs.pop("callback", None)
        logger = kwargs.pop("logger", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {names}")
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # PyTorch before the ``weights_only`` argument.
            checkpoint = torch.load(path, map_location=device)

        if not checkpoint["uses_default_network"] and network is None:
            raise ValueError(
                "loading a custom-network checkpoint requires network=nn.Module"
            )
        if (
            checkpoint.get("value_network") is not None
            and not checkpoint.get("uses_default_value_network", True)
            and value_network is None
        ):
            raise ValueError(
                "loading a custom value-network checkpoint requires "
                "value_network=nn.Module"
            )
        algorithm = cls(
            env,
            network=network,
            value_network=value_network,
            config=REINFORCEConfig(**checkpoint["config"]),
            device=device,
            callback=callback,
            logger=logger,
        )
        observation_matches = (
            tuple(checkpoint["observation_shape"]) == algorithm.observation_shape
        )
        action_state = checkpoint.get("action_space")
        if action_state is None:  # Backward compatibility with discrete checkpoints.
            action_matches = not algorithm.policy_action_space.continuous and (
                int(checkpoint["action_dim"]),
                int(checkpoint["action_start"]),
            ) == (algorithm.action_dim, algorithm.action_start)
        else:
            action_matches = algorithm.policy_action_space.matches_checkpoint(
                action_state
            )
        if not observation_matches or not action_matches:
            raise ValueError("checkpoint spaces do not match environment")

        algorithm.policy_network.load_state_dict(checkpoint["policy_network"])
        algorithm.optimizer.load_state_dict(checkpoint["optimizer"])
        if algorithm.value_network is not None:
            algorithm.value_network.load_state_dict(checkpoint["value_network"])
            assert algorithm.value_optimizer is not None
            algorithm.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        algorithm.num_timesteps = int(checkpoint["num_timesteps"])
        algorithm.num_updates = int(checkpoint["num_updates"])
        algorithm.episode_returns = list(checkpoint["episode_returns"])
        algorithm.episode_lengths = list(checkpoint["episode_lengths"])
        return algorithm

    def _sample_action(self, observation: np.ndarray) -> PolicyAction:
        return self.predict(observation, deterministic=False)

    def _update_from_transition(
        self, transition: Transition[np.ndarray, PolicyAction]
    ) -> dict[str, float | int]:
        self._episode_observations.append(
            np.asarray(transition.observation, dtype=np.float32)
        )
        self._episode_actions.append(
            self.policy_action_space.action_for_storage(transition.action)
        )
        self._episode_rewards.append(transition.reward)
        if not transition.done:
            return {}

        self._finish_episode()
        if self._batch_episodes >= self.config.episodes_per_update:
            return self._update_policy()
        return {}

    def _finish_episode(self) -> None:
        returns = self._discounted_returns(self._episode_rewards)
        self._batch_observations.extend(self._episode_observations)
        self._batch_actions.extend(self._episode_actions)
        self._batch_returns.extend(returns)
        self._batch_episodes += 1
        self._episode_observations.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()

    def _discounted_returns(self, rewards: list[float]) -> list[float]:
        returns = [0.0] * len(rewards)
        reward_to_go = 0.0
        for step in reversed(range(len(rewards))):
            reward_to_go = rewards[step] + self.config.gamma * reward_to_go
            returns[step] = reward_to_go
        return returns

    def _update_policy(self) -> dict[str, float | int]:
        observations = torch.as_tensor(
            np.asarray(self._batch_observations), device=self.device
        )
        actions = self.policy_action_space.action_batch_tensor(self._batch_actions)
        returns = torch.as_tensor(
            self._batch_returns, dtype=torch.float32, device=self.device
        )
        raw_return_mean = returns.mean()
        value_loss: torch.Tensor | None = None
        values: torch.Tensor | None = None
        if self.value_network is not None:
            values = self.value_network(observations)
            policy_weights = returns - values.detach()
            value_loss = nn.functional.mse_loss(values, returns)
        else:
            policy_weights = returns
        raw_weight_mean = policy_weights.mean()
        if self.config.normalize_returns and policy_weights.numel() > 1:
            policy_weights = (policy_weights - raw_weight_mean) / (
                policy_weights.std(unbiased=False) + 1e-8
            )

        distribution = self.policy_action_space.distribution(
            self.policy_network(observations)
        )
        log_probabilities = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        policy_loss = -(log_probabilities * policy_weights).mean()
        policy_objective_loss = (
            policy_loss - self.config.entropy_coefficient * entropy
        )

        self.optimizer.zero_grad()
        policy_objective_loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        value_gradient_norm: torch.Tensor | None = None
        if value_loss is not None:
            assert self.value_network is not None
            assert self.value_optimizer is not None
            self.value_optimizer.zero_grad()
            value_loss.backward()
            value_gradient_norm = nn.utils.clip_grad_norm_(
                self.value_network.parameters(), self.config.max_grad_norm
            )
            self.value_optimizer.step()
        self.num_updates += 1

        metrics: dict[str, float | int] = {
            "train/loss": float(policy_objective_loss.item()),
            "train/policy_loss": float(policy_loss.item()),
            "train/entropy": float(entropy.item()),
            "train/gradient_norm": float(gradient_norm),
            "train/return_mean": float(raw_return_mean.item()),
            "train/updates": self.num_updates,
        }
        if value_loss is not None:
            assert values is not None
            assert value_gradient_norm is not None
            metrics.update(
                {
                    "train/value_loss": float(value_loss.item()),
                    "train/value_mean": float(values.detach().mean().item()),
                    "train/advantage_mean": float(raw_weight_mean.item()),
                    "train/value_gradient_norm": float(value_gradient_norm),
                }
            )
        self._batch_observations.clear()
        self._batch_actions.clear()
        self._batch_returns.clear()
        self._batch_episodes = 0
        return metrics

    def _progress_metrics(
        self,
        training_metrics: Mapping[str, float | int] | None = None,
    ) -> dict[str, str | int]:
        metrics = super()._progress_metrics(training_metrics)
        if training_metrics and "train/entropy" in training_metrics:
            metrics["entropy"] = f"{float(training_metrics['train/entropy']):.3f}"
        return metrics

    def _validate_network(self, network: nn.Module) -> None:
        self.policy_action_space.validate_network(
            network, self.observation_shape, "network"
        )

    def _validate_value_network(self, network: nn.Module) -> None:
        try:
            with torch.no_grad():
                output = network(
                    torch.zeros((1, *self.observation_shape), device=self.device)
                )
        except Exception as error:
            raise ValueError(
                "value_network could not process one environment observation"
            ) from error
        expected_shape = (1,)
        if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
            actual_shape = getattr(output, "shape", None)
            raise ValueError(
                f"value_network must return shape {expected_shape}, got {actual_shape}"
            )

    def _as_observation(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != self.observation_shape:
            raise ValueError(
                f"expected observation shape {self.observation_shape}, "
                f"got {array.shape}"
            )
        return array
