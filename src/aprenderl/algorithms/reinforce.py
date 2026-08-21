"""Monte Carlo REINFORCE for discrete action spaces."""

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
from aprenderl.distributions import CategoricalDistribution
from aprenderl.logging import TrainingLogger
from aprenderl.networks import PolicyNetwork
from aprenderl.types import Transition
from aprenderl.utils import resolve_device


@dataclass(frozen=True)
class REINFORCEConfig:
    """Hyperparameters for :class:`REINFORCE`."""

    learning_rate: float = 1e-2
    gamma: float = 0.99
    episodes_per_update: int = 5
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
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


class REINFORCE(OnPolicyAlgorithm[np.ndarray, int]):
    """Vanilla episodic policy gradient for discrete actions.

    The policy is updated only from complete episodes. Each action is weighted
    by its discounted reward-to-go, and returns can be normalized across an
    update batch to reduce gradient variance. A custom policy network must map
    a batch of observations to one logit per action.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: REINFORCEConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config or REINFORCEConfig()
        self.device = resolve_device(device)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("REINFORCE requires a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("REINFORCE requires a Discrete action space")
        if env.observation_space.shape is None or not env.observation_space.shape:
            raise ValueError("the observation space must have a non-empty shape")

        self.observation_shape = tuple(env.observation_space.shape)
        self.observation_dim = int(np.prod(self.observation_shape))
        self.action_dim = int(env.action_space.n)
        self.action_start = int(env.action_space.start)
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
            else PolicyNetwork(self.observation_dim, self.action_dim)
        ).to(self.device)
        self._validate_network(self.policy_network)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.config.learning_rate
        )

        self._episode_observations: list[np.ndarray] = []
        self._episode_actions: list[int] = []
        self._episode_rewards: list[float] = []
        self._batch_observations: list[np.ndarray] = []
        self._batch_actions: list[int] = []
        self._batch_returns: list[float] = []
        self._batch_episodes = 0

    def predict(self, observation: np.ndarray, *, deterministic: bool = False) -> int:
        """Choose the modal action or sample from the categorical policy."""

        array = self._as_observation(observation)
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution = CategoricalDistribution(self.policy_network(tensor))
            action = distribution.mode() if deterministic else distribution.sample()
        return int(action.item()) + self.action_start

    def save(self, path: str | Path) -> None:
        """Save policy, optimizer, configuration, and training counters."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "config": asdict(self.config),
                "uses_default_network": self._uses_default_network,
                "observation_shape": self.observation_shape,
                "action_dim": self.action_dim,
                "action_start": self.action_start,
                "policy_network": self.policy_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
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
        algorithm = cls(
            env,
            network=network,
            config=REINFORCEConfig(**checkpoint["config"]),
            device=device,
            callback=callback,
            logger=logger,
        )
        checkpoint_space = (
            tuple(checkpoint["observation_shape"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["action_start"]),
        )
        environment_space = (
            algorithm.observation_shape,
            algorithm.action_dim,
            algorithm.action_start,
        )
        if checkpoint_space != environment_space:
            raise ValueError("checkpoint spaces do not match environment")

        algorithm.policy_network.load_state_dict(checkpoint["policy_network"])
        algorithm.optimizer.load_state_dict(checkpoint["optimizer"])
        algorithm.num_timesteps = int(checkpoint["num_timesteps"])
        algorithm.num_updates = int(checkpoint["num_updates"])
        algorithm.episode_returns = list(checkpoint["episode_returns"])
        algorithm.episode_lengths = list(checkpoint["episode_lengths"])
        return algorithm

    def _sample_action(self, observation: np.ndarray) -> int:
        return self.predict(observation, deterministic=False)

    def _update_from_transition(
        self, transition: Transition[np.ndarray, int]
    ) -> dict[str, float | int]:
        self._episode_observations.append(
            np.asarray(transition.observation, dtype=np.float32)
        )
        self._episode_actions.append(transition.action - self.action_start)
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
        actions = torch.as_tensor(
            self._batch_actions, dtype=torch.int64, device=self.device
        )
        returns = torch.as_tensor(
            self._batch_returns, dtype=torch.float32, device=self.device
        )
        raw_return_mean = returns.mean()
        if self.config.normalize_returns and returns.numel() > 1:
            returns = (returns - raw_return_mean) / (returns.std(unbiased=False) + 1e-8)

        distribution = CategoricalDistribution(self.policy_network(observations))
        log_probabilities = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        policy_loss = -(log_probabilities * returns).mean()
        loss = policy_loss - self.config.entropy_coefficient * entropy

        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.num_updates += 1

        metrics: dict[str, float | int] = {
            "train/loss": float(loss.item()),
            "train/policy_loss": float(policy_loss.item()),
            "train/entropy": float(entropy.item()),
            "train/gradient_norm": float(gradient_norm),
            "train/return_mean": float(raw_return_mean.item()),
            "train/updates": self.num_updates,
        }
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
        try:
            with torch.no_grad():
                output = network(
                    torch.zeros((1, *self.observation_shape), device=self.device)
                )
        except Exception as error:
            raise ValueError(
                "network could not process one environment observation"
            ) from error
        expected_shape = (1, self.action_dim)
        if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
            actual_shape = getattr(output, "shape", None)
            raise ValueError(
                f"network must return shape {expected_shape}, got {actual_shape}"
            )

    def _as_observation(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != self.observation_shape:
            raise ValueError(
                f"expected observation shape {self.observation_shape}, "
                f"got {array.shape}"
            )
        return array
