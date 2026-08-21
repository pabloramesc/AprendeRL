"""One-step actor-critic for discrete action spaces."""

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
from aprenderl.buffers import RolloutBuffer
from aprenderl.callbacks import BaseCallback
from aprenderl.distributions import CategoricalDistribution
from aprenderl.logging import TrainingLogger
from aprenderl.networks import PolicyNetwork, ValueNetwork
from aprenderl.types import Transition
from aprenderl.utils import resolve_device


@dataclass(frozen=True)
class ActorCriticConfig:
    """Hyperparameters for :class:`ActorCritic`."""

    learning_rate: float = 1e-3
    value_learning_rate: float = 1e-3
    gamma: float = 0.99
    n_steps: int = 32
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 1.0
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.value_learning_rate <= 0:
            raise ValueError("value_learning_rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


class ActorCritic(OnPolicyAlgorithm[np.ndarray, int]):
    """One-step actor-critic for discrete actions.

    The actor is a categorical policy and the critic estimates state values.
    Each small rollout is updated using one-step TD targets, and the actor is
    weighted by the detached TD errors. Custom networks must map observation
    batches to policy logits and scalar values, respectively.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        policy_network: nn.Module | None = None,
        *,
        value_network: nn.Module | None = None,
        config: ActorCriticConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config or ActorCriticConfig()
        self.device = resolve_device(device)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("ActorCritic requires a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("ActorCritic requires a Discrete action space")
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

        self._uses_default_policy_network = policy_network is None
        self.policy_network = (
            policy_network
            if policy_network is not None
            else PolicyNetwork(self.observation_dim, self.action_dim)
        ).to(self.device)
        self._validate_policy_network(self.policy_network)

        self._uses_default_value_network = value_network is None
        self.value_network = (
            value_network
            if value_network is not None
            else ValueNetwork(self.observation_dim)
        ).to(self.device)
        self._validate_value_network(self.value_network)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=self.config.value_learning_rate
        )
        self.rollout_buffer = RolloutBuffer(
            self.config.n_steps,
            self.observation_shape,
            gamma=self.config.gamma,
            gae_lambda=0.0,
        )

    def predict(self, observation: np.ndarray, *, deterministic: bool = False) -> int:
        """Choose the modal action or sample from the categorical actor."""

        array = self._as_observation(observation)
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution = CategoricalDistribution(self.policy_network(tensor))
            action = distribution.mode() if deterministic else distribution.sample()
        return int(action.item()) + self.action_start

    def save(self, path: str | Path) -> None:
        """Save both networks, optimizers, configuration, and training counters."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "config": asdict(self.config),
                "uses_default_policy_network": self._uses_default_policy_network,
                "uses_default_value_network": self._uses_default_value_network,
                "observation_shape": self.observation_shape,
                "action_dim": self.action_dim,
                "action_start": self.action_start,
                "policy_network": self.policy_network.state_dict(),
                "value_network": self.value_network.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
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
        policy_network = kwargs.pop("policy_network", None)
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

        if not checkpoint["uses_default_policy_network"] and policy_network is None:
            raise ValueError(
                "loading a custom-policy checkpoint requires policy_network=nn.Module"
            )
        if not checkpoint["uses_default_value_network"] and value_network is None:
            raise ValueError(
                "loading a custom-value checkpoint requires value_network=nn.Module"
            )
        algorithm = cls(
            env,
            policy_network=policy_network,
            value_network=value_network,
            config=ActorCriticConfig(**checkpoint["config"]),
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
        algorithm.value_network.load_state_dict(checkpoint["value_network"])
        algorithm.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        algorithm.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
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
        observation = self._as_observation(transition.observation)
        observation_tensor = torch.as_tensor(
            observation, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            value = self.value_network(observation_tensor).item()
            next_value = self._next_value(transition).item()
            distribution = CategoricalDistribution(
                self.policy_network(observation_tensor)
            )
            action = torch.tensor(
                [transition.action - self.action_start],
                dtype=torch.int64,
                device=self.device,
            )
            log_probability = distribution.log_prob(action).item()
        self.rollout_buffer.add(
            observation,
            transition.action - self.action_start,
            transition.reward,
            transition.terminated,
            transition.truncated,
            value,
            next_value,
            log_probability,
        )
        if not self.rollout_buffer.full:
            return {}
        self.rollout_buffer.compute_returns_and_advantages()
        metrics = self._train_step()
        self.rollout_buffer.reset()
        return metrics

    def _train_step(self) -> dict[str, float | int]:
        batch = self.rollout_buffer.batch(self.device)
        values = self.value_network(batch.observations)

        distribution = CategoricalDistribution(self.policy_network(batch.observations))
        log_probabilities = distribution.log_prob(batch.actions.to(torch.int64))
        entropy = distribution.entropy().mean()
        policy_loss = -(log_probabilities * batch.advantages.detach()).mean()
        policy_objective_loss = (
            policy_loss - self.config.entropy_coefficient * entropy
        )
        value_loss = nn.functional.mse_loss(values, batch.returns)

        self.policy_optimizer.zero_grad()
        policy_objective_loss.backward()
        policy_gradient_norm = nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.config.max_grad_norm
        )
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        value_gradient_norm = nn.utils.clip_grad_norm_(
            self.value_network.parameters(), self.config.max_grad_norm
        )
        self.value_optimizer.step()
        self.num_updates += 1

        metrics: dict[str, float | int] = {
            "train/loss": float(policy_objective_loss.item() + value_loss.item()),
            "train/policy_loss": float(policy_loss.item()),
            "train/value_loss": float(value_loss.item()),
            "train/entropy": float(entropy.item()),
            "train/advantage": float(batch.advantages.mean().item()),
            "train/value": float(values.detach().mean().item()),
            "train/td_target": float(batch.returns.mean().item()),
            "train/policy_gradient_norm": float(policy_gradient_norm),
            "train/value_gradient_norm": float(value_gradient_norm),
            "train/updates": self.num_updates,
        }
        return metrics

    @torch.no_grad()
    def _td_target(self, transition: Transition[np.ndarray, int]) -> torch.Tensor:
        reward = torch.tensor(
            [transition.reward], dtype=torch.float32, device=self.device
        )
        return reward + self.config.gamma * self._next_value(transition)

    @torch.no_grad()
    def _next_value(self, transition: Transition[np.ndarray, int]) -> torch.Tensor:
        if transition.terminated:
            return torch.zeros(1, dtype=torch.float32, device=self.device)
        next_observation = torch.as_tensor(
            self._as_observation(transition.next_observation), device=self.device
        ).unsqueeze(0)
        return self.value_network(next_observation)

    def _progress_metrics(
        self,
        training_metrics: Mapping[str, float | int] | None = None,
    ) -> dict[str, str | int]:
        metrics = super()._progress_metrics(training_metrics)
        if training_metrics and "train/entropy" in training_metrics:
            metrics["entropy"] = f"{float(training_metrics['train/entropy']):.3f}"
        return metrics

    def _validate_policy_network(self, network: nn.Module) -> None:
        try:
            with torch.no_grad():
                output = network(
                    torch.zeros((1, *self.observation_shape), device=self.device)
                )
        except Exception as error:
            raise ValueError(
                "policy_network could not process one environment observation"
            ) from error
        expected_shape = (1, self.action_dim)
        if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
            actual_shape = getattr(output, "shape", None)
            raise ValueError(
                f"policy_network must return shape {expected_shape}, "
                f"got {actual_shape}"
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
