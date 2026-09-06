"""Replay, network construction, and persistence for off-policy actor-critics."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from typing_extensions import Self

from aprenderl.algorithms.base import OffPolicyAlgorithm, TrainingMetrics
from aprenderl.algorithms.policy_gradient import PolicyGradientAlgorithm
from aprenderl.buffers import ReplayBuffer
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import PolicyNetwork, QNetwork
from aprenderl.networks.continuous import (
    ContinuousQNetwork,
    DeterministicPolicyNetwork,
    SACPolicyNetwork,
)
from aprenderl.types import Transition
from aprenderl.utils import load_torch_checkpoint, resolve_device


@dataclass(frozen=True)
class OffPolicyActorCriticConfig:
    """Common replay and optimization settings for a single environment."""

    learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 100_000
    batch_size: int = 128
    learning_starts: int = 1_000
    train_freq: int = 1
    gradient_steps: int = 1
    max_grad_norm: float = 10.0
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "buffer_size",
            "batch_size",
            "train_freq",
            "gradient_steps",
            "log_interval",
            "learning_starts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name == "learning_starts" else 1):
                raise ValueError(f"invalid {name}")
        for name in ("learning_rate", "critic_learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size cannot exceed buffer_size")


class OffPolicyActorCritic(PolicyGradientAlgorithm, OffPolicyAlgorithm):
    """Share auxiliary plumbing while keeping each learning rule local."""

    config_type = OffPolicyActorCriticConfig
    continuous = True
    stochastic = False
    n_critics = 1

    def __init__(
        self,
        env: gym.Env[Any, Any],
        policy_network: nn.Module | None = None,
        *,
        critic_networks: Sequence[nn.Module] | None = None,
        config: OffPolicyActorCriticConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        action_space = env.action_space
        if self.continuous:
            if not isinstance(action_space, gym.spaces.Box):
                raise TypeError(f"{type(self).__name__} requires a Box action space")
            if not (
                np.isfinite(action_space.low).all()
                and np.isfinite(action_space.high).all()
            ):
                raise ValueError("continuous off-policy methods require finite bounds")
        elif not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError(f"{type(self).__name__} requires a Discrete action space")
        super().__init__(
            env,
            policy_network,
            config=config or self.config_type(),
            device=device,
            callback=callback,
            logger=logger,
            policy_network_name="policy_network",
        )
        self._rng = np.random.default_rng(self.config.seed)
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size,
            self.observation_shape,
            seed=self.config.seed,
            action_shape=(self.action_dim,) if self.continuous else None,
        )
        self._uses_default_critics = critic_networks is None
        if critic_networks is None:
            factory = ContinuousQNetwork if self.continuous else QNetwork
            critic_networks = [
                factory(self.observation_dim, self.action_dim)
                for _ in range(self.n_critics)
            ]
        if len(critic_networks) != self.n_critics:
            raise ValueError(f"critic_networks must contain {self.n_critics} modules")
        self.critic_networks = nn.ModuleList(critic_networks).to(self.device).eval()
        self._validate_critics()
        # Disjoint parameters avoid stepping the same shared encoder twice.
        seen = {id(p) for p in self.policy_network.parameters()}
        for critic in self.critic_networks:
            current = {id(p) for p in critic.parameters()}
            if seen & current:
                raise ValueError("actor and critics must have disjoint parameters")
            seen |= current
        self.target_critics = copy.deepcopy(self.critic_networks)
        self.target_critics.eval().requires_grad_(False)
        self.target_policy_network = None
        if not self.stochastic:
            self.target_policy_network = copy.deepcopy(self.policy_network)
            self.target_policy_network.eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_networks.parameters(), lr=self.config.critic_learning_rate
        )

    def _default_policy_network(self) -> nn.Module:
        if not self.continuous:
            factory = PolicyNetwork
        elif self.stochastic:
            factory = SACPolicyNetwork
        else:
            factory = DeterministicPolicyNetwork
        return factory(self.observation_dim, self.action_dim)

    def _validate_policy_network(self, name: str) -> None:
        # Fixed evaluation mode keeps dropout/batchnorm custom nets predictable,
        # including this single-observation validation. Autograd stays enabled.
        self.policy_network.eval()
        if self.stochastic:
            super()._validate_policy_network(name)
            return
        with torch.no_grad():
            try:
                output = self.policy_network(
                    torch.zeros((1, *self.observation_shape), device=self.device)
                )
            except Exception as error:
                raise ValueError(
                    "policy_network cannot process observations"
                ) from error
        if (
            not isinstance(output, torch.Tensor)
            or output.shape != (1, self.action_dim)
            or not torch.isfinite(output).all()
            or (output.abs() > 1).any()
        ):
            raise ValueError("policy_network must return (batch, actions) in [-1, 1]")

    def _validate_critics(self) -> None:
        observations = torch.zeros((1, *self.observation_shape), device=self.device)
        actions = torch.zeros((1, self.action_dim), device=self.device)
        for critic in self.critic_networks:
            try:
                with torch.no_grad():
                    output = (
                        critic(observations, actions)
                        if self.continuous
                        else critic(observations)
                    )
            except Exception as error:
                raise ValueError(
                    "critic_networks cannot process observations/actions"
                ) from error
            expected = (1, 1 if self.continuous else self.action_dim)
            if not isinstance(output, torch.Tensor) or output.shape != expected:
                raise ValueError(f"each critic must return shape {expected}")

    def _scaled_action(
        self, network: nn.Module, observations: torch.Tensor
    ) -> torch.Tensor:
        low, high = self.policy_action_space.low, self.policy_action_space.high
        return (high + low) / 2 + (high - low) / 2 * network(observations).clamp(-1, 1)

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = False
    ) -> int | np.ndarray:
        if self.stochastic:
            return super().predict(observation, deterministic=deterministic)
        observations = torch.as_tensor(
            self._as_observation(observation), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self._scaled_action(self.policy_network, observations)
            if not deterministic:
                noise = self._rng.normal(0, self.config.action_noise, action.shape)
                scale = (
                    self.policy_action_space.high - self.policy_action_space.low
                ) / 2
                action = (
                    action
                    + torch.as_tensor(noise, dtype=action.dtype, device=self.device)
                    * scale
                )
        return self.policy_action_space.action_from_tensor(action)

    def _sample_action(self, observation: np.ndarray) -> int | np.ndarray:
        if self.num_timesteps < self.config.learning_starts:
            return self.env.action_space.sample()
        return self.predict(observation)

    def _update_from_transition(self, transition: Transition) -> TrainingMetrics:
        action = self.policy_action_space.action_for_storage(transition.action)
        if self.continuous:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.replay_buffer.add(
            self._as_observation(transition.observation),
            action,
            transition.reward,
            self._as_observation(transition.next_observation),
            transition.terminated,
            transition.truncated,
        )
        step = self.num_timesteps + 1
        metrics = {}
        if (
            step >= self.config.learning_starts
            and step % self.config.train_freq == 0
            and len(self.replay_buffer) >= self.config.batch_size
        ):
            for _ in range(self.config.gradient_steps):
                metrics = self._train_step()
        return metrics

    def _train_step(self) -> TrainingMetrics:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Save networks, optimizers and counters; replay/environment are omitted."""
        checkpoint = {
            "version": 1,
            "algorithm": type(self).__name__,
            "config": asdict(self.config),
            "observation_shape": self.observation_shape,
            "action_space": self.policy_action_space.checkpoint_state(),
            "uses_default_policy": self._uses_default_policy_network,
            "uses_default_critics": self._uses_default_critics,
            "policy": self.policy_network.state_dict(),
            "critics": self.critic_networks.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "target_policy": (
                self.target_policy_network.state_dict()
                if self.target_policy_network is not None
                else None
            ),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "rng": self._rng.bit_generator.state,
            "extra": self._checkpoint_extra(),
            **self._training_state(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path, env: gym.Env[Any, Any], **kwargs: Any) -> Self:
        device = resolve_device(kwargs.pop("device", "auto"))
        checkpoint = load_torch_checkpoint(path, device)
        if checkpoint["algorithm"] != cls.__name__ or checkpoint["version"] != 1:
            raise ValueError("checkpoint algorithm or version does not match")
        policy = kwargs.pop("policy_network", None)
        critics = kwargs.pop("critic_networks", None)
        callback, logger = kwargs.pop("callback", None), kwargs.pop("logger", None)
        cls._reject_unknown_arguments(kwargs)
        if not checkpoint["uses_default_policy"] and policy is None:
            raise ValueError("custom checkpoint requires policy_network")
        if not checkpoint["uses_default_critics"] and critics is None:
            raise ValueError("custom checkpoint requires critic_networks")
        agent = cls(
            env,
            policy,
            critic_networks=critics,
            config=cls.config_type(**checkpoint["config"]),
            device=device,
            callback=callback,
            logger=logger,
        )
        if not agent._checkpoint_spaces_match(checkpoint):
            raise ValueError("checkpoint spaces do not match environment")
        agent.policy_network.load_state_dict(checkpoint["policy"])
        agent.critic_networks.load_state_dict(checkpoint["critics"])
        agent.target_critics.load_state_dict(checkpoint["target_critics"])
        if agent.target_policy_network is not None:
            agent.target_policy_network.load_state_dict(checkpoint["target_policy"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        agent._restore_training_state(checkpoint)
        agent._rng.bit_generator.state = checkpoint["rng"]
        agent._restore_extra(checkpoint["extra"])
        return agent

    def _checkpoint_extra(self) -> dict[str, Any]:
        return {}

    def _restore_extra(self, state: dict[str, Any]) -> None:
        del state


@contextmanager
def frozen_parameters(module: nn.Module):
    """Freeze critic weights while retaining gradients with respect to actions."""
    parameters = list(module.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags, strict=True):
            parameter.requires_grad_(flag)


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Average parameters and copy non-parameter state such as running stats."""
    for current, delayed in zip(source.parameters(), target.parameters(), strict=True):
        delayed.lerp_(current, tau)
    for current, delayed in zip(source.buffers(), target.buffers(), strict=True):
        delayed.copy_(current)
