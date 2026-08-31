"""Categorical, quantile-regression, implicit-quantile, and Rainbow DQN."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from aprenderl.algorithms.dqn import DQN, DQNConfig
from aprenderl.buffers import PrioritizedReplayBuffer, ReplayBatch
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import (
    CategoricalQNetwork,
    ImplicitQuantileNetwork,
    QuantileQNetwork,
    reset_noise,
)
from aprenderl.policies import EpsilonGreedyPolicy, LinearSchedule
from aprenderl.types import Transition


@dataclass(frozen=True)
class C51Config(DQNConfig):
    """DQN settings plus the fixed categorical value support."""

    atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.atoms <= 1:
            raise ValueError("atoms must be greater than one")
        if self.v_min >= self.v_max:
            raise ValueError("v_min must be less than v_max")


class C51(DQN):
    """Categorical DQN projecting Bellman targets onto a fixed support."""

    config_type = C51Config

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: C51Config | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        actual_config = config or C51Config()
        provided_network = network is not None
        if network is None:
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise TypeError("C51 requires a Box observation space")
            if not isinstance(env.action_space, gym.spaces.Discrete):
                raise TypeError("C51 requires a Discrete action space")
            network = CategoricalQNetwork(
                int(np.prod(env.observation_space.shape)),
                int(env.action_space.n),
                actual_config.atoms,
            )
        super().__init__(
            env,
            network,
            config=actual_config,
            device=device,
            callback=callback,
            logger=logger,
        )
        self._uses_default_network = not provided_network
        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atoms,
            device=self.device,
        )

    def predict(self, observation: np.ndarray, *, deterministic: bool = False) -> int:
        array = self._as_observation(observation)
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self._q_values(self.q_network, tensor).squeeze(0)
        action = self.exploration.select(
            q_values,
            step=self.num_timesteps,
            deterministic=deterministic,
        )
        return action + self.action_start

    def _train_step(self) -> dict[str, float | int]:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        target_distribution = self._project_distribution(batch)
        logits = self.q_network(batch.observations)
        action_indices = batch.actions.unsqueeze(-1).expand(
            -1, 1, self.config.atoms
        )
        chosen_logits = logits.gather(1, action_indices).squeeze(1)
        element_loss = -(
            target_distribution * chosen_logits.log_softmax(dim=-1)
        ).sum(dim=-1)
        loss = element_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.num_updates += 1
        return {
            "train/loss": float(loss.item()),
            "train/gradient_norm": float(gradient_norm),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }

    @torch.no_grad()
    def _project_distribution(self, batch: ReplayBatch) -> torch.Tensor:
        chosen_probabilities = self._next_action_probabilities(batch)
        gamma = self.config.gamma
        discounts = getattr(batch, "discounts", None)
        if discounts is not None:
            gamma = discounts
        target_atoms = batch.rewards + gamma * (
            1 - batch.terminated
        ) * self.support.unsqueeze(0)
        return self._project_atoms(target_atoms, chosen_probabilities)

    def _next_action_probabilities(self, batch: ReplayBatch) -> torch.Tensor:
        next_probabilities = self._probabilities(
            self.target_network, batch.next_observations
        )
        next_values = (next_probabilities * self.support).sum(dim=-1)
        next_actions = next_values.argmax(dim=1)
        row_indices = torch.arange(batch.rewards.shape[0], device=self.device)
        return next_probabilities[row_indices, next_actions]

    def _project_atoms(
        self,
        target_atoms: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        target_atoms.clamp_(self.config.v_min, self.config.v_max)
        atom_width = (self.config.v_max - self.config.v_min) / (
            self.config.atoms - 1
        )
        locations = (target_atoms - self.config.v_min) / atom_width
        lower = locations.floor().long()
        upper = locations.ceil().long()
        projection = torch.zeros_like(probabilities)
        offset = (
            torch.arange(target_atoms.shape[0], device=self.device)
            * self.config.atoms
        ).unsqueeze(1)
        same_bin = lower == upper
        lower_mass = probabilities * (
            upper.float() - locations + same_bin.float()
        )
        upper_mass = probabilities * (locations - lower.float())
        projection.view(-1).index_add_(
            0, (lower + offset).view(-1), lower_mass.view(-1)
        )
        projection.view(-1).index_add_(
            0, (upper + offset).view(-1), upper_mass.view(-1)
        )
        return projection

    def _q_values(self, network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        probabilities = self._probabilities(network, observations)
        return (probabilities * self.support).sum(dim=-1)

    @staticmethod
    def _probabilities(
        network: nn.Module, observations: torch.Tensor
    ) -> torch.Tensor:
        return network(observations).softmax(dim=-1)

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
        expected = (1, self.action_dim, self.config.atoms)
        if not isinstance(output, torch.Tensor) or output.shape != expected:
            actual = getattr(output, "shape", None)
            raise ValueError(f"network must return shape {expected}, got {actual}")


@dataclass(frozen=True)
class QRDQNConfig(DQNConfig):
    """DQN settings for quantile-regression distributional learning."""

    quantiles: int = 51
    huber_threshold: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantiles <= 1:
            raise ValueError("quantiles must be greater than one")
        if self.huber_threshold <= 0:
            raise ValueError("huber_threshold must be positive")


class QRDQN(DQN):
    """Quantile Regression DQN with a fixed uniform quantile grid."""

    config_type = QRDQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: QRDQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        actual_config = config or QRDQNConfig()
        provided_network = network is not None
        if network is None:
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise TypeError("QRDQN requires a Box observation space")
            if not isinstance(env.action_space, gym.spaces.Discrete):
                raise TypeError("QRDQN requires a Discrete action space")
            network = QuantileQNetwork(
                int(np.prod(env.observation_space.shape)),
                int(env.action_space.n),
                actual_config.quantiles,
            )
        super().__init__(
            env,
            network,
            config=actual_config,
            device=device,
            callback=callback,
            logger=logger,
        )
        self._uses_default_network = not provided_network

    def predict(self, observation: np.ndarray, *, deterministic: bool = False) -> int:
        array = self._as_observation(observation)
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(tensor).mean(dim=-1).squeeze(0)
        action = self.exploration.select(
            q_values,
            step=self.num_timesteps,
            deterministic=deterministic,
        )
        return action + self.action_start

    def _train_step(self) -> dict[str, float | int]:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        with torch.no_grad():
            online_next = self.q_network(batch.next_observations).mean(dim=-1)
            next_actions = online_next.argmax(dim=1)
            target_quantiles = self.target_network(batch.next_observations)
            rows = torch.arange(self.config.batch_size, device=self.device)
            target_quantiles = target_quantiles[rows, next_actions]
            target_quantiles = batch.rewards + self.config.gamma * (
                1 - batch.terminated
            ) * target_quantiles
        predicted = self.q_network(batch.observations)
        action_indices = batch.actions.unsqueeze(-1).expand(
            -1, 1, self.config.quantiles
        )
        predicted = predicted.gather(1, action_indices).squeeze(1)
        taus = (
            torch.arange(self.config.quantiles, device=self.device) + 0.5
        ) / self.config.quantiles
        loss = _quantile_huber_loss(
            predicted,
            target_quantiles,
            taus,
            self.config.huber_threshold,
        )
        return self._optimize_distributional(loss)

    def _optimize_distributional(
        self, loss: torch.Tensor
    ) -> dict[str, float | int]:
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.num_updates += 1
        return {
            "train/loss": float(loss.item()),
            "train/gradient_norm": float(gradient_norm),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }

    def _validate_network(self, network: nn.Module) -> None:
        with torch.no_grad():
            output = network(
                torch.zeros((1, *self.observation_shape), device=self.device)
            )
        expected = (1, self.action_dim, self.config.quantiles)
        if not isinstance(output, torch.Tensor) or output.shape != expected:
            actual = getattr(output, "shape", None)
            raise ValueError(f"network must return shape {expected}, got {actual}")


@dataclass(frozen=True)
class IQNConfig(DQNConfig):
    """DQN settings for implicit quantile sampling."""

    quantiles: int = 32
    target_quantiles: int = 32
    embedding_dim: int = 64
    huber_threshold: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantiles <= 0 or self.target_quantiles <= 0:
            raise ValueError("quantile counts must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.huber_threshold <= 0:
            raise ValueError("huber_threshold must be positive")


class IQN(DQN):
    """Implicit Quantile Network with freshly sampled fractions per update."""

    config_type = IQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: IQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        actual_config = config or IQNConfig()
        provided_network = network is not None
        if network is None:
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise TypeError("IQN requires a Box observation space")
            if not isinstance(env.action_space, gym.spaces.Discrete):
                raise TypeError("IQN requires a Discrete action space")
            network = ImplicitQuantileNetwork(
                int(np.prod(env.observation_space.shape)),
                int(env.action_space.n),
                embedding_dim=actual_config.embedding_dim,
            )
        super().__init__(
            env,
            network,
            config=actual_config,
            device=device,
            callback=callback,
            logger=logger,
        )
        self._uses_default_network = not provided_network

    def predict(self, observation: np.ndarray, *, deterministic: bool = False) -> int:
        array = self._as_observation(observation)
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network.q_values(
                tensor, self.config.quantiles
            ).squeeze(0)
        action = self.exploration.select(
            q_values,
            step=self.num_timesteps,
            deterministic=deterministic,
        )
        return action + self.action_start

    def _train_step(self) -> dict[str, float | int]:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        predicted, taus = self.q_network(
            batch.observations, self.config.quantiles
        )
        action_indices = batch.actions.unsqueeze(-1).expand(
            -1, 1, self.config.quantiles
        )
        predicted = predicted.gather(1, action_indices).squeeze(1)
        with torch.no_grad():
            next_values = self.q_network.q_values(
                batch.next_observations, self.config.quantiles
            )
            next_actions = next_values.argmax(dim=1)
            target_quantiles, _ = self.target_network(
                batch.next_observations, self.config.target_quantiles
            )
            rows = torch.arange(self.config.batch_size, device=self.device)
            target_quantiles = target_quantiles[rows, next_actions]
            target_quantiles = batch.rewards + self.config.gamma * (
                1 - batch.terminated
            ) * target_quantiles
        loss = _quantile_huber_loss(
            predicted,
            target_quantiles,
            taus,
            self.config.huber_threshold,
        )
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.num_updates += 1
        return {
            "train/loss": float(loss.item()),
            "train/gradient_norm": float(gradient_norm),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }

    def _validate_network(self, network: nn.Module) -> None:
        observations = torch.zeros(
            (1, *self.observation_shape), device=self.device
        )
        try:
            with torch.no_grad():
                output, taus = network(observations, self.config.quantiles)
        except Exception as error:
            raise ValueError("network must implement the IQN forward API") from error
        expected = (1, self.action_dim, self.config.quantiles)
        if output.shape != expected or taus.shape != (1, self.config.quantiles):
            raise ValueError("network returned invalid IQN tensor shapes")


@dataclass(frozen=True)
class RainbowDQNConfig(C51Config):
    """Combined Double, dueling, PER, n-step, NoisyNet, and C51 settings."""

    n_steps: int = 3
    priority_alpha: float = 0.5
    priority_beta: float = 0.4
    priority_beta_steps: int = 100_000
    priority_epsilon: float = 1e-6
    noise_sigma: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.priority_alpha < 0:
            raise ValueError("priority_alpha cannot be negative")
        if not 0 <= self.priority_beta <= 1:
            raise ValueError("priority_beta must be between 0 and 1")
        if self.priority_beta_steps <= 0:
            raise ValueError("priority_beta_steps must be positive")
        if self.priority_epsilon <= 0 or self.noise_sigma <= 0:
            raise ValueError("priority_epsilon and noise_sigma must be positive")


class RainbowDQN(C51):
    """Readable Rainbow composition of its six canonical DQN improvements."""

    config_type = RainbowDQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: RainbowDQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        actual_config = config or RainbowDQNConfig()
        provided_network = network is not None
        if network is None:
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise TypeError("RainbowDQN requires a Box observation space")
            if not isinstance(env.action_space, gym.spaces.Discrete):
                raise TypeError("RainbowDQN requires a Discrete action space")
            network = CategoricalQNetwork(
                int(np.prod(env.observation_space.shape)),
                int(env.action_space.n),
                actual_config.atoms,
                dueling=True,
                noisy=True,
                initial_sigma=actual_config.noise_sigma,
            )
        super().__init__(
            env,
            network,
            config=actual_config,
            device=device,
            callback=callback,
            logger=logger,
        )
        self._uses_default_network = not provided_network
        self.replay_buffer = PrioritizedReplayBuffer(
            self.config.buffer_size,
            self.observation_shape,
            alpha=self.config.priority_alpha,
            seed=self.config.seed,
        )
        self._pending: deque[Transition[np.ndarray, int]] = deque()

    @property
    def priority_beta(self) -> float:
        progress = min(self.num_timesteps / self.config.priority_beta_steps, 1.0)
        return self.config.priority_beta + progress * (
            1.0 - self.config.priority_beta
        )

    def _make_exploration(self) -> EpsilonGreedyPolicy:
        return EpsilonGreedyPolicy(LinearSchedule(0.0, 0.0, 1), seed=self.config.seed)

    def predict(self, observation: np.ndarray, *, deterministic: bool = False) -> int:
        array = self._as_observation(observation)
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        was_training = self.q_network.training
        self.q_network.train(not deterministic)
        if not deterministic:
            reset_noise(self.q_network)
        with torch.no_grad():
            q_values = self._q_values(self.q_network, tensor).squeeze(0)
        self.q_network.train(was_training)
        return int(q_values.argmax().item()) + self.action_start

    def _update_from_transition(
        self, transition: Transition[np.ndarray, int]
    ) -> dict[str, float | int]:
        self._pending.append(transition)
        if len(self._pending) >= self.config.n_steps:
            self._store_return(self.config.n_steps)
            self._pending.popleft()
        if transition.done:
            while self._pending:
                self._store_return(len(self._pending))
                self._pending.popleft()
        next_timestep = self.num_timesteps + 1
        ready = (
            next_timestep >= self.config.learning_starts
            and next_timestep % self.config.train_freq == 0
            and len(self.replay_buffer) >= self.config.batch_size
        )
        latest: dict[str, float | int] = {}
        if ready:
            for _ in range(self.config.gradient_steps):
                latest = self._train_step()
        return latest

    def _store_return(self, horizon: int) -> None:
        items = list(self._pending)[:horizon]
        reward = 0.0
        discount = 1.0
        for transition in items:
            reward += discount * transition.reward
            discount *= self.config.gamma
        first = items[0]
        last = items[-1]
        self.replay_buffer.add(
            np.asarray(first.observation, dtype=np.float32),
            first.action - self.action_start,
            reward,
            np.asarray(last.next_observation, dtype=np.float32),
            last.terminated,
            last.truncated,
            discount=discount,
        )

    def _train_step(self) -> dict[str, float | int]:
        reset_noise(self.q_network)
        reset_noise(self.target_network)
        batch = self.replay_buffer.sample(
            self.config.batch_size, self.device, beta=self.priority_beta
        )
        target_distribution = self._project_distribution(batch)
        logits = self.q_network(batch.observations)
        action_indices = batch.actions.unsqueeze(-1).expand(
            -1, 1, self.config.atoms
        )
        chosen_logits = logits.gather(1, action_indices).squeeze(1)
        element_loss = -(
            target_distribution * chosen_logits.log_softmax(dim=-1)
        ).sum(dim=-1)
        loss = (batch.weights.squeeze(1) * element_loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        priorities = element_loss.detach().cpu().numpy() + self.config.priority_epsilon
        self.replay_buffer.update_priorities(batch.indices, priorities)
        self.num_updates += 1
        return {
            "train/loss": float(loss.item()),
            "train/gradient_norm": float(gradient_norm),
            "train/priority_mean": float(priorities.mean()),
            "train/updates": self.num_updates,
            "rollout/epsilon": 0.0,
        }

    @torch.no_grad()
    def _next_action_probabilities(self, batch: ReplayBatch) -> torch.Tensor:
        online_values = self._q_values(
            self.q_network, batch.next_observations
        )
        next_actions = online_values.argmax(dim=1)
        probabilities = self._probabilities(
            self.target_network, batch.next_observations
        )
        rows = torch.arange(batch.rewards.shape[0], device=self.device)
        return probabilities[rows, next_actions]


def _quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Pairwise quantile Huber loss for fixed or sampled fractions."""

    errors = target.unsqueeze(1) - predicted.unsqueeze(2)
    absolute = errors.abs()
    huber = torch.where(
        absolute <= threshold,
        0.5 * errors.square(),
        threshold * (absolute - 0.5 * threshold),
    )
    if taus.ndim == 1:
        taus = taus.unsqueeze(0).expand(predicted.shape[0], -1)
    weights = (taus.unsqueeze(2) - (errors.detach() < 0).float()).abs()
    return (weights * huber / threshold).mean()
