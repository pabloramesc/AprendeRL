"""Double, dueling, prioritized, multi-step, and noisy DQN variants."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from aprenderl.algorithms.dqn import DQN, DQNConfig
from aprenderl.buffers import (
    NStepReplayBatch,
    NStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBatch,
)
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import DuelingQNetwork, NoisyQNetwork, reset_noise
from aprenderl.policies import EpsilonGreedyPolicy, LinearSchedule
from aprenderl.types import Transition


@dataclass(frozen=True)
class DoubleDQNConfig(DQNConfig):
    """Hyperparameters for :class:`DoubleDQN`."""


class DoubleDQN(DQN):
    """DQN that selects target actions online and evaluates them by target net."""

    config_type = DoubleDQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: DoubleDQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            network,
            config=config or DoubleDQNConfig(),
            device=device,
            callback=callback,
            logger=logger,
        )

    @torch.no_grad()
    def _td_target(self, batch: ReplayBatch) -> torch.Tensor:
        next_actions = self.q_network(batch.next_observations).argmax(
            dim=1, keepdim=True
        )
        next_q_values = self.target_network(batch.next_observations).gather(
            1, next_actions
        )
        return (
            batch.rewards
            + self.config.gamma * (1 - batch.terminated) * next_q_values
        )


@dataclass(frozen=True)
class DuelingDQNConfig(DQNConfig):
    """Hyperparameters for :class:`DuelingDQN`."""


class DuelingDQN(DQN):
    """DQN with separate state-value and centered action-advantage streams."""

    config_type = DuelingDQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: DuelingDQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        provided_network = network is not None
        if network is None:
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise TypeError("DuelingDQN requires a Box observation space")
            if not isinstance(env.action_space, gym.spaces.Discrete):
                raise TypeError("DuelingDQN requires a Discrete action space")
            network = DuelingQNetwork(
                int(np.prod(env.observation_space.shape)), int(env.action_space.n)
            )
        super().__init__(
            env,
            network,
            config=config or DuelingDQNConfig(),
            device=device,
            callback=callback,
            logger=logger,
        )
        self._uses_default_network = not provided_network


@dataclass(frozen=True)
class PrioritizedDQNConfig(DQNConfig):
    """DQN and proportional prioritized-replay settings."""

    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_steps: int = 100_000
    priority_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.priority_alpha < 0:
            raise ValueError("priority_alpha cannot be negative")
        if not 0 <= self.priority_beta <= 1:
            raise ValueError("priority_beta must be between 0 and 1")
        if self.priority_beta_steps <= 0:
            raise ValueError("priority_beta_steps must be positive")
        if self.priority_epsilon <= 0:
            raise ValueError("priority_epsilon must be positive")


class PrioritizedDQN(DQN):
    """DQN trained from proportional prioritized experience replay."""

    config_type = PrioritizedDQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: PrioritizedDQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            network,
            config=config or PrioritizedDQNConfig(),
            device=device,
            callback=callback,
            logger=logger,
        )
        self.replay_buffer = PrioritizedReplayBuffer(
            self.config.buffer_size,
            self.observation_shape,
            alpha=self.config.priority_alpha,
            seed=self.config.seed,
        )

    @property
    def priority_beta(self) -> float:
        progress = min(self.num_timesteps / self.config.priority_beta_steps, 1.0)
        return self.config.priority_beta + progress * (
            1.0 - self.config.priority_beta
        )

    def _train_step(self) -> dict[str, float | int]:
        batch = self.replay_buffer.sample(
            self.config.batch_size, self.device, beta=self.priority_beta
        )
        target = self._td_target(batch)
        predicted = self.q_network(batch.observations).gather(1, batch.actions)
        element_loss = nn.functional.smooth_l1_loss(
            predicted, target, reduction="none"
        )
        loss = (batch.weights * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        priorities = (
            (target - predicted).detach().abs().squeeze(1).cpu().numpy()
            + self.config.priority_epsilon
        )
        self.replay_buffer.update_priorities(batch.indices, priorities)
        self.num_updates += 1
        return {
            "train/loss": float(loss.item()),
            "train/gradient_norm": float(gradient_norm),
            "train/priority_mean": float(priorities.mean()),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class NStepDQNConfig(DQNConfig):
    """DQN settings with an n-step replay return."""

    n_steps: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")


class NStepDQN(DQN):
    """DQN storing n-step returns and their exact bootstrap discounts."""

    config_type = NStepDQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: NStepDQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            network,
            config=config or NStepDQNConfig(),
            device=device,
            callback=callback,
            logger=logger,
        )
        self.replay_buffer = NStepReplayBuffer(
            self.config.buffer_size,
            self.observation_shape,
            seed=self.config.seed,
        )
        self._pending: deque[Transition[np.ndarray, int]] = deque()

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
        return self._maybe_train()

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

    def _maybe_train(self) -> dict[str, float | int]:
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

    @torch.no_grad()
    def _td_target(self, batch: NStepReplayBatch) -> torch.Tensor:
        next_q_values = (
            self.target_network(batch.next_observations)
            .max(dim=1, keepdim=True)
            .values
        )
        return batch.rewards + batch.discounts * (
            1 - batch.terminated
        ) * next_q_values


@dataclass(frozen=True)
class NoisyDQNConfig(DQNConfig):
    """DQN settings for parameter-space NoisyNet exploration."""

    noise_sigma: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.noise_sigma <= 0:
            raise ValueError("noise_sigma must be positive")


class NoisyDQN(DQN):
    """DQN using factorized parameter noise instead of epsilon-greedy noise."""

    config_type = NoisyDQNConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: NoisyDQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        actual_config = config or NoisyDQNConfig()
        provided_network = network is not None
        if network is None:
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise TypeError("NoisyDQN requires a Box observation space")
            if not isinstance(env.action_space, gym.spaces.Discrete):
                raise TypeError("NoisyDQN requires a Discrete action space")
            network = NoisyQNetwork(
                int(np.prod(env.observation_space.shape)),
                int(env.action_space.n),
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
            q_values = self.q_network(tensor).squeeze(0)
        self.q_network.train(was_training)
        return int(q_values.argmax().item()) + self.action_start

    def _train_step(self) -> dict[str, float | int]:
        reset_noise(self.q_network)
        reset_noise(self.target_network)
        return super()._train_step()
