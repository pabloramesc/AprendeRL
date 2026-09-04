"""Rainbow DQN as the paper-defined combination of six improvements."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from aprenderl.algorithms.dqn import DQN, DQNConfig, _environment_dimensions
from aprenderl.buffers import PrioritizedReplayBuffer, ReplayBatch
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import CategoricalQNetwork
from aprenderl.networks.q_network import reset_noise
from aprenderl.policies import EpsilonGreedyPolicy, LinearSchedule
from aprenderl.types import Transition


@dataclass(frozen=True)
class RainbowDQNConfig(DQNConfig):
    """Settings for the canonical Rainbow combination."""

    double_dqn: bool = True
    atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    n_steps: int = 3
    priority_alpha: float = 0.5
    priority_beta: float = 0.4
    priority_beta_steps: int = 100_000
    priority_epsilon: float = 1e-6
    noise_sigma: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.double_dqn:
            raise ValueError("Rainbow always uses Double DQN targets")
        if self.atoms <= 1:
            raise ValueError("atoms must be greater than one")
        if self.v_min >= self.v_max:
            raise ValueError("v_min must be less than v_max")
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


class RainbowDQN(DQN):
    """Combine Double DQN, dueling heads, PER, n-step returns, NoisyNet, and C51."""

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
            observation_shape, action_dim, _ = _environment_dimensions(
                env, "RainbowDQN"
            )
            network = CategoricalQNetwork(
                int(np.prod(observation_shape)),
                action_dim,
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
        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atoms,
            device=self.device,
        )
        self._pending: deque[Transition[np.ndarray, int]] = deque()

    @property
    def priority_beta(self) -> float:
        progress = min(self.num_timesteps / self.config.priority_beta_steps, 1.0)
        return self.config.priority_beta + progress * (1.0 - self.config.priority_beta)

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

    def _make_replay_buffer(self) -> PrioritizedReplayBuffer:
        return PrioritizedReplayBuffer(
            self.config.buffer_size,
            self.observation_shape,
            alpha=self.config.priority_alpha,
            seed=self.config.seed,
        )

    def _make_exploration(self) -> EpsilonGreedyPolicy:
        return EpsilonGreedyPolicy(LinearSchedule(0.0, 0.0, 1), seed=self.config.seed)

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
        transitions = list(self._pending)[:horizon]
        reward = 0.0
        discount = 1.0
        for transition in transitions:
            reward += discount * transition.reward
            discount *= self.config.gamma
        first = transitions[0]
        last = transitions[-1]
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
        target_distribution = self._target_distribution(batch)
        logits = self.q_network(batch.observations)
        action_indices = batch.actions.unsqueeze(-1).expand(-1, 1, self.config.atoms)
        chosen_logits = logits.gather(1, action_indices).squeeze(1)
        element_loss = -(target_distribution * chosen_logits.log_softmax(dim=-1)).sum(
            dim=-1
        )
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
    def _target_distribution(self, batch: ReplayBatch) -> torch.Tensor:
        online_values = self._q_values(self.q_network, batch.next_observations)
        next_actions = online_values.argmax(dim=1)
        target_probabilities = self._probabilities(
            self.target_network, batch.next_observations
        )
        rows = torch.arange(batch.rewards.shape[0], device=self.device)
        chosen_probabilities = target_probabilities[rows, next_actions]
        discounts = batch.discounts
        target_atoms = batch.rewards + discounts * (
            1 - batch.terminated
        ) * self.support.unsqueeze(0)
        return categorical_projection(
            target_atoms,
            chosen_probabilities,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
        )

    def _q_values(self, network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        probabilities = self._probabilities(network, observations)
        return (probabilities * self.support).sum(dim=-1)

    @staticmethod
    def _probabilities(network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        return network(observations).softmax(dim=-1)

    def _validate_network(self, network: nn.Module) -> None:
        try:
            with torch.no_grad():
                output = network(
                    torch.zeros((1, *self.observation_shape), device=self.device)
                )
        except Exception as error:
            raise ValueError(
                "network must implement the Rainbow network API"
            ) from error
        expected = (1, self.action_dim, self.config.atoms)
        if not isinstance(output, torch.Tensor) or output.shape != expected:
            actual = getattr(output, "shape", None)
            raise ValueError(f"network must return shape {expected}, got {actual}")


def categorical_projection(
    target_atoms: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    """Project Rainbow's categorical targets onto their fixed support."""

    atoms = probabilities.shape[1]
    clipped_atoms = target_atoms.clamp(v_min, v_max)
    atom_width = (v_max - v_min) / (atoms - 1)
    locations = (clipped_atoms - v_min) / atom_width
    lower = locations.floor().long()
    upper = locations.ceil().long()
    same_bin = lower == upper

    projection = torch.zeros_like(probabilities)
    offset = (
        torch.arange(target_atoms.shape[0], device=target_atoms.device) * atoms
    ).unsqueeze(1)
    lower_mass = probabilities * (
        upper.to(locations.dtype) - locations + same_bin.to(locations.dtype)
    )
    upper_mass = probabilities * (locations - lower.to(locations.dtype))
    projection.view(-1).index_add_(
        0, (lower + offset).view(-1), lower_mass.view(-1)
    )
    projection.view(-1).index_add_(
        0, (upper + offset).view(-1), upper_mass.view(-1)
    )
    return projection
