"""A readable vanilla Deep Q-Network implementation."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from typing_extensions import Self

from aprenderl.algorithms.base import OffPolicyAlgorithm
from aprenderl.buffers import ReplayBatch, ReplayBuffer
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import QNetwork
from aprenderl.policies import EpsilonGreedyPolicy, LinearSchedule
from aprenderl.utils import resolve_device


@dataclass(frozen=True)
class DQNConfig:
    """Typed hyperparameters for :class:`DQN`."""

    learning_rate: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_frequency: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 500
    exploration_initial_epsilon: float = 1.0
    exploration_final_epsilon: float = 0.05
    exploration_fraction: float = 0.2
    max_grad_norm: float = 10.0
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.buffer_size <= 0 or self.batch_size <= 0:
            raise ValueError("buffer_size and batch_size must be positive")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size cannot exceed buffer_size")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.train_frequency <= 0 or self.gradient_steps <= 0:
            raise ValueError("training frequencies must be positive")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if not 0 <= self.exploration_final_epsilon <= 1:
            raise ValueError("exploration_final_epsilon must be in [0, 1]")
        if not self.exploration_final_epsilon <= self.exploration_initial_epsilon <= 1:
            raise ValueError(
                "exploration_initial_epsilon must be between final epsilon and 1"
            )
        if not 0 < self.exploration_fraction <= 1:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


class DQN(OffPolicyAlgorithm[np.ndarray, int]):
    """Vanilla DQN for one Gymnasium environment with discrete actions.

    A custom Q-network can be supplied as any ``nn.Module`` mapping a batch of
    observations to one value per action. When omitted, a small MLP is created.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        network: nn.Module | None = None,
        *,
        config: DQNConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.device = resolve_device(device)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("DQN requires a Box observation space")
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("DQN requires a Discrete action space")
        if env.observation_space.shape is None or not env.observation_space.shape:
            raise ValueError("the observation space must have a non-empty shape")

        self.observation_shape = tuple(env.observation_space.shape)
        self.observation_dim = int(np.prod(self.observation_shape))
        self.action_dim = int(env.action_space.n)
        replay_buffer = ReplayBuffer(
            self.config.buffer_size,
            self.observation_shape,
            seed=self.config.seed,
        )
        super().__init__(
            env,
            replay_buffer,
            learning_starts=self.config.learning_starts,
            train_frequency=self.config.train_frequency,
            gradient_steps=self.config.gradient_steps,
            seed=self.config.seed,
            callback=callback,
            logger=logger,
            log_interval=self.config.log_interval,
        )

        self._uses_default_network = network is None
        self.q_network = (
            network
            if network is not None
            else QNetwork(self.observation_dim, self.action_dim)
        ).to(self.device)
        self._validate_network(self.q_network)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )
        self._exploration_duration: int | None = None
        self.exploration = self._make_exploration(duration=1)

    @property
    def epsilon(self) -> float:
        """Return the current epsilon used for behavior actions."""

        return self.exploration.schedule.value(self.num_timesteps)

    def predict(self, observation: np.ndarray, *, deterministic: bool = True) -> int:
        """Choose a greedy action or an epsilon-greedy behavior action."""

        array = self._as_observation(observation)
        tensor = torch.as_tensor(array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(tensor).squeeze(0)
        return self.exploration.select(
            q_values,
            lambda: int(self.env.action_space.sample()),
            step=self.num_timesteps,
            deterministic=deterministic,
        )

    def save(self, path: str | Path) -> None:
        """Save model, optimizer, configuration, and training counters."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 2,
                "config": asdict(self.config),
                "uses_default_network": self._uses_default_network,
                "observation_shape": self.observation_shape,
                "action_dim": self.action_dim,
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "num_timesteps": self.num_timesteps,
                "num_updates": self.num_updates,
                "exploration_duration": self._exploration_duration,
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

        config_data = dict(checkpoint["config"])
        if not checkpoint["uses_default_network"] and network is None:
            raise ValueError(
                "loading a custom-network checkpoint requires network=nn.Module"
            )
        algorithm = cls(
            env,
            network=network,
            config=DQNConfig(**config_data),
            device=device,
            callback=callback,
            logger=logger,
        )
        if tuple(checkpoint["observation_shape"]) != algorithm.observation_shape:
            raise ValueError("checkpoint observation shape does not match environment")
        if int(checkpoint["action_dim"]) != algorithm.action_dim:
            raise ValueError("checkpoint action count does not match environment")

        algorithm.q_network.load_state_dict(checkpoint["q_network"])
        algorithm.target_network.load_state_dict(checkpoint["target_network"])
        algorithm.optimizer.load_state_dict(checkpoint["optimizer"])
        algorithm.num_timesteps = int(checkpoint["num_timesteps"])
        algorithm.num_updates = int(checkpoint["num_updates"])
        algorithm._exploration_duration = checkpoint["exploration_duration"]
        if algorithm._exploration_duration is not None:
            algorithm.exploration = algorithm._make_exploration(
                algorithm._exploration_duration
            )
        algorithm.episode_returns = list(checkpoint["episode_returns"])
        algorithm.episode_lengths = list(checkpoint["episode_lengths"])
        return algorithm

    def _make_exploration(self, duration: int) -> EpsilonGreedyPolicy:
        return EpsilonGreedyPolicy(
            LinearSchedule(
                self.config.exploration_initial_epsilon,
                self.config.exploration_final_epsilon,
                duration,
            ),
            seed=self.config.seed,
        )

    def _before_learn(self, total_timesteps: int) -> None:
        if self._exploration_duration is None:
            self._exploration_duration = max(
                1, int(total_timesteps * self.config.exploration_fraction)
            )
            self.exploration = self._make_exploration(self._exploration_duration)

    def _sample_action(self, observation: np.ndarray) -> int:
        return self.predict(observation, deterministic=False)

    def _has_enough_replay(self) -> bool:
        return len(self.replay_buffer) >= self.config.batch_size

    def _train_step(self) -> dict[str, float | int]:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        target = self._td_target(batch)
        predicted = self.q_network(batch.observations).gather(1, batch.actions)
        loss = nn.functional.smooth_l1_loss(predicted, target)

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

    def _after_step(self) -> None:
        if self.num_timesteps % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    @torch.no_grad()
    def _td_target(self, batch: ReplayBatch) -> torch.Tensor:
        next_q_values = (
            self.target_network(batch.next_observations).max(dim=1, keepdim=True).values
        )
        return (
            batch.rewards + self.config.gamma * (1 - batch.terminated) * next_q_values
        )

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
            message = (
                f"expected observation shape {self.observation_shape}, "
                f"got {array.shape}"
            )
            raise ValueError(message)
        return array
