"""A readable Deep Q-Network implementation with optional Double DQN targets."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from typing_extensions import Self

from aprenderl.algorithms.base import OffPolicyAlgorithm, TrainingMetrics
from aprenderl.buffers import ReplayBatch, ReplayBuffer
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger
from aprenderl.networks import QNetwork
from aprenderl.policies import EpsilonGreedyPolicy, LinearSchedule
from aprenderl.types import Transition
from aprenderl.utils import load_torch_checkpoint, resolve_device

CHECKPOINT_VERSION = 5


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for DQN and Double DQN."""

    learning_rate: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 500
    exploration_initial_epsilon: float = 1.0
    exploration_final_epsilon: float = 0.05
    exploration_steps: int = 10_000
    double_dqn: bool = False
    max_grad_norm: float = 10.0
    log_interval: int = 10
    seed: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "exploration_steps": self.exploration_steps,
            "max_grad_norm": self.max_grad_norm,
            "log_interval": self.log_interval,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size cannot exceed buffer_size")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if not 0 <= self.exploration_final_epsilon <= 1:
            raise ValueError("exploration_final_epsilon must be in [0, 1]")
        if not self.exploration_final_epsilon <= self.exploration_initial_epsilon <= 1:
            raise ValueError(
                "exploration_initial_epsilon must be between final epsilon and 1"
            )


class DQN(OffPolicyAlgorithm[np.ndarray, int]):
    """A small DQN implementation for one discrete-action environment."""

    config_type = DQNConfig

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
        self.config = config or self.config_type()
        self.device = resolve_device(device)
        self.observation_shape, self.action_dim, self.action_start = (
            _environment_dimensions(env, type(self).__name__)
        )
        self.observation_dim = int(np.prod(self.observation_shape))

        super().__init__(
            env,
            seed=self.config.seed,
            callback=callback,
            logger=logger,
            log_interval=self.config.log_interval,
        )

        self.replay_buffer = self._make_replay_buffer()
        self.exploration = self._make_exploration()
        self._uses_default_network = network is None
        self.q_network = self._make_network(network)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )

    @property
    def epsilon(self) -> float:
        """Return the epsilon used for the next behavior action."""

        return self.exploration.schedule.value(self.num_timesteps)

    def predict(self, observation: np.ndarray, *, deterministic: bool = False) -> int:
        """Select one environment action."""

        observation = self._as_observation(observation)
        observation_tensor = torch.as_tensor(
            observation, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self._q_values(self.q_network, observation_tensor).squeeze(0)
        zero_based_action = self.exploration.select(
            q_values,
            step=self.num_timesteps,
            deterministic=deterministic,
        )
        return zero_based_action + self.action_start

    # ------------------------------------------------------------------
    # DQN learning rule
    # ------------------------------------------------------------------

    def _train_step(self) -> TrainingMetrics:
        """Apply one DQN gradient update from a replay-buffer batch."""

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        all_q_values = self.q_network(batch.observations)
        predicted_q_values = all_q_values.gather(dim=1, index=batch.actions)
        td_targets = self._td_target(batch)
        loss = nn.functional.smooth_l1_loss(predicted_q_values, td_targets)

        return self._optimize(loss)

    @torch.no_grad()
    def _td_target(self, batch: ReplayBatch) -> torch.Tensor:
        """Calculate ``reward + gamma * (1 - terminated) * next_q_value``.

        Truncated episodes still bootstrap because truncation is an external
        time limit rather than a terminal state in the underlying MDP.
        """

        next_q_values = self._next_q_values(batch.next_observations)
        not_terminated = 1.0 - batch.terminated
        discounted_next_q_values = (
            self.config.gamma * not_terminated * next_q_values
        )
        return batch.rewards + discounted_next_q_values

    def _next_q_values(self, next_observations: torch.Tensor) -> torch.Tensor:
        """Evaluate the next-state value using DQN or Double DQN."""

        if self.config.double_dqn:
            # Double DQN: the online network selects the action; the target
            # network evaluates that selected action.
            online_q_values = self.q_network(next_observations)
            best_actions = online_q_values.argmax(dim=1, keepdim=True)
            target_q_values = self.target_network(next_observations)
            return target_q_values.gather(dim=1, index=best_actions)

        # Vanilla DQN: the target network selects and evaluates the action.
        return self.target_network(next_observations).max(dim=1, keepdim=True).values

    def _optimize(self, loss: torch.Tensor) -> TrainingMetrics:
        """Backpropagate the loss and update the online Q-network."""

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.num_updates += 1
        return {
            "train/loss": loss.item(),
            "train/gradient_norm": float(gradient_norm),
            "train/updates": self.num_updates,
            "rollout/epsilon": self.epsilon,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the model and all state needed to resume training."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint(), checkpoint_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env[Any, Any],
        **kwargs: Any,
    ) -> Self:
        """Restore a checkpoint after checking it against ``env``."""

        device = resolve_device(kwargs.pop("device", "auto"))
        network = kwargs.pop("network", None)
        callback = kwargs.pop("callback", None)
        logger = kwargs.pop("logger", None)
        cls._reject_unknown_arguments(kwargs)

        checkpoint = load_torch_checkpoint(path, device)
        if not checkpoint["uses_default_network"] and network is None:
            raise ValueError(
                "loading a custom-network checkpoint requires network=nn.Module"
            )

        algorithm = cls(
            env,
            network=network,
            config=cls.config_type(**_config_from_checkpoint(checkpoint)),
            device=device,
            callback=callback,
            logger=logger,
        )
        algorithm._validate_checkpoint_environment(checkpoint)
        algorithm._restore_checkpoint(checkpoint)
        return algorithm

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _make_network(self, network: nn.Module | None) -> nn.Module:
        model = network if network is not None else self._make_default_network()
        model = model.to(self.device)
        self._validate_network(model)
        return model

    def _make_default_network(self) -> nn.Module:
        """Build the online network; DQN variants override this factory."""

        return QNetwork(self.observation_dim, self.action_dim)

    def _make_replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(
            self.config.buffer_size,
            self.observation_shape,
            seed=self.config.seed,
        )

    def _make_exploration(self) -> EpsilonGreedyPolicy:
        schedule = LinearSchedule(
            self.config.exploration_initial_epsilon,
            self.config.exploration_final_epsilon,
            self.config.exploration_steps,
        )
        return EpsilonGreedyPolicy(schedule, seed=self.config.seed)

    # ------------------------------------------------------------------
    # Environment interaction and replay
    # ------------------------------------------------------------------

    def _update_from_transition(
        self, transition: Transition[np.ndarray, int]
    ) -> TrainingMetrics:
        self._store_transition(transition)
        return self._maybe_train()

    def _maybe_train(self) -> TrainingMetrics:
        """Run the configured number of updates when training is due."""

        next_timestep = self.num_timesteps + 1
        training_is_due = (
            next_timestep >= self.config.learning_starts
            and next_timestep % self.config.train_freq == 0
            and len(self.replay_buffer) >= self.config.batch_size
        )
        if not training_is_due:
            return {}

        metrics: TrainingMetrics = {}
        for _ in range(self.config.gradient_steps):
            metrics = self._train_step()
        return metrics

    def _store_transition(self, transition: Transition[np.ndarray, int]) -> None:
        self.replay_buffer.add(
            np.asarray(transition.observation, dtype=np.float32),
            transition.action - self.action_start,
            transition.reward,
            np.asarray(transition.next_observation, dtype=np.float32),
            transition.terminated,
            transition.truncated,
        )

    def _after_step(self, transition: Transition[np.ndarray, int]) -> None:
        del transition
        if self.num_timesteps % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _progress_metrics(
        self,
        training_metrics: Mapping[str, float | int] | None = None,
    ) -> dict[str, str | int]:
        metrics = super()._progress_metrics(training_metrics)
        metrics["epsilon"] = f"{self.epsilon:.3f}"
        return metrics

    def _q_values(self, network: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        return network(observations)

    # ------------------------------------------------------------------
    # Validation and checkpoint details
    # ------------------------------------------------------------------

    def _validate_network(self, network: nn.Module) -> None:
        sample = torch.zeros((1, *self.observation_shape), device=self.device)
        try:
            with torch.no_grad():
                output = network(sample)
        except Exception as error:
            raise ValueError(
                "network could not process one environment observation"
            ) from error

        expected_shape = (1, self.action_dim)
        actual_shape = getattr(output, "shape", None)
        if not isinstance(output, torch.Tensor) or output.shape != expected_shape:
            raise ValueError(
                f"network must return shape {expected_shape}, got {actual_shape}"
            )

    def _as_observation(self, observation: Any) -> np.ndarray:
        """Validate and convert one environment observation."""

        array = np.asarray(observation, dtype=np.float32)
        if array.shape != self.observation_shape:
            raise ValueError(
                f"expected observation shape {self.observation_shape}, "
                f"got {array.shape}"
            )
        return array

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "uses_default_network": self._uses_default_network,
            "observation_shape": self.observation_shape,
            "action_dim": self.action_dim,
            "action_start": self.action_start,
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            **self._training_state(),
            "exploration_rng_state": self.exploration.rng_state,
            "algorithm_state": self._checkpoint_state(),
        }

    def _validate_checkpoint_environment(self, checkpoint: dict[str, Any]) -> None:
        expected = (
            ("observation_shape", self.observation_shape, tuple),
            ("action_dim", self.action_dim, int),
            ("action_start", self.action_start, int),
        )
        for name, current_value, convert in expected:
            saved_value = (
                checkpoint.get(name, 0)
                if name == "action_start"
                else checkpoint[name]
            )
            if convert(saved_value) != current_value:
                label = name.replace("_", " ")
                raise ValueError(f"checkpoint {label} does not match environment")

    def _restore_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._restore_training_state(checkpoint)
        if "exploration_rng_state" in checkpoint:
            self.exploration.rng_state = checkpoint["exploration_rng_state"]
        self._restore_checkpoint_state(checkpoint.get("algorithm_state", {}))

    def _checkpoint_state(self) -> dict[str, Any]:
        return {}

    def _restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        del state


def _environment_dimensions(
    env: gym.Env[Any, Any], algorithm_name: str
) -> tuple[tuple[int, ...], int, int]:
    """Validate the environment spaces and return their relevant dimensions."""

    observation_space = env.observation_space
    action_space = env.action_space
    if not isinstance(observation_space, gym.spaces.Box):
        raise TypeError(f"{algorithm_name} requires a Box observation space")
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError(f"{algorithm_name} requires a Discrete action space")
    if not observation_space.shape:
        raise ValueError("the observation space must have a non-empty shape")
    return tuple(observation_space.shape), int(action_space.n), int(action_space.start)


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Translate configuration keys used by older DQN checkpoints."""

    config = dict(checkpoint["config"])
    if "train_frequency" in config:
        config["train_freq"] = config.pop("train_frequency")
    if "exploration_fraction" in config:
        config.pop("exploration_fraction")
        config["exploration_steps"] = checkpoint.get("exploration_duration") or 1
    return config
