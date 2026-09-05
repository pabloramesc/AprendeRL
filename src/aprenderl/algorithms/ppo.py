"""Proximal Policy Optimization with a clipped surrogate objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import torch
from torch import nn

from aprenderl.algorithms.actor_critic import ActorCritic, ActorCriticConfig
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger


@dataclass(frozen=True)
class PPOConfig(ActorCriticConfig):
    """Hyperparameters for :class:`PPO`."""

    learning_rate: float = 3e-4
    n_steps: int = 1024
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("batch_size and n_epochs must be positive")
        if not 0 < self.clip_range < 1:
            raise ValueError("clip_range must be between 0 and 1, exclusively")


class PPO(ActorCritic):
    """PPO-Clip for discrete or continuous actions using separate networks.

    Collection, prediction, and checkpoints follow ActorCritic. Each completed
    rollout receives multiple shuffled minibatch passes with fixed GAE targets
    and collecting-policy log probabilities. Clipping is not a hard KL bound.
    """

    config_class: ClassVar[type[PPOConfig]] = PPOConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        policy_network: nn.Module | None = None,
        *,
        value_network: nn.Module | None = None,
        config: PPOConfig | None = None,
        device: str | torch.device = "auto",
        callback: BaseCallback | list[BaseCallback] | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        super().__init__(
            env,
            policy_network,
            value_network=value_network,
            config=config,
            device=device,
            callback=callback,
            logger=logger,
        )
        actor_ids = {id(p) for p in self.policy_network.parameters()}
        if any(id(p) in actor_ids for p in self.value_network.parameters()):
            raise ValueError("PPO actor and critic must not share parameters")
        # Keep dropout and batch-normalization state fixed across policy ratios.
        # Evaluation mode still permits parameter gradients during optimization.
        self.policy_network.eval()
        self.value_network.eval()

    # ------------------------------------------------------------------
    # PPO learning rule
    # ------------------------------------------------------------------

    def _train_step(self) -> dict[str, float | int]:
        batch = self.rollout_buffer.batch(self.device)
        actions = self.policy_action_space.action_batch_tensor(batch.actions)
        advantages = batch.advantages.detach()
        if self.config.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        old_log_probabilities = batch.log_probabilities.detach()
        targets = batch.returns.detach()
        size = len(advantages)
        for _ in range(self.config.n_epochs):
            indices = torch.randperm(size, device=self.device)
            for start in range(0, size, self.config.batch_size):
                selected = indices[start : start + self.config.batch_size]
                distribution = self.policy_action_space.distribution(
                    self.policy_network(batch.observations[selected])
                )
                ratio = torch.exp(
                    distribution.log_prob(actions[selected])
                    - old_log_probabilities[selected]
                )
                surrogate = ratio * advantages[selected]
                clipped = (
                    ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range)
                    * advantages[selected]
                )
                policy_loss = -torch.minimum(surrogate, clipped).mean()
                entropy = distribution.entropy().mean()
                actor_loss = policy_loss - self.config.entropy_coefficient * entropy
                self.policy_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.config.max_grad_norm
                )
                self.policy_optimizer.step()

                value_loss = nn.functional.mse_loss(
                    self.value_network(batch.observations[selected]), targets[selected]
                )
                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.value_network.parameters(), self.config.max_grad_norm
                )
                self.value_optimizer.step()

        self.num_updates += 1
        with torch.no_grad():
            distribution = self.policy_action_space.distribution(
                self.policy_network(batch.observations)
            )
            log_ratio = distribution.log_prob(actions) - old_log_probabilities
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(
                ratio * advantages,
                ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range)
                * advantages,
            ).mean()
            entropy = distribution.entropy().mean()
            value_loss = nn.functional.mse_loss(
                self.value_network(batch.observations), targets
            )
        return {
            "train/loss": (
                policy_loss + value_loss - self.config.entropy_coefficient * entropy
            ).item(),
            "train/policy_loss": policy_loss.item(),
            "train/value_loss": value_loss.item(),
            "train/entropy": entropy.item(),
            "train/approx_kl": ((ratio - 1) - log_ratio).mean().item(),
            "train/clip_fraction": ((ratio - 1).abs() > self.config.clip_range)
            .float()
            .mean()
            .item(),
            "train/updates": self.num_updates,
        }

    def _gae_lambda(self) -> float:
        return self.config.gae_lambda
