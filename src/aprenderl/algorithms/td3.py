"""Twin Delayed DDPG with clipped double Q targets and target-policy smoothing."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from aprenderl.algorithms.base import TrainingMetrics
from aprenderl.algorithms.ddpg import DDPG, DDPGConfig
from aprenderl.algorithms.off_policy_actor_critic import (
    frozen_parameters,
    polyak_update,
)
from aprenderl.buffers import ReplayBatch


@dataclass(frozen=True)
class TD3Config(DDPGConfig):
    """Noise scales are measured in normalized [-1, 1] action coordinates."""

    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            isinstance(self.policy_delay, bool)
            or not isinstance(self.policy_delay, int)
            or self.policy_delay <= 0
        ):
            raise ValueError("policy_delay must be a positive integer")
        for name in ("target_policy_noise", "target_noise_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class TD3(DDPG):
    """DDPG with two independent critics and delayed actor/target updates."""

    config_type = TD3Config
    n_critics = 2

    # ------------------------------------------------------------------
    # TD3 learning rule
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _td_target(self, batch: ReplayBatch) -> torch.Tensor:
        normalized = self.target_policy_network(batch.next_observations)
        noise = (torch.randn_like(normalized) * self.config.target_policy_noise).clamp(
            -self.config.target_noise_clip, self.config.target_noise_clip
        )
        normalized = (normalized + noise).clamp(-1, 1)
        low, high = self.policy_action_space.low, self.policy_action_space.high
        actions = (high + low) / 2 + (high - low) / 2 * normalized
        next_q = torch.minimum(
            self.target_critics[0](batch.next_observations, actions),
            self.target_critics[1](batch.next_observations, actions),
        )
        return batch.rewards + self.config.gamma * (1 - batch.terminated) * next_q

    def _train_step(self) -> TrainingMetrics:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        targets = self._td_target(batch)
        critic_loss = sum(
            nn.functional.mse_loss(critic(batch.observations, batch.actions), targets)
            for critic in self.critic_networks
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic_networks.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.num_updates += 1
        metrics = {"train/loss": critic_loss.item(), "train/updates": self.num_updates}

        if self.num_updates % self.config.policy_delay == 0:
            with frozen_parameters(self.critic_networks):
                actions = self._scaled_action(self.policy_network, batch.observations)
                actor_loss = -self.critic_networks[0](
                    batch.observations, actions
                ).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()
            polyak_update(
                self.policy_network, self.target_policy_network, self.config.tau
            )
            polyak_update(self.critic_networks, self.target_critics, self.config.tau)
            metrics["train/actor_loss"] = actor_loss.item()
        return metrics
