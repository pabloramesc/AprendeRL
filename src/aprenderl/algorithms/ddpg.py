"""Deep Deterministic Policy Gradient: replay and slowly moving targets."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from aprenderl.algorithms.base import TrainingMetrics
from aprenderl.algorithms.off_policy_actor_critic import (
    OffPolicyActorCritic,
    OffPolicyActorCriticConfig,
    frozen_parameters,
    polyak_update,
)
from aprenderl.buffers import ReplayBatch


@dataclass(frozen=True)
class DDPGConfig(OffPolicyActorCriticConfig):
    """DDPG settings; action_noise is relative to the action half-range."""

    action_noise: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.action_noise) or self.action_noise < 0:
            raise ValueError("action_noise must be finite and nonnegative")


class DDPG(OffPolicyActorCritic):
    """A deterministic actor and one action-conditioned critic for finite Box actions.

    Custom policy_network outputs normalized actions (batch, action_dim).
    critic_networks contains one module mapping (observations, environment_actions)
    to (batch, 1). See DDPGConfig for replay and optimization settings.
    """

    config_type = DDPGConfig

    # ------------------------------------------------------------------
    # DDPG learning rule
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _td_target(self, batch: ReplayBatch) -> torch.Tensor:
        next_actions = self._scaled_action(
            self.target_policy_network, batch.next_observations
        )
        next_q = self.target_critics[0](batch.next_observations, next_actions)
        return batch.rewards + self.config.gamma * (1 - batch.terminated) * next_q

    def _train_step(self) -> TrainingMetrics:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        targets = self._td_target(batch)
        predictions = self.critic_networks[0](batch.observations, batch.actions)
        critic_loss = nn.functional.mse_loss(predictions, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic_networks.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        # Autograd applies dQ/da * dmu/dtheta; only actor weights are stepped.
        with frozen_parameters(self.critic_networks):
            actions = self._scaled_action(self.policy_network, batch.observations)
            actor_loss = -self.critic_networks[0](batch.observations, actions).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), self.config.max_grad_norm
            )
            self.actor_optimizer.step()

        polyak_update(self.policy_network, self.target_policy_network, self.config.tau)
        polyak_update(self.critic_networks, self.target_critics, self.config.tau)
        self.num_updates += 1
        return {
            "train/loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/updates": self.num_updates,
        }
