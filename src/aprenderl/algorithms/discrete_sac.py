"""Discrete SAC: compute policy expectations exactly over all actions."""

from dataclasses import dataclass

import torch
from torch import nn

from aprenderl.algorithms.base import TrainingMetrics
from aprenderl.algorithms.off_policy_actor_critic import polyak_update
from aprenderl.algorithms.sac import SAC, SACConfig
from aprenderl.buffers import ReplayBatch


@dataclass(frozen=True)
class DiscreteSACConfig(SACConfig):
    """Discrete SAC settings; default entropy target is 0.98 * log(n_actions)."""


class DiscreteSAC(SAC):
    """Categorical actor and two Q-vector critics for Discrete action spaces.

    policy_network returns logits, and each critic returns one Q per action.
    All three outputs have shape (batch, action_dim), using zero-based indices
    internally even when the environment's Discrete space has a nonzero start.
    """

    config_type = DiscreteSACConfig
    continuous = False

    # ------------------------------------------------------------------
    # Discrete SAC learning rule
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _td_target(self, batch: ReplayBatch) -> torch.Tensor:
        log_probs = self.policy_network(batch.next_observations).log_softmax(dim=-1)
        probs = log_probs.exp()
        q = torch.minimum(
            self.target_critics[0](batch.next_observations),
            self.target_critics[1](batch.next_observations),
        )
        soft_value = (probs * (q - self.ent_coef * log_probs)).sum(dim=-1, keepdim=True)
        return batch.rewards + self.config.gamma * (1 - batch.terminated) * soft_value

    def _train_step(self) -> TrainingMetrics:
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)
        targets = self._td_target(batch)
        critic_loss = sum(
            nn.functional.mse_loss(
                critic(batch.observations).gather(1, batch.actions), targets
            )
            for critic in self.critic_networks
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic_networks.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)

        log_probs = self.policy_network(batch.observations).log_softmax(dim=-1)
        probs = log_probs.exp()
        with torch.no_grad():
            q = torch.minimum(
                self.critic_networks[0](batch.observations),
                self.critic_networks[1](batch.observations),
            )
        actor_loss = (probs * (self.ent_coef * log_probs - q)).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.config.max_grad_norm
        )
        self.actor_optimizer.step()

        expected_log_prob = (probs.detach() * log_probs.detach()).sum(dim=-1)
        metrics = {}
        if self.log_ent_coef is not None:
            temperature_loss = -(
                self.log_ent_coef * (expected_log_prob + self.target_entropy)
            ).mean()
            self.ent_coef_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.ent_coef_optimizer.step()
            metrics["train/temperature_loss"] = temperature_loss.item()
        polyak_update(self.critic_networks, self.target_critics, self.config.tau)
        self.num_updates += 1
        metrics.update(
            {
                "train/loss": critic_loss.item(),
                "train/actor_loss": actor_loss.item(),
                "train/entropy": -expected_log_prob.mean().item(),
                "train/ent_coef": self.ent_coef.item(),
                "train/updates": self.num_updates,
            }
        )
        return metrics
