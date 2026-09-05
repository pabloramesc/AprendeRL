"""Trust Region Policy Optimization with conjugate gradients and line search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from aprenderl.algorithms.actor_critic import ActorCritic, ActorCriticConfig
from aprenderl.callbacks import BaseCallback
from aprenderl.logging import TrainingLogger


@dataclass(frozen=True)
class TRPOConfig(ActorCriticConfig):
    """TRPO settings; inherited learning_rate is unused by the actor."""

    n_steps: int = 1024
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    max_kl: float = 0.01
    damping: float = 0.1
    cg_steps: int = 10
    cg_tolerance: float = 1e-10
    backtrack_steps: int = 10
    backtrack_coefficient: float = 0.5
    accept_ratio: float = 0.1
    value_epochs: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if not self.max_kl > 0 or not self.damping >= 0:
            raise ValueError("max_kl must be positive and damping nonnegative")
        if self.cg_steps <= 0 or self.backtrack_steps <= 0 or self.value_epochs <= 0:
            raise ValueError(
                "cg_steps, backtrack_steps and value_epochs must be positive"
            )
        if not self.cg_tolerance > 0:
            raise ValueError("cg_tolerance must be positive")
        if not 0 < self.backtrack_coefficient < 1:
            raise ValueError("backtrack_coefficient must be between 0 and 1")
        if not 0 <= self.accept_ratio < 1:
            raise ValueError("accept_ratio must be in [0, 1)")
        if self.entropy_coefficient != 0:
            raise ValueError("TRPO requires entropy_coefficient=0")


class TRPO(ActorCritic):
    """A sampled mean-KL constrained actor update and an Adam value fit.

    Supports categorical and diagonal Gaussian policies, including fixed tanh
    action transforms. The inherited policy optimizer is saved for the common
    checkpoint format but never stepped. Actor and critic must not share weights.
    """

    config_class: ClassVar[type[TRPOConfig]] = TRPOConfig

    def __init__(
        self,
        env: gym.Env[Any, Any],
        policy_network: nn.Module | None = None,
        *,
        value_network: nn.Module | None = None,
        config: TRPOConfig | None = None,
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
            raise ValueError("TRPO actor and critic must not share parameters")
        # Keep dropout and batch-normalization state fixed across policy ratios.
        # Evaluation mode still permits parameter gradients during optimization.
        self.policy_network.eval()
        self.value_network.eval()

    # ------------------------------------------------------------------
    # TRPO learning rule
    # ------------------------------------------------------------------

    def _update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probabilities: torch.Tensor,
    ) -> dict[str, float]:
        parameters = [p for p in self.policy_network.parameters() if p.requires_grad]
        old_parameters = parameters_to_vector(parameters).detach().clone()
        with torch.no_grad():
            output = self.policy_network(observations)
            frozen_output = (
                tuple(item.detach().clone() for item in output)
                if isinstance(output, tuple)
                else output.detach().clone()
            )
            old_distribution = self.policy_action_space.distribution(frozen_output)
        advantages = advantages.detach()
        old_log_probabilities = old_log_probabilities.detach()

        def objective_and_kl() -> tuple[torch.Tensor, torch.Tensor]:
            distribution = self.policy_action_space.distribution(
                self.policy_network(observations)
            )
            ratio = (distribution.log_prob(actions) - old_log_probabilities).exp()
            objective = (ratio * advantages).mean()
            divergence = kl_divergence(
                old_distribution.distribution, distribution.distribution
            )
            # A common invertible action transform preserves Gaussian KL.
            if self.policy_action_space.continuous:
                divergence = divergence.sum(dim=-1)
            return objective, divergence.mean()

        def flat_gradient(
            scalar: torch.Tensor, create_graph: bool = False
        ) -> torch.Tensor:
            gradients = torch.autograd.grad(
                scalar, parameters, create_graph=create_graph, allow_unused=True
            )
            return torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(parameters, gradients, strict=True)
                ]
            )

        objective, _ = objective_and_kl()
        old_objective = objective.detach()
        gradient = flat_gradient(objective).detach()

        def hessian_vector_product(vector: torch.Tensor) -> torch.Tensor:
            _, mean_kl = objective_and_kl()
            kl_gradient = flat_gradient(mean_kl, create_graph=True)
            product = flat_gradient(torch.dot(kl_gradient, vector)).detach()
            return product + self.config.damping * vector

        direction = self._conjugate_gradient(hessian_vector_product, gradient)
        curvature = torch.dot(direction, hessian_vector_product(direction))
        result = {
            "train/kl": 0.0,
            "train/surrogate_gain": 0.0,
            "train/step_fraction": 0.0,
        }
        if not torch.isfinite(curvature) or curvature <= 1e-8:
            return result
        full_step = torch.sqrt(2 * self.config.max_kl / curvature) * direction
        expected_gain = torch.dot(gradient, full_step)
        if not torch.isfinite(expected_gain) or expected_gain <= 0:
            return result

        accepted = False
        try:
            with torch.no_grad():
                for backtrack in range(self.config.backtrack_steps):
                    fraction = self.config.backtrack_coefficient**backtrack
                    vector_to_parameters(
                        old_parameters + fraction * full_step, parameters
                    )
                    # Reject invalid candidates, such as overflowing Gaussian scales.
                    try:
                        candidate_objective, candidate_kl = objective_and_kl()
                    except ValueError:
                        continue
                    gain = candidate_objective - old_objective
                    if (
                        torch.isfinite(candidate_kl)
                        and torch.isfinite(gain)
                        and candidate_kl <= self.config.max_kl
                        and gain > 0
                        and gain >= self.config.accept_ratio * fraction * expected_gain
                    ):
                        accepted = True
                        result = {
                            "train/kl": candidate_kl.item(),
                            "train/surrogate_gain": gain.item(),
                            "train/step_fraction": fraction,
                        }
                        break
        finally:
            if not accepted:
                with torch.no_grad():
                    vector_to_parameters(old_parameters, parameters)
        return result

    def _conjugate_gradient(
        self,
        matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Approximately solve the damped Fisher system without a dense matrix."""
        x = torch.zeros_like(b)
        residual = b.clone()
        direction = residual.clone()
        squared = torch.dot(residual, residual)
        for _ in range(self.config.cg_steps):
            if squared <= self.config.cg_tolerance:
                break
            product = matrix_vector_product(direction)
            curvature = torch.dot(direction, product)
            if not torch.isfinite(curvature) or curvature <= 0:
                break
            alpha = squared / curvature
            x = x + alpha * direction
            residual = residual - alpha * product
            next_squared = torch.dot(residual, residual)
            direction = residual + (next_squared / squared) * direction
            squared = next_squared
        return x

    def _train_step(self) -> dict[str, float | int]:
        batch = self.rollout_buffer.batch(self.device)
        advantages = batch.advantages.detach()
        if self.config.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        metrics = self._update_actor(
            batch.observations,
            self.policy_action_space.action_batch_tensor(batch.actions),
            advantages,
            batch.log_probabilities,
        )
        for _ in range(self.config.value_epochs):
            value_loss = nn.functional.mse_loss(
                self.value_network(batch.observations), batch.returns.detach()
            )
            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.value_network.parameters(), self.config.max_grad_norm
            )
            self.value_optimizer.step()
        self.num_updates += 1
        with torch.no_grad():
            entropy = (
                self.policy_action_space.distribution(
                    self.policy_network(batch.observations)
                )
                .entropy()
                .mean()
            )
        return {
            **metrics,
            "train/value_loss": value_loss.item(),
            "train/entropy": entropy.item(),
            "train/updates": self.num_updates,
        }

    def _gae_lambda(self) -> float:
        return self.config.gae_lambda
