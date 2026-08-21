"""Tests for callbacks, distributions, evaluation, and wrappers."""

import gymnasium as gym
import torch

from aprenderl import DQN, DQNConfig
from aprenderl.callbacks import BaseCallback
from aprenderl.distributions import (
    CategoricalDistribution,
    DiagonalGaussianDistribution,
    SquashedGaussianDistribution,
)
from aprenderl.envs import EpisodeStatsWrapper
from aprenderl.types import Transition
from aprenderl.utils import evaluate_policy


class StopAfterOneStep(BaseCallback):
    def on_step(self, algorithm: DQN, transition: Transition) -> bool:
        return False


def test_callback_can_stop_training() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = DQN(
            env,
            config=DQNConfig(buffer_size=8, batch_size=2, seed=1),
            callback=StopAfterOneStep(),
            device="cpu",
        ).learn(10)
    finally:
        env.close()

    assert agent.num_timesteps == 1


def test_categorical_distribution_shapes() -> None:
    distribution = CategoricalDistribution(torch.tensor([[0.0, 2.0], [3.0, 0.0]]))

    assert distribution.sample().shape == (2,)
    assert distribution.mode().tolist() == [1, 0]
    assert distribution.log_prob(torch.tensor([1, 0])).shape == (2,)
    assert distribution.entropy().shape == (2,)


def test_squashed_gaussian_distribution_is_bounded_and_differentiable() -> None:
    means = torch.zeros((4, 2), requires_grad=True)
    log_stds = torch.zeros((4, 2), requires_grad=True)
    low = torch.tensor([-2.0, -1.0])
    high = torch.tensor([2.0, 3.0])
    distribution = SquashedGaussianDistribution(means, log_stds, low, high)

    actions = distribution.sample()
    modes = distribution.mode()
    log_probabilities = distribution.log_prob(actions.detach())
    entropy = distribution.entropy()
    (-log_probabilities.mean()).backward()

    assert actions.shape == (4, 2)
    assert torch.all(actions >= low)
    assert torch.all(actions <= high)
    torch.testing.assert_close(modes, torch.tensor([[0.0, 1.0]]).expand(4, -1))
    assert log_probabilities.shape == (4,)
    assert entropy.shape == (4,)
    assert torch.isfinite(log_probabilities).all()
    assert means.grad is not None
    assert log_stds.grad is not None


def test_diagonal_gaussian_distribution_is_unsquashed() -> None:
    means = torch.tensor([[2.0, -3.0], [1.0, 4.0]], requires_grad=True)
    log_stds = torch.zeros((2, 2), requires_grad=True)
    distribution = DiagonalGaussianDistribution(means, log_stds)

    actions = distribution.sample()
    modes = distribution.mode()
    log_probabilities = distribution.log_prob(actions.detach())
    entropy = distribution.entropy()
    (-log_probabilities.mean()).backward()

    torch.testing.assert_close(modes, means.detach())
    assert actions.shape == (2, 2)
    assert log_probabilities.shape == (2,)
    assert entropy.shape == (2,)
    assert torch.isfinite(actions).all()
    assert means.grad is not None
    assert log_stds.grad is not None


def test_wrapper_and_evaluation_handle_episode_end() -> None:
    env = EpisodeStatsWrapper(gym.make("CartPole-v1", max_episode_steps=1))
    evaluation_env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        observation, _ = env.reset(seed=2)
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        agent = DQN(
            evaluation_env,
            config=DQNConfig(buffer_size=8, batch_size=2, seed=2),
            device="cpu",
        )
        result = evaluate_policy(agent, evaluation_env, episodes=2, seed=10)
    finally:
        env.close()
        evaluation_env.close()

    assert observation.shape == (4,)
    assert terminated or truncated
    assert info["episode"]["length"] == 1
    assert result.lengths == (2, 2)
