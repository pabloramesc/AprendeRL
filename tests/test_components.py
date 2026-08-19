"""Tests for callbacks, distributions, evaluation, and wrappers."""

import gymnasium as gym
import torch

from aprenderl import DoubleDQN, DoubleDQNConfig
from aprenderl.callbacks import BaseCallback
from aprenderl.distributions import CategoricalDistribution
from aprenderl.envs import EpisodeStatsWrapper
from aprenderl.types import Transition
from aprenderl.utils import evaluate_policy


class StopAfterOneStep(BaseCallback):
    def on_step(self, algorithm: DoubleDQN, transition: Transition) -> bool:
        return False


def test_callback_can_stop_training() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = DoubleDQN(
            env,
            DoubleDQNConfig(buffer_size=8, batch_size=2, seed=1),
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


def test_wrapper_and_evaluation_handle_episode_end() -> None:
    env = EpisodeStatsWrapper(gym.make("CartPole-v1", max_episode_steps=1))
    evaluation_env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        observation, _ = env.reset(seed=2)
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        agent = DoubleDQN(
            evaluation_env,
            DoubleDQNConfig(buffer_size=8, batch_size=2, seed=2),
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
