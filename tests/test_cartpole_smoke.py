"""Learning smoke test for Double DQN on CartPole."""

import gymnasium as gym
import pytest

from aprenderl import DoubleDQN, DoubleDQNConfig
from aprenderl.utils import evaluate_policy


@pytest.mark.smoke
def test_double_dqn_improves_on_cartpole() -> None:
    train_env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    config = DoubleDQNConfig(
        learning_rate=1e-3,
        buffer_size=8_000,
        batch_size=64,
        learning_starts=500,
        train_frequency=4,
        target_update_interval=250,
        exploration_fraction=0.35,
        hidden_sizes=(64, 64),
        seed=11,
    )
    try:
        agent = DoubleDQN(train_env, config, device="cpu").learn(6_000)
        result = evaluate_policy(agent, eval_env, episodes=5, seed=100)
    finally:
        train_env.close()
        eval_env.close()

    assert agent.num_updates > 0
    assert result.mean_return >= 25.0
