"""Unit and interaction tests for Double DQN."""

import gymnasium as gym
import torch

from aprenderl import DoubleDQN, DoubleDQNConfig
from aprenderl.buffers import ReplayBatch
from aprenderl.networks import QNetwork


def small_config(**overrides: object) -> DoubleDQNConfig:
    values: dict[str, object] = {
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 8,
        "train_frequency": 1,
        "target_update_interval": 5,
        "exploration_fraction": 0.5,
        "hidden_sizes": (16,),
        "seed": 3,
    }
    values.update(overrides)
    return DoubleDQNConfig(**values)  # type: ignore[arg-type]


def test_double_dqn_uses_q_network() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = DoubleDQN(env, small_config(), device="cpu")
    finally:
        env.close()

    assert isinstance(agent.q_network, QNetwork)


def test_dqn_updates_and_round_trips_checkpoint(tmp_path) -> None:
    env = gym.make("CartPole-v1", max_episode_steps=20)
    restored_env = gym.make("CartPole-v1", max_episode_steps=20)
    try:
        agent = DoubleDQN(env, small_config(), device="cpu").learn(24)
        observation, _ = env.reset(seed=9)
        action = agent.predict(observation)
        checkpoint = tmp_path / "agent.pt"
        agent.save(checkpoint)
        restored = DoubleDQN.load(checkpoint, restored_env, device="cpu")
    finally:
        env.close()
        restored_env.close()

    assert agent.num_updates > 0
    assert action in (0, 1)
    assert restored.num_timesteps == agent.num_timesteps
    assert restored.num_updates == agent.num_updates
    for expected, actual in zip(
        agent.q_network.parameters(), restored.q_network.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_target_network_is_updated_on_schedule() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = DoubleDQN(
            env,
            small_config(
                learning_starts=10,
                target_update_interval=2,
            ),
            device="cpu",
        )
        with torch.no_grad():
            next(agent.q_network.parameters()).add_(1.0)
        agent.learn(1)
        assert any(
            not torch.equal(online, target)
            for online, target in zip(
                agent.q_network.parameters(),
                agent.target_network.parameters(),
                strict=True,
            )
        )
        agent.learn(1)
    finally:
        env.close()

    for online, target in zip(
        agent.q_network.parameters(), agent.target_network.parameters(), strict=True
    ):
        torch.testing.assert_close(online, target)


def test_double_target_selects_online_and_bootstraps_only_on_truncation() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = DoubleDQN(
            env,
            small_config(gamma=0.5, hidden_sizes=()),
            device="cpu",
        )
        with torch.no_grad():
            for parameter in agent.q_network.parameters():
                parameter.zero_()
            for parameter in agent.target_network.parameters():
                parameter.zero_()
            agent.q_network.q_head.bias.copy_(torch.tensor([10.0, 0.0]))
            agent.target_network.q_head.bias.copy_(torch.tensor([1.0, 5.0]))
        batch = ReplayBatch(
            observations=torch.zeros(2, 4),
            actions=torch.zeros(2, 1, dtype=torch.int64),
            rewards=torch.ones(2, 1),
            next_observations=torch.zeros(2, 4),
            terminated=torch.tensor([[1.0], [0.0]]),
            truncated=torch.tensor([[0.0], [1.0]]),
        )
        target = agent._td_target(batch)
    finally:
        env.close()

    torch.testing.assert_close(target, torch.tensor([[1.0], [1.5]]))


def test_environment_truncation_is_recorded_separately() -> None:
    env = gym.make("CartPole-v1", max_episode_steps=1)
    try:
        agent = DoubleDQN(
            env,
            small_config(learning_starts=10),
            device="cpu",
        ).learn(1)
    finally:
        env.close()

    assert agent.replay_buffer.terminated[0, 0] == 0
    assert agent.replay_buffer.truncated[0, 0] == 1
    assert agent.episode_lengths == [1]
