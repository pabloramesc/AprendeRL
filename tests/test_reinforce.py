"""Unit and interaction tests for REINFORCE."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from aprenderl import REINFORCE, REINFORCEConfig
from aprenderl.networks import PolicyNetwork, ValueNetwork


def small_config(**overrides: object) -> REINFORCEConfig:
    values: dict[str, object] = {
        "learning_rate": 1e-2,
        "episodes_per_update": 2,
        "seed": 7,
    }
    values.update(overrides)
    return REINFORCEConfig(**values)  # type: ignore[arg-type]


def test_reinforce_uses_default_policy_network() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = REINFORCE(env, device="cpu")
    finally:
        env.close()

    assert isinstance(agent.policy_network, PolicyNetwork)
    assert agent.value_network is None


def test_reinforce_can_use_learned_value_baseline() -> None:
    env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        agent = REINFORCE(
            env,
            config=small_config(use_baseline=True),
            device="cpu",
        )
        assert isinstance(agent.value_network, ValueNetwork)
        assert agent.value_optimizer is not None
        value_parameters_before = [
            parameter.detach().clone()
            for parameter in agent.value_network.parameters()
        ]

        agent.learn(4, progress_bar=False)
        metrics = agent.logger.dump(agent.num_timesteps)
    finally:
        env.close()

    assert agent.num_updates == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            value_parameters_before,
            agent.value_network.parameters(),
            strict=True,
        )
    )
    assert "train/value_loss" in metrics
    assert "train/advantage_mean" in metrics


def test_discounted_returns_are_reward_to_go() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = REINFORCE(
            env,
            config=small_config(gamma=0.5),
            device="cpu",
        )
        returns = agent._discounted_returns([1.0, 2.0, 3.0])
    finally:
        env.close()

    assert returns == pytest.approx([2.75, 3.5, 3.0])


def test_baseline_uses_return_minus_value_as_advantage() -> None:
    env = gym.make("CartPole-v1")
    value_network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 1),
        nn.Flatten(start_dim=0),
    )
    with torch.no_grad():
        linear = value_network[1]
        assert isinstance(linear, nn.Linear)
        linear.weight.zero_()
        linear.bias.fill_(0.5)
    try:
        agent = REINFORCE(
            env,
            value_network=value_network,
            config=small_config(use_baseline=True, normalize_returns=False),
            device="cpu",
        )
        agent._batch_observations = [
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        ]
        agent._batch_actions = [0, 1]
        agent._batch_returns = [1.0, 2.0]
        agent._batch_episodes = 2

        metrics = agent._update_policy()
    finally:
        env.close()

    assert metrics["train/advantage_mean"] == pytest.approx(1.0)
    assert metrics["train/value_loss"] == pytest.approx(1.25)


def test_policy_updates_only_after_complete_episode_batch() -> None:
    env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        agent = REINFORCE(env, config=small_config(), device="cpu")
        agent.learn(3, progress_bar=False)
        assert agent.num_updates == 0
        agent.learn(1, progress_bar=False)
    finally:
        env.close()

    assert agent.num_timesteps == 4
    assert agent.num_updates == 1
    assert agent.episode_lengths == [2, 2]


def test_learning_and_checkpoint_round_trip(tmp_path) -> None:
    env = gym.make("CartPole-v1", max_episode_steps=2)
    restored_env = gym.make("CartPole-v1", max_episode_steps=2)
    checkpoint = tmp_path / "reinforce.pt"
    try:
        agent = REINFORCE(env, config=small_config(), device="cpu")
        parameters_before = [
            parameter.detach().clone()
            for parameter in agent.policy_network.parameters()
        ]
        agent.learn(4, progress_bar=False)
        agent.save(checkpoint)
        restored = REINFORCE.load(checkpoint, restored_env, device="cpu")
    finally:
        env.close()
        restored_env.close()

    assert agent.num_updates == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before, agent.policy_network.parameters(), strict=True
        )
    )
    assert restored.num_timesteps == agent.num_timesteps
    assert restored.num_updates == agent.num_updates
    for expected, actual in zip(
        agent.policy_network.parameters(),
        restored.policy_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_baseline_checkpoint_round_trip(tmp_path) -> None:
    env = gym.make("CartPole-v1", max_episode_steps=2)
    restored_env = gym.make("CartPole-v1", max_episode_steps=2)
    checkpoint = tmp_path / "reinforce_baseline.pt"
    try:
        agent = REINFORCE(
            env,
            config=small_config(use_baseline=True, value_learning_rate=3e-3),
            device="cpu",
        )
        agent.learn(4, progress_bar=False)
        agent.save(checkpoint)
        restored = REINFORCE.load(checkpoint, restored_env, device="cpu")
    finally:
        env.close()
        restored_env.close()

    assert restored.config.use_baseline
    assert restored.config.value_learning_rate == pytest.approx(3e-3)
    assert agent.value_network is not None
    assert restored.value_network is not None
    for expected, actual in zip(
        agent.value_network.parameters(),
        restored.value_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_custom_network_and_output_shape_validation(tmp_path) -> None:
    env = gym.make("CartPole-v1")
    restored_env = gym.make("CartPole-v1")
    checkpoint = tmp_path / "custom.pt"
    try:
        network = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        agent = REINFORCE(env, network, config=small_config(), device="cpu")
        agent.save(checkpoint)
        with pytest.raises(ValueError, match="requires network"):
            REINFORCE.load(checkpoint, restored_env, device="cpu")
        with pytest.raises(ValueError, match="must return shape"):
            REINFORCE(
                env,
                nn.Sequential(nn.Flatten(), nn.Linear(4, 3)),
                config=small_config(),
                device="cpu",
            )
        with pytest.raises(ValueError, match="requires use_baseline"):
            REINFORCE(
                env,
                value_network=ValueNetwork(4),
                config=small_config(),
                device="cpu",
            )
        with pytest.raises(ValueError, match="value_network must return shape"):
            REINFORCE(
                env,
                value_network=nn.Sequential(nn.Flatten(), nn.Linear(4, 2)),
                config=small_config(use_baseline=True),
                device="cpu",
            )
    finally:
        env.close()
        restored_env.close()


def test_reinforce_requires_box_observations_and_discrete_actions() -> None:
    discrete_observation_env = gym.make("FrozenLake-v1")
    continuous_action_env = gym.make("Pendulum-v1")
    try:
        with pytest.raises(TypeError, match="Box observation"):
            REINFORCE(discrete_observation_env, device="cpu")
        with pytest.raises(TypeError, match="Discrete action"):
            REINFORCE(continuous_action_env, device="cpu")
    finally:
        discrete_observation_env.close()
        continuous_action_env.close()


def test_reinforce_config_validates_value_learning_rate() -> None:
    with pytest.raises(ValueError, match="value_learning_rate must be positive"):
        REINFORCEConfig(value_learning_rate=0.0)
