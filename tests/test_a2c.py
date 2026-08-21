"""Unit and interaction tests for Advantage Actor-Critic."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from aprenderl import A2C, A2CConfig
from aprenderl.networks import PolicyNetwork, ValueNetwork
from aprenderl.types import Transition


def small_config(**overrides: object) -> A2CConfig:
    values: dict[str, object] = {
        "learning_rate": 1e-2,
        "value_learning_rate": 1e-2,
        "n_steps": 3,
        "seed": 7,
    }
    values.update(overrides)
    return A2CConfig(**values)  # type: ignore[arg-type]


def transition(
    reward: float,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> Transition[np.ndarray, int]:
    return Transition(
        observation=np.zeros(4, dtype=np.float32),
        action=0,
        reward=reward,
        next_observation=np.ones(4, dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
        info={},
    )


def test_a2c_uses_default_networks_and_configures_gae() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = A2C(env, config=small_config(gae_lambda=0.75), device="cpu")
    finally:
        env.close()

    assert isinstance(agent.policy_network, PolicyNetwork)
    assert isinstance(agent.value_network, ValueNetwork)
    assert agent.rollout_buffer.gae_lambda == pytest.approx(0.75)


def test_a2c_uses_multi_step_gae_advantages() -> None:
    env = gym.make("CartPole-v1")
    value_network = ValueNetwork(4, hidden_sizes=())
    try:
        agent = A2C(
            env,
            value_network=value_network,
            config=small_config(
                gamma=1.0,
                gae_lambda=1.0,
                value_loss_coefficient=0.5,
            ),
            device="cpu",
        )
        with torch.no_grad():
            value_network.value_head.weight.zero_()
            value_network.value_head.bias.zero_()

        assert agent._update_from_transition(transition(1.0)) == {}
        assert agent._update_from_transition(transition(1.0)) == {}
        metrics = agent._update_from_transition(
            transition(1.0, terminated=True)
        )
    finally:
        env.close()

    # With zero values and lambda=1, the targets are [3, 2, 1].
    assert metrics["train/advantage"] == pytest.approx(2.0)
    assert metrics["train/td_target"] == pytest.approx(2.0)
    assert metrics["train/value_loss"] == pytest.approx(14 / 3)
    expected_loss = (
        metrics["train/policy_loss"]
        - agent.config.entropy_coefficient * metrics["train/entropy"]
        + 0.5 * metrics["train/value_loss"]
    )
    assert metrics["train/loss"] == pytest.approx(expected_loss)


def test_a2c_normalizes_policy_advantages() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = A2C(
            env,
            config=small_config(normalize_advantage=True),
            device="cpu",
        )
        advantages = agent._prepare_advantages(torch.tensor([1.0, 2.0, 3.0]))
    finally:
        env.close()

    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-7)
    assert advantages.std(unbiased=False).item() == pytest.approx(1.0)


def test_a2c_updates_after_each_complete_rollout() -> None:
    env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        agent = A2C(env, config=small_config(), device="cpu")
        policy_parameters_before = [
            parameter.detach().clone()
            for parameter in agent.policy_network.parameters()
        ]
        value_parameters_before = [
            parameter.detach().clone() for parameter in agent.value_network.parameters()
        ]
        agent.learn(7, progress_bar=False)
    finally:
        env.close()

    assert agent.num_timesteps == 7
    assert agent.num_updates == 2
    assert len(agent.rollout_buffer) == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            policy_parameters_before,
            agent.policy_network.parameters(),
            strict=True,
        )
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            value_parameters_before,
            agent.value_network.parameters(),
            strict=True,
        )
    )


def test_a2c_checkpoint_round_trip_uses_a2c_config(tmp_path) -> None:
    env = gym.make("CartPole-v1", max_episode_steps=2)
    restored_env = gym.make("CartPole-v1", max_episode_steps=2)
    checkpoint = tmp_path / "a2c.pt"
    try:
        agent = A2C(
            env,
            config=small_config(
                gae_lambda=0.8,
                normalize_advantage=True,
                value_loss_coefficient=0.25,
            ),
            device="cpu",
        )
        agent.learn(6, progress_bar=False)
        agent.save(checkpoint)
        restored = A2C.load(checkpoint, restored_env, device="cpu")
    finally:
        env.close()
        restored_env.close()

    assert isinstance(restored.config, A2CConfig)
    assert restored.config.gae_lambda == pytest.approx(0.8)
    assert restored.config.normalize_advantage
    assert restored.config.value_loss_coefficient == pytest.approx(0.25)
    assert restored.num_timesteps == agent.num_timesteps
    assert restored.num_updates == agent.num_updates
    for expected, actual in zip(
        agent.policy_network.parameters(),
        restored.policy_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(
        agent.value_network.parameters(),
        restored.value_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_custom_a2c_network_checkpoints_require_matching_modules(tmp_path) -> None:
    env = gym.make("CartPole-v1")
    restored_env = gym.make("CartPole-v1")
    checkpoint = tmp_path / "custom_a2c.pt"
    try:
        agent = A2C(
            env,
            nn.Sequential(nn.Flatten(), nn.Linear(4, 2)),
            value_network=ValueNetwork(4, hidden_sizes=()),
            config=small_config(),
            device="cpu",
        )
        agent.save(checkpoint)
        with pytest.raises(ValueError, match="requires policy_network"):
            A2C.load(
                checkpoint,
                restored_env,
                value_network=ValueNetwork(4, hidden_sizes=()),
                device="cpu",
            )
        with pytest.raises(ValueError, match="requires value_network"):
            A2C.load(
                checkpoint,
                restored_env,
                policy_network=nn.Sequential(nn.Flatten(), nn.Linear(4, 2)),
                device="cpu",
            )
    finally:
        env.close()
        restored_env.close()


def test_a2c_requires_box_observations() -> None:
    discrete_observation_env = gym.make("FrozenLake-v1")
    try:
        with pytest.raises(TypeError, match="A2C requires a Box observation"):
            A2C(discrete_observation_env, device="cpu")
    finally:
        discrete_observation_env.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gae_lambda": -0.1}, "gae_lambda must be between 0 and 1"),
        (
            {"value_loss_coefficient": -0.1},
            "value_loss_coefficient cannot be negative",
        ),
    ],
)
def test_a2c_config_validation(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        A2CConfig(**overrides)  # type: ignore[arg-type]
