"""Unit and interaction tests for one-step actor-critic."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from aprenderl import ActorCritic, ActorCriticConfig
from aprenderl.networks import PolicyNetwork, ValueNetwork
from aprenderl.types import Transition


def small_config(**overrides: object) -> ActorCriticConfig:
    values: dict[str, object] = {
        "learning_rate": 1e-2,
        "value_learning_rate": 1e-2,
        "n_steps": 2,
        "seed": 7,
    }
    values.update(overrides)
    return ActorCriticConfig(**values)  # type: ignore[arg-type]


def transition(*, terminated: bool, truncated: bool) -> Transition[np.ndarray, int]:
    return Transition(
        observation=np.zeros(4, dtype=np.float32),
        action=0,
        reward=1.0,
        next_observation=np.ones(4, dtype=np.float32),
        terminated=terminated,
        truncated=truncated,
        info={},
    )


def test_actor_critic_uses_default_policy_and_value_networks() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = ActorCritic(env, device="cpu")
    finally:
        env.close()

    assert isinstance(agent.policy_network, PolicyNetwork)
    assert isinstance(agent.value_network, ValueNetwork)


def test_actor_and_critic_update_after_each_rollout() -> None:
    env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        agent = ActorCritic(env, config=small_config(), device="cpu")
        policy_parameters_before = [
            parameter.detach().clone()
            for parameter in agent.policy_network.parameters()
        ]
        value_parameters_before = [
            parameter.detach().clone() for parameter in agent.value_network.parameters()
        ]

        agent.learn(3, progress_bar=False)
        assert agent.num_updates == 1
        agent.learn(1, progress_bar=False)
        metrics = agent.logger.dump(agent.num_timesteps)
    finally:
        env.close()

    assert agent.num_timesteps == 4
    assert agent.num_updates == 2
    assert agent.episode_lengths == [2, 2]
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
    assert "train/policy_loss" in metrics
    assert "train/value_loss" in metrics
    assert "train/advantage" in metrics


def test_td_target_stops_only_at_true_termination() -> None:
    env = gym.make("CartPole-v1")
    value_network = ValueNetwork(4, hidden_sizes=())
    try:
        agent = ActorCritic(
            env,
            value_network=value_network,
            config=small_config(gamma=0.5, n_steps=1),
            device="cpu",
        )
        with torch.no_grad():
            value_network.value_head.weight.zero_()
            value_network.value_head.bias.fill_(4.0)

        terminal_target = agent._td_target(
            transition(terminated=True, truncated=False)
        )
        truncated_target = agent._td_target(
            transition(terminated=False, truncated=True)
        )
    finally:
        env.close()

    torch.testing.assert_close(terminal_target, torch.tensor([1.0]))
    torch.testing.assert_close(truncated_target, torch.tensor([3.0]))


def test_td_error_is_used_as_the_actor_advantage() -> None:
    env = gym.make("CartPole-v1")
    value_network = ValueNetwork(4, hidden_sizes=())
    try:
        agent = ActorCritic(
            env,
            value_network=value_network,
            config=small_config(gamma=0.5, n_steps=1),
            device="cpu",
        )
        with torch.no_grad():
            value_network.value_head.weight.zero_()
            value_network.value_head.bias.fill_(2.0)

        metrics = agent._update_from_transition(
            transition(terminated=False, truncated=False)
        )
    finally:
        env.close()

    # target = 1 + 0.5 * 2 = 2 and V(s) = 2 before the update.
    assert metrics["train/td_target"] == pytest.approx(2.0)
    assert metrics["train/value"] == pytest.approx(2.0)
    assert metrics["train/advantage"] == pytest.approx(0.0)
    assert metrics["train/value_loss"] == pytest.approx(0.0)


def test_checkpoint_round_trip_restores_both_networks(tmp_path) -> None:
    env = gym.make("CartPole-v1", max_episode_steps=2)
    restored_env = gym.make("CartPole-v1", max_episode_steps=2)
    checkpoint = tmp_path / "actor_critic.pt"
    try:
        agent = ActorCritic(env, config=small_config(), device="cpu")
        agent.learn(4, progress_bar=False)
        agent.save(checkpoint)
        restored = ActorCritic.load(checkpoint, restored_env, device="cpu")
    finally:
        env.close()
        restored_env.close()

    assert restored.num_timesteps == agent.num_timesteps
    assert restored.num_updates == agent.num_updates
    assert restored.episode_returns == agent.episode_returns
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


def test_custom_network_checkpoints_require_matching_modules(tmp_path) -> None:
    env = gym.make("CartPole-v1")
    restored_env = gym.make("CartPole-v1")
    checkpoint = tmp_path / "custom_actor_critic.pt"
    policy_network = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    value_network = ValueNetwork(4, hidden_sizes=())
    try:
        agent = ActorCritic(
            env,
            policy_network,
            value_network=value_network,
            config=small_config(),
            device="cpu",
        )
        agent.save(checkpoint)
        with pytest.raises(ValueError, match="requires policy_network"):
            ActorCritic.load(
                checkpoint,
                restored_env,
                value_network=ValueNetwork(4, hidden_sizes=()),
                device="cpu",
            )
        with pytest.raises(ValueError, match="requires value_network"):
            ActorCritic.load(
                checkpoint,
                restored_env,
                policy_network=nn.Sequential(nn.Flatten(), nn.Linear(4, 2)),
                device="cpu",
            )
    finally:
        env.close()
        restored_env.close()


def test_network_output_shapes_are_validated() -> None:
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="policy_network must return shape"):
            ActorCritic(
                env,
                nn.Sequential(nn.Flatten(), nn.Linear(4, 3)),
                config=small_config(),
                device="cpu",
            )
        with pytest.raises(ValueError, match="value_network must return shape"):
            ActorCritic(
                env,
                value_network=nn.Sequential(nn.Flatten(), nn.Linear(4, 2)),
                config=small_config(),
                device="cpu",
            )
    finally:
        env.close()


def test_discrete_action_space_offsets_are_supported() -> None:
    env = gym.Env()
    env.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
    env.action_space = gym.spaces.Discrete(2, start=5)
    policy_network = nn.Sequential(nn.Flatten(), nn.Linear(1, 2))
    try:
        agent = ActorCritic(
            env,
            policy_network,
            config=small_config(),
            device="cpu",
        )
        with torch.no_grad():
            policy_network[1].weight.zero_()
            policy_network[1].bias.copy_(torch.tensor([0.0, 1.0]))
        action = agent.predict(np.zeros(1, dtype=np.float32), deterministic=True)
    finally:
        env.close()

    assert action == 6


def test_actor_critic_requires_box_observations() -> None:
    discrete_observation_env = gym.make("FrozenLake-v1")
    try:
        with pytest.raises(TypeError, match="Box observation"):
            ActorCritic(discrete_observation_env, device="cpu")
    finally:
        discrete_observation_env.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"value_learning_rate": 0.0}, "value_learning_rate must be positive"),
        ({"gamma": 1.1}, "gamma must be between 0 and 1"),
        ({"n_steps": 0}, "n_steps must be positive"),
        ({"entropy_coefficient": -0.1}, "entropy_coefficient cannot be negative"),
        ({"max_grad_norm": 0.0}, "max_grad_norm must be positive"),
    ],
)
def test_actor_critic_config_validation(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ActorCriticConfig(**overrides)  # type: ignore[arg-type]
