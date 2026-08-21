"""Continuous-action coverage shared by all policy-gradient algorithms."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from aprenderl import (
    A2C,
    REINFORCE,
    A2CConfig,
    ActorCritic,
    ActorCriticConfig,
    REINFORCEConfig,
)
from aprenderl.networks import GaussianPolicyNetwork, ValueNetwork

AgentFactory = Callable[[gym.Env[Any, Any]], Any]


class ContinuousBandit(gym.Env[np.ndarray, np.ndarray]):
    """One-step task whose optimal bounded action is 0.6."""

    observation_space = gym.spaces.Box(
        -1.0, 1.0, shape=(1,), dtype=np.float32
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = 1.0 - float((action[0] - 0.6) ** 2)
        return np.zeros(1, dtype=np.float32), reward, True, False, {}


class UnboundedContinuousBandit(gym.Env[np.ndarray, np.ndarray]):
    """One-step continuous task without action bounds."""

    observation_space = gym.spaces.Box(
        -1.0, 1.0, shape=(3,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(1,), dtype=np.float32
    )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = 1.0 - float((action[0] - 0.5) ** 2)
        return np.zeros(3, dtype=np.float32), reward, True, False, {}


def reinforce(env: gym.Env[Any, Any]) -> REINFORCE:
    return REINFORCE(
        env,
        config=REINFORCEConfig(
            learning_rate=1e-2,
            value_learning_rate=1e-2,
            episodes_per_update=1,
            use_baseline=True,
            seed=7,
        ),
        device="cpu",
    )


def actor_critic(env: gym.Env[Any, Any]) -> ActorCritic:
    return ActorCritic(
        env,
        config=ActorCriticConfig(
            learning_rate=1e-2,
            value_learning_rate=1e-2,
            n_steps=2,
            seed=7,
        ),
        device="cpu",
    )


def a2c(env: gym.Env[Any, Any]) -> A2C:
    return A2C(
        env,
        config=A2CConfig(
            learning_rate=1e-2,
            value_learning_rate=1e-2,
            n_steps=2,
            seed=7,
        ),
        device="cpu",
    )


@pytest.mark.parametrize("factory", [reinforce, actor_critic, a2c])
def test_continuous_policy_predicts_actions_inside_box(factory: AgentFactory) -> None:
    env = gym.make("Pendulum-v1")
    try:
        agent = factory(env)
        observation, _ = env.reset(seed=3)
        deterministic_action = agent.predict(observation, deterministic=True)
        sampled_action = agent.predict(observation, deterministic=False)
    finally:
        env.close()

    assert isinstance(agent.policy_network, GaussianPolicyNetwork)
    assert agent.policy_action_space.squashed
    assert isinstance(deterministic_action, np.ndarray)
    assert env.action_space.contains(deterministic_action)
    assert env.action_space.contains(sampled_action)


@pytest.mark.parametrize("factory", [reinforce, actor_critic, a2c])
def test_continuous_policy_and_value_networks_update(factory: AgentFactory) -> None:
    env = gym.make("Pendulum-v1", max_episode_steps=2)
    try:
        agent = factory(env)
        policy_before = [
            parameter.detach().clone()
            for parameter in agent.policy_network.parameters()
        ]
        assert agent.value_network is not None
        value_before = [
            parameter.detach().clone() for parameter in agent.value_network.parameters()
        ]

        agent.learn(2, progress_bar=False)
        metrics = agent.logger.dump(agent.num_timesteps)
    finally:
        env.close()

    assert agent.num_updates == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            policy_before, agent.policy_network.parameters(), strict=True
        )
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            value_before, agent.value_network.parameters(), strict=True
        )
    )
    assert all(
        parameter.isfinite().all()
        for parameter in agent.policy_network.parameters()
    )
    assert all(
        parameter.isfinite().all()
        for parameter in agent.value_network.parameters()
    )
    assert all(
        np.isfinite(value)
        for key, value in metrics.items()
        if key.startswith("train/")
    )


@pytest.mark.parametrize(
    ("algorithm_class", "factory"),
    [
        (REINFORCE, reinforce),
        (ActorCritic, actor_critic),
        (A2C, a2c),
    ],
)
def test_continuous_checkpoint_round_trip(
    algorithm_class: type[Any],
    factory: AgentFactory,
    tmp_path: Path,
) -> None:
    env = gym.make("Pendulum-v1", max_episode_steps=2)
    restored_env = gym.make("Pendulum-v1", max_episode_steps=2)
    checkpoint = tmp_path / f"{algorithm_class.__name__.lower()}.pt"
    try:
        agent = factory(env)
        agent.learn(2, progress_bar=False)
        agent.save(checkpoint)
        restored = algorithm_class.load(checkpoint, restored_env, device="cpu")
        observation, _ = restored_env.reset(seed=11)
        expected_action = agent.predict(observation, deterministic=True)
        actual_action = restored.predict(observation, deterministic=True)
    finally:
        env.close()
        restored_env.close()

    assert restored.num_timesteps == agent.num_timesteps
    assert restored.num_updates == agent.num_updates
    np.testing.assert_allclose(actual_action, expected_action)
    for expected, actual in zip(
        agent.policy_network.parameters(),
        restored.policy_network.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("factory", [reinforce, actor_critic, a2c])
def test_continuous_policy_validates_network_output(factory: AgentFactory) -> None:
    env = gym.make("Pendulum-v1")
    try:
        bad_network = nn.Sequential(nn.Flatten(), nn.Linear(3, 1))
        with pytest.raises(ValueError, match="means, log_stds"):
            if factory is reinforce:
                REINFORCE(env, bad_network, device="cpu")
            elif factory is actor_critic:
                ActorCritic(env, bad_network, device="cpu")
            else:
                A2C(env, bad_network, device="cpu")
    finally:
        env.close()


@pytest.mark.parametrize("factory", [reinforce, actor_critic, a2c])
def test_policy_gradients_reject_unsupported_action_spaces(
    factory: AgentFactory,
) -> None:
    env = gym.Env()
    env.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
    env.action_space = gym.spaces.MultiDiscrete([2, 2])
    try:
        with pytest.raises(TypeError, match="Discrete or Box action space"):
            factory(env)
    finally:
        env.close()


@pytest.mark.parametrize("factory", [reinforce, actor_critic, a2c])
def test_policy_gradients_reject_partially_bounded_action_spaces(
    factory: AgentFactory,
) -> None:
    env = gym.Env()
    env.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
    env.action_space = gym.spaces.Box(
        np.array([-np.inf, 0.0], dtype=np.float32),
        np.array([np.inf, np.inf], dtype=np.float32),
    )
    try:
        with pytest.raises(ValueError, match="finite or fully unbounded"):
            factory(env)
    finally:
        env.close()


@pytest.mark.parametrize("factory", [reinforce, actor_critic, a2c])
def test_unbounded_actions_use_plain_gaussian_and_update(
    factory: AgentFactory,
) -> None:
    env = UnboundedContinuousBandit()
    try:
        agent = factory(env)
        observation, _ = env.reset(seed=3)
        action = agent.predict(observation)
        agent.learn(2, progress_bar=False)
    finally:
        env.close()

    assert not agent.policy_action_space.squashed
    assert env.action_space.contains(action)
    assert agent.num_updates >= 1
    assert all(
        parameter.isfinite().all()
        for parameter in agent.policy_network.parameters()
    )


def test_continuous_policy_preserves_multidimensional_action_shape() -> None:
    env = gym.Env()
    env.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
    env.action_space = gym.spaces.Box(-2.0, 2.0, shape=(2, 2))
    try:
        agent = A2C(env, config=A2CConfig(n_steps=2), device="cpu")
        action = agent.predict(np.zeros(3, dtype=np.float32))
    finally:
        env.close()

    assert action.shape == (2, 2)
    assert env.action_space.contains(action)


def test_continuous_policy_respects_exact_float64_bounds() -> None:
    env = gym.Env()
    env.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
    env.action_space = gym.spaces.Box(
        np.array([0.1], dtype=np.float64),
        np.array([0.3], dtype=np.float64),
        dtype=np.float64,
    )
    try:
        agent = A2C(env, config=A2CConfig(n_steps=2), device="cpu")
        action = agent.policy_action_space.action_from_tensor(torch.tensor([[1.0]]))
    finally:
        env.close()

    assert env.action_space.contains(action)
    assert action[0] == pytest.approx(0.3)


def test_continuous_checkpoint_validates_action_bounds(tmp_path: Path) -> None:
    env = gym.make("Pendulum-v1")
    mismatched_env = gym.Env()
    mismatched_env.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,))
    mismatched_env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))
    checkpoint = tmp_path / "bounded_a2c.pt"
    try:
        agent = a2c(env)
        agent.save(checkpoint)
        with pytest.raises(ValueError, match="spaces do not match"):
            A2C.load(checkpoint, mismatched_env, device="cpu")
    finally:
        env.close()
        mismatched_env.close()


@pytest.mark.parametrize(
    ("algorithm_class", "factory"),
    [
        (REINFORCE, reinforce),
        (ActorCritic, actor_critic),
        (A2C, a2c),
    ],
)
def test_plain_gaussian_checkpoint_round_trip(
    algorithm_class: type[Any],
    factory: AgentFactory,
    tmp_path: Path,
) -> None:
    env = UnboundedContinuousBandit()
    restored_env = UnboundedContinuousBandit()
    checkpoint = tmp_path / f"plain_{algorithm_class.__name__.lower()}.pt"
    try:
        agent = factory(env)
        agent.learn(2, progress_bar=False)
        agent.save(checkpoint)
        restored = algorithm_class.load(checkpoint, restored_env, device="cpu")
        observation, _ = restored_env.reset(seed=11)
        expected_action = agent.predict(observation, deterministic=True)
        actual_action = restored.predict(observation, deterministic=True)
    finally:
        env.close()
        restored_env.close()

    assert not restored.policy_action_space.squashed
    np.testing.assert_allclose(actual_action, expected_action)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("algorithm_class", "config"),
    [
        (
            REINFORCE,
            REINFORCEConfig(
                learning_rate=1e-2,
                value_learning_rate=1e-2,
                episodes_per_update=16,
                use_baseline=True,
                seed=7,
            ),
        ),
        (
            ActorCritic,
            ActorCriticConfig(
                learning_rate=1e-2,
                value_learning_rate=1e-2,
                n_steps=16,
                seed=7,
            ),
        ),
        (
            A2C,
            A2CConfig(
                learning_rate=1e-2,
                value_learning_rate=1e-2,
                n_steps=16,
                normalize_advantage=True,
                seed=7,
            ),
        ),
    ],
)
def test_continuous_policy_learns_a_bounded_target_action(
    algorithm_class: type[Any],
    config: REINFORCEConfig | ActorCriticConfig | A2CConfig,
) -> None:
    env = ContinuousBandit()
    torch.manual_seed(3)
    policy_network = GaussianPolicyNetwork(1, 1, hidden_sizes=())
    value_network = ValueNetwork(1, hidden_sizes=())
    try:
        agent = algorithm_class(
            env,
            policy_network,
            value_network=value_network,
            config=config,
            device="cpu",
        )
        observation = np.zeros(1, dtype=np.float32)
        initial_action = agent.predict(observation, deterministic=True)
        agent.learn(2_000, progress_bar=False)
        learned_action = agent.predict(observation, deterministic=True)
    finally:
        env.close()

    assert abs(float(initial_action[0]) - 0.6) > 1.0
    assert abs(float(learned_action[0]) - 0.6) < 0.35
