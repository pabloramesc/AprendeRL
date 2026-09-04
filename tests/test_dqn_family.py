"""Training, checkpoint, and target tests for paper-level DQN agents."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from aprenderl import (
    C51,
    DQN,
    FQF,
    IQN,
    QRDQN,
    C51Config,
    FQFConfig,
    IQNConfig,
    QRDQNConfig,
    RainbowDQN,
    RainbowDQNConfig,
)
from aprenderl.buffers import PrioritizedReplayBuffer, ReplayBatch
from aprenderl.types import Transition


def config_values() -> dict[str, int]:
    return {
        "buffer_size": 32,
        "batch_size": 4,
        "learning_starts": 4,
        "train_freq": 1,
        "target_update_interval": 5,
        "exploration_steps": 8,
        "seed": 3,
    }


@pytest.mark.parametrize(
    ("algorithm_type", "module_name"),
    [
        (C51, "aprenderl.algorithms.c51"),
        (DQN, "aprenderl.algorithms.dqn"),
        (FQF, "aprenderl.algorithms.fqf"),
        (IQN, "aprenderl.algorithms.iqn"),
        (QRDQN, "aprenderl.algorithms.qr_dqn"),
        (RainbowDQN, "aprenderl.algorithms.rainbow"),
    ],
)
def test_distributional_agents_live_in_flat_modules(
    algorithm_type,
    module_name: str,
) -> None:
    assert algorithm_type.__module__ == module_name


@pytest.mark.parametrize(
    ("algorithm_type", "config"),
    [
        (C51, C51Config(**config_values(), atoms=11)),
        (QRDQN, QRDQNConfig(**config_values(), quantiles=8)),
        (
            IQN,
            IQNConfig(**config_values(), quantiles=8, target_quantiles=8),
        ),
        (RainbowDQN, RainbowDQNConfig(**config_values(), atoms=11, n_steps=2)),
        (
            FQF,
            FQFConfig(
                **config_values(),
                quantiles=8,
                fraction_learning_rate=1e-3,
            ),
        ),
    ],
)
def test_distributional_agents_train_and_checkpoint(
    algorithm_type,
    config,
    tmp_path,
) -> None:
    env = gym.make("CartPole-v1", max_episode_steps=8)
    restored_env = gym.make("CartPole-v1", max_episode_steps=8)
    try:
        agent = algorithm_type(env, config=config, device="cpu").learn(
            12, progress_bar=False
        )
        checkpoint = tmp_path / f"{algorithm_type.__name__}.pt"
        agent.save(checkpoint)
        restored = algorithm_type.load(checkpoint, restored_env, device="cpu")
    finally:
        env.close()
        restored_env.close()

    assert agent.num_updates > 0
    assert restored.num_timesteps == 12
    assert restored.num_updates == agent.num_updates


def test_c51_projection_is_a_probability_distribution() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = C51(
            env,
            config=C51Config(**config_values(), atoms=11),
            device="cpu",
        )
        batch = ReplayBatch(
            observations=torch.zeros(2, 4),
            actions=torch.zeros(2, 1, dtype=torch.int64),
            rewards=torch.tensor([[0.0], [1.0]]),
            next_observations=torch.zeros(2, 4),
            terminated=torch.tensor([[0.0], [1.0]]),
            truncated=torch.zeros(2, 1),
        )
        projection = agent._target_distribution(batch)
    finally:
        env.close()

    torch.testing.assert_close(projection.sum(dim=1), torch.ones(2))


def test_rainbow_owns_prioritized_n_step_replay() -> None:
    env = gym.make("CartPole-v1")
    config = RainbowDQNConfig(
        **config_values(), atoms=11, gamma=0.5, n_steps=2
    )
    observation = np.zeros(4, dtype=np.float32)
    try:
        agent = RainbowDQN(env, config=config, device="cpu")
        agent._pending.extend(
            [
                Transition(observation, 0, 1.0, observation.copy(), False, False, {}),
                Transition(observation, 0, 2.0, observation.copy(), False, False, {}),
            ]
        )
        agent._store_return(2)
    finally:
        env.close()

    assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)
    assert agent.replay_buffer.rewards[0, 0] == pytest.approx(2.0)
    assert agent.replay_buffer.discounts[0, 0] == pytest.approx(0.25)
    assert agent.epsilon == 0.0


def test_fqf_uses_disjoint_fraction_and_quantile_optimizers() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = FQF(
            env,
            config=FQFConfig(**config_values(), quantiles=8),
            device="cpu",
        )
    finally:
        env.close()

    fraction_parameters = {
        id(parameter)
        for group in agent.fraction_optimizer.param_groups
        for parameter in group["params"]
    }
    quantile_parameters = {
        id(parameter)
        for group in agent.optimizer.param_groups
        for parameter in group["params"]
    }
    assert fraction_parameters
    assert quantile_parameters
    assert fraction_parameters.isdisjoint(quantile_parameters)
