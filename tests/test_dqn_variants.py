"""Target, replay, smoke, and checkpoint tests for the DQN family."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from aprenderl import (
    C51,
    IQN,
    QRDQN,
    C51Config,
    DoubleDQN,
    DoubleDQNConfig,
    DuelingDQN,
    DuelingDQNConfig,
    IQNConfig,
    NoisyDQN,
    NoisyDQNConfig,
    NStepDQN,
    NStepDQNConfig,
    PrioritizedDQN,
    PrioritizedDQNConfig,
    QRDQNConfig,
    RainbowDQN,
    RainbowDQNConfig,
)
from aprenderl.buffers import PrioritizedReplayBuffer, ReplayBatch
from aprenderl.networks import DuelingQNetwork, NoisyQNetwork


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


def test_double_dqn_selects_online_and_evaluates_target() -> None:
    env = gym.make("CartPole-v1")
    try:
        agent = DoubleDQN(
            env, config=DoubleDQNConfig(**config_values()), device="cpu"
        )
        with torch.no_grad():
            for parameter in agent.q_network.parameters():
                parameter.zero_()
            for parameter in agent.target_network.parameters():
                parameter.zero_()
            agent.q_network.q_head.bias.copy_(torch.tensor([5.0, 1.0]))
            agent.target_network.q_head.bias.copy_(torch.tensor([2.0, 9.0]))
        batch = ReplayBatch(
            observations=torch.zeros(1, 4),
            actions=torch.zeros(1, 1, dtype=torch.int64),
            rewards=torch.ones(1, 1),
            next_observations=torch.zeros(1, 4),
            terminated=torch.zeros(1, 1),
            truncated=torch.zeros(1, 1),
        )
        target = agent._td_target(batch)
    finally:
        env.close()

    assert target.item() == pytest.approx(1.0 + agent.config.gamma * 2.0)


def test_dueling_and_noisy_variants_use_corresponding_networks() -> None:
    env = gym.make("CartPole-v1")
    try:
        dueling = DuelingDQN(
            env, config=DuelingDQNConfig(**config_values()), device="cpu"
        )
        noisy = NoisyDQN(
            env, config=NoisyDQNConfig(**config_values()), device="cpu"
        )
    finally:
        env.close()

    assert isinstance(dueling.q_network, DuelingQNetwork)
    assert isinstance(noisy.q_network, NoisyQNetwork)
    assert noisy.epsilon == 0.0


def test_prioritized_replay_updates_sample_priorities() -> None:
    buffer = PrioritizedReplayBuffer(8, (2,), seed=1)
    for index in range(4):
        observation = np.full(2, index, dtype=np.float32)
        buffer.add(observation, 0, 0.0, observation, False)
    batch = buffer.sample(4, torch.device("cpu"), beta=0.4)
    buffer.update_priorities(batch.indices, np.full(4, 3.0))

    assert batch.weights.shape == (4, 1)
    assert np.all(buffer.priorities[batch.indices] == 3.0)


def test_n_step_dqn_stores_discounted_return() -> None:
    env = gym.make("CartPole-v1")
    config = NStepDQNConfig(**config_values(), gamma=0.5, n_steps=2)
    try:
        agent = NStepDQN(env, config=config, device="cpu")
        observation = np.zeros(4, dtype=np.float32)
        agent._pending.extend(
            [
                agent_transition(observation, 1.0, False),
                agent_transition(observation, 2.0, False),
            ]
        )
        agent._store_return(2)
    finally:
        env.close()

    assert agent.replay_buffer.rewards[0, 0] == pytest.approx(2.0)
    assert agent.replay_buffer.discounts[0, 0] == pytest.approx(0.25)


def agent_transition(
    observation: np.ndarray, reward: float, terminated: bool
):
    from aprenderl.types import Transition

    return Transition(
        observation,
        0,
        reward,
        observation.copy(),
        terminated,
        False,
        {},
    )


@pytest.mark.parametrize(
    ("algorithm_type", "config"),
    [
        (PrioritizedDQN, PrioritizedDQNConfig(**config_values())),
        (C51, C51Config(**config_values(), atoms=11)),
        (QRDQN, QRDQNConfig(**config_values(), quantiles=8)),
        (
            IQN,
            IQNConfig(**config_values(), quantiles=8, target_quantiles=8),
        ),
        (RainbowDQN, RainbowDQNConfig(**config_values(), atoms=11)),
    ],
)
def test_advanced_dqn_variants_train_and_checkpoint(
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
        projection = agent._project_distribution(batch)
    finally:
        env.close()

    torch.testing.assert_close(projection.sum(dim=1), torch.ones(2))
