"""Tests for tabular SARSA."""

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from aprenderl import SARSA, SARSAConfig
from aprenderl.types import Transition


class TwoStepEnv(gym.Env[int, int]):
    """Small deterministic environment that records behavior actions."""

    observation_space = gym.spaces.Discrete(3)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.state = 0
        self.actions: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        self.state += 1
        return self.state, 0.0, self.state == 2, False, {}


def test_q_table_is_compact_float32_array() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        agent = SARSA(env)
    finally:
        env.close()

    assert agent.q_table.shape == (16, 4)
    assert agent.q_table.dtype == np.float32
    assert agent.q_table.flags.c_contiguous


def test_update_uses_selected_behavior_action_and_terminal_mask() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        agent = SARSA(env, config=SARSAConfig(learning_rate=0.5, gamma=0.9))
        agent.q_table[1] = [8.0, 4.0, 2.0, 1.0]
        agent._update(Transition(0, 1, 2.0, 1, False, False, {}), next_action=1)
        agent._update(Transition(2, 1, 2.0, 1, True, False, {}), next_action=None)
    finally:
        env.close()

    assert agent.q_table[0, 1] == pytest.approx(2.8)
    assert agent.q_table[2, 1] == pytest.approx(1.0)
    assert agent.num_updates == 2


def test_sampled_next_action_is_reused_for_the_next_step() -> None:
    env = TwoStepEnv()
    config = SARSAConfig(
        exploration_initial_epsilon=0.0,
        exploration_final_epsilon=0.0,
    )
    try:
        agent = SARSA(env, config=config)
        agent.q_table[0] = [2.0, 1.0]
        agent.q_table[1] = [1.0, 2.0]
        agent.learn(1, progress_bar=False)

        # The next action was already sampled as action 1. Changing the greedy
        # action now must not change the behavior action used by SARSA.
        agent.q_table[1] = [3.0, 0.0]
        agent.learn(1, progress_bar=False)
    finally:
        env.close()

    assert env.actions == [0, 1]


def test_truncation_still_bootstraps() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        agent = SARSA(env, config=SARSAConfig(learning_rate=1.0, gamma=0.5))
        agent.q_table[1, 3] = 6.0
        agent._update(
            Transition(0, 2, 1.0, 1, False, True, {}),
            next_action=3,
        )
    finally:
        env.close()

    assert agent.q_table[0, 2] == pytest.approx(4.0)


def test_learning_and_checkpoint_round_trip(tmp_path) -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=20)
    restored_env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=20)
    checkpoint = tmp_path / "frozen_lake.sarsa"
    config = SARSAConfig(exploration_steps=20, seed=7)
    try:
        agent = SARSA(env, config=config).learn(40, progress_bar=False)
        agent.save(checkpoint)
        restored = SARSA.load(checkpoint, restored_env)
    finally:
        env.close()
        restored_env.close()

    np.testing.assert_array_equal(restored.q_table, agent.q_table)
    assert restored.num_timesteps == agent.num_timesteps == 40
    assert restored.num_updates == agent.num_updates == 40
    assert restored.episode_returns == agent.episode_returns
    assert checkpoint.is_file()


def test_discrete_space_offsets_are_supported() -> None:
    env = gym.Env()
    env.observation_space = gym.spaces.Discrete(3, start=5)
    env.action_space = gym.spaces.Discrete(2, start=10)
    try:
        agent = SARSA(env)
        agent.q_table[1] = [1.0, 3.0]
        action = agent.predict(6, deterministic=True)
    finally:
        env.close()

    assert action == 11


def test_requires_discrete_spaces() -> None:
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(TypeError, match="Discrete observation"):
            SARSA(env)
    finally:
        env.close()
