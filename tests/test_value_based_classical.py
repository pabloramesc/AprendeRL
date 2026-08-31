"""Focused tests for the classical value-based algorithms."""

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from aprenderl import (
    DynaQ,
    DynaQConfig,
    ExpectedSARSA,
    ExpectedSARSAConfig,
    MonteCarloControl,
    MonteCarloControlConfig,
    MonteCarloPrediction,
    MultiArmedBandit,
    MultiArmedBanditConfig,
    NStepSARSA,
    NStepSARSAConfig,
    PolicyIteration,
    SARSALambda,
    SARSALambdaConfig,
    ValueIteration,
)
from aprenderl.types import Transition


class BanditEnv(gym.Env[int, int]):
    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(3)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        return 0, float(action), True, False, {}


def test_bandit_uses_incremental_sample_averages() -> None:
    env = BanditEnv()
    config = MultiArmedBanditConfig(
        exploration_initial_epsilon=1.0,
        exploration_final_epsilon=1.0,
        seed=2,
    )
    try:
        agent = MultiArmedBandit(env, config=config).learn(100, progress_bar=False)
    finally:
        env.close()

    np.testing.assert_allclose(agent.q_values, [0.0, 1.0, 2.0])
    assert agent.action_counts.sum() == 100


@pytest.mark.parametrize("algorithm_type", [ValueIteration, PolicyIteration])
def test_dynamic_programming_solves_deterministic_frozen_lake(
    algorithm_type: type[ValueIteration] | type[PolicyIteration],
) -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        agent = algorithm_type(env).learn(100, progress_bar=False)
    finally:
        env.close()

    assert agent.converged
    assert agent.values[0] > 0
    assert agent.predict(0, deterministic=True) in (1, 2)


def test_monte_carlo_prediction_and_control_update_complete_episode() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=1)
    policy = np.zeros(16, dtype=np.int64)
    try:
        prediction = MonteCarloPrediction(policy, env).learn(
            1, progress_bar=False
        )
        control = MonteCarloControl(
            env,
            config=MonteCarloControlConfig(
                exploration_initial_epsilon=0.0,
                exploration_final_epsilon=0.0,
            ),
        ).learn(1, progress_bar=False)
    finally:
        env.close()

    assert prediction.visit_counts[0] == 1
    assert control.visit_counts.sum() == 1


def test_expected_sarsa_uses_epsilon_greedy_expectation() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    config = ExpectedSARSAConfig(
        learning_rate=1.0,
        gamma=0.5,
        exploration_initial_epsilon=0.2,
        exploration_final_epsilon=0.2,
    )
    try:
        agent = ExpectedSARSA(env, config=config)
        agent.q_table[1] = [1.0, 2.0, 3.0, 4.0]
        agent._update_from_transition(
            Transition(0, 0, 1.0, 1, False, False, {})
        )
    finally:
        env.close()

    expected_next = 0.8 * 4.0 + 0.2 * 2.5
    assert agent.q_table[0, 0] == pytest.approx(1.0 + 0.5 * expected_next)


def test_n_step_sarsa_uses_n_rewards_and_bootstrap() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    config = NStepSARSAConfig(
        learning_rate=1.0,
        gamma=0.5,
        n_steps=2,
        exploration_initial_epsilon=0.0,
        exploration_final_epsilon=0.0,
    )
    try:
        agent = NStepSARSA(env, config=config)
        agent.q_table[2, 1] = 8.0
        agent._pending.extend(
            [
                (Transition(0, 0, 1.0, 1, False, False, {}), 0),
                (Transition(1, 0, 2.0, 2, False, False, {}), 1),
            ]
        )
        agent._update_oldest(2)
    finally:
        env.close()

    assert agent.q_table[0, 0] == pytest.approx(4.0)


def test_sarsa_lambda_updates_earlier_eligible_pair() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    config = SARSALambdaConfig(
        learning_rate=1.0,
        gamma=1.0,
        trace_decay=0.5,
        exploration_initial_epsilon=0.0,
        exploration_final_epsilon=0.0,
    )
    try:
        agent = SARSALambda(env, config=config)
        agent._update_from_transition(
            Transition(0, 0, 0.0, 1, False, False, {})
        )
        agent._update_from_transition(
            Transition(1, 0, 2.0, 2, True, False, {})
        )
    finally:
        env.close()

    assert agent.q_table[0, 0] == pytest.approx(1.0)
    assert agent.q_table[1, 0] == pytest.approx(2.0)
    assert not agent.eligibility.any()


def test_dyna_q_runs_requested_planning_updates() -> None:
    env = gym.make("FrozenLake-v1", is_slippery=False)
    try:
        agent = DynaQ(
            env,
            config=DynaQConfig(
                planning_steps=4,
                exploration_initial_epsilon=0.0,
                exploration_final_epsilon=0.0,
            ),
        )
        agent._update_from_transition(
            Transition(0, 0, 1.0, 1, False, False, {})
        )
    finally:
        env.close()

    assert agent.num_updates == 5
    assert len(agent.model) == 1
