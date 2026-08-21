"""Tests for exploration schedules."""

import numpy as np
import pytest
import torch

from aprenderl.policies import EpsilonGreedyPolicy, LinearSchedule


def test_linear_schedule_interpolates_and_clamps() -> None:
    schedule = LinearSchedule(start=1.0, end=0.1, duration=100)

    assert schedule.value(-1) == pytest.approx(1.0)
    assert schedule.value(50) == pytest.approx(0.55)
    assert schedule.value(100) == pytest.approx(0.1)
    assert schedule.value(200) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "q_values",
    [np.asarray([1.0, 3.0, 2.0]), torch.tensor([1.0, 3.0, 2.0])],
)
def test_epsilon_greedy_policy_accepts_tabular_and_neural_values(
    q_values: np.ndarray | torch.Tensor,
) -> None:
    policy = EpsilonGreedyPolicy(LinearSchedule(1.0, 1.0, 1), seed=4)

    assert policy.select(q_values, step=0, deterministic=True) == 1
    assert policy.select(q_values, step=0, deterministic=False) in range(3)
