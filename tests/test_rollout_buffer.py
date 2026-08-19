"""Tests for ordered on-policy rollout storage."""

import numpy as np
import pytest
import torch

from aprenderl.buffers import RolloutBuffer


def test_rollout_buffer_computes_returns_and_advantages() -> None:
    buffer = RolloutBuffer(
        capacity=3,
        observation_shape=(2,),
        gamma=1.0,
        gae_lambda=1.0,
    )
    buffer.add(np.array([0, 1]), 0, 1.0, False, False, 0.5, 0.5, -0.2)
    buffer.add(np.array([1, 2]), 1, 1.0, False, False, 0.5, 0.5, -0.3)
    buffer.add(np.array([2, 3]), 0, 1.0, True, False, 0.5, 0.0, -0.4)

    buffer.compute_returns_and_advantages()
    batch = buffer.batch(torch.device("cpu"))

    assert buffer.full
    torch.testing.assert_close(batch.returns, torch.tensor([3.0, 2.0, 1.0]))
    torch.testing.assert_close(batch.advantages, torch.tensor([2.5, 1.5, 0.5]))
    assert batch.actions.shape == (3,)

    with pytest.raises(ValueError, match="full"):
        buffer.add(np.array([3, 4]), 1, 1.0, False, False, 0.0, 0.0, -0.1)


def test_rollout_buffer_requires_targets_before_batching() -> None:
    buffer = RolloutBuffer(capacity=2, observation_shape=(1,))
    buffer.add(np.array([0]), 0, 1.0, False, False, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="compute returns"):
        buffer.batch(torch.device("cpu"))


def test_rollout_bootstraps_truncation_but_stops_gae_recursion() -> None:
    buffer = RolloutBuffer(
        capacity=2,
        observation_shape=(1,),
        gamma=1.0,
        gae_lambda=1.0,
    )
    buffer.add(np.array([0]), 0, 1.0, False, True, 0.0, 4.0, 0.0)
    buffer.add(np.array([1]), 0, 2.0, True, False, 0.0, 0.0, 0.0)

    buffer.compute_returns_and_advantages()
    batch = buffer.batch(torch.device("cpu"))

    torch.testing.assert_close(batch.returns, torch.tensor([5.0, 2.0]))
