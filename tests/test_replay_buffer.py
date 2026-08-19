"""Tests for replay storage and sampling."""

import numpy as np
import torch

from aprenderl.buffers import ReplayBuffer


def test_replay_buffer_wraps_and_samples_tensors() -> None:
    buffer = ReplayBuffer(capacity=3, observation_shape=(2,), seed=7)
    for step in range(5):
        observation = np.array([step, step + 1], dtype=np.float32)
        buffer.add(
            observation,
            step % 2,
            float(step),
            observation + 1,
            terminated=step == 4,
            truncated=step == 3,
        )

    batch = buffer.sample(batch_size=4, device=torch.device("cpu"))

    assert len(buffer) == 3
    assert batch.observations.shape == (4, 2)
    assert batch.actions.shape == (4, 1)
    assert batch.actions.dtype == torch.int64
    assert batch.rewards.dtype == torch.float32
    assert batch.terminated.shape == (4, 1)
    assert batch.truncated.shape == (4, 1)
