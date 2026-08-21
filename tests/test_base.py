"""Tests for the algorithm interfaces."""

import inspect

from aprenderl.algorithms import (
    DQN,
    REINFORCE,
    SARSA,
    BaseAlgorithm,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
    QLearning,
)


def test_algorithm_interfaces_are_abstract() -> None:
    assert inspect.isabstract(BaseAlgorithm)
    assert inspect.isabstract(OnPolicyAlgorithm)
    assert inspect.isabstract(OffPolicyAlgorithm)
    assert {"learn", "predict", "save", "load"}.issubset(
        BaseAlgorithm.__abstractmethods__
    )


def test_algorithms_use_their_policy_family_base_class() -> None:
    assert issubclass(DQN, OffPolicyAlgorithm)
    assert issubclass(QLearning, OffPolicyAlgorithm)
    assert issubclass(SARSA, OnPolicyAlgorithm)
    assert issubclass(REINFORCE, OnPolicyAlgorithm)
