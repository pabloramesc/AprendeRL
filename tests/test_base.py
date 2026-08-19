"""Tests for the algorithm interfaces."""

import inspect

from aprenderl.algorithms import (
    BaseAlgorithm,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
)


def test_algorithm_interfaces_are_abstract() -> None:
    assert inspect.isabstract(BaseAlgorithm)
    assert inspect.isabstract(OnPolicyAlgorithm)
    assert inspect.isabstract(OffPolicyAlgorithm)
    assert {"learn", "predict", "save", "load"}.issubset(
        BaseAlgorithm.__abstractmethods__
    )
