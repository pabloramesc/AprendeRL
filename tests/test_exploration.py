"""Tests for exploration schedules."""

import pytest

from aprenderl.policies import LinearSchedule


def test_linear_schedule_interpolates_and_clamps() -> None:
    schedule = LinearSchedule(start=1.0, end=0.1, duration=100)

    assert schedule.value(-1) == pytest.approx(1.0)
    assert schedule.value(50) == pytest.approx(0.55)
    assert schedule.value(100) == pytest.approx(0.1)
    assert schedule.value(200) == pytest.approx(0.1)
