"""Tests for the custom metric decorator and registry."""

from __future__ import annotations

import pytest

from iso_obs.metrics import metric, registered_metrics
from iso_obs_schemas import MetricDefinition, MetricDirection, MetricValueKind


def test_metric_registers_and_returns_function_unchanged() -> None:
    @metric("tracking_error", "minimize")
    def tracking_error(errors: list[float]) -> float:
        """Return the worst absolute tracking error."""
        return max(abs(e) for e in errors)

    assert tracking_error([-2.0, 1.0]) == 2.0
    registered = registered_metrics()["tracking_error"]
    assert registered.compute is tracking_error
    definition = registered.definition
    assert isinstance(definition, MetricDefinition)
    assert definition.key == "tracking_error"
    assert definition.kind is MetricValueKind.SCALAR
    assert definition.direction is MetricDirection.MINIMIZE
    attached = tracking_error.__iso_obs_metric__  # type: ignore[attr-defined]
    assert attached == definition


def test_metric_accepts_enum_direction() -> None:
    @metric("success_rate", MetricDirection.MAXIMIZE)
    def success_rate(outcomes: list[bool]) -> float:
        """Return the fraction of successful outcomes."""
        return sum(outcomes) / len(outcomes)

    direction = registered_metrics()["success_rate"].definition.direction
    assert direction is MetricDirection.MAXIMIZE


def test_metric_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError):
        metric("bogus", "sideways")
