"""Focused tests for the executable pendulum world-model workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "pendulum_world_model_workflow.py"
SPEC = importlib.util.spec_from_file_location("pendulum_world_model_workflow", EXAMPLE)
assert SPEC is not None and SPEC.loader is not None
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


def test_reference_step_and_feature_contract_are_finite() -> None:
    """Declared dynamics produce a bounded state and stable feature width."""
    state = WORKFLOW.reference_step(
        (0.4, 0.2),
        0.8,
        mass=1.1,
        damping=0.08,
    )

    assert all(-8.0 <= value <= 8.0 for value in state)
    assert len(WORKFLOW.features(state, 0.8, 1.1, 0.08)) == len(WORKFLOW.FEATURE_NAMES)


def test_peak_memory_uses_platform_specific_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MacOS byte and Linux kilobyte RSS values both convert to MiB."""
    usage = type("Usage", (), {"ru_maxrss": 64 * 1024 * 1024})()
    monkeypatch.setattr(WORKFLOW.resource, "getrusage", lambda _: usage)
    monkeypatch.setattr(WORKFLOW.sys, "platform", "darwin")
    assert WORKFLOW.peak_memory_mb() == pytest.approx(64.0)

    usage.ru_maxrss = 64 * 1024
    monkeypatch.setattr(WORKFLOW.sys, "platform", "linux")
    assert WORKFLOW.peak_memory_mb() == pytest.approx(64.0)
