"""Tests for the ReliabilityEnvironment adapter interface."""

from __future__ import annotations

from typing import Any

import pytest

from iso_obs.environments import ReliabilityEnvironment


class Pendulum(ReliabilityEnvironment):
    """Minimal concrete adapter used to exercise the ABC."""

    def __init__(self) -> None:
        """Initialize the toy state."""
        self.angle = 0.0
        self.closed = False

    def reset(self, scenario: str, seed: int) -> dict[str, float]:
        """Reset the pendulum to the seeded initial angle."""
        self.angle = float(seed)
        return {"angle": self.angle}

    def observe(self) -> dict[str, float]:
        """Return the current angle."""
        return {"angle": self.angle}

    def step(self, action: Any) -> dict[str, float]:
        """Apply a torque and return the new observation."""
        self.angle += float(action)
        return {"angle": self.angle}

    def get_state(self) -> dict[str, Any]:
        """Return the ground-truth state."""
        return {"angle": self.angle, "closed": self.closed}

    def collect_artifacts(self) -> list[str]:
        """Return no artifacts."""
        return []

    def close(self) -> None:
        """Mark the environment closed."""
        self.closed = True


def test_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ReliabilityEnvironment()  # type: ignore[abstract]


def test_concrete_adapter_protocol() -> None:
    env = Pendulum()
    assert env.reset("swing-up", seed=3) == {"angle": 3.0}
    assert env.observe() == {"angle": 3.0}
    assert env.step(0.5) == {"angle": 3.5}
    assert env.get_state()["closed"] is False
    assert env.collect_artifacts() == []
    env.close()
    assert env.get_state()["closed"] is True
