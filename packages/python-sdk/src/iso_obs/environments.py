"""Environment adapter interface for the iso-obs SDK.

Reliability Studio is simulator-agnostic: to evaluate a system in your own
simulator (Isaac Sim, MuJoCo, Gazebo, CARLA, PyBullet, Gymnasium, or custom),
implement :class:`ReliabilityEnvironment`. The worker drives the adapter with a
``reset -> (observe/step)* -> collect_artifacts -> close`` protocol, so the
platform never needs to know simulator internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ReliabilityEnvironment(ABC):
    """Abstract adapter between Reliability Studio and a simulator."""

    @abstractmethod
    def reset(self, scenario: str, seed: int) -> Any:
        """Reset the environment to the start of a scenario.

        Args:
            scenario: Scenario to load (name or id).
            seed: Random seed making the episode reproducible.

        Returns:
            The initial observation.
        """

    @abstractmethod
    def observe(self) -> Any:
        """Return the current observation without advancing the simulation.

        Returns:
            The observation the system under evaluation would receive now.
        """

    @abstractmethod
    def step(self, action: Any) -> Any:
        """Apply an action and advance the simulation by one step.

        Args:
            action: The action produced by the system under evaluation.

        Returns:
            The simulator's step result (e.g. observation, reward, done).
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return a ground-truth state snapshot for failure analysis.

        Returns:
            Privileged simulator state (poses, contacts, internals) that the
            system under evaluation does not observe.
        """

    @abstractmethod
    def collect_artifacts(self) -> list[str]:
        """Return artifacts produced during the episode.

        Returns:
            Local paths or URIs of artifacts (videos, plots, logs) to attach
            to the run.
        """

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources; called exactly once per episode."""
