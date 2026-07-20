"""Versioned contracts for truthful environment and run rendering."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .core import SchemaModel

ENVIRONMENT_RENDER_SCHEMA_VERSION = "iso-obs.environment-render.v1"
RUN_PLAYBACK_SCHEMA_VERSION = "iso-obs.run-playback.v1"


class RenderPoint(SchemaModel):
    """A two-dimensional point in the declared environment coordinate frame."""

    x: float
    y: float


class RenderBoundary(SchemaModel):
    """A simulator-declared polygon with explicit semantics."""

    id: str
    kind: Literal["world", "navigable", "obstacle", "restricted", "spawn", "goal"]
    points: list[RenderPoint] = Field(min_length=3)
    label: str | None = None


class EnvironmentRenderManifest(SchemaModel):
    """Exact renderable geometry pinned to one environment version."""

    schema_version: Literal["iso-obs.environment-render.v1"] = (
        "iso-obs.environment-render.v1"
    )
    coordinate_frame: str = Field(description="Simulator coordinate-frame name.")
    units: str = Field(description="Coordinate unit, for example 'm'.")
    boundaries: list[RenderBoundary] = Field(min_length=1)
    source_digest: str = Field(
        description="Digest of the simulator artifact that declared the geometry."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlaybackSample(SchemaModel):
    """One observed trace sample; values are preserved without interpolation."""

    step: int | None = None
    monotonic_ns: int
    event_type: str
    observation: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)


class RunPlayback(SchemaModel):
    """A run's recorded samples bound to its pinned environment version."""

    schema_version: Literal["iso-obs.run-playback.v1"] = "iso-obs.run-playback.v1"
    run_id: str
    environment_version_id: str
    samples: list[PlaybackSample] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
