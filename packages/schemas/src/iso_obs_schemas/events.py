"""Trace event schema for Reliability Studio.

During a run, the SDK emits a stream of :class:`BaseEvent` records that together
form the run's *trace*. Every event carries the full set of ids needed to place
it in the object model (workspace → project → suite → run, plus the pinned
system / environment / scenario it exercised), a pair of clocks (wall-clock
``timestamp`` and monotonic ``monotonic_ns``), an ``event_type`` drawn from
:class:`~iso_obs_schemas.enums.EventType`, and a free-form ``payload``.

The ``payload`` is intentionally an open ``dict`` on :class:`BaseEvent` so the
trace format is forward-compatible. The typed payload models in this module
(:class:`ObservationPayload`, :class:`ActionPayload`,
:class:`ConstraintViolationPayload`) describe the expected shape of that dict for
the common event types and are used by producers and consumers to validate it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .core import SchemaModel, utcnow
from .enums import EventType, Severity
from .ids import (
    EnvironmentId,
    EnvironmentVersionId,
    EvaluationSuiteId,
    EventId,
    ProjectId,
    RunId,
    ScenarioId,
    SystemId,
    SystemVersionId,
    WorkspaceId,
)

# Current version of the trace event envelope. Bump on breaking changes to
# :class:`BaseEvent` so consumers can dispatch on the shape they received.
TRACE_VERSION = "1.0"


class BaseEvent(SchemaModel):
    """A single event in a run's trace.

    Field order mirrors the canonical trace event schema. The id fields reference
    the same validated id types defined in :mod:`iso_obs_schemas.ids`, so an
    event is guaranteed to be internally consistent with the core object model.
    """

    event_id: EventId
    workspace_id: WorkspaceId
    project_id: ProjectId
    suite_id: EvaluationSuiteId
    run_id: RunId
    system_id: SystemId
    system_version: SystemVersionId = Field(
        description="Pinned system version that produced this event."
    )
    environment_id: EnvironmentId
    environment_version: EnvironmentVersionId = Field(
        description="Pinned environment version this event was produced in."
    )
    scenario_id: ScenarioId
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="Wall-clock time (timezone-aware UTC) the event occurred.",
    )
    monotonic_ns: int = Field(
        description="Monotonic clock reading in nanoseconds for precise ordering."
    )
    step: int | None = Field(
        default=None, description="Environment step index, if applicable."
    )
    event_type: EventType
    source: str = Field(description="Emitter of the event, e.g. 'sdk' or 'worker'.")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Event-type-specific body."
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Structured tags for filtering/indexing."
    )
    trace_version: str = Field(
        default=TRACE_VERSION, description="Version of the trace event envelope."
    )


class ArtifactRef(BaseModel):
    """A reference to a stored artifact (image, video, tensor, file).

    Artifacts live in object storage (Spaces/S3); events and payloads carry a
    lightweight reference rather than the bytes themselves.
    """

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(description="Storage URI of the artifact.")
    kind: str = Field(description="Logical artifact kind, e.g. 'frame' or 'video'.")
    media_type: str | None = Field(
        default=None, description="MIME type, e.g. 'image/png'."
    )
    size_bytes: int | None = Field(
        default=None, description="Artifact size in bytes, if known."
    )
    checksum: str | None = Field(
        default=None, description="Content checksum for integrity/dedup."
    )


class ObservationPayload(BaseModel):
    """Payload for an ``observation.received`` event."""

    model_config = ConfigDict(extra="forbid")

    observation: dict[str, Any] = Field(
        description="The observation delivered to the system this step."
    )
    artifact_refs: list[ArtifactRef] = Field(
        default_factory=list,
        description="References to heavy observation artifacts (e.g. frames).",
    )


class ActionPayload(BaseModel):
    """Payload for ``action.*`` events (requested / emitted / applied)."""

    model_config = ConfigDict(extra="forbid")

    action: dict[str, Any] = Field(description="The action produced by the system.")
    latency_ms: float | None = Field(
        default=None, description="Time taken to produce the action, milliseconds."
    )
    policy_confidence: float | None = Field(
        default=None, description="Optional policy confidence in [0, 1]."
    )


class ConstraintViolationPayload(BaseModel):
    """Payload for a ``constraint.violated`` event."""

    model_config = ConfigDict(extra="forbid")

    constraint: str = Field(description="Name of the violated constraint.")
    observed_value: float = Field(description="Value that breached the constraint.")
    threshold: float = Field(description="Threshold that was breached.")
    severity: Severity = Field(
        default=Severity.WARNING, description="Severity of the violation."
    )
    first_occurrence: bool = Field(
        default=True,
        description="Whether this is the run's first violation of this constraint.",
    )
