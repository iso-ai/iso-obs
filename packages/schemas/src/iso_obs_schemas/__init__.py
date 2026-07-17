"""iso-obs-schemas: the authoritative shared schema contract for Reliability Studio.

This package defines the Pydantic v2 models that the Python SDK (``iso_obs``),
the FastAPI backend (``apps/api``), and the web app all depend on. It is the
single source of truth for object shapes, the typed-id convention, the trace
event schema, metrics/gates, and reliability reports. Every other package treats
these models as a contract: if a shape changes here, it changes everywhere.

The public surface is re-exported at the top level so consumers can write
``from iso_obs_schemas import Run, BaseEvent`` without knowing the module layout.
"""

from __future__ import annotations

from .core import (
    Environment,
    EnvironmentVersion,
    EvaluationSuite,
    PerturbationSpec,
    Project,
    Run,
    Scenario,
    ScenarioVersion,
    SchemaModel,
    System,
    SystemVersion,
    Workspace,
    utcnow,
)
from .enums import (
    EventType,
    FailureCategory,
    MetricDirection,
    PerturbationFamily,
    RunStatus,
    Severity,
    SystemType,
    WorkspaceRole,
)
from .events import (
    TRACE_VERSION,
    ActionPayload,
    ArtifactRef,
    BaseEvent,
    ConstraintViolationPayload,
    ObservationPayload,
)
from .ids import (
    ApiKeyId,
    ComparisonId,
    EnvironmentId,
    EnvironmentVersionId,
    EvaluationSuiteId,
    EventId,
    FailureId,
    IdPrefix,
    ProjectId,
    ReportId,
    RunId,
    ScenarioId,
    ScenarioVersionId,
    SuiteExecutionId,
    SystemId,
    SystemVersionId,
    WebhookId,
    WorkspaceId,
    generate_id,
    prefix_of,
)
from .metrics import (
    ComparisonOperator,
    Gate,
    GateRequirement,
    Invariant,
    InvariantMode,
    InvariantScope,
    MetricDefinition,
    MetricValue,
    MetricValueKind,
    RegressionRequirement,
)
from .reports import Finding, Provenance, ReliabilityReport

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # ids
    "IdPrefix",
    "generate_id",
    "prefix_of",
    "WorkspaceId",
    "ProjectId",
    "SystemId",
    "SystemVersionId",
    "EnvironmentId",
    "EnvironmentVersionId",
    "ScenarioId",
    "ScenarioVersionId",
    "EvaluationSuiteId",
    "SuiteExecutionId",
    "RunId",
    "EventId",
    "FailureId",
    "ComparisonId",
    "ReportId",
    "ApiKeyId",
    "WebhookId",
    # enums
    "EventType",
    "RunStatus",
    "FailureCategory",
    "Severity",
    "SystemType",
    "PerturbationFamily",
    "MetricDirection",
    "WorkspaceRole",
    # core
    "SchemaModel",
    "utcnow",
    "Workspace",
    "Project",
    "System",
    "SystemVersion",
    "Environment",
    "EnvironmentVersion",
    "Scenario",
    "ScenarioVersion",
    "PerturbationSpec",
    "EvaluationSuite",
    "Run",
    # events
    "TRACE_VERSION",
    "BaseEvent",
    "ArtifactRef",
    "ObservationPayload",
    "ActionPayload",
    "ConstraintViolationPayload",
    # metrics
    "MetricValue",
    "MetricValueKind",
    "MetricDefinition",
    "Invariant",
    "InvariantScope",
    "InvariantMode",
    "ComparisonOperator",
    "Gate",
    "GateRequirement",
    "RegressionRequirement",
    # reports
    "Finding",
    "Provenance",
    "ReliabilityReport",
]
