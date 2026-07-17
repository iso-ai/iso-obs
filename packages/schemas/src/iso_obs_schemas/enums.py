"""Shared enumerations for the Reliability Studio schema contract.

These string enumerations are the canonical vocabularies used across the SDK,
API, workers, and web app. They are :class:`enum.StrEnum` subclasses so that
each member is a real ``str`` — it serializes to its value directly in JSON and
compares equal to that value — while still giving callers a typed, discoverable
name.
"""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """Kind of a trace event emitted during a run.

    The values follow a ``<domain>.<verb>`` dotted convention so that consumers
    can filter by prefix (e.g. every ``action.*`` event) without an explicit
    grouping enum.
    """

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    ENVIRONMENT_RESET = "environment.reset"
    OBSERVATION_RECEIVED = "observation.received"
    STATE_RECORDED = "state.recorded"
    ACTION_REQUESTED = "action.requested"
    ACTION_EMITTED = "action.emitted"
    ACTION_APPLIED = "action.applied"
    REWARD_RECEIVED = "reward.received"
    METRIC_RECORDED = "metric.recorded"
    INVARIANT_VIOLATED = "invariant.violated"
    CONSTRAINT_VIOLATED = "constraint.violated"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_ERROR = "system.error"
    ENVIRONMENT_EVENT = "environment.event"
    ARTIFACT_CREATED = "artifact.created"
    ANNOTATION_CREATED = "annotation.created"


class RunStatus(StrEnum):
    """Lifecycle state of a single evaluation run.

    The states form a mostly linear progression from ``queued`` through
    ``processing`` to a terminal ``completed``/``failed``/``cancelled`` state.
    """

    QUEUED = "queued"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureCategory(StrEnum):
    """Taxonomy of why a run or system behavior is considered a failure.

    Categories span the perception → planning → control stack plus outcome-level
    failures (non-completion, timeout) and dynamical failures (oscillation,
    instability, divergence).
    """

    UNSAFE_ACTION = "unsafe_action"
    CONSTRAINT_VIOLATION = "constraint_violation"
    NON_COMPLETION = "non_completion"
    TIMEOUT = "timeout"
    OSCILLATION = "oscillation"
    INSTABILITY = "instability"
    DEGRADED_PERFORMANCE = "degraded_performance"
    PERCEPTION_FAILURE = "perception_failure"
    PLANNING_FAILURE = "planning_failure"
    CONTROL_FAILURE = "control_failure"
    RECOVERY_FAILURE = "recovery_failure"
    DIVERGENCE = "divergence"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    """Severity level for violations, findings, and system events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SystemType(StrEnum):
    """Kind of system under evaluation.

    Reliability Studio evaluates decision-making systems across the RL, control,
    and agentic spectrum; this enum names the broad category so metrics and
    default invariants can be specialized per type.
    """

    POLICY = "policy"
    AGENT = "agent"
    CONTROLLER = "controller"
    PLANNER = "planner"
    PERCEPTION = "perception"
    PIPELINE = "pipeline"
    MODEL = "model"
    OTHER = "other"


class PerturbationFamily(StrEnum):
    """Family of perturbation applied to stress-test a system.

    A perturbation family names *how* the world is disturbed; concrete
    parameters live in :class:`iso_obs_schemas.core.PerturbationSpec`.
    """

    OBSERVATION_NOISE = "observation_noise"
    ACTION_NOISE = "action_noise"
    SENSOR_DROPOUT = "sensor_dropout"
    LATENCY = "latency"
    PACKET_LOSS = "packet_loss"
    DYNAMICS_SHIFT = "dynamics_shift"
    DISTURBANCE_FORCE = "disturbance_force"
    INITIAL_CONDITION = "initial_condition"
    REWARD_CORRUPTION = "reward_corruption"
    DOMAIN_RANDOMIZATION = "domain_randomization"
    ADVERSARIAL = "adversarial"
    OTHER = "other"


class MetricDirection(StrEnum):
    """Direction in which a metric improves.

    Determines the sense of threshold comparisons and regression checks: a
    ``maximize`` metric regresses when it drops, a ``minimize`` metric regresses
    when it rises.
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class WorkspaceRole(StrEnum):
    """Role a principal holds within a workspace.

    Roles are ordered from most to least privileged; ``service_account`` is a
    non-human principal used by CI and the SDK.
    """

    OWNER = "owner"
    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"
    SERVICE_ACCOUNT = "service_account"
