"""Core domain objects for the Reliability Studio contract.

This module defines the eight core objects and their immutable versions that
make up the Reliability Studio object model:

* :class:`Workspace` and :class:`Project` — tenancy and grouping.
* :class:`System` / :class:`SystemVersion` — the system under evaluation and a
  pinned build of it.
* :class:`Environment` / :class:`EnvironmentVersion` — where a system runs and a
  pinned build of that environment.
* :class:`Scenario` / :class:`ScenarioVersion` — a specific evaluation situation
  (including its :class:`PerturbationSpec` set) and a pinned build of it.
* :class:`EvaluationSuite` — a named collection of scenario versions to run.
* :class:`Run` — a single execution of one scenario version by one system
  version inside a suite execution.

The versioned split is deliberate: definitions (``System``, ``Environment``,
``Scenario``) carry mutable, human-facing metadata, while their ``*Version``
counterparts are immutable, content-pinned snapshots that a :class:`Run`
references so results stay reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import PerturbationFamily, RunStatus, SystemType
from .ids import (
    EnvironmentId,
    EnvironmentVersionId,
    EvaluationSuiteId,
    ProjectId,
    RunId,
    ScenarioId,
    ScenarioVersionId,
    SuiteExecutionId,
    SystemId,
    SystemVersionId,
    WorkspaceId,
)


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Returns:
        A ``datetime`` in UTC. Used as the default factory for creation
        timestamps so every persisted timestamp is unambiguous.
    """
    return datetime.now(UTC)


class SchemaModel(BaseModel):
    """Base class for every Reliability Studio schema model.

    Configures strict, contract-friendly behavior shared by all models: unknown
    fields are rejected so drift is caught at the boundary, enum values are kept
    as enum members (not coerced to bare strings), and assignment is validated.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        ser_json_timedelta="float",
    )


class Workspace(SchemaModel):
    """Top-level tenant that owns projects, members, and billing."""

    id: WorkspaceId
    name: str = Field(description="Human-readable workspace name.")
    slug: str = Field(description="URL-safe unique short name within Iso AI.")
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Project(SchemaModel):
    """A body of evaluation work inside a workspace."""

    id: ProjectId
    workspace_id: WorkspaceId
    name: str = Field(description="Human-readable project name.")
    slug: str = Field(description="URL-safe unique short name within the workspace.")
    description: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class System(SchemaModel):
    """A decision-making system under evaluation (policy, agent, controller...)."""

    id: SystemId
    project_id: ProjectId
    name: str = Field(description="Human-readable system name.")
    system_type: SystemType = Field(description="Broad category of the system.")
    description: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemVersion(SchemaModel):
    """An immutable, content-pinned build of a :class:`System`."""

    id: SystemVersionId
    system_id: SystemId
    version: str = Field(description="Version label, e.g. a semver or build tag.")
    commit_sha: str | None = Field(
        default=None, description="Source revision that produced this build."
    )
    artifact_uri: str | None = Field(
        default=None, description="Location of the system artifact (weights/image)."
    )
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Environment(SchemaModel):
    """A world in which systems are evaluated (a simulator, benchmark, or rig)."""

    id: EnvironmentId
    project_id: ProjectId
    name: str = Field(description="Human-readable environment name.")
    description: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentVersion(SchemaModel):
    """An immutable, content-pinned build of an :class:`Environment`."""

    id: EnvironmentVersionId
    environment_id: EnvironmentId
    version: str = Field(description="Version label for the environment build.")
    image_uri: str | None = Field(
        default=None, description="Container image or artifact for this build."
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Frozen environment configuration."
    )
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerturbationSpec(SchemaModel):
    """A single perturbation applied within a scenario.

    Perturbations are embedded value objects (not standalone persisted entities),
    so they carry no id. They describe *how* the world is disturbed and *when*.
    """

    family: PerturbationFamily = Field(description="Perturbation family.")
    name: str = Field(description="Short label for this perturbation instance.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Family-specific parameters, e.g. noise scale or force.",
    )
    start_step: int | None = Field(
        default=None, description="First step the perturbation is active (inclusive)."
    )
    end_step: int | None = Field(
        default=None, description="Last step the perturbation is active (inclusive)."
    )


class Scenario(SchemaModel):
    """A named evaluation situation defined against an environment."""

    id: ScenarioId
    environment_id: EnvironmentId
    name: str = Field(description="Human-readable scenario name.")
    description: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioVersion(SchemaModel):
    """An immutable, content-pinned build of a :class:`Scenario`.

    Pins the exact perturbation set and configuration so a run against this
    version is reproducible.
    """

    id: ScenarioVersionId
    scenario_id: ScenarioId
    version: str = Field(description="Version label for the scenario build.")
    perturbations: list[PerturbationSpec] = Field(
        default_factory=list, description="Perturbations applied in this scenario."
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Frozen scenario configuration."
    )
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSuite(SchemaModel):
    """A named collection of scenario versions evaluated together.

    Running a suite produces a *suite execution* (identified by a
    :data:`~iso_obs_schemas.ids.SuiteExecutionId`) that groups the individual
    :class:`Run` results.
    """

    id: EvaluationSuiteId
    project_id: ProjectId
    name: str = Field(description="Human-readable suite name.")
    description: str | None = None
    scenario_version_ids: list[ScenarioVersionId] = Field(
        default_factory=list,
        description="Scenario versions included in this suite.",
    )
    seeds: list[int] = Field(
        default_factory=list,
        description="Seeds to run each scenario with; one run per (scenario, seed).",
    )
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Run(SchemaModel):
    """A single execution of one scenario version by one system version.

    A run is the atomic unit of evaluation. It references the exact pinned
    versions it exercised, records its lifecycle status and timing, and points
    at the trace it produced. All timestamps are timezone-aware UTC.
    """

    id: RunId
    workspace_id: WorkspaceId
    project_id: ProjectId
    suite_execution_id: SuiteExecutionId = Field(
        description="Suite execution this run belongs to."
    )
    system_version_id: SystemVersionId
    environment_version_id: EnvironmentVersionId
    scenario_version_id: ScenarioVersionId
    seed: int = Field(description="Random seed used for this run.")
    status: RunStatus = Field(
        default=RunStatus.QUEUED, description="Current lifecycle state."
    )
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = Field(
        default=None, description="When execution began, if it has started."
    )
    finished_at: datetime | None = Field(
        default=None, description="When the run reached a terminal state, if it has."
    )
    failure_reason: str | None = Field(
        default=None, description="Human-readable reason when status is failed."
    )
    worker_id: str | None = Field(
        default=None, description="Identifier of the worker that executed the run."
    )
    trace_uri: str | None = Field(
        default=None, description="Location of the raw event trace for this run."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
