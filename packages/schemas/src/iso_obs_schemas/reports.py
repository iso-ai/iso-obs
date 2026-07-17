"""Reliability report models for the Reliability Studio contract.

A :class:`ReliabilityReport` is the human-facing summary produced after a suite
execution: a set of :class:`Finding` statements, each backed by the runs that
support it and tagged with how confident and how directly evidenced it is.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from .core import SchemaModel, utcnow
from .ids import ProjectId, ReportId, RunId, SuiteExecutionId, WorkspaceId


class Provenance(StrEnum):
    """Whether a finding is directly observed or inferred from evidence."""

    OBSERVED = "observed"
    INFERRED = "inferred"


class Finding(SchemaModel):
    """A single conclusion in a reliability report.

    A finding pairs a natural-language ``statement`` with the concrete runs that
    back it, so every claim in a report is traceable to evidence.
    """

    statement: str = Field(description="The conclusion, in plain language.")
    evidence: str | None = Field(
        default=None, description="Narrative or quantitative support for the claim."
    )
    supporting_run_ids: list[RunId] = Field(
        default_factory=list, description="Runs that substantiate this finding."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the finding, in [0, 1].",
    )
    observed_or_inferred: Provenance = Field(
        default=Provenance.OBSERVED,
        description="Whether the finding is directly observed or inferred.",
    )


class ReliabilityReport(SchemaModel):
    """A report summarizing the reliability of a suite execution."""

    id: ReportId
    workspace_id: WorkspaceId
    project_id: ProjectId
    suite_execution_id: SuiteExecutionId = Field(
        description="Suite execution this report summarizes."
    )
    title: str = Field(description="Human-readable report title.")
    summary: str | None = Field(
        default=None, description="Executive summary of the report."
    )
    findings: list[Finding] = Field(
        default_factory=list, description="Findings that make up the report."
    )
    created_at: datetime = Field(default_factory=utcnow)
