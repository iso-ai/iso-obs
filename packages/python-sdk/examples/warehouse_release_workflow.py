"""Run an end-to-end reliability decision for a warehouse autonomy release.

The individual study modules contain the domain-to-SDK adaptation seams. This
orchestrator shows how a team can run those studies together, preserve each
report as immutable JSON, and make a noncompensatory CI decision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from causal_perturbation_campaign import (
    build_observations as build_causal_observations,
)
from causal_perturbation_campaign import build_plan as build_causal_plan
from failure_boundary_map import (
    build_observations as build_boundary_observations,
)
from failure_boundary_map import build_plan as build_boundary_plan
from sim_to_real_transport import (
    build_observations as build_transport_observations,
)
from sim_to_real_transport import build_plan as build_transport_plan

from iso_obs.causal_perturbations import (
    CausalCampaignDisposition,
    CausalPerturbationReport,
    assess_causal_perturbations,
)
from iso_obs.failure_boundaries import (
    BoundaryCellDisposition,
    BoundaryMapDisposition,
    FailureBoundaryMap,
    fit_failure_boundary_model,
    map_failure_boundary,
)
from iso_obs.transportability import (
    TransportConclusion,
    TransportValidationReport,
    validate_transportability,
)


class JsonReport(Protocol):
    """Protocol shared by content-addressed SDK reports."""

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""

    def content_digest(self) -> str:
        """Return the stable digest of the complete report."""


class ReleaseStatus(StrEnum):
    """Noncompensatory release status for the example workflow."""

    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    """Human-readable release decision derived from scientific reports."""

    status: ReleaseStatus
    blockers: tuple[str, ...]
    next_actions: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the decision for CI logs and build artifacts."""
        return json.dumps(
            {
                "status": self.status.value,
                "blockers": self.blockers,
                "next_actions": self.next_actions,
            },
            indent=2,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class WorkflowReports:
    """Reports produced by the three-stage release workflow."""

    causal: CausalPerturbationReport
    boundary: FailureBoundaryMap
    transport: TransportValidationReport


def run_studies() -> WorkflowReports:
    """Run the causal, boundary, and transport studies.

    Returns:
        Reports that preserve the plan, evidence scope, uncertainty, and
        limitations for each distinct engineering claim.
    """
    causal_plan = build_causal_plan()
    causal = assess_causal_perturbations(
        causal_plan,
        build_causal_observations(causal_plan),
    )

    boundary_plan = build_boundary_plan()
    boundary_model = fit_failure_boundary_model(
        boundary_plan,
        build_boundary_observations(boundary_plan),
    )
    boundary = map_failure_boundary(boundary_plan, boundary_model)

    transport_plan = build_transport_plan()
    transport = validate_transportability(
        transport_plan,
        build_transport_observations(transport_plan),
    )
    return WorkflowReports(
        causal=causal,
        boundary=boundary,
        transport=transport,
    )


def decide_release(reports: WorkflowReports) -> ReleaseDecision:
    """Apply a noncompensatory release policy to the study reports.

    Args:
        reports: Completed scientific reports for the candidate system.

    Returns:
        A blocked decision when any critical evidentiary requirement fails.
    """
    blockers: list[str] = []
    next_actions: list[str] = []

    if (
        reports.causal.disposition
        is not CausalCampaignDisposition.NO_HARMFUL_EFFECT_SUPPORTED
    ):
        if (
            reports.causal.disposition
            is CausalCampaignDisposition.HARMFUL_EFFECT_IDENTIFIED
        ):
            blockers.append("a controlled perturbation has a supported harmful effect")
            next_actions.append(
                "reduce observation-latency sensitivity, then rerun matched replays"
            )
        else:
            blockers.append("the perturbation campaign has not excluded material harm")
            next_actions.append(
                "collect the preregistered matched evidence needed for a conclusion"
            )

    if reports.boundary.disposition is not BoundaryMapDisposition.COMPLETE:
        blockers.append("the operating boundary contains unresolved conditions")
        next_actions.append(
            "sample the unresolved visibility band with predeclared anchors"
        )

    if any(
        cell.disposition is BoundaryCellDisposition.UNRELIABLE_CERTIFIED
        for cell in reports.boundary.cells
    ):
        blockers.append(
            "the intended operating envelope contains unreliable conditions"
        )
        next_actions.append(
            "resolve the blackout failure or exclude that region operationally"
        )

    if (
        reports.transport.conclusion
        is not TransportConclusion.SUPPORTED_WITHIN_ANCHOR_ENVELOPE
    ):
        limiting = ", ".join(reports.transport.limiting_stratum_ids) or "unknown"
        blockers.append(
            f"sim-to-real transport is unsupported in critical strata: {limiting}"
        )
        next_actions.append(
            "improve low-friction physics or narrow the validated operating envelope"
        )

    return ReleaseDecision(
        status=ReleaseStatus.BLOCKED if blockers else ReleaseStatus.READY,
        blockers=tuple(blockers),
        next_actions=tuple(next_actions),
    )


def write_evidence(
    output_dir: Path,
    reports: WorkflowReports,
    decision: ReleaseDecision,
) -> tuple[Path, ...]:
    """Write canonical reports and a CI decision without overwriting evidence.

    Args:
        output_dir: New or empty directory for generated evidence.
        reports: Scientific reports to preserve.
        decision: Derived release decision.

    Returns:
        Paths written by the workflow.

    Raises:
        FileExistsError: If a target evidence file already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: tuple[tuple[str, JsonReport | ReleaseDecision], ...] = (
        ("causal-perturbation-report.json", reports.causal),
        ("failure-boundary-report.json", reports.boundary),
        ("transportability-report.json", reports.transport),
        ("release-decision.json", decision),
    )
    written: list[Path] = []
    for filename, artifact in artifacts:
        path = output_dir / filename
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(artifact.to_json())
            stream.write("\n")
        written.append(path)
    return tuple(written)


def print_summary(
    reports: WorkflowReports,
    decision: ReleaseDecision,
) -> None:
    """Print a decision-oriented summary for engineers and CI logs.

    Args:
        reports: Scientific reports behind the decision.
        decision: Noncompensatory release decision.
    """
    print("Warehouse autonomy release evidence")
    print(f"  release: {decision.status.value}")
    print(
        "  causal campaign: "
        f"{reports.causal.disposition.value} "
        f"({reports.causal.content_digest()})"
    )
    print(
        "  failure boundary: "
        f"{reports.boundary.disposition.value} "
        f"({reports.boundary.content_digest()})"
    )
    print(
        "  sim-to-real transport: "
        f"{reports.transport.conclusion.value} "
        f"({reports.transport.content_digest()})"
    )
    print("  blockers:")
    for blocker in decision.blockers:
        print(f"    - {blocker}")
    print("  next experiments:")
    for action in decision.next_actions:
        print(f"    - {action}")


def parse_args() -> argparse.Namespace:
    """Parse workflow command-line arguments.

    Returns:
        Parsed output and CI behavior options.
    """
    parser = argparse.ArgumentParser(
        description="Run the warehouse autonomy reliability release workflow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write canonical reports to a new or empty evidence directory",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit with status 2 when the release evidence is not ready",
    )
    return parser.parse_args()


def main() -> int:
    """Run the workflow, optionally persist reports, and return a CI status."""
    arguments = parse_args()
    reports = run_studies()
    decision = decide_release(reports)
    print_summary(reports, decision)

    if arguments.output_dir is not None:
        paths = write_evidence(arguments.output_dir, reports, decision)
        print("  evidence artifacts:")
        for path in paths:
            print(f"    - {path}")

    if arguments.require_ready and decision.status is not ReleaseStatus.READY:
        return 2
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
