"""Generate a small, versioned reliability-boundary report in under two minutes."""

from __future__ import annotations

import argparse
from pathlib import Path

from iso_obs.evidence import sha256_digest
from iso_obs.failure_boundaries import (
    BoundaryAnchor,
    BoundaryDimension,
    BoundaryObservation,
    FailureBoundaryMap,
    FailureBoundaryPlan,
    fit_failure_boundary_model,
    map_failure_boundary,
)


def build_quickstart_report() -> FailureBoundaryMap:
    """Build the deterministic warehouse-visibility quickstart report.

    Returns:
        A versioned boundary map with reliable, unresolved, and unreliable cells.
    """
    plan = FailureBoundaryPlan(
        plan_id="iso-obs-quickstart-visibility-boundary",
        plan_version="1",
        target_population_digest=sha256_digest(
            "quickstart low-speed warehouse intersections"
        ),
        simulation_evidence_digest=sha256_digest(
            "quickstart deterministic warehouse simulator"
        ),
        system_artifact_digest=sha256_digest("quickstart navigation policy v1"),
        failure_outcome_definition_digest=sha256_digest(
            "minimum stopping margin below zero"
        ),
        coordinate_extraction_digest=sha256_digest(
            "normalized visibility degradation: 0 clear, 1 blackout"
        ),
        dimensions=(
            BoundaryDimension(
                dimension_id="visibility_degradation",
                definition_digest=sha256_digest(
                    "normalized visibility degradation: 0 clear, 1 blackout"
                ),
                lower_bound=0.0,
                upper_bound=1.0,
                distance_weight=1.0,
                grid_points=(0.0, 0.25, 0.75, 1.0),
            ),
        ),
        anchors=(
            BoundaryAnchor(
                anchor_id="clear", coordinates=(0.0,), planned_sample_count=200
            ),
            BoundaryAnchor(
                anchor_id="blackout", coordinates=(1.0,), planned_sample_count=200
            ),
        ),
        maximum_acceptable_failure_probability=0.5,
        failure_probability_lipschitz_constant=1.0,
        familywise_error_rate=0.05,
        maximum_grid_cell_count=10,
        limitations=(
            "Quickstart data are deterministic demonstration evidence, not a "
            "claim about a production robot.",
            "Replace the declared anchors and outcomes with fixed-count evidence "
            "from the system under evaluation.",
        ),
    )
    observations = tuple(
        BoundaryObservation(
            plan_content_digest=plan.content_digest(),
            anchor_id=anchor_id,
            sample_index=index,
            trial_id=f"{anchor_id}-trial-{index:04d}",
            evidence_digest=sha256_digest(
                f"quickstart-{anchor_id}-evidence-{index:04d}"
            ),
            failed=anchor_id == "blackout",
        )
        for anchor_id in ("clear", "blackout")
        for index in range(200)
    )
    model = fit_failure_boundary_model(plan, observations)
    return map_failure_boundary(plan, model)


def write_quickstart_report(output: Path) -> FailureBoundaryMap:
    """Write the quickstart report without overwriting existing evidence.

    Args:
        output: Destination JSON path, which must not already exist.

    Returns:
        The report written to ``output``.
    """
    report = build_quickstart_report()
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report.to_json())
        stream.write("\n")
    return report


def main() -> None:
    """Generate the quickstart artifact and print its interpretation path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("failure-boundary-report.json"),
        help="new JSON report path (default: failure-boundary-report.json)",
    )
    args = parser.parse_args()
    report = write_quickstart_report(args.output)
    print(f"Wrote {args.output}")
    print(f"Disposition: {report.disposition.value}")
    print(f"Schema: {report.report_schema_version}")
    print(f"Digest: {report.content_digest()}")
    print("Next: publish this exact report to Reliability Studio")
    print("  1. Create a key: https://iso-obs.studio/settings")
    print("  2. iso auth login --api-key <YOUR_KEY>")
    print(f"  3. iso evidence submit {args.output}")
    print("  4. Inspect it: https://iso-obs.studio/evidence")


if __name__ == "__main__":
    main()
