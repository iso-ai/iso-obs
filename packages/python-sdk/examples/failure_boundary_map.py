"""Map certified and unresolved regions of a continuous failure surface.

The example uses fixed sampling at declared anchors and a Lipschitz assumption
to propagate simultaneous failure-probability bounds across a continuous
visibility-degradation axis.
"""

from __future__ import annotations

from iso_obs.evidence import sha256_digest
from iso_obs.failure_boundaries import (
    BoundaryAnchor,
    BoundaryDimension,
    BoundaryObservation,
    FailureBoundaryPlan,
    fit_failure_boundary_model,
    map_failure_boundary,
)


def build_plan() -> FailureBoundaryPlan:
    """Declare the operating dimension, anchors, threshold, and fixed design."""
    return FailureBoundaryPlan(
        plan_id="visibility-failure-boundary",
        plan_version="1",
        target_population_digest=sha256_digest("low-speed warehouse intersections"),
        simulation_evidence_digest=sha256_digest("warehouse-twin-v4.2"),
        system_artifact_digest=sha256_digest("navigation-policy-v18"),
        failure_outcome_definition_digest=sha256_digest(
            "minimum stopping margin below zero"
        ),
        coordinate_extraction_digest=sha256_digest(
            "normalized visibility degradation extractor v1"
        ),
        dimensions=(
            BoundaryDimension(
                dimension_id="visibility_degradation",
                definition_digest=sha256_digest(
                    "zero is clear and one is sensor blackout"
                ),
                lower_bound=0.0,
                upper_bound=1.0,
                distance_weight=1.0,
                grid_points=(0.0, 0.25, 0.75, 1.0),
            ),
        ),
        anchors=(
            BoundaryAnchor(
                anchor_id="clear",
                coordinates=(0.0,),
                planned_sample_count=1_000,
            ),
            BoundaryAnchor(
                anchor_id="blackout",
                coordinates=(1.0,),
                planned_sample_count=1_000,
            ),
        ),
        maximum_acceptable_failure_probability=0.5,
        failure_probability_lipschitz_constant=1.0,
        familywise_error_rate=0.05,
        maximum_grid_cell_count=10,
        limitations=(
            "The smoothness bound is a declared modeling assumption.",
            "The map applies only to the declared system and simulator evidence.",
        ),
    )


def build_observations(
    plan: FailureBoundaryPlan,
) -> tuple[BoundaryObservation, ...]:
    """Build fixed-count outcomes at the two predeclared anchors."""
    plan_digest = plan.content_digest()
    observations: list[BoundaryObservation] = []
    for anchor_id, failure_count in (("clear", 0), ("blackout", 1_000)):
        for index in range(1_000):
            observations.append(
                BoundaryObservation(
                    plan_content_digest=plan_digest,
                    anchor_id=anchor_id,
                    sample_index=index,
                    trial_id=f"{anchor_id}-trial-{index:04d}",
                    evidence_digest=sha256_digest(f"{anchor_id}-evidence-{index:04d}"),
                    failed=index < failure_count,
                )
            )
    return tuple(observations)


def main() -> None:
    """Fit the fixed design and print the three-way boundary partition."""
    plan = build_plan()
    model = fit_failure_boundary_model(plan, build_observations(plan))
    report = map_failure_boundary(plan, model)

    print("Failure boundary map")
    print(f"  model: {model.disposition.value}")
    print(f"  map: {report.disposition.value}")
    for cell in report.cells:
        interval = cell.ranges[0]
        print(
            f"    {interval.lower_bound:.2f}–{interval.upper_bound:.2f}: "
            f"failure probability "
            f"[{cell.failure_probability_lower_bound:.3f}, "
            f"{cell.failure_probability_upper_bound:.3f}] "
            f"→ {cell.disposition.value}"
        )
    print(
        "  geometric partition: "
        f"{report.reliable_geometric_volume:.0%} reliable, "
        f"{report.unresolved_geometric_volume:.0%} unresolved, "
        f"{report.unreliable_geometric_volume:.0%} unreliable"
    )
    print(f"  report digest: {report.content_digest()}")
    print("  claim boundary: the unresolved band is preserved, not interpolated away")


if __name__ == "__main__":
    main()
