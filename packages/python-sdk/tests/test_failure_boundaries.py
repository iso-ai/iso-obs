"""Tests for Lipschitz-certified continuous failure-boundary mapping."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.evidence import sha256_digest
from iso_obs.failure_boundaries import (
    BoundaryAnchor,
    BoundaryCellDisposition,
    BoundaryDimension,
    BoundaryMapDisposition,
    BoundaryModelDisposition,
    BoundaryObservation,
    FailureBoundaryPlan,
    fit_failure_boundary_model,
    map_failure_boundary,
)
from iso_obs.simulation import EvidenceUse


def dimension(
    dimension_id: str = "friction",
    *,
    grid_points: tuple[float, ...] = (0.0, 0.25, 0.75, 1.0),
) -> BoundaryDimension:
    """Build one normalized continuous boundary dimension."""
    return BoundaryDimension(
        dimension_id=dimension_id,
        definition_digest=sha256_digest(f"definition-{dimension_id}"),
        lower_bound=0.0,
        upper_bound=1.0,
        distance_weight=1.0,
        grid_points=grid_points,
    )


def plan(
    *,
    lipschitz_constant: float = 1.0,
    dimensions: tuple[BoundaryDimension, ...] | None = None,
    anchors: tuple[BoundaryAnchor, ...] | None = None,
    maximum_cells: int = 10,
) -> FailureBoundaryPlan:
    """Build a deterministic fixed-sample boundary plan."""
    resolved_dimensions = dimensions or (dimension(),)
    resolved_anchors = anchors or (
        BoundaryAnchor(
            anchor_id="low",
            coordinates=(0.0,) * len(resolved_dimensions),
            planned_sample_count=1_000,
        ),
        BoundaryAnchor(
            anchor_id="high",
            coordinates=(1.0,) * len(resolved_dimensions),
            planned_sample_count=1_000,
        ),
    )
    return FailureBoundaryPlan(
        plan_id="warehouse-failure-boundary",
        plan_version="1",
        target_population_digest=sha256_digest("target-population"),
        simulation_evidence_digest=sha256_digest("simulator"),
        system_artifact_digest=sha256_digest("system"),
        failure_outcome_definition_digest=sha256_digest("collision"),
        coordinate_extraction_digest=sha256_digest("coordinate-extractor"),
        dimensions=resolved_dimensions,
        anchors=resolved_anchors,
        maximum_acceptable_failure_probability=0.5,
        failure_probability_lipschitz_constant=lipschitz_constant,
        familywise_error_rate=0.1,
        maximum_grid_cell_count=maximum_cells,
        limitations=("Normalized friction study only.",),
    )


def observations(
    boundary_plan: FailureBoundaryPlan,
    anchor_id: str,
    count: int,
    *,
    failure_count: int,
) -> tuple[BoundaryObservation, ...]:
    """Build exact indexed binary observations for one anchor."""
    plan_digest = boundary_plan.content_digest()
    return tuple(
        BoundaryObservation(
            plan_content_digest=plan_digest,
            anchor_id=anchor_id,
            sample_index=index,
            trial_id=f"{anchor_id}-trial-{index}",
            evidence_digest=sha256_digest(f"evidence-{anchor_id}-{index}"),
            failed=index < failure_count,
        )
        for index in range(count)
    )


def polarized_evidence(
    boundary_plan: FailureBoundaryPlan,
    *,
    count: int = 1_000,
) -> tuple[BoundaryObservation, ...]:
    """Build low-risk and high-risk anchor outcomes."""
    return (
        *observations(
            boundary_plan,
            "low",
            count,
            failure_count=0,
        ),
        *observations(
            boundary_plan,
            "high",
            count,
            failure_count=count,
        ),
    )


def test_partial_map_certifies_tails_and_preserves_boundary() -> None:
    """Evidence certifies far tails while withholding the transition region."""
    boundary_plan = plan()
    model = fit_failure_boundary_model(
        boundary_plan,
        polarized_evidence(boundary_plan),
    )

    report = map_failure_boundary(boundary_plan, model)

    assert model.disposition is BoundaryModelDisposition.READY
    assert report.disposition is BoundaryMapDisposition.PARTIAL
    assert [item.disposition for item in report.cells] == [
        BoundaryCellDisposition.RELIABLE_CERTIFIED,
        BoundaryCellDisposition.UNRESOLVED_BOUNDARY,
        BoundaryCellDisposition.UNRELIABLE_CERTIFIED,
    ]
    assert report.reliable_geometric_volume == pytest.approx(0.25)
    assert report.unresolved_geometric_volume == pytest.approx(0.5)
    assert report.unreliable_geometric_volume == pytest.approx(0.25)
    assert sum(
        item.normalized_geometric_volume for item in report.cells
    ) == pytest.approx(1.0)


def test_complete_map_requires_every_cell_to_clear_threshold() -> None:
    """A validated constant-low-risk model can certify the full grid."""
    boundary_plan = plan(lipschitz_constant=0.0)
    evidence = (
        *observations(boundary_plan, "low", 1_000, failure_count=0),
        *observations(boundary_plan, "high", 1_000, failure_count=0),
    )
    model = fit_failure_boundary_model(boundary_plan, evidence)

    report = map_failure_boundary(boundary_plan, model)

    assert report.disposition is BoundaryMapDisposition.COMPLETE
    assert report.reliable_geometric_volume == pytest.approx(1.0)
    assert report.unresolved_geometric_volume == 0.0


def test_anchor_contradiction_invalidates_smoothness_assumption() -> None:
    """Observed anchor intervals can falsify an undersized Lipschitz bound."""
    boundary_plan = plan(lipschitz_constant=0.1)

    model = fit_failure_boundary_model(
        boundary_plan,
        polarized_evidence(boundary_plan),
    )
    report = map_failure_boundary(boundary_plan, model)

    assert model.disposition is BoundaryModelDisposition.ASSUMPTIONS_VIOLATED
    assert model.maximum_lipschitz_violation > 0.0
    assert model.violating_anchor_pair == ("high", "low")
    assert report.disposition is BoundaryMapDisposition.ASSUMPTIONS_VIOLATED
    assert report.cells == ()


def test_incomplete_fixed_sample_design_withholds_map() -> None:
    """Stopping below a planned anchor count produces no inferential cells."""
    boundary_plan = plan()
    evidence = (
        *observations(boundary_plan, "low", 999, failure_count=0),
        *observations(boundary_plan, "high", 1_000, failure_count=1_000),
    )

    model = fit_failure_boundary_model(boundary_plan, evidence)
    report = map_failure_boundary(boundary_plan, model)

    assert model.disposition is BoundaryModelDisposition.INSUFFICIENT_EVIDENCE
    assert model.anchor_estimates == ()
    assert report.disposition is BoundaryMapDisposition.INSUFFICIENT_EVIDENCE


def test_exceeding_fixed_sample_design_is_a_violation() -> None:
    """Optional continuation beyond the frozen count invalidates inference."""
    boundary_plan = plan()
    evidence = (
        *observations(boundary_plan, "low", 1_001, failure_count=0),
        *observations(boundary_plan, "high", 1_000, failure_count=1_000),
    )

    model = fit_failure_boundary_model(boundary_plan, evidence)
    report = map_failure_boundary(boundary_plan, model)

    assert model.disposition is BoundaryModelDisposition.DESIGN_VIOLATED
    assert report.disposition is BoundaryMapDisposition.DESIGN_VIOLATED


def test_fit_and_map_are_order_invariant_and_content_addressed() -> None:
    """Observation order cannot alter the exact model or map."""
    boundary_plan = plan()
    evidence = polarized_evidence(boundary_plan)

    first_model = fit_failure_boundary_model(boundary_plan, evidence)
    second_model = fit_failure_boundary_model(
        boundary_plan,
        tuple(reversed(evidence)),
    )
    first_map = map_failure_boundary(boundary_plan, first_model)
    second_map = map_failure_boundary(boundary_plan, second_model)

    assert first_model == second_model
    assert first_model.content_digest() == second_model.content_digest()
    assert first_map == second_map
    assert first_map.content_digest() == second_map.content_digest()
    assert first_map.evidence_use is EvidenceUse.CONFIRMATORY


def test_sample_gaps_and_duplicate_provenance_are_rejected() -> None:
    """Anchor indexing and evidence identity prevent silent row loss or reuse."""
    boundary_plan = plan()
    first = observations(
        boundary_plan,
        "low",
        1,
        failure_count=0,
    )[0]
    gap = replace(first, sample_index=1)
    with pytest.raises(ValueError, match="contiguous"):
        fit_failure_boundary_model(boundary_plan, (gap,))

    duplicate_trial = replace(
        first,
        evidence_digest=sha256_digest("other-evidence"),
    )
    with pytest.raises(ValueError, match="trial IDs"):
        fit_failure_boundary_model(
            boundary_plan,
            (first, duplicate_trial),
        )

    duplicate_evidence = replace(first, trial_id="other-trial")
    with pytest.raises(ValueError, match="evidence digests"):
        fit_failure_boundary_model(
            boundary_plan,
            (first, duplicate_evidence),
        )


def test_plan_rejects_invalid_grid_and_anchor_geometry() -> None:
    """Grid and anchor coordinates must respect the frozen continuous space."""
    with pytest.raises(ValueError, match="strictly increasing"):
        dimension(grid_points=(0.0, 0.5, 0.5, 1.0))

    with pytest.raises(ValueError, match="outside dimension bounds"):
        plan(
            anchors=(
                BoundaryAnchor(
                    anchor_id="outside",
                    coordinates=(1.1,),
                    planned_sample_count=10,
                ),
            )
        )


def test_grid_explosion_is_rejected_before_mapping() -> None:
    """A declared complexity cap prevents accidental Cartesian explosion."""
    dimensions = (
        dimension("x", grid_points=(0.0, 0.5, 1.0)),
        dimension("y", grid_points=(0.0, 0.5, 1.0)),
    )

    with pytest.raises(ValueError, match="exceeds maximum cell count"):
        plan(
            dimensions=dimensions,
            maximum_cells=3,
        )


def test_model_cannot_be_reused_under_changed_smoothness() -> None:
    """A fitted boundary state is bound to its exact threshold and assumptions."""
    boundary_plan = plan()
    model = fit_failure_boundary_model(
        boundary_plan,
        polarized_evidence(boundary_plan),
    )
    changed_plan = replace(
        boundary_plan,
        failure_probability_lipschitz_constant=2.0,
    )

    with pytest.raises(ValueError, match="does not belong"):
        map_failure_boundary(changed_plan, model)


def test_plan_requires_confirmatory_evidence_use() -> None:
    """Continuous certification requires a frozen confirmatory design."""
    with pytest.raises(ValueError, match="must be confirmatory"):
        replace(plan(), evidence_use=EvidenceUse.DISCOVERY_ONLY)
