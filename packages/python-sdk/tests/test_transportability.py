"""Tests for fixed-design simulator-to-real transport validation."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from iso_obs.evidence import NamedDigest, ValidityRange, sha256_digest
from iso_obs.transportability import (
    DiscrepancyKind,
    DiscrepancySource,
    OperatingCondition,
    PairedAnchorObservation,
    TransportApplicability,
    TransportConclusion,
    TransportStratum,
    TransportStratumConclusion,
    TransportValidationPlan,
    assess_transport_applicability,
    validate_transportability,
)


def digest(value: str) -> str:
    """Build a valid digest for test content."""
    return sha256_digest(value)


def discrepancy_source(
    name: str = "contact-dynamics",
    *,
    quantified: bool = True,
) -> DiscrepancySource:
    """Build one declared discrepancy source."""
    return DiscrepancySource(
        name=name,
        kind=DiscrepancyKind.PHYSICS_MODEL_FORM,
        characterization="Rigid contact differs from compliant hardware.",
        evidence_digest=digest(f"discrepancy-{name}"),
        quantified=quantified,
    )


def stratum(
    stratum_id: str,
    mass: float,
    *,
    pairs: int = 1_000,
    acceptable: float = 0.1,
    possible: float = 1.0,
    allocated_error_rate: float = 0.025,
) -> TransportStratum:
    """Build one fixed paired-anchor stratum."""
    return TransportStratum(
        stratum_id=stratum_id,
        condition_definition_digest=digest(f"condition-{stratum_id}"),
        target_population_mass=mass,
        planned_pair_count=pairs,
        maximum_acceptable_mean_expanded_discrepancy=acceptable,
        maximum_possible_expanded_discrepancy=possible,
        allocated_error_rate=allocated_error_rate,
    )


def plan(
    *,
    strata: tuple[TransportStratum, ...] | None = None,
    sources: tuple[DiscrepancySource, ...] | None = None,
    overall_limit: float = 0.1,
    family_error_rate: float = 0.05,
) -> TransportValidationPlan:
    """Build a representative simulator-to-real validation plan."""
    return TransportValidationPlan(
        plan_id="warehouse-twin-transport",
        plan_version="1",
        simulator_manifest_digest=digest("warehouse-twin-v4"),
        real_anchor_dataset_digest=digest("warehouse-anchor-runs"),
        pairing_protocol_digest=digest("matched-initial-state-protocol"),
        target_population_digest=digest("warehouse-exposure-population"),
        partition_definition_digest=digest("friction-speed-partition"),
        metric_name="stopping margin",
        metric_unit="m",
        outcome_definition_digest=digest("stopping-margin-measurement"),
        maximum_acceptable_overall_mean_expanded_discrepancy=overall_limit,
        family_error_rate=family_error_rate,
        anchor_envelope=(
            ValidityRange("floor_friction", 0.45, 0.9, "coefficient"),
            ValidityRange("robot_speed", 0.0, 2.0, "m/s"),
        ),
        discrepancy_sources=sources or (discrepancy_source(),),
        physics_evidence=(NamedDigest("brake wear study", digest("brake-wear")),),
        strata=strata
        or (
            stratum("common", 0.99),
            stratum("rare-low-friction", 0.01),
        ),
        limitations=("Anchors use the production sensor calibration.",),
    )


def observations(
    validation_plan: TransportValidationPlan,
    stratum_id: str,
    count: int,
    *,
    residual: float = 0.01,
    simulation_uncertainty: float = 0.01,
    measurement_uncertainty: float = 0.01,
) -> tuple[PairedAnchorObservation, ...]:
    """Build contiguous matched anchors with a constant residual."""
    plan_digest = validation_plan.content_digest()
    return tuple(
        PairedAnchorObservation(
            plan_content_digest=plan_digest,
            stratum_id=stratum_id,
            pair_index=index,
            anchor_id=f"{stratum_id}-anchor-{index}",
            simulation_run_id=f"{stratum_id}-sim-{index}",
            real_run_id=f"{stratum_id}-real-{index}",
            simulation_value=10.0,
            real_value=10.0 + residual,
            simulation_uncertainty_bound=simulation_uncertainty,
            measurement_uncertainty_bound=measurement_uncertainty,
            evidence_digest=digest(f"{stratum_id}-evidence-{index}"),
        )
        for index in range(count)
    )


def complete_observations(
    validation_plan: TransportValidationPlan,
    *,
    residuals: dict[str, float] | None = None,
) -> tuple[PairedAnchorObservation, ...]:
    """Build the exact fixed sample in every stratum."""
    resolved = residuals or {}
    return tuple(
        observation
        for item in validation_plan.strata
        for observation in observations(
            validation_plan,
            item.stratum_id,
            item.planned_pair_count,
            residual=resolved.get(item.stratum_id, 0.01),
        )
    )


def estimate_by_id(report, stratum_id: str):
    """Select one stratum estimate by stable ID."""
    return next(
        item for item in report.stratum_estimates if item.stratum_id == stratum_id
    )


def operating_point(
    *,
    friction: float = 0.6,
    speed: float = 1.0,
) -> tuple[OperatingCondition, ...]:
    """Build a complete point in the anchor envelope."""
    return (
        OperatingCondition("floor_friction", friction, "coefficient"),
        OperatingCondition("robot_speed", speed, "m/s"),
    )


def test_complete_low_discrepancy_design_supports_transport_in_envelope() -> None:
    validation_plan = plan()

    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )

    assert report.design_complete
    assert report.assumptions_satisfied
    assert report.completed_target_population_mass == pytest.approx(1.0)
    assert report.overall_mean_expanded_discrepancy == pytest.approx(0.03)
    assert report.overall_confidence_upper is not None
    assert report.overall_confidence_upper < 0.1
    assert report.conclusion is TransportConclusion.SUPPORTED_WITHIN_ANCHOR_ENVELOPE
    assert all(
        item.conclusion is TransportStratumConclusion.WITHIN_LIMIT
        for item in report.stratum_estimates
    )


def test_rare_bad_transport_cell_cannot_hide_in_weighted_average() -> None:
    validation_plan = plan()

    report = validate_transportability(
        validation_plan,
        complete_observations(
            validation_plan,
            residuals={"rare-low-friction": 0.3},
        ),
    )
    rare = estimate_by_id(report, "rare-low-friction")

    assert report.overall_mean_expanded_discrepancy == pytest.approx(
        0.99 * 0.03 + 0.01 * 0.32
    )
    assert report.overall_confidence_upper is not None
    assert report.overall_confidence_upper < 0.1
    assert rare.confidence_lower is not None
    assert rare.confidence_lower > 0.1
    assert report.conclusion is TransportConclusion.NOT_SUPPORTED
    assert report.limiting_stratum_ids == ("rare-low-friction",)


def test_global_discrepancy_lower_bound_can_fail_overall_limit() -> None:
    validation_plan = plan(overall_limit=0.1)

    report = validate_transportability(
        validation_plan,
        complete_observations(
            validation_plan,
            residuals={"common": 0.3, "rare-low-friction": 0.3},
        ),
    )

    assert report.overall_confidence_lower is not None
    assert report.overall_confidence_lower > 0.1
    assert report.conclusion is TransportConclusion.NOT_SUPPORTED


def test_absolute_discrepancy_prevents_signed_residual_cancellation() -> None:
    one_stratum = plan(
        strata=(
            stratum(
                "only",
                1.0,
                pairs=2,
                acceptable=0.5,
                allocated_error_rate=0.05,
            ),
        ),
        family_error_rate=0.05,
        overall_limit=0.5,
    )
    observed = list(
        observations(
            one_stratum,
            "only",
            2,
            residual=0.2,
            simulation_uncertainty=0.0,
            measurement_uncertainty=0.0,
        )
    )
    observed[1] = replace(observed[1], real_value=9.8)

    report = validate_transportability(one_stratum, observed)
    estimate = report.stratum_estimates[0]

    assert estimate.mean_signed_residual == pytest.approx(0.0)
    assert estimate.mean_expanded_discrepancy == pytest.approx(0.2)


def test_incomplete_design_withholds_all_inference() -> None:
    validation_plan = plan()
    common = next(
        item for item in validation_plan.strata if item.stratum_id == "common"
    )

    report = validate_transportability(
        validation_plan,
        observations(
            validation_plan,
            common.stratum_id,
            common.planned_pair_count,
        ),
    )

    assert not report.design_complete
    assert report.completed_target_population_mass == pytest.approx(0.99)
    assert report.overall_mean_expanded_discrepancy is None
    assert report.overall_confidence_lower is None
    assert report.conclusion is TransportConclusion.INCOMPLETE
    assert estimate_by_id(report, "common").mean_expanded_discrepancy == pytest.approx(
        0.03
    )
    assert estimate_by_id(report, "common").confidence_upper is None
    assert all(
        item.conclusion is TransportStratumConclusion.INCOMPLETE
        for item in report.stratum_estimates
    )


def test_declared_discrepancy_bound_violation_invalidates_inference() -> None:
    validation_plan = plan()
    observed = list(complete_observations(validation_plan))
    observed[0] = replace(
        observed[0],
        real_value=12.0,
    )

    report = validate_transportability(validation_plan, observed)

    assert report.design_complete
    assert not report.assumptions_satisfied
    assert report.conclusion is TransportConclusion.ASSUMPTIONS_VIOLATED
    assert report.overall_mean_expanded_discrepancy is None
    assert report.violating_anchor_ids == ("common-anchor-0",)
    assert all(item.confidence_upper is None for item in report.stratum_estimates)


def test_unquantified_known_discrepancy_prevents_support() -> None:
    validation_plan = plan(sources=(discrepancy_source(quantified=False),))

    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )

    assert report.overall_confidence_upper is not None
    assert report.overall_confidence_upper < 0.1
    assert report.conclusion is TransportConclusion.INCONCLUSIVE
    assert report.unquantified_discrepancy_source_names == ("contact-dynamics",)


def test_low_power_complete_design_is_inconclusive() -> None:
    small_plan = plan(
        strata=(
            stratum("common", 0.99, pairs=10),
            stratum("rare-low-friction", 0.01, pairs=10),
        )
    )

    report = validate_transportability(
        small_plan,
        complete_observations(small_plan),
    )

    assert report.design_complete
    assert report.conclusion is TransportConclusion.INCONCLUSIVE
    assert report.limiting_stratum_ids == (
        "common",
        "rare-low-friction",
    )


def test_hoeffding_radius_uses_declared_possible_discrepancy() -> None:
    validation_plan = plan()

    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )
    common = estimate_by_id(report, "common")
    expected_radius = math.sqrt(math.log(2.0 / 0.025) / 2_000.0)

    assert common.confidence_lower == 0.0
    assert common.confidence_upper == pytest.approx(0.03 + expected_radius)
    assert report.confidence_level == 0.95


def test_input_order_and_plan_order_are_canonical() -> None:
    validation_plan = plan()
    reordered_plan = replace(
        validation_plan,
        anchor_envelope=tuple(reversed(validation_plan.anchor_envelope)),
        discrepancy_sources=tuple(reversed(validation_plan.discrepancy_sources)),
        physics_evidence=tuple(reversed(validation_plan.physics_evidence)),
        strata=tuple(reversed(validation_plan.strata)),
        limitations=tuple(reversed(validation_plan.limitations)),
    )
    observed = complete_observations(validation_plan)

    forward = validate_transportability(validation_plan, observed)
    reverse = validate_transportability(
        reordered_plan,
        tuple(reversed(observed)),
    )

    assert validation_plan == reordered_plan
    assert validation_plan.content_digest() == reordered_plan.content_digest()
    assert forward == reverse
    assert forward.content_digest() == reverse.content_digest()


def test_observation_plan_stratum_anchor_and_order_integrity() -> None:
    validation_plan = plan()
    observed = observations(validation_plan, "common", 2)

    with pytest.raises(ValueError, match="plan content digest mismatch"):
        validate_transportability(
            validation_plan,
            (replace(observed[0], plan_content_digest=digest("other")),),
        )
    with pytest.raises(ValueError, match="unknown transport stratum"):
        validate_transportability(
            validation_plan,
            (replace(observed[0], stratum_id="unknown"),),
        )
    with pytest.raises(ValueError, match="duplicate paired anchor"):
        validate_transportability(
            validation_plan,
            (
                observed[0],
                replace(observed[1], anchor_id=observed[0].anchor_id),
            ),
        )
    with pytest.raises(ValueError, match="contiguous from zero"):
        validate_transportability(
            validation_plan,
            (replace(observed[0], pair_index=1),),
        )


def test_pair_count_cannot_exceed_fixed_design() -> None:
    tiny_plan = plan(
        strata=(
            stratum(
                "only",
                1.0,
                pairs=1,
                allocated_error_rate=0.05,
            ),
        ),
        family_error_rate=0.05,
    )

    with pytest.raises(ValueError, match="more pairs"):
        validate_transportability(
            tiny_plan,
            observations(tiny_plan, "only", 2),
        )


def test_paired_observation_validation() -> None:
    validation_plan = plan()
    observed = observations(validation_plan, "common", 1)[0]

    with pytest.raises(ValueError, match="must be different"):
        replace(observed, real_run_id=observed.simulation_run_id)
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(observed, pair_index=True)
    with pytest.raises(ValueError, match="must be finite"):
        replace(observed, real_value=float("nan"))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(observed, measurement_uncertainty_bound=-0.1)


def test_plan_requires_partition_mass_and_error_budget() -> None:
    validation_plan = plan()

    with pytest.raises(ValueError, match="masses must sum to one"):
        replace(
            validation_plan,
            strata=(
                stratum("one", 0.8),
                stratum("two", 0.1),
            ),
        )
    with pytest.raises(ValueError, match="must not exceed"):
        replace(
            validation_plan,
            strata=(
                stratum("one", 0.5, allocated_error_rate=0.04),
                stratum("two", 0.5, allocated_error_rate=0.04),
            ),
        )


def test_plan_requires_sources_envelope_and_valid_bounds() -> None:
    validation_plan = plan()
    base = stratum("only", 1.0, allocated_error_rate=0.05)

    with pytest.raises(ValueError, match="requires discrepancy sources"):
        replace(validation_plan, discrepancy_sources=())
    with pytest.raises(ValueError, match="requires an anchor envelope"):
        replace(validation_plan, anchor_envelope=())
    with pytest.raises(ValueError, match="must not exceed"):
        replace(
            base,
            maximum_acceptable_mean_expanded_discrepancy=2.0,
        )
    with pytest.raises(ValueError, match="positive integer"):
        replace(base, planned_pair_count=True)


def test_supported_report_applies_only_inside_anchor_envelope() -> None:
    validation_plan = plan()
    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )

    applicability = assess_transport_applicability(
        validation_plan,
        report,
        operating_point(),
    )

    assert (
        applicability.applicability
        is TransportApplicability.APPLICABLE_WITHIN_ANCHOR_ENVELOPE
    )
    assert applicability.out_of_range_parameters == ()


def test_operating_point_outside_anchor_envelope_is_rejected_for_use() -> None:
    validation_plan = plan()
    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )

    applicability = assess_transport_applicability(
        validation_plan,
        report,
        operating_point(speed=3.0),
    )

    assert applicability.applicability is TransportApplicability.OUTSIDE_ANCHOR_ENVELOPE
    assert applicability.out_of_range_parameters == ("robot_speed",)


def test_inconclusive_evidence_is_not_applicable_despite_in_range_point() -> None:
    validation_plan = plan(sources=(discrepancy_source(quantified=False),))
    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )

    applicability = assess_transport_applicability(
        validation_plan,
        report,
        operating_point(),
    )

    assert applicability.applicability is TransportApplicability.EVIDENCE_NOT_SUPPORTIVE


def test_applicability_requires_exact_parameters_units_and_report_plan() -> None:
    validation_plan = plan()
    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )

    with pytest.raises(ValueError, match="exactly match"):
        assess_transport_applicability(
            validation_plan,
            report,
            operating_point()[:1],
        )
    with pytest.raises(ValueError, match="unit mismatch"):
        assess_transport_applicability(
            validation_plan,
            report,
            (
                OperatingCondition(
                    "floor_friction",
                    0.6,
                    "wrong-unit",
                ),
                operating_point()[1],
            ),
        )
    with pytest.raises(ValueError, match="plan digest mismatch"):
        assess_transport_applicability(
            replace(
                validation_plan,
                limitations=("Changed plan.",),
            ),
            report,
            operating_point(),
        )


def test_applicability_is_order_invariant_and_content_addressed() -> None:
    validation_plan = plan()
    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )
    conditions = operating_point()

    forward = assess_transport_applicability(
        validation_plan,
        report,
        conditions,
    )
    reverse = assess_transport_applicability(
        validation_plan,
        report,
        tuple(reversed(conditions)),
    )

    assert forward == reverse
    assert forward.content_digest() == reverse.content_digest()


def test_report_preserves_vvuq_and_extrapolation_boundaries() -> None:
    validation_plan = plan()

    report = validate_transportability(
        validation_plan,
        complete_observations(validation_plan),
    )
    limitations = " ".join(report.limitations)

    assert "not universal simulator certification" in limitations
    assert "adaptive stopping" in limitations
    assert "omitted or underestimated uncertainty" in limitations
    assert "population drift" in limitations
    assert "does not establish accuracy, safety, or causality outside" in limitations
    assert "Anchors use the production sensor calibration." in limitations
