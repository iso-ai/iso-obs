"""Tests for fixed-design stratified failure-surface estimation."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from iso_obs.evidence import sha256_digest
from iso_obs.failure_surface import (
    FailureSurfaceConclusion,
    FailureSurfaceObservation,
    FailureSurfacePlan,
    FailureSurfaceStratum,
    StratumConclusion,
    estimate_failure_surface,
)


def digest(value: str) -> str:
    """Build a valid digest for test content."""
    return sha256_digest(value)


def stratum(
    stratum_id: str,
    mass: float,
    *,
    samples: int = 1_000,
    maximum_rate: float = 0.1,
    allocated_error_rate: float = 0.025,
) -> FailureSurfaceStratum:
    """Build one fixed operating-condition stratum."""
    return FailureSurfaceStratum(
        stratum_id=stratum_id,
        condition_definition_digest=digest(f"condition-{stratum_id}"),
        target_population_mass=mass,
        planned_sample_count=samples,
        maximum_acceptable_failure_rate=maximum_rate,
        allocated_error_rate=allocated_error_rate,
    )


def plan(
    *,
    strata: tuple[FailureSurfaceStratum, ...] | None = None,
    overall_limit: float = 0.1,
    family_error_rate: float = 0.05,
) -> FailureSurfacePlan:
    """Build a two-cell operational failure-surface design."""
    return FailureSurfacePlan(
        plan_id="warehouse-failure-surface",
        plan_version="1",
        target_population_digest=digest("warehouse-exposure-population"),
        partition_definition_digest=digest("visibility-friction-partition"),
        simulation_manifest_digest=digest("warehouse-simulation"),
        system_artifact_digest=digest("policy-v18"),
        outcome_definition_digest=digest("collision-or-safety-stop"),
        maximum_acceptable_overall_failure_rate=overall_limit,
        family_error_rate=family_error_rate,
        strata=strata
        or (
            stratum("common", 0.99),
            stratum("rare-critical", 0.01),
        ),
        limitations=("Target weights come from the 2026-Q2 exposure model.",),
    )


def observations(
    surface_plan: FailureSurfacePlan,
    stratum_id: str,
    count: int,
    *,
    failures: int = 0,
) -> tuple[FailureSurfaceObservation, ...]:
    """Build a contiguous stratum sample with failures first."""
    plan_digest = surface_plan.content_digest()
    return tuple(
        FailureSurfaceObservation(
            plan_content_digest=plan_digest,
            stratum_id=stratum_id,
            sample_index=index,
            trial_id=f"{stratum_id}-trial-{index}",
            failed=index < failures,
            evidence_digest=digest(f"{stratum_id}-evidence-{index}"),
        )
        for index in range(count)
    )


def complete_observations(
    surface_plan: FailureSurfacePlan,
    *,
    failures_by_stratum: dict[str, int] | None = None,
) -> tuple[FailureSurfaceObservation, ...]:
    """Build the exact fixed sample for every stratum."""
    resolved = failures_by_stratum or {}
    return tuple(
        observation
        for item in surface_plan.strata
        for observation in observations(
            surface_plan,
            item.stratum_id,
            item.planned_sample_count,
            failures=resolved.get(item.stratum_id, 0),
        )
    )


def estimate_by_id(report, stratum_id: str):
    """Select one stratum estimate by stable ID."""
    return next(
        item for item in report.stratum_estimates if item.stratum_id == stratum_id
    )


def test_complete_zero_failure_surface_is_within_all_limits() -> None:
    surface_plan = plan()

    report = estimate_failure_surface(
        surface_plan,
        complete_observations(surface_plan),
    )

    assert report.design_complete
    assert report.completed_target_population_mass == pytest.approx(1.0)
    assert report.overall_failure_probability == 0.0
    assert report.overall_confidence_upper is not None
    assert report.overall_confidence_upper < 0.1
    assert report.conclusion is FailureSurfaceConclusion.WITHIN_LIMITS
    assert all(
        item.conclusion is StratumConclusion.WITHIN_LIMIT
        for item in report.stratum_estimates
    )


def test_rare_critical_failure_cannot_hide_in_favorable_average() -> None:
    surface_plan = plan()

    report = estimate_failure_surface(
        surface_plan,
        complete_observations(
            surface_plan,
            failures_by_stratum={"rare-critical": 200},
        ),
    )
    rare = estimate_by_id(report, "rare-critical")

    assert report.overall_failure_probability == pytest.approx(0.002)
    assert report.overall_confidence_upper is not None
    assert report.overall_confidence_upper < 0.1
    assert rare.confidence_lower is not None
    assert rare.confidence_lower > rare.maximum_acceptable_failure_rate
    assert rare.conclusion is StratumConclusion.EXCEEDS_LIMIT
    assert report.conclusion is FailureSurfaceConclusion.EXCEEDS_LIMITS
    assert report.limiting_stratum_ids == ("rare-critical",)


def test_overall_lower_bound_can_exceed_global_limit() -> None:
    surface_plan = plan(overall_limit=0.1)

    report = estimate_failure_surface(
        surface_plan,
        complete_observations(
            surface_plan,
            failures_by_stratum={"common": 300, "rare-critical": 300},
        ),
    )

    assert report.overall_confidence_lower is not None
    assert report.overall_confidence_lower > 0.1
    assert report.conclusion is FailureSurfaceConclusion.EXCEEDS_LIMITS


def test_complete_but_low_power_design_is_inconclusive() -> None:
    small_plan = plan(
        strata=(
            stratum("common", 0.99, samples=10),
            stratum("rare-critical", 0.01, samples=10),
        )
    )

    report = estimate_failure_surface(
        small_plan,
        complete_observations(small_plan),
    )

    assert report.design_complete
    assert report.conclusion is FailureSurfaceConclusion.INCONCLUSIVE
    assert report.limiting_stratum_ids == ("common", "rare-critical")


def test_incomplete_design_withholds_all_confirmatory_surface_values() -> None:
    surface_plan = plan()
    partial = observations(surface_plan, "common", 999)

    report = estimate_failure_surface(surface_plan, partial)
    common = estimate_by_id(report, "common")
    rare = estimate_by_id(report, "rare-critical")

    assert not report.design_complete
    assert report.completed_target_population_mass == 0.0
    assert report.overall_failure_probability is None
    assert report.overall_confidence_lower is None
    assert report.overall_confidence_upper is None
    assert report.conclusion is FailureSurfaceConclusion.INCOMPLETE
    assert common.empirical_failure_rate == 0.0
    assert common.confidence_upper is None
    assert rare.empirical_failure_rate is None
    assert set(report.limiting_stratum_ids) == {"common", "rare-critical"}


def test_completed_cells_report_coverage_during_partial_design() -> None:
    surface_plan = plan()
    common = next(item for item in surface_plan.strata if item.stratum_id == "common")

    report = estimate_failure_surface(
        surface_plan,
        observations(
            surface_plan,
            common.stratum_id,
            common.planned_sample_count,
        ),
    )

    assert not report.design_complete
    assert report.completed_target_population_mass == pytest.approx(0.99)
    assert estimate_by_id(report, "common").empirical_failure_rate == 0.0
    assert estimate_by_id(report, "common").confidence_upper is None
    assert estimate_by_id(report, "common").conclusion is StratumConclusion.INCOMPLETE
    assert estimate_by_id(report, "rare-critical").confidence_upper is None


def test_hoeffding_radius_uses_allocated_stratum_error() -> None:
    surface_plan = plan()

    report = estimate_failure_surface(
        surface_plan,
        complete_observations(surface_plan),
    )
    common = estimate_by_id(report, "common")
    expected_radius = math.sqrt(math.log(2.0 / 0.025) / 2_000.0)

    assert common.confidence_lower == 0.0
    assert common.confidence_upper == pytest.approx(expected_radius)
    assert report.confidence_level == 0.95


def test_target_weighting_corrects_deliberate_rare_stratum_oversampling() -> None:
    surface_plan = plan()

    report = estimate_failure_surface(
        surface_plan,
        complete_observations(
            surface_plan,
            failures_by_stratum={"common": 10, "rare-critical": 500},
        ),
    )
    common = estimate_by_id(report, "common")
    rare = estimate_by_id(report, "rare-critical")

    assert common.empirical_failure_rate == pytest.approx(0.01)
    assert rare.empirical_failure_rate == pytest.approx(0.5)
    assert report.overall_failure_probability == pytest.approx(0.99 * 0.01 + 0.01 * 0.5)


def test_input_order_and_plan_stratum_order_are_canonical() -> None:
    surface_plan = plan()
    reordered_plan = replace(
        surface_plan,
        strata=tuple(reversed(surface_plan.strata)),
        limitations=tuple(reversed(surface_plan.limitations)),
    )
    observed = complete_observations(surface_plan)

    forward = estimate_failure_surface(surface_plan, observed)
    reverse = estimate_failure_surface(
        reordered_plan,
        tuple(reversed(observed)),
    )

    assert surface_plan == reordered_plan
    assert surface_plan.content_digest() == reordered_plan.content_digest()
    assert forward == reverse
    assert forward.content_digest() == reverse.content_digest()


def test_observations_must_echo_exact_plan_identity() -> None:
    surface_plan = plan()
    observed = observations(surface_plan, "common", 1)[0]

    with pytest.raises(ValueError, match="plan content digest mismatch"):
        estimate_failure_surface(
            surface_plan,
            (replace(observed, plan_content_digest=digest("other-plan")),),
        )


def test_unknown_strata_duplicate_trials_and_sample_gaps_are_rejected() -> None:
    surface_plan = plan()
    observed = observations(surface_plan, "common", 2)

    with pytest.raises(ValueError, match="unknown failure-surface stratum"):
        estimate_failure_surface(
            surface_plan,
            (replace(observed[0], stratum_id="unknown"),),
        )
    with pytest.raises(ValueError, match="duplicate failure-surface trial"):
        estimate_failure_surface(
            surface_plan,
            (observed[0], replace(observed[1], trial_id=observed[0].trial_id)),
        )
    with pytest.raises(ValueError, match="contiguous from zero"):
        estimate_failure_surface(
            surface_plan,
            (replace(observed[0], sample_index=1),),
        )


def test_observations_cannot_exceed_fixed_sample_count() -> None:
    tiny_plan = plan(
        strata=(
            stratum("common", 0.99, samples=1),
            stratum("rare-critical", 0.01, samples=1),
        )
    )

    with pytest.raises(ValueError, match="more observations"):
        estimate_failure_surface(
            tiny_plan,
            observations(tiny_plan, "common", 2),
        )


def test_observation_validation_rejects_invalid_index_and_outcome() -> None:
    surface_plan = plan()
    observed = observations(surface_plan, "common", 1)[0]

    with pytest.raises(ValueError, match="non-negative integer"):
        replace(observed, sample_index=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        replace(observed, failed=1)


def test_plan_requires_partition_mass_and_error_budget() -> None:
    surface_plan = plan()

    with pytest.raises(ValueError, match="masses must sum to one"):
        replace(
            surface_plan,
            strata=(
                stratum("one", 0.8),
                stratum("two", 0.1),
            ),
        )
    with pytest.raises(ValueError, match="must not exceed"):
        replace(
            surface_plan,
            strata=(
                stratum("one", 0.5, allocated_error_rate=0.04),
                stratum("two", 0.5, allocated_error_rate=0.04),
            ),
        )


def test_plan_rejects_duplicate_strata_and_condition_definitions() -> None:
    surface_plan = plan()
    first = surface_plan.strata[0]

    with pytest.raises(ValueError, match="stratum IDs must be unique"):
        replace(
            surface_plan,
            strata=(
                first,
                replace(surface_plan.strata[1], stratum_id=first.stratum_id),
            ),
        )
    with pytest.raises(ValueError, match="condition definitions must be unique"):
        replace(
            surface_plan,
            strata=(
                first,
                replace(
                    surface_plan.strata[1],
                    condition_definition_digest=first.condition_definition_digest,
                ),
            ),
        )


def test_stratum_and_plan_probability_validation() -> None:
    base = stratum("test", 1.0)

    with pytest.raises(ValueError, match="positive integer"):
        replace(base, planned_sample_count=True)
    with pytest.raises(ValueError, match="between zero and one"):
        replace(base, target_population_mass=0.0)
    with pytest.raises(ValueError, match="between zero and one"):
        replace(base, maximum_acceptable_failure_rate=1.1)
    with pytest.raises(ValueError, match="between zero and one"):
        replace(plan(strata=(base,)), family_error_rate=0.0)


def test_zero_tolerance_stratum_remains_inconclusive_without_power() -> None:
    zero_tolerance_plan = plan(
        strata=(
            stratum(
                "only",
                1.0,
                samples=1_000,
                maximum_rate=0.0,
                allocated_error_rate=0.05,
            ),
        ),
        family_error_rate=0.05,
    )

    report = estimate_failure_surface(
        zero_tolerance_plan,
        complete_observations(zero_tolerance_plan),
    )

    assert report.conclusion is FailureSurfaceConclusion.INCONCLUSIVE
    assert report.stratum_estimates[0].conclusion is StratumConclusion.INCONCLUSIVE


def test_report_preserves_sampling_and_scope_limitations() -> None:
    surface_plan = plan()

    report = estimate_failure_surface(
        surface_plan,
        complete_observations(surface_plan),
    )
    limitations = " ".join(report.limitations)

    assert "adaptive stopping" in limitations
    assert "population drift" in limitations
    assert "cannot compensate for a stratum" in limitations
    assert "Incomplete designs provide only sampling coverage" in limitations
    assert "Target weights come from the 2026-Q2 exposure model." in limitations
