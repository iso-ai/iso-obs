"""Tests for the Bayesian exploratory experiment-selection model."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.evidence import sha256_digest
from iso_obs.experiment_selection import (
    BayesianSelectorConfig,
    ExperimentRegion,
    ExperimentSelectionObservation,
    ExperimentSelectionPlan,
    SelectionDisposition,
    SelectionReason,
    fit_bayesian_experiment_selector,
    recommend_next_experiment,
)
from iso_obs.simulation import EvidenceUse


def region(
    region_id: str,
    *,
    mass: float,
    severity: float = 1.0,
    cost: float = 1.0,
    minimum: int = 1,
    maximum: int = 5,
) -> ExperimentRegion:
    """Build one compact test region."""
    return ExperimentRegion(
        region_id=region_id,
        condition_definition_digest=sha256_digest(f"condition-{region_id}"),
        target_population_mass=mass,
        severity_weight=severity,
        expected_execution_cost=cost,
        minimum_exploration_count=minimum,
        maximum_experiment_count=maximum,
    )


def plan(
    *,
    regions: tuple[ExperimentRegion, ...] | None = None,
    budget: float = 20.0,
) -> ExperimentSelectionPlan:
    """Build a deterministic exploratory selection plan."""
    return ExperimentSelectionPlan(
        plan_id="warehouse-failure-search",
        plan_version="1",
        target_population_digest=sha256_digest("target-population"),
        partition_definition_digest=sha256_digest("partition"),
        simulation_manifest_digest=sha256_digest("simulator"),
        system_artifact_digest=sha256_digest("system"),
        outcome_definition_digest=sha256_digest("collision"),
        total_execution_budget=budget,
        regions=regions
        or (
            region("nominal", mass=0.8),
            region("rain", mass=0.15, severity=2.0),
            region("occluded", mass=0.05, severity=20.0),
        ),
        config=BayesianSelectorConfig(),
        limitations=("Low-speed warehouse operation only.",),
    )


def observation(
    selection_plan: ExperimentSelectionPlan,
    region_id: str,
    sample_index: int,
    *,
    failed: bool,
    trial_id: str | None = None,
) -> ExperimentSelectionObservation:
    """Build one identity-linked exploratory outcome."""
    resolved_trial_id = trial_id or f"{region_id}-{sample_index}"
    return ExperimentSelectionObservation(
        plan_content_digest=selection_plan.content_digest(),
        region_id=region_id,
        sample_index=sample_index,
        trial_id=resolved_trial_id,
        failed=failed,
        evidence_digest=sha256_digest(f"evidence-{resolved_trial_id}"),
    )


def test_fit_is_order_invariant_and_content_addressed() -> None:
    """Training order cannot change fitted state or its exact identity."""
    selection_plan = plan()
    observations = (
        observation(selection_plan, "nominal", 0, failed=False),
        observation(selection_plan, "rain", 0, failed=True),
        observation(selection_plan, "rain", 1, failed=False),
    )

    first = fit_bayesian_experiment_selector(selection_plan, observations)
    second = fit_bayesian_experiment_selector(
        selection_plan,
        tuple(reversed(observations)),
    )

    assert first == second
    assert first.content_digest() == second.content_digest()
    assert first.evidence_use is EvidenceUse.DISCOVERY_ONLY
    rain = next(item for item in first.region_posteriors if item.region_id == "rain")
    assert rain.observation_count == 2
    assert rain.failure_count == 1
    assert rain.posterior_failure_count == 1.5
    assert rain.posterior_success_count == 1.5
    assert rain.mean_failure_probability == 0.5
    assert rain.posterior_standard_deviation > 0.0
    assert rain.expected_variance_reduction > 0.0


def test_required_coverage_precedes_failure_exploitation() -> None:
    """An unobserved required region outranks a known high-failure region."""
    selection_plan = plan()
    observations = (
        observation(selection_plan, "nominal", 0, failed=True),
        observation(selection_plan, "rain", 0, failed=True),
    )
    state = fit_bayesian_experiment_selector(selection_plan, observations)

    report = recommend_next_experiment(selection_plan, state)

    assert report.disposition is SelectionDisposition.RECOMMENDED
    assert report.selected_region_id == "occluded"
    assert len(report.ranked_candidates) == 1
    assert report.ranked_candidates[0].reason is SelectionReason.REQUIRED_COVERAGE


def test_acquisition_favors_severe_high_failure_region_after_coverage() -> None:
    """Severity and observed failures affect post-coverage selection."""
    selection_plan = plan()
    observations = (
        observation(selection_plan, "nominal", 0, failed=False),
        observation(selection_plan, "rain", 0, failed=False),
        observation(selection_plan, "occluded", 0, failed=True),
    )
    state = fit_bayesian_experiment_selector(selection_plan, observations)

    report = recommend_next_experiment(selection_plan, state)

    assert report.selected_region_id == "occluded"
    selected = report.ranked_candidates[0]
    assert selected.reason is SelectionReason.ACQUISITION_SCORE
    assert selected.posterior_failure_probability == 0.75
    nominal = next(
        item for item in report.ranked_candidates if item.region_id == "nominal"
    )
    assert selected.target_population_mass < nominal.target_population_mass
    assert selected.severity_weight > nominal.severity_weight
    assert selected.unadjusted_acquisition_score > 0.0
    assert selected.cost_adjusted_acquisition_score > 0.0
    assert tuple(item.rank for item in report.ranked_candidates) == (1, 2, 3)


def test_cost_adjustment_can_change_the_selected_region() -> None:
    """Acquisition ranks scientific value per expected execution cost."""
    selection_plan = plan(
        regions=(
            region(
                "expensive",
                mass=0.5,
                severity=2.0,
                cost=10.0,
                minimum=0,
            ),
            region(
                "cheap",
                mass=0.5,
                severity=1.0,
                cost=1.0,
                minimum=0,
            ),
        )
    )
    state = fit_bayesian_experiment_selector(selection_plan, ())

    report = recommend_next_experiment(selection_plan, state)

    expensive = next(
        item for item in report.ranked_candidates if item.region_id == "expensive"
    )
    cheap = next(item for item in report.ranked_candidates if item.region_id == "cheap")
    assert expensive.unadjusted_acquisition_score > (cheap.unadjusted_acquisition_score)
    assert cheap.cost_adjusted_acquisition_score > (
        expensive.cost_adjusted_acquisition_score
    )
    assert report.selected_region_id == "cheap"


def test_budget_exhaustion_is_an_explicit_abstention() -> None:
    """The model recommends nothing when no incomplete region is affordable."""
    selection_plan = plan(
        regions=(
            region(
                "only",
                mass=1.0,
                cost=2.0,
                minimum=0,
                maximum=3,
            ),
        ),
        budget=3.0,
    )
    state = fit_bayesian_experiment_selector(
        selection_plan,
        (observation(selection_plan, "only", 0, failed=False),),
    )

    report = recommend_next_experiment(selection_plan, state)

    assert report.disposition is SelectionDisposition.BUDGET_EXHAUSTED
    assert report.selected_region_id is None
    assert report.ranked_candidates == ()


def test_unaffordable_required_coverage_blocks_exploitation() -> None:
    """The model abstains rather than bypassing required regional coverage."""
    selection_plan = plan(
        regions=(
            region(
                "required-expensive",
                mass=0.5,
                cost=10.0,
                minimum=1,
            ),
            region(
                "covered-cheap",
                mass=0.5,
                cost=1.0,
                minimum=1,
            ),
        ),
        budget=5.0,
    )
    state = fit_bayesian_experiment_selector(
        selection_plan,
        (
            observation(
                selection_plan,
                "covered-cheap",
                0,
                failed=True,
            ),
        ),
    )

    report = recommend_next_experiment(selection_plan, state)

    assert report.disposition is SelectionDisposition.BUDGET_EXHAUSTED
    assert report.selected_region_id is None
    assert report.ranked_candidates == ()


def test_design_completion_is_distinct_from_budget_exhaustion() -> None:
    """Reaching every region cap produces a complete-design disposition."""
    selection_plan = plan(
        regions=(
            region(
                "only",
                mass=1.0,
                minimum=1,
                maximum=1,
            ),
        )
    )
    state = fit_bayesian_experiment_selector(
        selection_plan,
        (observation(selection_plan, "only", 0, failed=False),),
    )

    report = recommend_next_experiment(selection_plan, state)

    assert report.disposition is SelectionDisposition.DESIGN_COMPLETE
    assert report.selected_region_id is None


def test_adaptive_plan_rejects_confirmatory_evidence_use() -> None:
    """Adaptive search cannot be relabeled as confirmatory evidence."""
    with pytest.raises(ValueError, match="discovery-only"):
        replace(plan(), evidence_use=EvidenceUse.CONFIRMATORY)


def test_observation_integrity_failures_are_rejected() -> None:
    """Wrong identity, duplicates, and sample gaps cannot train the model."""
    selection_plan = plan()
    valid = observation(selection_plan, "nominal", 0, failed=False)
    wrong_plan = replace(valid, plan_content_digest=sha256_digest("other-plan"))
    with pytest.raises(ValueError, match="plan content digest mismatch"):
        fit_bayesian_experiment_selector(selection_plan, (wrong_plan,))

    duplicate = replace(
        observation(selection_plan, "rain", 0, failed=False),
        trial_id=valid.trial_id,
    )
    with pytest.raises(ValueError, match="duplicate"):
        fit_bayesian_experiment_selector(selection_plan, (valid, duplicate))

    gap = observation(selection_plan, "nominal", 1, failed=False)
    with pytest.raises(ValueError, match="contiguous"):
        fit_bayesian_experiment_selector(selection_plan, (gap,))


def test_plan_rejects_invalid_partition_and_objective() -> None:
    """Target masses and acquisition weights define a valid model objective."""
    with pytest.raises(ValueError, match="target masses must sum to one"):
        plan(
            regions=(
                region("a", mass=0.4),
                region("b", mass=0.4),
            )
        )
    with pytest.raises(ValueError, match="weights must sum to one"):
        BayesianSelectorConfig(
            failure_probability_weight=0.5,
            uncertainty_weight=0.5,
            information_gain_weight=0.5,
        )


def test_recommendation_rejects_state_from_another_plan() -> None:
    """A fitted state cannot be reused under a changed selection objective."""
    first_plan = plan()
    state = fit_bayesian_experiment_selector(first_plan, ())
    second_plan = replace(first_plan, total_execution_budget=25.0)

    with pytest.raises(ValueError, match="does not belong"):
        recommend_next_experiment(second_plan, state)
