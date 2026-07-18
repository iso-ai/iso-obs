"""Tests for controlled counterfactual replay and counterexample minimization."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.counterfactual import (
    ConditionSetTrial,
    CounterfactualDesign,
    CounterfactualEstimand,
    CounterfactualEvidenceLevel,
    CounterfactualObservation,
    CounterfactualPair,
    EffectDirection,
    FailureCondition,
    InterventionOperation,
    InterventionScope,
    InterventionSpec,
    ReplayControl,
    analyze_counterfactual_design,
    assess_minimal_counterexample,
)
from iso_obs.evidence import NamedDigest, sha256_digest
from iso_obs.replication import PairedAnalysisConfig


def digest(value: str) -> str:
    """Build a valid digest for test content."""
    return sha256_digest(value)


def replay_control(index: int) -> ReplayControl:
    """Build one exact replay control."""
    return ReplayControl(
        scenario_id=f"scenario-{index}",
        scenario_version="1",
        simulation_manifest_digest=digest("simulator"),
        initial_state_digest=digest(f"state-{index}"),
        seed=index,
        random_stream_digest=digest(f"random-stream-{index}"),
        nuisance_configuration_digests=(
            NamedDigest("weather", digest(f"weather-{index}")),
            NamedDigest("traffic", digest(f"traffic-{index}")),
        ),
    )


def intervention() -> InterventionSpec:
    """Build an atomic braking intervention."""
    return InterventionSpec(
        intervention_id="restore-brake-controller",
        mechanism_hypothesis_id="late-braking-causes-collision",
        scope=InterventionScope.COMPONENT,
        operation=InterventionOperation.REPLACE,
        target="controller.braking",
        baseline_value_digest=digest("candidate-controller"),
        intervention_value_digest=digest("reference-controller"),
        rationale="Test whether the changed brake controller drives impact risk.",
        start_step=20,
        end_step=80,
    )


def design(
    *,
    direction: EffectDirection = EffectDirection.DECREASE,
    threshold: float = 1.0,
) -> CounterfactualDesign:
    """Build a six-pair preregistered design."""
    pairs = tuple(
        CounterfactualPair(
            pair_id=f"pair-{index}",
            control=replay_control(index),
            source_failure_event_ids=(f"failure-{index}",),
        )
        for index in range(6)
    )
    return CounterfactualDesign(
        design_id="braking-ablation",
        design_version="1",
        preregistration_digest=digest("preregistration"),
        simulation_evidence_manifest_digest=digest("simulation-evidence"),
        intervention=intervention(),
        estimand=CounterfactualEstimand(
            outcome_name="minimum time to collision",
            summary_statistic="minimum over episode",
            expected_direction=direction,
            minimum_meaningful_effect=threshold,
            unit="seconds",
        ),
        pairs=pairs,
        analysis_config=PairedAnalysisConfig(
            bootstrap_resamples=1_000,
            randomization_resamples=1_000,
            random_seed=42,
        ),
    )


def observations(
    experiment: CounterfactualDesign,
    differences: tuple[float, ...],
) -> tuple[CounterfactualObservation, ...]:
    """Build identity-matched observations with requested paired differences."""
    return tuple(
        CounterfactualObservation(
            pair_id=pair.pair_id,
            baseline=10.0,
            intervention=10.0 + difference,
            baseline_run_id=f"baseline-{pair.pair_id}",
            intervention_run_id=f"intervention-{pair.pair_id}",
            replay_control_digest=pair.control.content_digest(),
            intervention_spec_digest=experiment.intervention.content_digest(),
        )
        for pair, difference in zip(experiment.pairs, differences, strict=True)
    )


def test_replicated_decrease_requires_practical_and_statistical_support() -> None:
    experiment = design()

    report = analyze_counterfactual_design(
        experiment,
        observations(experiment, (-2.0,) * 6),
    )

    assert report.paired_effect.mean_difference == pytest.approx(-2.0)
    assert report.directionally_consistent
    assert report.meaningfully_supported
    assert (
        report.evidence_level
        is CounterfactualEvidenceLevel.REPLICATED_INTERVENTION_ASSOCIATION
    )
    assert report.paired_effect.randomization_p_value == pytest.approx(0.03125)


def test_replicated_increase_uses_the_preregistered_direction() -> None:
    experiment = design(direction=EffectDirection.INCREASE)

    report = analyze_counterfactual_design(
        experiment,
        observations(experiment, (2.0,) * 6),
    )

    assert report.meaningfully_supported
    assert (
        report.evidence_level
        is CounterfactualEvidenceLevel.REPLICATED_INTERVENTION_ASSOCIATION
    )


def test_small_directional_effect_is_not_promoted() -> None:
    experiment = design(threshold=1.0)

    report = analyze_counterfactual_design(
        experiment,
        observations(experiment, (-0.2,) * 6),
    )

    assert report.directionally_consistent
    assert not report.meaningfully_supported
    assert report.evidence_level is CounterfactualEvidenceLevel.DIRECTIONALLY_CONSISTENT


def test_wrong_direction_is_inconclusive() -> None:
    experiment = design()

    report = analyze_counterfactual_design(
        experiment,
        observations(experiment, (2.0,) * 6),
    )

    assert not report.directionally_consistent
    assert report.evidence_level is CounterfactualEvidenceLevel.INCONCLUSIVE


def test_observation_order_does_not_change_report() -> None:
    experiment = design()
    results = observations(experiment, (-1.0, -2.0, -1.5, -2.5, -1.2, -2.2))

    forward = analyze_counterfactual_design(experiment, results)
    reverse = analyze_counterfactual_design(experiment, tuple(reversed(results)))

    assert forward == reverse
    assert forward.content_digest() == reverse.content_digest()


def test_observations_must_exactly_match_design_pairs() -> None:
    experiment = design()
    results = observations(experiment, (-2.0,) * 6)

    with pytest.raises(ValueError, match="missing: pair-5"):
        analyze_counterfactual_design(experiment, results[:-1])
    with pytest.raises(ValueError, match="unique"):
        analyze_counterfactual_design(experiment, results + (results[0],))
    with pytest.raises(ValueError, match="unknown: other"):
        analyze_counterfactual_design(
            experiment,
            results[:-1] + (replace(results[-1], pair_id="other"),),
        )


def test_observations_must_match_control_and_intervention_digests() -> None:
    experiment = design()
    results = observations(experiment, (-2.0,) * 6)

    with pytest.raises(ValueError, match="replay control digest mismatch"):
        analyze_counterfactual_design(
            experiment,
            (replace(results[0], replay_control_digest=digest("wrong")),) + results[1:],
        )
    with pytest.raises(ValueError, match="intervention digest mismatch"):
        analyze_counterfactual_design(
            experiment,
            (replace(results[0], intervention_spec_digest=digest("wrong")),)
            + results[1:],
        )


def test_design_and_observation_validation() -> None:
    experiment = design()
    result = observations(experiment, (-2.0,) * 6)[0]

    with pytest.raises(ValueError, match="at least two pairs"):
        replace(experiment, pairs=experiment.pairs[:1])
    with pytest.raises(ValueError, match="pair IDs must be unique"):
        replace(
            experiment,
            pairs=experiment.pairs + (experiment.pairs[0],),
        )
    with pytest.raises(ValueError, match="must be different"):
        replace(result, intervention_run_id=result.baseline_run_id)
    with pytest.raises(ValueError, match="finite"):
        replace(result, intervention=float("nan"))


def test_intervention_and_estimand_validation() -> None:
    change = intervention()
    estimate = design().estimand

    assert replace(change, scope="state").scope is InterventionScope.STATE
    assert replace(change, operation="clamp").operation is InterventionOperation.CLAMP
    with pytest.raises(ValueError, match="must change"):
        replace(
            change,
            intervention_value_digest=change.baseline_value_digest,
        )
    with pytest.raises(ValueError, match="supplied together"):
        replace(change, end_step=None)
    with pytest.raises(ValueError, match="ordered range"):
        replace(change, start_step=81)
    with pytest.raises(ValueError, match="finite and positive"):
        replace(estimate, minimum_meaningful_effect=0.0)


def test_replay_control_is_order_invariant_and_validated() -> None:
    control = replay_control(0)

    reversed_control = replace(
        control,
        nuisance_configuration_digests=tuple(
            reversed(control.nuisance_configuration_digests)
        ),
    )

    assert control == reversed_control
    assert control.content_digest() == reversed_control.content_digest()
    with pytest.raises(ValueError, match="must be an integer"):
        replace(control, seed=True)
    with pytest.raises(ValueError, match="must be unique"):
        replace(
            control,
            nuisance_configuration_digests=(control.nuisance_configuration_digests[0],)
            * 2,
        )


def test_report_explicitly_bounds_causal_and_simulator_claims() -> None:
    experiment = design()

    report = analyze_counterfactual_design(
        experiment,
        observations(experiment, (-2.0,) * 6),
    )

    limitations = " ".join(report.limitations)
    assert "does not establish a real-world causal mechanism" in limitations
    assert "common-mode simulator or model-form bias" in limitations
    assert "multiplicity control" in limitations
    assert report.design_content_digest == experiment.content_digest()


def failure_conditions() -> tuple[FailureCondition, ...]:
    """Build three source-failure conditions."""
    return tuple(
        FailureCondition(
            condition_id=condition_id,
            value_digest=digest(condition_id),
            category="environment",
        )
        for condition_id in ("fog", "glare", "wet-road")
    )


def trial(
    retained: tuple[str, ...],
    reproduced: bool,
    run_id: str,
) -> ConditionSetTrial:
    """Build one condition-subset reproduction trial."""
    return ConditionSetTrial(
        retained_condition_ids=retained,
        failure_reproduced=reproduced,
        run_id=run_id,
        evidence_digest=digest(run_id),
    )


def test_minimal_counterexample_can_be_verified_as_one_minimal() -> None:
    trials = (
        trial(("fog", "glare", "wet-road"), True, "full"),
        trial(("fog", "glare"), True, "reduced"),
        trial(("fog",), False, "remove-glare"),
        trial(("glare",), False, "remove-fog"),
    )

    report = assess_minimal_counterexample(failure_conditions(), trials)

    assert report.retained_condition_ids == ("fog", "glare")
    assert report.eliminated_condition_ids == ("wet-road",)
    assert report.one_minimal
    assert report.untested_single_removals == ()
    assert report.reproducing_run_ids == ("full", "reduced")


def test_missing_single_removal_prevents_one_minimal_label() -> None:
    trials = (
        trial(("fog", "glare", "wet-road"), True, "full"),
        trial(("fog", "glare"), True, "reduced"),
        trial(("fog",), False, "remove-glare"),
    )

    report = assess_minimal_counterexample(failure_conditions(), trials)

    assert not report.one_minimal
    assert report.untested_single_removals == ("fog",)


def test_smallest_reproducing_set_has_deterministic_tie_break() -> None:
    trials = (
        trial(("fog", "glare", "wet-road"), True, "full"),
        trial(("fog", "wet-road"), True, "second"),
        trial(("fog", "glare"), True, "first"),
        trial(("fog",), False, "fog"),
        trial(("glare",), False, "glare"),
    )

    report = assess_minimal_counterexample(
        tuple(reversed(failure_conditions())),
        tuple(reversed(trials)),
    )

    assert report.retained_condition_ids == ("fog", "glare")


def test_empty_reproducing_set_is_vacuously_one_minimal() -> None:
    trials = (
        trial(("fog", "glare", "wet-road"), True, "full"),
        trial((), True, "empty"),
    )

    report = assess_minimal_counterexample(failure_conditions(), trials)

    assert report.retained_condition_ids == ()
    assert report.one_minimal
    assert report.eliminated_condition_ids == ("fog", "glare", "wet-road")


def test_minimal_counterexample_requires_reproduced_source_anchor() -> None:
    conditions = failure_conditions()

    with pytest.raises(ValueError, match="full source condition set"):
        assess_minimal_counterexample(
            conditions,
            (trial(("fog", "glare"), True, "reduced"),),
        )
    with pytest.raises(ValueError, match="full source condition set"):
        assess_minimal_counterexample(
            conditions,
            (trial(("fog", "glare", "wet-road"), False, "full"),),
        )


def test_minimal_counterexample_rejects_unknown_and_duplicate_sets() -> None:
    conditions = failure_conditions()
    full = trial(("fog", "glare", "wet-road"), True, "full")

    with pytest.raises(ValueError, match="unknown conditions"):
        assess_minimal_counterexample(
            conditions,
            (full, trial(("fog", "snow"), True, "unknown")),
        )
    with pytest.raises(ValueError, match="retained condition sets"):
        assess_minimal_counterexample(
            conditions,
            (full, replace(full, run_id="duplicate")),
        )
    with pytest.raises(ValueError, match="source condition IDs"):
        assess_minimal_counterexample(
            conditions + (conditions[0],),
            (full,),
        )


def test_minimal_report_is_order_invariant_and_bounded() -> None:
    trials = (
        trial(("fog", "glare", "wet-road"), True, "full"),
        trial(("fog", "glare"), True, "reduced"),
        trial(("fog",), False, "remove-glare"),
        trial(("glare",), False, "remove-fog"),
    )

    forward = assess_minimal_counterexample(failure_conditions(), trials)
    reverse = assess_minimal_counterexample(
        tuple(reversed(failure_conditions())),
        tuple(reversed(trials)),
    )

    assert forward == reverse
    assert forward.content_digest() == reverse.content_digest()
    limitations = " ".join(forward.limitations)
    assert "does not prove global minimality or causality" in limitations
    assert "simulator validity" in limitations
