"""Tests for anytime-valid sequential reliability surveillance."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from iso_obs.assurance import (
    AssuranceDisposition,
    ReleaseAssuranceReport,
)
from iso_obs.evidence import sha256_digest
from iso_obs.surveillance import (
    AlternativeRate,
    SequentialMonitoringPlan,
    SignalMonitoringSpec,
    SurveillanceDisposition,
    SurveillanceEvidenceState,
    SurveillanceObservation,
    evaluate_sequential_surveillance,
)


def digest(value: str) -> str:
    """Build a valid digest for test content."""
    return sha256_digest(value)


def assurance_report(
    *,
    disposition: AssuranceDisposition = AssuranceDisposition.SUPPORTED_WITHIN_SCOPE,
) -> ReleaseAssuranceReport:
    """Build a scoped assurance report for surveillance."""
    return ReleaseAssuranceReport(
        dossier_content_digest=digest("dossier"),
        release_scope_digest=digest("release-scope"),
        policy_content_digest=digest("assurance-policy"),
        regression_results=(),
        independent_passing_challenge_count=1,
        independent_passing_challenge_kind_counts=(),
        open_defeater_ids=(),
        blocking_reasons=(),
        restriction_reasons=(),
        disposition=disposition,
        limitations=("Release evidence remains scoped.",),
    )


def signal(
    signal_id: str = "safety-intervention",
    *,
    maximum_rate: float = 0.1,
    allocated_error_rate: float = 0.025,
    alternatives: tuple[AlternativeRate, ...] | None = None,
) -> SignalMonitoringSpec:
    """Build one pre-specified binary signal test."""
    return SignalMonitoringSpec(
        signal_id=signal_id,
        description=f"Monitor {signal_id.replace('-', ' ')} events.",
        outcome_definition_digest=digest(f"definition-{signal_id}"),
        maximum_acceptable_event_rate=maximum_rate,
        alternatives=alternatives
        or (AlternativeRate(event_probability=0.5, weight=1.0),),
        allocated_error_rate=allocated_error_rate,
    )


def plan(
    report: ReleaseAssuranceReport,
    *,
    signals: tuple[SignalMonitoringSpec, ...] | None = None,
    family_error_rate: float = 0.05,
) -> SequentialMonitoringPlan:
    """Build a monitoring plan linked to an assurance report."""
    return SequentialMonitoringPlan(
        plan_id="policy-v18-field-monitoring",
        plan_version="1",
        assurance_report_digest=report.content_digest(),
        release_scope_digest=report.release_scope_digest,
        family_error_rate=family_error_rate,
        signals=signals
        or (
            signal(),
            signal("scope-exit"),
        ),
        limitations=("Events depend on validated production labeling.",),
    )


def observations(
    report: ReleaseAssuranceReport,
    signal_id: str,
    outcomes: tuple[bool, ...],
) -> tuple[SurveillanceObservation, ...]:
    """Build a contiguous observation sequence."""
    return tuple(
        SurveillanceObservation(
            signal_id=signal_id,
            sequence_index=index,
            observation_id=f"{signal_id}-observation-{index}",
            exposure_id=f"{signal_id}-exposure-{index}",
            occurred=occurred,
            release_scope_digest=report.release_scope_digest,
            evidence_digest=digest(f"{signal_id}-evidence-{index}"),
        )
        for index, occurred in enumerate(outcomes)
    )


def result_by_id(report, signal_id: str):
    """Select a signal result by stable ID."""
    return next(item for item in report.signal_results if item.signal_id == signal_id)


def test_two_events_cross_the_anytime_valid_threshold() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)
    observed = observations(
        release_report,
        "safety-intervention",
        (True, True),
    )

    report = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        observed,
    )
    result = result_by_id(report, "safety-intervention")

    assert result.current_log_e_value == pytest.approx(math.log(25.0))
    assert result.log_e_value_threshold == pytest.approx(math.log(40.0))
    assert result.evidence_state is SurveillanceEvidenceState.NO_THRESHOLD_CROSSING
    assert report.disposition is SurveillanceDisposition.CONTINUE_MONITORING


def test_three_events_trigger_scope_review() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)

    report = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        observations(
            release_report,
            "safety-intervention",
            (True, True, True),
        ),
    )
    result = result_by_id(report, "safety-intervention")

    assert result.first_crossing_observation_count == 3
    assert result.evidence_state is SurveillanceEvidenceState.THRESHOLD_CROSSED
    assert report.crossed_signal_ids == ("safety-intervention",)
    assert report.disposition is SurveillanceDisposition.ESCALATE_SCOPE_REVIEW


def test_historical_crossing_persists_after_current_evidence_declines() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)
    outcomes = (True, True, True) + (False,) * 20

    report = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        observations(release_report, "safety-intervention", outcomes),
    )
    result = result_by_id(report, "safety-intervention")

    assert result.current_log_e_value < result.log_e_value_threshold
    assert result.maximum_log_e_value >= result.log_e_value_threshold
    assert result.first_crossing_observation_count == 3
    assert result.evidence_state is SurveillanceEvidenceState.THRESHOLD_CROSSED


def test_many_non_events_do_not_claim_acceptance() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)

    report = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        observations(
            release_report,
            "safety-intervention",
            (False,) * 100,
        ),
    )
    result = result_by_id(report, "safety-intervention")

    assert result.empirical_event_rate == 0.0
    assert result.evidence_state is SurveillanceEvidenceState.NO_THRESHOLD_CROSSING
    assert report.disposition is SurveillanceDisposition.CONTINUE_MONITORING
    assert "not evidence that the system is safe" in " ".join(report.limitations)


def test_empty_signal_stream_has_neutral_e_value() -> None:
    release_report = assurance_report()

    report = evaluate_sequential_surveillance(
        plan(release_report),
        release_report,
        (),
    )

    for result in report.signal_results:
        assert result.observation_count == 0
        assert result.empirical_event_rate is None
        assert result.current_log_e_value == 0.0
        assert result.maximum_log_e_value == 0.0


def test_weighted_alternative_mixture_matches_likelihood_ratio_sum() -> None:
    release_report = assurance_report()
    mixture_signal = signal(
        alternatives=(
            AlternativeRate(0.2, 0.25),
            AlternativeRate(0.5, 0.75),
        ),
    )
    monitoring_plan = plan(release_report, signals=(mixture_signal,))

    report = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        observations(
            release_report,
            mixture_signal.signal_id,
            (True, False, True),
        ),
    )
    result = report.signal_results[0]
    expected = 0.25 * (0.2 / 0.1) ** 2 * (0.8 / 0.9) + 0.75 * (0.5 / 0.1) ** 2 * (
        0.5 / 0.9
    )

    assert result.current_log_e_value == pytest.approx(math.log(expected))


def test_input_order_does_not_change_report() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)
    observed = observations(
        release_report,
        "safety-intervention",
        (False, True, False, True, True),
    )

    forward = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        observed,
    )
    reverse = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        tuple(reversed(observed)),
    )

    assert forward == reverse
    assert forward.content_digest() == reverse.content_digest()


def test_multiple_signals_are_evaluated_without_independence_assumption() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)
    observed = observations(
        release_report,
        "safety-intervention",
        (True, True, True),
    ) + observations(
        release_report,
        "scope-exit",
        (False, False, False),
    )

    report = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        observed,
    )

    assert report.crossed_signal_ids == ("safety-intervention",)
    assert "does not require signal independence" not in " ".join(report.limitations)
    assert "union bound" in " ".join(report.limitations)


def test_plan_rejects_signal_allocations_above_family_budget() -> None:
    release_report = assurance_report()
    signals = (
        signal("one", allocated_error_rate=0.03),
        signal("two", allocated_error_rate=0.03),
    )

    with pytest.raises(ValueError, match="must not exceed"):
        plan(release_report, signals=signals, family_error_rate=0.05)


def test_signal_rejects_invalid_null_alternatives_and_weights() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        signal(alternatives=(AlternativeRate(0.05, 1.0),))
    with pytest.raises(ValueError, match="sum to one"):
        signal(
            alternatives=(
                AlternativeRate(0.2, 0.2),
                AlternativeRate(0.5, 0.7),
            )
        )
    with pytest.raises(ValueError, match="must be unique"):
        signal(
            alternatives=(
                AlternativeRate(0.2, 0.5),
                AlternativeRate(0.2, 0.5),
            )
        )
    with pytest.raises(ValueError, match="between zero and one"):
        signal(maximum_rate=0.0)


def test_plan_and_signal_order_are_canonical() -> None:
    release_report = assurance_report()
    original = plan(release_report)
    reordered = replace(
        original,
        signals=tuple(reversed(original.signals)),
        limitations=tuple(reversed(original.limitations)),
    )

    assert original == reordered
    assert original.content_digest() == reordered.content_digest()


def test_assurance_report_and_scope_must_match_plan() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)

    with pytest.raises(ValueError, match="assurance report content"):
        evaluate_sequential_surveillance(
            monitoring_plan,
            replace(
                release_report,
                limitations=("Changed assurance evidence.",),
            ),
            (),
        )
    changed_scope_report = replace(
        release_report,
        release_scope_digest=digest("other-scope"),
    )
    changed_plan = replace(
        monitoring_plan,
        assurance_report_digest=changed_scope_report.content_digest(),
    )
    with pytest.raises(ValueError, match="release scope"):
        evaluate_sequential_surveillance(
            changed_plan,
            changed_scope_report,
            (),
        )


def test_blocked_assurance_cannot_be_used_for_release_surveillance() -> None:
    blocked = assurance_report(disposition=AssuranceDisposition.BLOCKED)

    with pytest.raises(ValueError, match="blocked assurance scope"):
        evaluate_sequential_surveillance(plan(blocked), blocked, ())


def test_unknown_or_cross_scope_observation_is_rejected() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)
    observed = observations(
        release_report,
        "safety-intervention",
        (False,),
    )[0]

    with pytest.raises(ValueError, match="unknown surveillance signal"):
        evaluate_sequential_surveillance(
            monitoring_plan,
            release_report,
            (replace(observed, signal_id="unknown"),),
        )
    with pytest.raises(ValueError, match="release scope"):
        evaluate_sequential_surveillance(
            monitoring_plan,
            release_report,
            (
                replace(
                    observed,
                    release_scope_digest=digest("other-scope"),
                ),
            ),
        )


def test_duplicate_gap_and_reused_exposure_are_rejected() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)
    observed = observations(
        release_report,
        "safety-intervention",
        (False, True),
    )

    with pytest.raises(ValueError, match="duplicate surveillance observation"):
        evaluate_sequential_surveillance(
            monitoring_plan,
            release_report,
            (observed[0], observed[0]),
        )
    with pytest.raises(ValueError, match="contiguous from zero"):
        evaluate_sequential_surveillance(
            monitoring_plan,
            release_report,
            (replace(observed[0], sequence_index=1),),
        )
    with pytest.raises(ValueError, match="exposure IDs"):
        evaluate_sequential_surveillance(
            monitoring_plan,
            release_report,
            (
                observed[0],
                replace(observed[1], exposure_id=observed[0].exposure_id),
            ),
        )


def test_observation_validation_rejects_invalid_order_and_outcome() -> None:
    release_report = assurance_report()
    observed = observations(
        release_report,
        "safety-intervention",
        (False,),
    )[0]

    with pytest.raises(ValueError, match="non-negative integer"):
        replace(observed, sequence_index=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        replace(observed, occurred=1)


def test_long_sequences_remain_finite_in_log_space() -> None:
    release_report = assurance_report()
    monitoring_plan = plan(release_report)

    report = evaluate_sequential_surveillance(
        monitoring_plan,
        release_report,
        observations(
            release_report,
            "safety-intervention",
            (True,) * 10_000,
        ),
    )
    result = result_by_id(report, "safety-intervention")

    assert math.isfinite(result.current_log_e_value)
    assert math.isfinite(result.maximum_log_e_value)


def test_report_preserves_statistical_and_operational_boundaries() -> None:
    release_report = assurance_report()

    report = evaluate_sequential_surveillance(
        plan(release_report),
        release_report,
        observations(
            release_report,
            "safety-intervention",
            (True, True, True),
        ),
    )
    limitations = " ".join(report.limitations)

    assert "does not identify a cause" in limitations
    assert "reporting, exposure, or label bias" in limitations
    assert "not itself an automated deployment or shutdown decision" in limitations
    assert "Events depend on validated production labeling." in limitations
