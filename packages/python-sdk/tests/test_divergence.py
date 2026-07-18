"""Tests for deterministic, evidence-linked trace divergence analysis."""

from __future__ import annotations

import math

import pytest

from iso_obs.divergence import (
    PairingAssessment,
    PairingQuality,
    SignalSpec,
    TraceStep,
    compare_traces,
)


def exact_pairing() -> PairingAssessment:
    """Return a fully controlled pairing assessment."""
    return PairingAssessment.assess(
        same_scenario_version=True,
        same_environment_version=True,
        same_seed=True,
        same_perturbation_realization=True,
    )


def trace_step(step: int, **values: float) -> TraceStep:
    """Build a trace step with deterministic evidence IDs."""
    return TraceStep(
        step=step,
        values=values,
        evidence_event_ids={name: f"evt_{step}_{name}" for name in values},
    )


def test_identical_trace_has_no_divergence() -> None:
    trace = [trace_step(step, force=10.0) for step in range(5)]

    report = compare_traces(
        trace,
        trace,
        signals=[
            SignalSpec(
                path="force",
                category="control",
                absolute_tolerance=0.1,
                persistence_steps=2,
            )
        ],
        pairing=exact_pairing(),
    )

    assert report.earliest_numerical_difference_step is None
    assert report.earliest_meaningful_divergence is None
    assert report.findings == ()
    assert report.compared_step_count == 5


def test_transient_difference_does_not_become_meaningful() -> None:
    baseline = [trace_step(step, force=10.0) for step in range(5)]
    candidate = [
        trace_step(step, force=12.0 if step == 2 else 10.0) for step in range(5)
    ]

    report = compare_traces(
        baseline,
        candidate,
        signals=[
            SignalSpec(
                path="force",
                category="control",
                absolute_tolerance=0.5,
                persistence_steps=2,
            )
        ],
        pairing=exact_pairing(),
    )

    assert report.earliest_numerical_difference_step == 2
    assert report.earliest_meaningful_divergence is None


def test_sustained_difference_reports_start_confirmation_and_evidence() -> None:
    baseline = [trace_step(step, force=10.0) for step in range(6)]
    candidate = [
        trace_step(step, force=12.0 if step >= 2 else 10.0) for step in range(6)
    ]

    report = compare_traces(
        baseline,
        candidate,
        signals=[
            SignalSpec(
                path="force",
                category="control",
                absolute_tolerance=0.5,
                relative_tolerance=0.05,
                persistence_steps=3,
            )
        ],
        pairing=exact_pairing(),
    )

    finding = report.earliest_meaningful_divergence
    assert finding is not None
    assert finding.start_step == 2
    assert finding.confirmed_step == 4
    assert finding.persistence_steps == 3
    assert finding.evidence_event_ids == (
        "evt_2_force",
        "evt_3_force",
        "evt_4_force",
    )
    assert [delta.signed_delta for delta in finding.deltas] == [2.0, 2.0, 2.0]


def test_relative_tolerance_scales_with_signal_magnitude() -> None:
    baseline = [trace_step(0, force=100.0)]
    candidate = [trace_step(0, force=104.0)]

    report = compare_traces(
        baseline,
        candidate,
        signals=[
            SignalSpec(
                path="force",
                category="control",
                absolute_tolerance=1.0,
                relative_tolerance=0.05,
            )
        ],
        pairing=exact_pairing(),
    )

    assert report.earliest_numerical_difference_step == 0
    assert report.earliest_meaningful_divergence is None


def test_missing_data_breaks_persistence_and_is_reported() -> None:
    baseline = [trace_step(step, force=10.0) for step in range(4)]
    candidate = [
        trace_step(0, force=12.0),
        trace_step(1, other=1.0),
        trace_step(2, force=12.0),
        trace_step(3, force=12.0),
    ]

    report = compare_traces(
        baseline,
        candidate,
        signals=[
            SignalSpec(
                path="force",
                category="control",
                absolute_tolerance=0.5,
                persistence_steps=2,
            )
        ],
        pairing=exact_pairing(),
    )

    finding = report.earliest_meaningful_divergence
    assert finding is not None
    assert finding.start_step == 2
    assert report.limitations[-1] == ("1 aligned signal comparisons were missing.")


def test_unmatched_steps_and_weak_pairing_are_explicit_limitations() -> None:
    pairing = PairingAssessment.assess(
        same_scenario_version=True,
        same_environment_version=True,
        same_seed=False,
        same_perturbation_realization=False,
    )

    report = compare_traces(
        [trace_step(0, force=10.0), trace_step(1, force=10.0)],
        [trace_step(1, force=10.0), trace_step(2, force=10.0)],
        signals=[SignalSpec(path="force", category="control")],
        pairing=pairing,
    )

    assert pairing.quality is PairingQuality.CONFIGURATION_MATCHED
    assert report.baseline_only_steps == (0,)
    assert report.candidate_only_steps == (2,)
    assert len(report.limitations) == 3


def test_findings_are_ordered_by_earliest_step_then_signal() -> None:
    baseline = [trace_step(step, force=10.0, velocity=1.0) for step in range(4)]
    candidate = [
        trace_step(
            step,
            force=12.0 if step >= 2 else 10.0,
            velocity=2.0 if step >= 1 else 1.0,
        )
        for step in range(4)
    ]

    report = compare_traces(
        baseline,
        candidate,
        signals=[
            SignalSpec(path="force", category="control", absolute_tolerance=0.1),
            SignalSpec(
                path="velocity",
                category="state",
                absolute_tolerance=0.1,
            ),
        ],
        pairing=exact_pairing(),
    )

    assert [finding.signal for finding in report.findings] == [
        "velocity",
        "force",
    ]
    assert report.earliest_meaningful_divergence == report.findings[0]


def test_swapping_traces_preserves_step_and_reverses_signed_delta() -> None:
    baseline = [trace_step(0, force=10.0)]
    candidate = [trace_step(0, force=12.0)]
    signal = SignalSpec(
        path="force",
        category="control",
        absolute_tolerance=0.1,
    )

    forward = compare_traces(
        baseline,
        candidate,
        signals=[signal],
        pairing=exact_pairing(),
    )
    reverse = compare_traces(
        candidate,
        baseline,
        signals=[signal],
        pairing=exact_pairing(),
    )

    forward_finding = forward.earliest_meaningful_divergence
    reverse_finding = reverse.earliest_meaningful_divergence
    assert forward_finding is not None
    assert reverse_finding is not None
    assert forward_finding.start_step == reverse_finding.start_step
    assert forward_finding.deltas[0].signed_delta == 2.0
    assert reverse_finding.deltas[0].signed_delta == -2.0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_signal_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        trace_step(0, force=value)


def test_duplicate_trace_steps_are_rejected() -> None:
    duplicate = [trace_step(0, force=1.0), trace_step(0, force=2.0)]
    with pytest.raises(ValueError, match="duplicate step 0"):
        compare_traces(
            duplicate,
            [],
            signals=[SignalSpec(path="force", category="control")],
            pairing=exact_pairing(),
        )


def test_duplicate_signal_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate signal specs"):
        compare_traces(
            [],
            [],
            signals=[
                SignalSpec(path="force", category="control"),
                SignalSpec(path="force", category="safety"),
            ],
            pairing=exact_pairing(),
        )


def test_trace_step_defensively_freezes_input_mappings() -> None:
    values = {"force": 10.0}
    evidence = {"force": "evt_force"}
    step = TraceStep(
        step=0,
        values=values,
        evidence_event_ids=evidence,
    )

    values["force"] = 99.0
    evidence["force"] = "evt_changed"

    assert step.values["force"] == 10.0
    assert step.evidence_event_ids["force"] == "evt_force"
    with pytest.raises(TypeError):
        step.values["force"] = 11.0  # type: ignore[index]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_tolerance_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        SignalSpec(
            path="force",
            category="control",
            absolute_tolerance=value,
        )


def test_inconsistent_pairing_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="pairing quality must be unpaired"):
        PairingAssessment(
            quality=PairingQuality.EXACT_REPLAY,
            same_scenario_version=False,
            same_environment_version=False,
            same_seed=False,
            same_perturbation_realization=False,
        )
