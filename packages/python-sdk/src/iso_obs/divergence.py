"""Deterministic behavioral-divergence analysis for paired run traces.

This module locates the earliest sustained signal difference between a baseline
and candidate trace. It intentionally makes a narrow claim: a finding identifies
an observed divergence that preceded or accompanied an outcome; it does not
claim that the divergence caused that outcome.

The v0 analyzer uses exact environment-step alignment. More permissive temporal
alignment belongs in a future layer that reports alignment quality explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .evidence import _canonical_json, sha256_digest


class PairingQuality(StrEnum):
    """Strength of experimental control between two traces."""

    EXACT_REPLAY = "exact_replay"
    SEED_MATCHED = "seed_matched"
    CONFIGURATION_MATCHED = "configuration_matched"
    UNPAIRED = "unpaired"


@dataclass(frozen=True, slots=True)
class PairingAssessment:
    """Experimental-pairing metadata for a trace comparison."""

    quality: PairingQuality
    same_scenario_version: bool
    same_environment_version: bool
    same_seed: bool
    same_perturbation_realization: bool

    def __post_init__(self) -> None:
        """Reject a quality label inconsistent with the recorded controls."""
        expected = _pairing_quality(
            same_scenario_version=self.same_scenario_version,
            same_environment_version=self.same_environment_version,
            same_seed=self.same_seed,
            same_perturbation_realization=self.same_perturbation_realization,
        )
        if self.quality is not expected:
            raise ValueError(
                f"pairing quality must be {expected.value} for the recorded "
                "experimental controls"
            )

    @classmethod
    def assess(
        cls,
        *,
        same_scenario_version: bool,
        same_environment_version: bool,
        same_seed: bool,
        same_perturbation_realization: bool,
    ) -> PairingAssessment:
        """Classify pairing quality from controlled experimental factors.

        Args:
            same_scenario_version: Whether both runs use one scenario version.
            same_environment_version: Whether environment versions match.
            same_seed: Whether the declared run seeds match.
            same_perturbation_realization: Whether stochastic perturbations use
                the same realized random stream.

        Returns:
            Pairing metadata with the strongest justified quality level.
        """
        quality = _pairing_quality(
            same_scenario_version=same_scenario_version,
            same_environment_version=same_environment_version,
            same_seed=same_seed,
            same_perturbation_realization=same_perturbation_realization,
        )
        return cls(
            quality=quality,
            same_scenario_version=same_scenario_version,
            same_environment_version=same_environment_version,
            same_seed=same_seed,
            same_perturbation_realization=same_perturbation_realization,
        )


@dataclass(frozen=True, slots=True)
class TraceStep:
    """Numeric signals and evidence references at one environment step."""

    step: int
    values: Mapping[str, float]
    evidence_event_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate step coordinates and numeric evidence."""
        if self.step < 0:
            raise ValueError("trace step must be non-negative")
        frozen_values = {name: float(value) for name, value in self.values.items()}
        invalid = [
            name for name, value in frozen_values.items() if not math.isfinite(value)
        ]
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(f"trace step contains non-finite signals: {names}")
        object.__setattr__(
            self,
            "values",
            MappingProxyType(frozen_values),
        )
        object.__setattr__(
            self,
            "evidence_event_ids",
            MappingProxyType(dict(self.evidence_event_ids)),
        )


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Comparison policy for one numeric trace signal."""

    path: str
    category: str
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0
    persistence_steps: int = 1

    def __post_init__(self) -> None:
        """Validate tolerances and persistence requirements."""
        if not self.path:
            raise ValueError("signal path must not be empty")
        if not self.category:
            raise ValueError("signal category must not be empty")
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0.0:
            raise ValueError("absolute tolerance must be finite and non-negative")
        if not math.isfinite(self.relative_tolerance) or self.relative_tolerance < 0.0:
            raise ValueError("relative tolerance must be finite and non-negative")
        if self.persistence_steps < 1:
            raise ValueError("persistence_steps must be at least one")


@dataclass(frozen=True, slots=True)
class SignalDelta:
    """Observed baseline-to-candidate difference at one aligned step."""

    step: int
    baseline: float
    candidate: float
    signed_delta: float
    absolute_delta: float
    relative_delta: float
    threshold: float


@dataclass(frozen=True, slots=True)
class DivergenceFinding:
    """Earliest sustained meaningful divergence for one signal."""

    signal: str
    category: str
    start_step: int
    confirmed_step: int
    persistence_steps: int
    deltas: tuple[SignalDelta, ...]
    evidence_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DivergenceReport:
    """Deterministic result of comparing one baseline/candidate trace pair."""

    pairing: PairingAssessment
    compared_step_count: int
    baseline_only_steps: tuple[int, ...]
    candidate_only_steps: tuple[int, ...]
    earliest_numerical_difference_step: int | None
    earliest_meaningful_divergence: DivergenceFinding | None
    findings: tuple[DivergenceFinding, ...]
    limitations: tuple[str, ...]
    report_schema_version: str = "iso-obs.divergence-report.v1"

    def to_json(self) -> str:
        """Serialize the divergence report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact divergence-report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def _pairing_quality(
    *,
    same_scenario_version: bool,
    same_environment_version: bool,
    same_seed: bool,
    same_perturbation_realization: bool,
) -> PairingQuality:
    """Classify pairing strength from controlled experimental factors.

    Args:
        same_scenario_version: Whether scenario versions match.
        same_environment_version: Whether environment versions match.
        same_seed: Whether declared seeds match.
        same_perturbation_realization: Whether realized randomness matches.

    Returns:
        The strongest justified pairing quality.
    """
    if (
        same_scenario_version
        and same_environment_version
        and same_seed
        and same_perturbation_realization
    ):
        return PairingQuality.EXACT_REPLAY
    if same_scenario_version and same_environment_version and same_seed:
        return PairingQuality.SEED_MATCHED
    if same_scenario_version and same_environment_version:
        return PairingQuality.CONFIGURATION_MATCHED
    return PairingQuality.UNPAIRED


def compare_traces(
    baseline: Sequence[TraceStep],
    candidate: Sequence[TraceStep],
    *,
    signals: Sequence[SignalSpec],
    pairing: PairingAssessment,
) -> DivergenceReport:
    """Compare paired traces using exact-step sustained-change detection.

    A signal exceeds tolerance when its absolute difference is greater than the
    larger of its absolute threshold and its relative threshold multiplied by
    the larger observed magnitude. A meaningful finding requires that condition
    at consecutive environment steps for ``persistence_steps``.

    Args:
        baseline: Baseline trace steps.
        candidate: Candidate trace steps.
        signals: Signal-specific comparison policies.
        pairing: Experimental pairing assessment.

    Returns:
        A report with pairing, alignment diagnostics, findings, and limitations.

    Raises:
        ValueError: If a trace contains duplicate steps or signal specs repeat.
    """
    baseline_by_step = _index_trace(baseline, name="baseline")
    candidate_by_step = _index_trace(candidate, name="candidate")
    _validate_signal_specs(signals)

    baseline_steps = set(baseline_by_step)
    candidate_steps = set(candidate_by_step)
    aligned_steps = sorted(baseline_steps & candidate_steps)
    baseline_only = tuple(sorted(baseline_steps - candidate_steps))
    candidate_only = tuple(sorted(candidate_steps - baseline_steps))

    findings: list[DivergenceFinding] = []
    earliest_numerical: int | None = None
    missing_signal_pairs = 0
    for spec in signals:
        deltas: list[tuple[SignalDelta, tuple[str, ...]]] = []
        for step in aligned_steps:
            baseline_step = baseline_by_step[step]
            candidate_step = candidate_by_step[step]
            if (
                spec.path not in baseline_step.values
                or spec.path not in candidate_step.values
            ):
                missing_signal_pairs += 1
                deltas.append((_gap_delta(step), ()))
                continue
            baseline_value = float(baseline_step.values[spec.path])
            candidate_value = float(candidate_step.values[spec.path])
            delta = _calculate_delta(
                step=step,
                baseline=baseline_value,
                candidate=candidate_value,
                spec=spec,
            )
            if delta.absolute_delta > 0.0 and (
                earliest_numerical is None or step < earliest_numerical
            ):
                earliest_numerical = step
            evidence = _evidence_ids(baseline_step, candidate_step, spec.path)
            deltas.append((delta, evidence))

        finding = _first_sustained_finding(spec, deltas)
        if finding is not None:
            findings.append(finding)

    ordered_findings = tuple(
        sorted(findings, key=lambda item: (item.start_step, item.signal))
    )
    limitations = _limitations(
        pairing=pairing,
        baseline_only=baseline_only,
        candidate_only=candidate_only,
        missing_signal_pairs=missing_signal_pairs,
    )
    return DivergenceReport(
        pairing=pairing,
        compared_step_count=len(aligned_steps),
        baseline_only_steps=baseline_only,
        candidate_only_steps=candidate_only,
        earliest_numerical_difference_step=earliest_numerical,
        earliest_meaningful_divergence=(
            ordered_findings[0] if ordered_findings else None
        ),
        findings=ordered_findings,
        limitations=limitations,
    )


def _index_trace(trace: Sequence[TraceStep], *, name: str) -> dict[int, TraceStep]:
    """Index a trace by unique environment step.

    Args:
        trace: Trace steps to index.
        name: Trace label used in validation errors.

    Returns:
        Mapping from step number to trace evidence.

    Raises:
        ValueError: If more than one record carries the same step.
    """
    indexed: dict[int, TraceStep] = {}
    for item in trace:
        if item.step in indexed:
            raise ValueError(f"{name} trace contains duplicate step {item.step}")
        indexed[item.step] = item
    return indexed


def _validate_signal_specs(signals: Sequence[SignalSpec]) -> None:
    """Require one comparison policy per signal path.

    Args:
        signals: Signal policies to validate.

    Raises:
        ValueError: If a signal path is declared more than once.
    """
    paths = [spec.path for spec in signals]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        raise ValueError(f"duplicate signal specs: {', '.join(duplicates)}")


def _calculate_delta(
    *,
    step: int,
    baseline: float,
    candidate: float,
    spec: SignalSpec,
) -> SignalDelta:
    """Calculate a scale-aware difference for one aligned signal.

    Args:
        step: Aligned environment step.
        baseline: Baseline signal value.
        candidate: Candidate signal value.
        spec: Signal comparison policy.

    Returns:
        Difference values and the effective tolerance threshold.
    """
    signed = candidate - baseline
    absolute = abs(signed)
    scale = max(abs(baseline), abs(candidate))
    threshold = max(
        spec.absolute_tolerance,
        spec.relative_tolerance * scale,
    )
    relative = absolute / scale if scale > 0.0 else 0.0
    return SignalDelta(
        step=step,
        baseline=baseline,
        candidate=candidate,
        signed_delta=signed,
        absolute_delta=absolute,
        relative_delta=relative,
        threshold=threshold,
    )


def _gap_delta(step: int) -> SignalDelta:
    """Create a sentinel delta that breaks persistence across missing data.

    Args:
        step: Aligned step whose signal value is missing.

    Returns:
        A zero-valued delta that cannot exceed a tolerance.
    """
    return SignalDelta(
        step=step,
        baseline=0.0,
        candidate=0.0,
        signed_delta=0.0,
        absolute_delta=0.0,
        relative_delta=0.0,
        threshold=math.inf,
    )


def _evidence_ids(
    baseline: TraceStep,
    candidate: TraceStep,
    signal: str,
) -> tuple[str, ...]:
    """Collect stable evidence references for one signal comparison.

    Args:
        baseline: Baseline trace step.
        candidate: Candidate trace step.
        signal: Compared signal path.

    Returns:
        Existing baseline then candidate event IDs, without duplicates.
    """
    values = [
        baseline.evidence_event_ids.get(signal),
        candidate.evidence_event_ids.get(signal),
    ]
    return tuple(dict.fromkeys(value for value in values if value is not None))


def _first_sustained_finding(
    spec: SignalSpec,
    deltas: Sequence[tuple[SignalDelta, tuple[str, ...]]],
) -> DivergenceFinding | None:
    """Find the earliest consecutive window above a signal tolerance.

    Args:
        spec: Signal comparison policy.
        deltas: Ordered deltas and evidence IDs.

    Returns:
        The earliest sustained finding, or ``None``.
    """
    window: list[tuple[SignalDelta, tuple[str, ...]]] = []
    for delta, evidence in deltas:
        previous_step = window[-1][0].step if window else None
        consecutive = previous_step is None or delta.step == previous_step + 1
        exceeds = delta.absolute_delta > delta.threshold
        if not exceeds or not consecutive:
            window = []
        if exceeds:
            window.append((delta, evidence))
        if len(window) == spec.persistence_steps:
            evidence_ids = tuple(
                dict.fromkeys(event_id for _, ids in window for event_id in ids)
            )
            return DivergenceFinding(
                signal=spec.path,
                category=spec.category,
                start_step=window[0][0].step,
                confirmed_step=window[-1][0].step,
                persistence_steps=spec.persistence_steps,
                deltas=tuple(item[0] for item in window),
                evidence_event_ids=evidence_ids,
            )
    return None


def _limitations(
    *,
    pairing: PairingAssessment,
    baseline_only: tuple[int, ...],
    candidate_only: tuple[int, ...],
    missing_signal_pairs: int,
) -> tuple[str, ...]:
    """Describe conditions that limit interpretation of the comparison.

    Args:
        pairing: Experimental pairing assessment.
        baseline_only: Steps absent from the candidate trace.
        candidate_only: Steps absent from the baseline trace.
        missing_signal_pairs: Aligned steps missing at least one signal value.

    Returns:
        Deterministically ordered limitation statements.
    """
    limitations = ["Observed divergence does not, by itself, establish causality."]
    if pairing.quality is not PairingQuality.EXACT_REPLAY:
        limitations.append(
            f"Pairing quality is {pairing.quality.value}; uncontrolled "
            "variation may contribute to observed differences."
        )
    if baseline_only or candidate_only:
        limitations.append("Exact-step alignment excluded unmatched trace steps.")
    if missing_signal_pairs:
        limitations.append(
            f"{missing_signal_pairs} aligned signal comparisons were missing."
        )
    return tuple(limitations)
