"""Anytime-valid sequential surveillance for scoped reliability claims.

Each binary event stream is tested against a pre-specified maximum acceptable
conditional event rate using a mixture of likelihood-ratio e-processes. The
result may be inspected after every observation without ordinary repeated-
testing inflation. No threshold crossing means only that monitoring continues;
it does not prove that a system is safe or that its event rate is acceptable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .assurance import (
    AssuranceDisposition,
    ReleaseAssuranceReport,
)
from .evidence import NamedDigest, _canonical_json, sha256_digest


class SurveillanceEvidenceState(StrEnum):
    """Anytime-valid evidence state for one monitored signal."""

    NO_THRESHOLD_CROSSING = "no_threshold_crossing"
    THRESHOLD_CROSSED = "threshold_crossed"


class SurveillanceDisposition(StrEnum):
    """Operational interpretation of a surveillance report."""

    CONTINUE_MONITORING = "continue_monitoring"
    ESCALATE_SCOPE_REVIEW = "escalate_scope_review"


@dataclass(frozen=True, slots=True)
class AlternativeRate:
    """One higher event rate and its pre-specified mixture weight."""

    event_probability: float
    weight: float

    def __post_init__(self) -> None:
        """Validate a finite probability and positive mixture weight."""
        probability = _open_unit_interval(
            self.event_probability,
            "alternative event probability",
        )
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("alternative rate weight must be finite and positive")
        object.__setattr__(self, "event_probability", probability)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class SignalMonitoringSpec:
    """Pre-specified anytime-valid test for one binary event stream."""

    signal_id: str
    description: str
    outcome_definition_digest: str
    maximum_acceptable_event_rate: float
    alternatives: tuple[AlternativeRate, ...]
    allocated_error_rate: float

    def __post_init__(self) -> None:
        """Validate the null bound, alternatives, and error allocation."""
        _require_text(self.signal_id, "surveillance signal ID")
        _require_text(self.description, "surveillance signal description")
        NamedDigest(
            "surveillance outcome definition",
            self.outcome_definition_digest,
        )
        maximum = _open_unit_interval(
            self.maximum_acceptable_event_rate,
            "maximum acceptable event rate",
        )
        allocated = _open_unit_interval(
            self.allocated_error_rate,
            "allocated error rate",
        )
        if not self.alternatives:
            raise ValueError("signal monitoring requires alternative rates")
        _require_unique(
            (item.event_probability for item in self.alternatives),
            "alternative event probabilities",
        )
        invalid = sorted(
            item.event_probability
            for item in self.alternatives
            if item.event_probability <= maximum
        )
        if invalid:
            raise ValueError(
                "alternative event probabilities must exceed the maximum "
                f"acceptable rate: {invalid}"
            )
        weight_sum = math.fsum(item.weight for item in self.alternatives)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("alternative rate weights must sum to one")
        object.__setattr__(
            self,
            "maximum_acceptable_event_rate",
            maximum,
        )
        object.__setattr__(self, "allocated_error_rate", allocated)
        object.__setattr__(
            self,
            "alternatives",
            tuple(
                sorted(
                    self.alternatives,
                    key=lambda item: item.event_probability,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class SequentialMonitoringPlan:
    """Content-addressed surveillance plan for one assurance scope."""

    plan_id: str
    plan_version: str
    assurance_report_digest: str
    release_scope_digest: str
    family_error_rate: float
    signals: tuple[SignalMonitoringSpec, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate plan identity, signal family, and error budget."""
        _require_text(self.plan_id, "sequential monitoring plan ID")
        _require_text(self.plan_version, "sequential monitoring plan version")
        NamedDigest("assurance report", self.assurance_report_digest)
        NamedDigest("monitoring release scope", self.release_scope_digest)
        family_rate = _open_unit_interval(
            self.family_error_rate,
            "family error rate",
        )
        if not self.signals:
            raise ValueError("sequential monitoring plan requires signals")
        _require_unique(
            (item.signal_id for item in self.signals),
            "surveillance signal IDs",
        )
        allocated = math.fsum(item.allocated_error_rate for item in self.signals)
        if allocated > family_rate and not math.isclose(
            allocated,
            family_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "signal error allocations must not exceed the family error rate"
            )
        _require_unique_text(
            self.limitations,
            "sequential monitoring plan limitations",
        )
        object.__setattr__(self, "family_error_rate", family_rate)
        object.__setattr__(
            self,
            "signals",
            tuple(sorted(self.signals, key=lambda item: item.signal_id)),
        )
        object.__setattr__(self, "limitations", tuple(sorted(self.limitations)))

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact monitoring-plan content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class SurveillanceObservation:
    """One ordered binary outcome for a monitored signal and exposure."""

    signal_id: str
    sequence_index: int
    observation_id: str
    exposure_id: str
    occurred: bool
    release_scope_digest: str
    evidence_digest: str

    def __post_init__(self) -> None:
        """Validate observation order, identity, scope, and outcome."""
        for value, label in (
            (self.signal_id, "observation signal ID"),
            (self.observation_id, "surveillance observation ID"),
            (self.exposure_id, "surveillance exposure ID"),
        ):
            _require_text(value, label)
        if (
            isinstance(self.sequence_index, bool)
            or not isinstance(self.sequence_index, int)
            or self.sequence_index < 0
        ):
            raise ValueError(
                "surveillance sequence_index must be a non-negative integer"
            )
        if not isinstance(self.occurred, bool):
            raise ValueError("surveillance occurred must be a boolean")
        NamedDigest("observation release scope", self.release_scope_digest)
        NamedDigest("surveillance observation evidence", self.evidence_digest)


@dataclass(frozen=True, slots=True)
class SignalSurveillanceResult:
    """Anytime-valid evidence trajectory summary for one signal."""

    signal_id: str
    observation_count: int
    event_count: int
    empirical_event_rate: float | None
    current_log_e_value: float
    maximum_log_e_value: float
    log_e_value_threshold: float
    first_crossing_observation_count: int | None
    evidence_state: SurveillanceEvidenceState


@dataclass(frozen=True, slots=True)
class SequentialSurveillanceReport:
    """Deterministic, scope-bound summary of all monitored signals."""

    plan_content_digest: str
    assurance_report_digest: str
    release_scope_digest: str
    signal_results: tuple[SignalSurveillanceResult, ...]
    crossed_signal_ids: tuple[str, ...]
    disposition: SurveillanceDisposition
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact surveillance-report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def evaluate_sequential_surveillance(
    plan: SequentialMonitoringPlan,
    assurance_report: ReleaseAssuranceReport,
    observations: Sequence[SurveillanceObservation],
) -> SequentialSurveillanceReport:
    """Evaluate anytime-valid evidence for all planned binary signals.

    Each alternative likelihood ratio is a nonnegative supermartingale under
    the composite null that the conditional event probability never exceeds
    the accepted bound. Their pre-specified weighted sum is an e-process.
    Ville's inequality justifies threshold ``1 / allocated_error_rate`` at
    arbitrary stopping times. Family control uses the union bound across
    signals and does not require signal independence.

    Args:
        plan: Pre-specified signal family and error allocations.
        assurance_report: Exact scoped assurance report being monitored.
        observations: Ordered binary outcomes, supplied in any input order.

    Returns:
        Signal trajectories and a bounded operational disposition.

    Raises:
        ValueError: If assurance identity, observation scope, signal identity,
            ordering, or exposure uniqueness is invalid.
    """
    if assurance_report.content_digest() != plan.assurance_report_digest:
        raise ValueError("assurance report content digest mismatch")
    if assurance_report.release_scope_digest != plan.release_scope_digest:
        raise ValueError("assurance report release scope digest mismatch")
    if assurance_report.disposition is AssuranceDisposition.BLOCKED:
        raise ValueError(
            "blocked assurance scope cannot be used as a release monitoring plan"
        )

    observations_by_signal: dict[str, list[SurveillanceObservation]] = {
        item.signal_id: [] for item in plan.signals
    }
    observation_keys: set[tuple[str, str]] = set()
    for observation in observations:
        if observation.signal_id not in observations_by_signal:
            raise ValueError(f"unknown surveillance signal: {observation.signal_id}")
        if observation.release_scope_digest != plan.release_scope_digest:
            raise ValueError(
                f"observation '{observation.observation_id}' release scope "
                "digest mismatch"
            )
        key = (observation.signal_id, observation.observation_id)
        if key in observation_keys:
            raise ValueError(
                "duplicate surveillance observation: "
                f"{observation.signal_id}, {observation.observation_id}"
            )
        observation_keys.add(key)
        observations_by_signal[observation.signal_id].append(observation)

    results = []
    for spec in plan.signals:
        ordered = tuple(
            sorted(
                observations_by_signal[spec.signal_id],
                key=lambda item: item.sequence_index,
            )
        )
        expected_indices = tuple(range(len(ordered)))
        observed_indices = tuple(item.sequence_index for item in ordered)
        if observed_indices != expected_indices:
            raise ValueError(
                f"signal '{spec.signal_id}' sequence indices must be "
                "contiguous from zero"
            )
        _require_unique(
            (item.exposure_id for item in ordered),
            f"signal '{spec.signal_id}' exposure IDs",
        )
        results.append(_evaluate_signal(spec, ordered))

    crossed = tuple(
        item.signal_id
        for item in results
        if item.evidence_state is SurveillanceEvidenceState.THRESHOLD_CROSSED
    )
    disposition = (
        SurveillanceDisposition.ESCALATE_SCOPE_REVIEW
        if crossed
        else SurveillanceDisposition.CONTINUE_MONITORING
    )
    limitations = tuple(
        sorted(
            plan.limitations
            + (
                "No threshold crossing is not evidence that the system is "
                "safe or that the true event rate is below the accepted bound.",
                "A crossing is anytime-valid evidence against the declared "
                "conditional rate bound; it does not identify a cause, "
                "severity distribution, or corrective action.",
                "Validity requires complete, correctly ordered, scope-matched "
                "observations and pre-specified outcomes; reporting, exposure, "
                "or label bias can invalidate the evidence process.",
                "Family error control follows from allocated per-signal "
                "e-process thresholds and the union bound; it does not remove "
                "model-form, selection, measurement, or transfer uncertainty.",
                "Escalation triggers technical scope review and is not itself "
                "an automated deployment or shutdown decision.",
            )
        )
    )
    return SequentialSurveillanceReport(
        plan_content_digest=plan.content_digest(),
        assurance_report_digest=assurance_report.content_digest(),
        release_scope_digest=plan.release_scope_digest,
        signal_results=tuple(results),
        crossed_signal_ids=crossed,
        disposition=disposition,
        limitations=limitations,
    )


def _evaluate_signal(
    spec: SignalMonitoringSpec,
    observations: Sequence[SurveillanceObservation],
) -> SignalSurveillanceResult:
    """Calculate the mixture e-process trajectory for one signal.

    Args:
        spec: Null bound, alternatives, weights, and error allocation.
        observations: Contiguous observations in sequence order.

    Returns:
        Current, maximum, and first-crossing evidence summary.
    """
    threshold = math.log(1.0 / spec.allocated_error_rate)
    event_count = 0
    maximum = 0.0
    current = 0.0
    first_crossing = None
    for observation_count, observation in enumerate(observations, start=1):
        event_count += int(observation.occurred)
        current = _mixture_log_e_value(
            event_count,
            observation_count - event_count,
            spec,
        )
        maximum = max(maximum, current)
        if first_crossing is None and current >= threshold:
            first_crossing = observation_count
    count = len(observations)
    empirical_rate = event_count / count if count else None
    state = (
        SurveillanceEvidenceState.THRESHOLD_CROSSED
        if first_crossing is not None
        else SurveillanceEvidenceState.NO_THRESHOLD_CROSSING
    )
    return SignalSurveillanceResult(
        signal_id=spec.signal_id,
        observation_count=count,
        event_count=event_count,
        empirical_event_rate=empirical_rate,
        current_log_e_value=current,
        maximum_log_e_value=maximum,
        log_e_value_threshold=threshold,
        first_crossing_observation_count=first_crossing,
        evidence_state=state,
    )


def _mixture_log_e_value(
    event_count: int,
    non_event_count: int,
    spec: SignalMonitoringSpec,
) -> float:
    """Calculate a stable mixture log e-value for one sequence prefix.

    Args:
        event_count: Number of observed events.
        non_event_count: Number of observed non-events.
        spec: Null rate and weighted alternatives.

    Returns:
        Natural logarithm of the mixture likelihood ratio.
    """
    null_rate = spec.maximum_acceptable_event_rate
    components = tuple(
        math.log(item.weight)
        + event_count * math.log(item.event_probability / null_rate)
        + non_event_count * math.log((1.0 - item.event_probability) / (1.0 - null_rate))
        for item in spec.alternatives
    )
    maximum = max(components)
    return maximum + math.log(
        math.fsum(math.exp(item - maximum) for item in components)
    )


def _open_unit_interval(value: float, label: str) -> float:
    """Validate a finite value strictly between zero and one.

    Args:
        value: Candidate probability.
        label: Human-readable field label.

    Returns:
        Canonical float probability.
    """
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ValueError(f"{label} must be finite and between zero and one")
    return resolved


def _require_text(value: str, label: str) -> None:
    """Require non-empty text.

    Args:
        value: Candidate text.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_unique(values: Iterable[object], label: str) -> None:
    """Require unique values.

    Args:
        values: Candidate values.
        label: Human-readable collection label.
    """
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} must be unique")


def _require_unique_text(values: Sequence[str], label: str) -> None:
    """Require unique non-empty text values.

    Args:
        values: Candidate text values.
        label: Human-readable collection label.
    """
    if not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_text(value, label)
    _require_unique(values, label)
