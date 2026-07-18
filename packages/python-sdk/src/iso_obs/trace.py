"""Convert normalized run events into analysis-ready numeric trace steps.

Extraction is rule-driven and conservative. Callers identify the event type,
payload path, and optional payload filters for every numeric signal. The
extractor never guesses field semantics, silently coerces booleans, or combines
multiple observations without an explicit multiplicity policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from iso_obs_schemas import BaseEvent, EventType

from .divergence import TraceStep

type JsonScalar = str | int | float | bool | None

_MISSING = object()


class MultiplicityPolicy(StrEnum):
    """Resolution policy when multiple events provide one step signal."""

    ERROR = "error"
    FIRST = "first"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class EventSignalSpec:
    """Rule for extracting one numeric signal from normalized events."""

    signal: str
    event_type: EventType
    value_path: str
    where: Mapping[str, JsonScalar] = field(default_factory=dict)
    multiplicity: MultiplicityPolicy = MultiplicityPolicy.ERROR

    def __post_init__(self) -> None:
        """Validate and freeze the extraction rule."""
        if not self.signal.strip():
            raise ValueError("signal name must not be empty")
        if not self.value_path or any(
            not segment for segment in self.value_path.split(".")
        ):
            raise ValueError("value_path must be a valid dot-delimited path")
        if any(
            not path or any(not segment for segment in path.split("."))
            for path in self.where
        ):
            raise ValueError("filter paths must be valid dot-delimited paths")
        non_finite_filters = [
            path
            for path, value in self.where.items()
            if isinstance(value, float) and not math.isfinite(value)
        ]
        if non_finite_filters:
            paths = ", ".join(sorted(non_finite_filters))
            raise ValueError(f"filter values must be finite: {paths}")
        object.__setattr__(self, "event_type", EventType(self.event_type))
        object.__setattr__(
            self,
            "multiplicity",
            MultiplicityPolicy(self.multiplicity),
        )
        object.__setattr__(
            self,
            "where",
            MappingProxyType(dict(self.where)),
        )


@dataclass(frozen=True, slots=True)
class TraceExtraction:
    """Analysis-ready trace plus transparent extraction diagnostics."""

    run_id: str | None
    steps: tuple[TraceStep, ...]
    input_event_count: int
    extracted_value_count: int
    ignored_unstepped_event_count: int
    unmatched_event_count: int


def extract_trace_steps(
    events: Sequence[BaseEvent],
    *,
    signals: Sequence[EventSignalSpec],
) -> TraceExtraction:
    """Extract numeric step signals from one run's normalized events.

    Events are ordered deterministically by environment step, monotonic clock,
    and event ID before extraction. Every stepped event contributes its step
    coordinate even when no rule matches it, preserving missing-signal evidence
    for downstream comparison.

    Args:
        events: Normalized events from exactly one run.
        signals: Explicit signal extraction rules.

    Returns:
        Frozen trace steps and extraction diagnostics.

    Raises:
        ValueError: If run/event identities conflict, rules repeat a signal,
            a required payload path is absent, a value is non-numeric or
            non-finite, or multiplicity is ambiguous.
    """
    _validate_signal_specs(signals)
    run_id = _validate_event_identity(events)
    ordered = sorted(
        events,
        key=lambda event: (
            event.step is None,
            event.step if event.step is not None else 0,
            event.monotonic_ns,
            event.event_id,
        ),
    )
    step_values: dict[int, dict[str, float]] = {}
    step_evidence: dict[int, dict[str, str]] = {}
    unstepped_count = 0
    unmatched_count = 0

    for event in ordered:
        if event.step is None:
            unstepped_count += 1
            continue
        values = step_values.setdefault(event.step, {})
        evidence = step_evidence.setdefault(event.step, {})
        matching_specs = [spec for spec in signals if _matches(event, spec)]
        if not matching_specs:
            unmatched_count += 1
            continue
        for spec in matching_specs:
            value = _numeric_payload_value(event, spec)
            if spec.signal in values:
                if spec.multiplicity is MultiplicityPolicy.ERROR:
                    raise ValueError(
                        f"multiple events provide signal '{spec.signal}' "
                        f"at step {event.step}"
                    )
                if spec.multiplicity is MultiplicityPolicy.FIRST:
                    continue
            values[spec.signal] = value
            evidence[spec.signal] = event.event_id

    steps = tuple(
        TraceStep(
            step=step,
            values=step_values[step],
            evidence_event_ids=step_evidence[step],
        )
        for step in sorted(step_values)
    )
    return TraceExtraction(
        run_id=run_id,
        steps=steps,
        input_event_count=len(events),
        extracted_value_count=sum(len(step.values) for step in steps),
        ignored_unstepped_event_count=unstepped_count,
        unmatched_event_count=unmatched_count,
    )


def _validate_signal_specs(signals: Sequence[EventSignalSpec]) -> None:
    """Require one extraction rule per output signal.

    Args:
        signals: Extraction rules to validate.

    Raises:
        ValueError: If an output signal is declared more than once.
    """
    names = [spec.signal for spec in signals]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate extracted signals: {', '.join(duplicates)}")


def _validate_event_identity(events: Sequence[BaseEvent]) -> str | None:
    """Validate unique event IDs and a single run identity.

    Args:
        events: Events to validate.

    Returns:
        The shared run ID, or ``None`` for an empty event stream.

    Raises:
        ValueError: If event IDs repeat or events span multiple runs.
    """
    event_ids = [event.event_id for event in events]
    duplicate_ids = sorted(
        {event_id for event_id in event_ids if event_ids.count(event_id) > 1}
    )
    if duplicate_ids:
        raise ValueError(f"duplicate event IDs: {', '.join(duplicate_ids)}")
    run_ids = {event.run_id for event in events}
    if len(run_ids) > 1:
        raise ValueError("event stream contains multiple run IDs")
    return next(iter(run_ids), None)


def _matches(event: BaseEvent, spec: EventSignalSpec) -> bool:
    """Return whether an event satisfies an extraction rule.

    Args:
        event: Candidate event.
        spec: Signal extraction rule.

    Returns:
        ``True`` when event type and all payload filters match exactly.
    """
    if event.event_type is not spec.event_type:
        return False
    return all(
        _read_path(event.payload, path) == expected
        for path, expected in spec.where.items()
    )


def _numeric_payload_value(
    event: BaseEvent,
    spec: EventSignalSpec,
) -> float:
    """Read and validate one numeric event payload value.

    Args:
        event: Event selected by the extraction rule.
        spec: Rule identifying the payload path.

    Returns:
        Finite floating-point signal value.

    Raises:
        ValueError: If the path is missing, boolean/non-numeric, or non-finite.
    """
    value = _read_path(event.payload, spec.value_path)
    if value is _MISSING:
        raise ValueError(
            f"event {event.event_id} is missing payload path "
            f"'{spec.value_path}' for signal '{spec.signal}'"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"event {event.event_id} payload path '{spec.value_path}' "
            f"for signal '{spec.signal}' must be numeric"
        )
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"event {event.event_id} signal '{spec.signal}' is non-finite")
    return resolved


def _read_path(payload: Mapping[str, object], path: str) -> object:
    """Read a dot-delimited mapping path without executing user code.

    Args:
        payload: Event payload mapping.
        path: Dot-delimited mapping keys.

    Returns:
        The resolved value, or an internal missing-value sentinel.
    """
    value: object = payload
    for segment in path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            return _MISSING
        value = value[segment]
    return value
