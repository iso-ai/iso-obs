"""Tests for conservative normalized-event trace extraction."""

from __future__ import annotations

import math
from typing import Any

import pytest

from iso_obs.trace import (
    EventSignalSpec,
    MultiplicityPolicy,
    extract_trace_steps,
)
from iso_obs_schemas import BaseEvent, EventType, IdPrefix, generate_id

_WORKSPACE_ID = generate_id(IdPrefix.WORKSPACE)
_PROJECT_ID = generate_id(IdPrefix.PROJECT)
_SUITE_ID = generate_id(IdPrefix.EVALUATION_SUITE)
_RUN_ID = generate_id(IdPrefix.RUN)
_SYSTEM_ID = generate_id(IdPrefix.SYSTEM)
_SYSTEM_VERSION_ID = generate_id(IdPrefix.SYSTEM_VERSION)
_ENVIRONMENT_ID = generate_id(IdPrefix.ENVIRONMENT)
_ENVIRONMENT_VERSION_ID = generate_id(IdPrefix.ENVIRONMENT_VERSION)
_SCENARIO_ID = generate_id(IdPrefix.SCENARIO)


def event(
    *,
    step: int | None,
    event_type: EventType,
    payload: dict[str, Any],
    monotonic_ns: int,
    event_id: str | None = None,
    run_id: str = _RUN_ID,
) -> BaseEvent:
    """Build one contract-valid normalized event.

    Args:
        step: Environment step or ``None``.
        event_type: Normalized event type.
        payload: Event payload.
        monotonic_ns: Producer monotonic timestamp.
        event_id: Optional fixed event ID.
        run_id: Run identity.

    Returns:
        A normalized event suitable for extraction.
    """
    return BaseEvent(
        event_id=event_id or generate_id(IdPrefix.EVENT),
        workspace_id=_WORKSPACE_ID,
        project_id=_PROJECT_ID,
        suite_id=_SUITE_ID,
        run_id=run_id,
        system_id=_SYSTEM_ID,
        system_version=_SYSTEM_VERSION_ID,
        environment_id=_ENVIRONMENT_ID,
        environment_version=_ENVIRONMENT_VERSION_ID,
        scenario_id=_SCENARIO_ID,
        monotonic_ns=monotonic_ns,
        step=step,
        event_type=event_type,
        source="test",
        payload=payload,
    )


def metric_spec(name: str, *, signal: str | None = None) -> EventSignalSpec:
    """Build a metric-recorded extraction rule.

    Args:
        name: Metric payload name to match.
        signal: Optional output signal name.

    Returns:
        Metric extraction rule.
    """
    return EventSignalSpec(
        signal=signal or name,
        event_type=EventType.METRIC_RECORDED,
        value_path="value",
        where={"name": name},
    )


def test_extracts_filtered_nested_signals_in_step_order() -> None:
    events = [
        event(
            step=1,
            event_type=EventType.METRIC_RECORDED,
            payload={"name": "force", "value": 12.0},
            monotonic_ns=30,
        ),
        event(
            step=0,
            event_type=EventType.ACTION_EMITTED,
            payload={"action": {"acceleration": 0.4}},
            monotonic_ns=20,
        ),
        event(
            step=0,
            event_type=EventType.METRIC_RECORDED,
            payload={"name": "force", "value": 10.0},
            monotonic_ns=10,
        ),
    ]

    extraction = extract_trace_steps(
        events,
        signals=[
            metric_spec("force", signal="gripper_force_n"),
            EventSignalSpec(
                signal="acceleration",
                event_type=EventType.ACTION_EMITTED,
                value_path="action.acceleration",
            ),
        ],
    )

    assert extraction.run_id == _RUN_ID
    assert [step.step for step in extraction.steps] == [0, 1]
    assert dict(extraction.steps[0].values) == {
        "gripper_force_n": 10.0,
        "acceleration": 0.4,
    }
    assert dict(extraction.steps[1].values) == {"gripper_force_n": 12.0}
    assert extraction.extracted_value_count == 3
    assert extraction.unmatched_event_count == 0


def test_stepped_unmatched_event_preserves_empty_step() -> None:
    extraction = extract_trace_steps(
        [
            event(
                step=2,
                event_type=EventType.ENVIRONMENT_EVENT,
                payload={"kind": "contact"},
                monotonic_ns=1,
            )
        ],
        signals=[metric_spec("force")],
    )

    assert len(extraction.steps) == 1
    assert extraction.steps[0].step == 2
    assert dict(extraction.steps[0].values) == {}
    assert extraction.unmatched_event_count == 1


def test_unstepped_events_are_counted_but_not_extracted() -> None:
    extraction = extract_trace_steps(
        [
            event(
                step=None,
                event_type=EventType.METRIC_RECORDED,
                payload={"name": "force", "value": 10.0},
                monotonic_ns=1,
            )
        ],
        signals=[metric_spec("force")],
    )

    assert extraction.steps == ()
    assert extraction.ignored_unstepped_event_count == 1
    assert extraction.unmatched_event_count == 0


def test_multiple_matching_events_error_by_default() -> None:
    events = [
        event(
            step=0,
            event_type=EventType.METRIC_RECORDED,
            payload={"name": "force", "value": value},
            monotonic_ns=index,
        )
        for index, value in enumerate((10.0, 11.0), start=1)
    ]

    with pytest.raises(ValueError, match="multiple events provide signal"):
        extract_trace_steps(events, signals=[metric_spec("force")])


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (MultiplicityPolicy.FIRST, 10.0),
        (MultiplicityPolicy.LAST, 11.0),
    ],
)
def test_explicit_multiplicity_policy_is_deterministic(
    policy: MultiplicityPolicy,
    expected: float,
) -> None:
    events = [
        event(
            step=0,
            event_type=EventType.METRIC_RECORDED,
            payload={"name": "force", "value": 11.0},
            monotonic_ns=20,
        ),
        event(
            step=0,
            event_type=EventType.METRIC_RECORDED,
            payload={"name": "force", "value": 10.0},
            monotonic_ns=10,
        ),
    ]
    spec = EventSignalSpec(
        signal="force",
        event_type=EventType.METRIC_RECORDED,
        value_path="value",
        where={"name": "force"},
        multiplicity=policy,
    )

    extraction = extract_trace_steps(events, signals=[spec])

    assert extraction.steps[0].values["force"] == expected


def test_selected_evidence_id_matches_multiplicity_policy() -> None:
    first_id = generate_id(IdPrefix.EVENT)
    last_id = generate_id(IdPrefix.EVENT)
    events = [
        event(
            step=0,
            event_type=EventType.METRIC_RECORDED,
            payload={"name": "force", "value": 10.0},
            monotonic_ns=10,
            event_id=first_id,
        ),
        event(
            step=0,
            event_type=EventType.METRIC_RECORDED,
            payload={"name": "force", "value": 11.0},
            monotonic_ns=20,
            event_id=last_id,
        ),
    ]
    spec = EventSignalSpec(
        signal="force",
        event_type=EventType.METRIC_RECORDED,
        value_path="value",
        where={"name": "force"},
        multiplicity=MultiplicityPolicy.LAST,
    )

    extraction = extract_trace_steps(events, signals=[spec])

    assert extraction.steps[0].evidence_event_ids["force"] == last_id


def test_mixed_run_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="multiple run IDs"):
        extract_trace_steps(
            [
                event(
                    step=0,
                    event_type=EventType.METRIC_RECORDED,
                    payload={"name": "force", "value": 10.0},
                    monotonic_ns=1,
                ),
                event(
                    step=1,
                    event_type=EventType.METRIC_RECORDED,
                    payload={"name": "force", "value": 10.0},
                    monotonic_ns=2,
                    run_id=generate_id(IdPrefix.RUN),
                ),
            ],
            signals=[metric_spec("force")],
        )


def test_duplicate_event_ids_are_rejected() -> None:
    event_id = generate_id(IdPrefix.EVENT)
    with pytest.raises(ValueError, match="duplicate event IDs"):
        extract_trace_steps(
            [
                event(
                    step=0,
                    event_type=EventType.METRIC_RECORDED,
                    payload={"name": "force", "value": 10.0},
                    monotonic_ns=1,
                    event_id=event_id,
                ),
                event(
                    step=1,
                    event_type=EventType.METRIC_RECORDED,
                    payload={"name": "force", "value": 10.0},
                    monotonic_ns=2,
                    event_id=event_id,
                ),
            ],
            signals=[metric_spec("force")],
        )


def test_missing_payload_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing payload path"):
        extract_trace_steps(
            [
                event(
                    step=0,
                    event_type=EventType.METRIC_RECORDED,
                    payload={"name": "force"},
                    monotonic_ns=1,
                )
            ],
            signals=[metric_spec("force")],
        )


@pytest.mark.parametrize("value", [True, "10", math.nan, math.inf])
def test_invalid_numeric_value_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="must be numeric|non-finite"):
        extract_trace_steps(
            [
                event(
                    step=0,
                    event_type=EventType.METRIC_RECORDED,
                    payload={"name": "force", "value": value},
                    monotonic_ns=1,
                )
            ],
            signals=[metric_spec("force")],
        )


def test_duplicate_output_signal_rules_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate extracted signals"):
        extract_trace_steps(
            [],
            signals=[
                metric_spec("force", signal="control"),
                metric_spec("velocity", signal="control"),
            ],
        )


def test_signal_rule_defensively_freezes_filters() -> None:
    filters: dict[str, str] = {"name": "force"}
    spec = EventSignalSpec(
        signal="force",
        event_type=EventType.METRIC_RECORDED,
        value_path="value",
        where=filters,
    )

    filters["name"] = "velocity"

    assert spec.where["name"] == "force"
    with pytest.raises(TypeError):
        spec.where["name"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize("path", ["", ".value", "value.", "payload..value"])
def test_malformed_value_path_is_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="valid dot-delimited path"):
        EventSignalSpec(
            signal="force",
            event_type=EventType.METRIC_RECORDED,
            value_path=path,
        )


def test_non_finite_filter_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="filter values must be finite"):
        EventSignalSpec(
            signal="force",
            event_type=EventType.METRIC_RECORDED,
            value_path="value",
            where={"threshold": math.nan},
        )


def test_string_enum_configuration_is_canonicalized() -> None:
    spec = EventSignalSpec(
        signal="force",
        event_type="metric.recorded",  # type: ignore[arg-type]
        value_path="value",
        multiplicity="last",  # type: ignore[arg-type]
    )

    assert spec.event_type is EventType.METRIC_RECORDED
    assert spec.multiplicity is MultiplicityPolicy.LAST
