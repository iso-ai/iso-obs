"""Tests for RunContext: lifecycle, batching, failure paths, and spill."""

from __future__ import annotations

from pathlib import Path

import pytest
from sdk_doubles import FakeApi, make_client

from iso_obs.exceptions import ServerError
from iso_obs.run import BATCH_SIZE, SPILL_PATH, RunContext
from iso_obs_schemas import BaseEvent, EventType


def make_context(api: FakeApi) -> RunContext:
    """Build a RunContext wired to the given fake API."""
    return RunContext(
        client=make_client(api),
        project="robot-arm",
        system_version="policy-v17",
        environment="warehouse-v4",
        scenario="obstructed-pick",
        seed=42,
    )


def test_lifecycle_success_flushes_and_completes() -> None:
    api = FakeApi()
    with make_context(api) as run:
        assert run.seed == 42
        run.log_observation({"joint_pos": [0.1, 0.2]})
        run.log_action({"torque": [0.5]}, latency_ms=12.5)
        run.log_metric("reward", 1.5)
        run.log_state({"contacts": 0})
        run.log_artifact("replay.mp4")

    assert api.completed == [api.run_id]
    assert api.failed == []
    assert len(api.batches) == 1
    events = api.batches[0]
    assert [e["event_type"] for e in events] == [
        EventType.OBSERVATION_RECEIVED,
        EventType.ACTION_EMITTED,
        EventType.METRIC_RECORDED,
        EventType.STATE_RECORDED,
        EventType.ARTIFACT_CREATED,
    ]
    for event in events:
        # Round-trip through the schema proves the SDK emits contract-valid
        # events, including prefixed ids and the monotonic clock.
        parsed = BaseEvent.model_validate(event)
        assert parsed.event_id.startswith("evt_")
        assert parsed.run_id == api.run_id
        assert parsed.source == "sdk"
        assert parsed.monotonic_ns > 0
        assert parsed.step == 0
    assert events[0]["payload"]["observation"] == {"joint_pos": [0.1, 0.2]}
    assert events[1]["payload"]["latency_ms"] == 12.5
    assert events[2]["payload"] == {"name": "reward", "value": 1.5}
    assert events[4]["payload"]["artifact"]["uri"] == "replay.mp4"


def test_step_counter_advances_per_observation() -> None:
    api = FakeApi()
    with make_context(api) as run:
        run.log_metric("pre", 0.0)  # before any observation: no step yet
        for _ in range(3):
            run.log_observation({"x": 1})
            run.log_action({"u": 2})

    steps = [event["step"] for event in api.batches[0]]
    assert steps == [None, 0, 0, 1, 1, 2, 2]


def test_batching_at_500_events() -> None:
    api = FakeApi()
    with make_context(api) as run:
        for _ in range(BATCH_SIZE):
            run.log_observation({"x": 1})
        # The 500th event triggers an eager flush before the context exits.
        assert [len(batch) for batch in api.batches] == [BATCH_SIZE]
        for _ in range(250):
            run.log_observation({"x": 1})

    assert [len(batch) for batch in api.batches] == [BATCH_SIZE, 250]
    assert api.completed == [api.run_id]


def test_exception_fails_run_and_reraises() -> None:
    api = FakeApi()
    with pytest.raises(ValueError, match="kaboom"):
        with make_context(api) as run:
            run.log_observation({"x": 1})
            raise ValueError("kaboom")

    assert api.completed == []
    assert api.failed == [(api.run_id, "ValueError: kaboom")]
    # The buffered event was still flushed before the run was failed.
    assert [len(batch) for batch in api.batches] == [1]


def test_failed_flush_preserves_buffer_and_retries_later() -> None:
    # First flush: all three transport attempts return 500 -> buffer kept.
    api = FakeApi(fail_batch_requests=3)
    with make_context(api) as run:
        for _ in range(BATCH_SIZE):
            run.log_observation({"x": 1})
        assert api.batches == []  # first flush failed; nothing was dropped
        run.log_observation({"x": 1})  # refills past the threshold -> retry

    assert [len(batch) for batch in api.batches] == [BATCH_SIZE, 1]
    assert api.completed == [api.run_id]


def test_spill_file_written_when_close_flush_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    api = FakeApi(fail_batch_requests=10_000)
    with pytest.raises(ServerError):
        with make_context(api) as run:
            for _ in range(3):
                run.log_observation({"x": 1})

    assert api.completed == []
    spill = tmp_path / SPILL_PATH
    assert spill.is_file()
    lines = spill.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        BaseEvent.model_validate_json(line)


def test_spill_on_exception_path_keeps_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    api = FakeApi(fail_batch_requests=10_000)
    with pytest.raises(ValueError, match="kaboom"):
        with make_context(api) as run:
            run.log_observation({"x": 1})
            raise ValueError("kaboom")

    # The flush failure was spilled, the run failed, and the user's exception
    # propagated unmasked.
    assert api.failed == [(api.run_id, "ValueError: kaboom")]
    spill = tmp_path / SPILL_PATH
    assert spill.is_file()
    assert len(spill.read_text(encoding="utf-8").splitlines()) == 1


def test_logging_before_enter_raises() -> None:
    context = make_context(FakeApi())
    with pytest.raises(RuntimeError, match="entered"):
        context.log_metric("reward", 1.0)
