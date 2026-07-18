"""Tests for explicit MCAP and ROS 2 dataset ingestion."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from iso_obs.dataset_ingestion import (
    DatasetIngestionDisposition,
    DatasetIngestionError,
    DatasetReaderUnavailableError,
)
from iso_obs.dataset_mcap import (
    MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    McapDatasetIngestionPlan,
    McapDatasetRowReader,
    McapEpisodeWindow,
    McapPayloadMode,
    McapTimestampSource,
    McapTopicRule,
    ingest_mcap_source,
)
from iso_obs.dataset_reliability import DatasetEvidenceRole, DatasetScope
from iso_obs.evidence import sha256_digest


def plan(
    rules: tuple[McapTopicRule, ...],
    *,
    windows: tuple[McapEpisodeWindow, ...] | None = None,
) -> McapDatasetIngestionPlan:
    """Build a compact MCAP ingestion plan."""
    return McapDatasetIngestionPlan(
        schema_version=MCAP_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="robot-mcap-ingestion",
        plan_version="1",
        dataset_id="robot-mcap",
        dataset_version="2026-07",
        source_uri="file:///datasets/robot.mcap",
        license_id="Apache-2.0",
        scope=DatasetScope(
            domain_namespace="robotics",
            target_population_digest=sha256_digest("population"),
            collection_protocol_digest=sha256_digest("recording-protocol"),
            modality_ids=("camera", "state"),
            evidence_roles=(DatasetEvidenceRole.REPRESENTATION_LEARNING,),
        ),
        topic_rules=rules,
        episode_windows=windows
        or (
            McapEpisodeWindow(
                episode_id="episode-1",
                independence_unit_id="physical-run-1",
                source_split="unassigned",
            ),
        ),
        episode_window_timestamp_source=McapTimestampSource.LOG_TIME,
    )


def json_rule() -> McapTopicRule:
    """Build a JSON robot-state topic rule."""
    return McapTopicRule(
        topic="/robot/state",
        channel_id="robot-state",
        modality_id="state",
        clock_id="publisher-clock",
        payload_mode=McapPayloadMode.JSON,
        timestamp_source=McapTimestampSource.PUBLISH_TIME,
        timestamp_uncertainty_seconds=1e-6,
        expected_message_encoding="json",
        expected_schema_name="robot-state",
        expected_schema_encoding="jsonschema",
    )


def raw_rule() -> McapTopicRule:
    """Build a raw image topic rule."""
    return McapTopicRule(
        topic="/camera/front",
        channel_id="camera-front",
        modality_id="camera",
        clock_id="recorder-clock",
        payload_mode=McapPayloadMode.RAW_BYTES,
        timestamp_source=McapTimestampSource.LOG_TIME,
        timestamp_uncertainty_seconds=2e-6,
        expected_message_encoding="raw",
        required=False,
    )


def ros2_rule() -> McapTopicRule:
    """Build a decoded ROS 2 topic using its header timestamp."""
    return McapTopicRule(
        topic="/stamped-value",
        channel_id="stamped-value",
        modality_id="state",
        clock_id="ros-header-clock",
        payload_mode=McapPayloadMode.ROS2_DECODED,
        timestamp_source=McapTimestampSource.ROS_HEADER,
        timestamp_uncertainty_seconds=5e-7,
        expected_message_encoding="cdr",
        expected_schema_name="example_msgs/msg/StampedValue",
        expected_schema_encoding="ros2msg",
    )


def mcap_modules() -> tuple[Any, Any]:
    """Load optional MCAP writer modules or skip the integration test."""
    pytest.importorskip("mcap")
    writer = importlib.import_module("mcap.writer")
    return writer, writer.CompressionType


def write_raw_mcap(path: Path) -> Path:
    """Write JSON and raw topics in deliberately non-monotonic file order."""
    writer_module, compression_type = mcap_modules()
    with path.open("wb") as stream:
        writer = writer_module.Writer(
            stream,
            compression=compression_type.NONE,
            enable_crcs=True,
            enable_data_crcs=True,
        )
        writer.start(profile="ros2", library="iso-obs-test")
        schema_id = writer.register_schema(
            name="robot-state",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        state_channel = writer.register_channel(
            schema_id=schema_id,
            topic="/robot/state",
            message_encoding="json",
        )
        camera_channel = writer.register_channel(
            schema_id=0,
            topic="/camera/front",
            message_encoding="raw",
        )
        writer.add_message(
            channel_id=state_channel,
            log_time=200,
            publish_time=190,
            sequence=1,
            data=b'{"position":[0.2]}',
        )
        writer.add_message(
            channel_id=camera_channel,
            log_time=150,
            publish_time=140,
            sequence=1,
            data=b"\x89PNG",
        )
        writer.add_message(
            channel_id=state_channel,
            log_time=100,
            publish_time=90,
            sequence=2,
            data=b'{"position":[0.1]}',
        )
        writer.finish()
    return path


def write_ros2_mcap(path: Path, *, schema_encoding: str = "ros2msg") -> Path:
    """Write one CDR message with an embedded ROS 2 header stamp."""
    pytest.importorskip("mcap_ros2")
    if schema_encoding != "ros2msg":
        writer_module, compression_type = mcap_modules()
        with path.open("wb") as stream:
            writer = writer_module.Writer(
                stream,
                compression=compression_type.NONE,
            )
            writer.start(profile="ros2", library="iso-obs-test")
            schema_id = writer.register_schema(
                name="example_msgs/msg/StampedValue",
                encoding=schema_encoding,
                data=b"module example_msgs {}",
            )
            channel_id = writer.register_channel(
                schema_id=schema_id,
                topic="/stamped-value",
                message_encoding="cdr",
            )
            writer.add_message(
                channel_id=channel_id,
                log_time=1_000,
                publish_time=900,
                data=b"\x00\x01\x00\x00",
            )
            writer.finish()
        return path
    ros_writer_module = importlib.import_module("mcap_ros2.writer")
    schema_text = (
        "std_msgs/Header header\n"
        "float64 value\n"
        "===\n"
        "MSG: std_msgs/Header\n"
        "builtin_interfaces/Time stamp\n"
        "string frame_id\n"
    )
    with path.open("wb") as stream:
        writer = ros_writer_module.Writer(stream, enable_crcs=True)
        schema = writer.register_msgdef(
            "example_msgs/msg/StampedValue",
            schema_text,
        )
        writer.write_message(
            topic="/stamped-value",
            schema=schema,
            message={
                "header": {
                    "stamp": {"sec": 12, "nanosec": 34},
                    "frame_id": "base",
                },
                "value": 1.5,
            },
            log_time=1_000,
            publish_time=900,
            sequence=7,
        )
        writer.finish()
    return path


def test_raw_and_json_mcap_preserve_physical_message_order(
    tmp_path: Path,
) -> None:
    """Reader order is physical file order, not sorted MCAP log time."""
    source = write_raw_mcap(tmp_path / "robot.mcap")

    result = ingest_mcap_source(plan((json_rule(), raw_rule())), source)

    assert result.report.disposition is (
        DatasetIngestionDisposition.INGESTED_WITHIN_DECLARED_MAPPING
    )
    traces = {item.channel_id: item for item in result.bundle.timing_traces}
    assert traces["robot-state"].timestamps_seconds == (190e-9, 90e-9)
    assert traces["camera-front"].timestamps_seconds == (150e-9,)
    assert (
        result.report.plan_content_digest
        == plan((json_rule(), raw_rule())).content_digest()
    )
    assert "Physical MCAP message order was preserved." in result.report.limitations


def test_explicit_episode_windows_filter_without_inference(tmp_path: Path) -> None:
    """Only messages inside preregistered half-open windows become episodes."""
    source = write_raw_mcap(tmp_path / "windowed.mcap")
    windows = (
        McapEpisodeWindow("early", "run-early", 0, 150),
        McapEpisodeWindow("late", "run-late", 150, 201),
    )

    result = ingest_mcap_source(
        plan((json_rule(), raw_rule()), windows=windows),
        source,
    )

    assert tuple(item.episode_id for item in result.bundle.episodes) == (
        "early",
        "late",
    )
    traces = {
        (item.episode_id, item.channel_id): item for item in result.bundle.timing_traces
    }
    assert traces[("early", "robot-state")].timestamps_seconds == (90e-9,)
    assert traces[("late", "robot-state")].timestamps_seconds == (190e-9,)


def test_required_topic_is_enforced_within_each_episode(tmp_path: Path) -> None:
    """One observed camera cannot mask its absence from another episode."""
    source = write_raw_mcap(tmp_path / "episode-coverage.mcap")
    windows = (
        McapEpisodeWindow("early", "run-early", 0, 150),
        McapEpisodeWindow("late", "run-late", 150, 201),
    )
    required_camera = replace(raw_rule(), required=True)

    result = ingest_mcap_source(
        plan((json_rule(), required_camera), windows=windows),
        source,
    )

    assert result.report.disposition is DatasetIngestionDisposition.REVIEW_REQUIRED
    assert any("episode 'early'" in issue.description for issue in result.report.issues)


def test_schema_drift_is_rejected_before_payload_ingestion(tmp_path: Path) -> None:
    """A topic cannot silently change its declared schema identity."""
    source = write_raw_mcap(tmp_path / "schema-drift.mcap")
    changed = replace(
        json_rule(),
        expected_schema_name="other-schema",
    )

    with pytest.raises(DatasetIngestionError, match="schema name changed"):
        ingest_mcap_source(plan((changed, raw_rule())), source)


def test_crc_corruption_is_rejected(tmp_path: Path) -> None:
    """Changed message bytes cannot pass a CRC-validating ingestion plan."""
    source = write_raw_mcap(tmp_path / "corrupted.mcap")
    encoded = bytearray(source.read_bytes())
    marker = b'{"position":[0.2]}'
    offset = encoded.index(marker)
    encoded[offset + marker.index(b"2")] = ord("9")
    source.write_bytes(encoded)

    with pytest.raises(DatasetIngestionError, match="failed to read MCAP source"):
        ingest_mcap_source(plan((json_rule(), raw_rule())), source)


def test_ros2_header_timestamp_and_decoded_payload_round_trip(
    tmp_path: Path,
) -> None:
    """Embedded ros2msg evidence supports explicit ROS header timing."""
    source = write_ros2_mcap(tmp_path / "ros2.mcap")

    result = ingest_mcap_source(plan((ros2_rule(),)), source)

    trace = result.bundle.timing_traces[0]
    assert trace.timestamps_seconds == (12.000000034,)
    assert result.report.disposition is (
        DatasetIngestionDisposition.INGESTED_WITHIN_DECLARED_MAPPING
    )


def test_ros2_idl_schema_cannot_masquerade_as_decodable_ros2msg(
    tmp_path: Path,
) -> None:
    """Unsupported IDL decoding fails explicitly instead of using raw bytes."""
    source = write_ros2_mcap(
        tmp_path / "ros2-idl.mcap",
        schema_encoding="ros2idl",
    )
    idl_rule = McapTopicRule(
        topic="/stamped-value",
        channel_id="stamped-value",
        modality_id="state",
        clock_id="ros-header-clock",
        payload_mode=McapPayloadMode.ROS2_DECODED,
        timestamp_source=McapTimestampSource.ROS_HEADER,
        timestamp_uncertainty_seconds=5e-7,
        expected_message_encoding="cdr",
        expected_schema_name="example_msgs/msg/StampedValue",
        expected_schema_encoding="ros2idl",
    )

    with pytest.raises(DatasetIngestionError, match="embedded ros2msg schema"):
        ingest_mcap_source(plan((idl_rule,)), source)


def test_overlapping_episode_windows_are_rejected() -> None:
    """Ambiguous temporal membership cannot enter the ingestion plan."""
    with pytest.raises(ValueError, match="must not overlap"):
        plan(
            (json_rule(),),
            windows=(
                McapEpisodeWindow("one", "run-one", 0, 20),
                McapEpisodeWindow("two", "run-two", 10, 30),
            ),
        )


def test_missing_mcap_dependency_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base SDK installations receive the exact optional-extra instruction."""
    source = tmp_path / "source.mcap"
    source.write_bytes(b"not-read")
    ingestion_plan = plan((json_rule(),))

    def unavailable(name: str) -> None:
        """Simulate an environment without the optional reader."""
        if name == "mcap.reader":
            raise ImportError(name)

    monkeypatch.setattr(
        "iso_obs.dataset_mcap.importlib.import_module",
        unavailable,
    )

    with pytest.raises(DatasetReaderUnavailableError, match="iso-obs\\[mcap\\]"):
        tuple(McapDatasetRowReader(source, ingestion_plan).rows())
