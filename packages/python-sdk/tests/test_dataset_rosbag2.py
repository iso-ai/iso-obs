"""Tests for rosbag2 metadata and split-MCAP dataset ingestion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from iso_obs.dataset_ingestion import (
    DatasetIngestionDisposition,
    DatasetIngestionError,
    DatasetIngestionIssueKind,
    DatasetReaderUnavailableError,
)
from iso_obs.dataset_mcap import (
    MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    McapDatasetIngestionPlan,
    McapEpisodeWindow,
    McapPayloadMode,
    McapTimestampSource,
    McapTopicRule,
)
from iso_obs.dataset_reliability import DatasetEvidenceRole, DatasetScope
from iso_obs.dataset_rosbag2 import (
    ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    Rosbag2McapDatasetRowReader,
    Rosbag2McapIngestionPlan,
    Rosbag2RecordingDisposition,
    Rosbag2RecordingIssueKind,
    ingest_rosbag2_mcap_source,
    inspect_rosbag2_mcap_recording,
)
from iso_obs.evidence import sha256_digest


def plan() -> Rosbag2McapIngestionPlan:
    """Build a compact split-recording ingestion plan."""
    mcap_plan = McapDatasetIngestionPlan(
        schema_version=MCAP_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="split-robot-ingestion",
        plan_version="1",
        dataset_id="split-robot",
        dataset_version="2026-07",
        source_uri="file:///datasets/split-robot",
        license_id="Apache-2.0",
        scope=DatasetScope(
            domain_namespace="robotics",
            target_population_digest=sha256_digest("population"),
            collection_protocol_digest=sha256_digest("protocol"),
            modality_ids=("state",),
            evidence_roles=(DatasetEvidenceRole.REPRESENTATION_LEARNING,),
        ),
        topic_rules=(
            McapTopicRule(
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
            ),
        ),
        episode_windows=(
            McapEpisodeWindow(
                episode_id="episode-1",
                independence_unit_id="physical-run-1",
                source_split="unassigned",
            ),
        ),
        episode_window_timestamp_source=McapTimestampSource.LOG_TIME,
    )
    return Rosbag2McapIngestionPlan(
        schema_version=ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION,
        mcap_plan=mcap_plan,
    )


def write_mcap(path: Path, timestamps: tuple[int, ...]) -> Path:
    """Write JSON messages in the supplied physical order."""
    pytest.importorskip("mcap")
    writer_module = importlib.import_module("mcap.writer")
    with path.open("wb") as stream:
        writer = writer_module.Writer(
            stream,
            compression=writer_module.CompressionType.NONE,
            enable_crcs=True,
            enable_data_crcs=True,
        )
        writer.start(profile="ros2", library="iso-obs-test")
        schema_id = writer.register_schema(
            name="robot-state",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        channel_id = writer.register_channel(
            schema_id=schema_id,
            topic="/robot/state",
            message_encoding="json",
        )
        for sequence, timestamp in enumerate(timestamps):
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp,
                publish_time=timestamp - 10,
                sequence=sequence,
                data=f'{{"timestamp":{timestamp}}}'.encode(),
            )
        writer.finish()
    return path


def metadata_payload(
    files: tuple[tuple[str, int, int, int], ...],
    *,
    recording_start: int | None = None,
    recording_duration: int | None = None,
    recording_count: int | None = None,
    storage: str = "mcap",
    compression_mode: str = "",
    compression_format: str = "",
) -> dict[str, Any]:
    """Build modern rosbag2 metadata from file declarations."""
    start = (
        recording_start
        if recording_start is not None
        else min(item[1] for item in files)
    )
    end = max(item[1] + item[2] for item in files)
    return {
        "rosbag2_bagfile_information": {
            "version": 9,
            "storage_identifier": storage,
            "duration": {
                "nanoseconds": (
                    recording_duration
                    if recording_duration is not None
                    else end - start
                )
            },
            "starting_time": {"nanoseconds_since_epoch": start},
            "message_count": (
                recording_count
                if recording_count is not None
                else sum(item[3] for item in files)
            ),
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/robot/state",
                        "type": "robot-state",
                        "serialization_format": "json",
                        "offered_qos_profiles": [],
                        "type_description_hash": "",
                    },
                    "message_count": (
                        recording_count
                        if recording_count is not None
                        else sum(item[3] for item in files)
                    ),
                }
            ],
            "compression_format": compression_format,
            "compression_mode": compression_mode,
            "relative_file_paths": [item[0] for item in files],
            "files": [
                {
                    "path": path,
                    "starting_time": {"nanoseconds_since_epoch": file_start},
                    "duration": {"nanoseconds": duration},
                    "message_count": count,
                }
                for path, file_start, duration, count in files
            ],
            "custom_data": {},
            "ros_distro": "jazzy",
        }
    }


def write_metadata(root: Path, payload: Mapping[str, Any]) -> Path:
    """Write deterministic YAML metadata for a test recording."""
    yaml = importlib.import_module("yaml")
    path = root / "metadata.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def split_recording(root: Path) -> Path:
    """Create two chronological files with nonmonotonic physical messages."""
    root.mkdir()
    write_mcap(root / "robot_0.mcap", (200, 100))
    write_mcap(root / "robot_1.mcap", (400, 300))
    return write_metadata(
        root,
        metadata_payload(
            (
                ("robot_0.mcap", 100, 100, 2),
                ("robot_1.mcap", 300, 100, 2),
            )
        ),
    )


def test_split_recording_is_verified_and_preserves_both_orders(
    tmp_path: Path,
) -> None:
    """Ingest metadata-order files without sorting physical messages."""
    metadata_path = split_recording(tmp_path / "bag")

    root, _, evidence = inspect_rosbag2_mcap_recording(plan(), metadata_path)
    reader = Rosbag2McapDatasetRowReader(
        root,
        metadata_path,
        plan().mcap_plan,
        evidence,
    )
    rows = list(reader.rows())
    result = ingest_rosbag2_mcap_source(plan(), metadata_path.parent)

    assert evidence.disposition is Rosbag2RecordingDisposition.VERIFIED
    assert [row["window_timestamp_ns"] for row in rows] == [200, 100, 400, 300]
    assert [
        row["topics"]["/robot/state"]["mcap"]["source_file"]["relative_path"]
        for row in rows
    ] == ["robot_0.mcap", "robot_0.mcap", "robot_1.mcap", "robot_1.mcap"]
    assert result.report.row_count == 4
    assert (
        result.report.disposition
        is DatasetIngestionDisposition.INGESTED_WITHIN_DECLARED_MAPPING
    )
    assert (
        result.report.source_content_digest
        == result.recording_evidence.recording_content_digest
    )
    assert result.report.plan_content_digest == plan().content_digest()
    trace = result.bundle.timing_traces[0]
    assert trace.timestamps_seconds == pytest.approx((190e-9, 90e-9, 390e-9, 290e-9))


def test_manifest_count_disagreement_requires_review(tmp_path: Path) -> None:
    """Keep ingestible bytes but qualify claims when metadata counts disagree."""
    root = tmp_path / "bag"
    root.mkdir()
    write_mcap(root / "robot_0.mcap", (100, 200))
    write_metadata(
        root,
        metadata_payload((("robot_0.mcap", 100, 100, 3),), recording_count=3),
    )

    result = ingest_rosbag2_mcap_source(plan(), root)

    assert (
        result.recording_evidence.disposition
        is Rosbag2RecordingDisposition.REVIEW_REQUIRED
    )
    assert {item.kind for item in result.recording_evidence.issues} == {
        Rosbag2RecordingIssueKind.COUNT_MISMATCH
    }
    assert result.report.disposition is DatasetIngestionDisposition.REVIEW_REQUIRED
    assert {item.kind for item in result.report.issues} == {
        DatasetIngestionIssueKind.SOURCE_MANIFEST_MISMATCH
    }


def test_topic_type_disagreement_requires_review(tmp_path: Path) -> None:
    """Verify topic-level type claims against embedded MCAP schemas."""
    root = tmp_path / "bag"
    root.mkdir()
    write_mcap(root / "robot_0.mcap", (100,))
    payload = metadata_payload((("robot_0.mcap", 100, 0, 1),))
    topic_metadata = payload["rosbag2_bagfile_information"][
        "topics_with_message_count"
    ][0]["topic_metadata"]
    topic_metadata["type"] = "wrong_msgs/msg/State"
    write_metadata(root, payload)

    result = ingest_rosbag2_mcap_source(plan(), root)

    assert result.report.disposition is DatasetIngestionDisposition.REVIEW_REQUIRED
    assert any(
        item.kind is Rosbag2RecordingIssueKind.TOPIC_MISMATCH
        for item in result.recording_evidence.issues
    )
    assert result.recording_evidence.topics[0].observed_schema_names == ("robot-state",)


def test_file_time_overlap_requires_review_without_deduplication(
    tmp_path: Path,
) -> None:
    """Surface split-boundary overlap while retaining repeated messages."""
    root = tmp_path / "bag"
    root.mkdir()
    write_mcap(root / "robot_0.mcap", (100, 200))
    write_mcap(root / "robot_1.mcap", (200, 300))
    write_metadata(
        root,
        metadata_payload(
            (
                ("robot_0.mcap", 100, 100, 2),
                ("robot_1.mcap", 200, 100, 2),
            )
        ),
    )

    result = ingest_rosbag2_mcap_source(plan(), root)

    assert result.report.row_count == 4
    assert any(
        item.kind is Rosbag2RecordingIssueKind.FILE_TIME_OVERLAP
        for item in result.recording_evidence.issues
    )


def test_empty_split_file_has_unverifiable_time_claims(tmp_path: Path) -> None:
    """Do not treat empty-file time metadata as independently verified."""
    root = tmp_path / "bag"
    root.mkdir()
    write_mcap(root / "robot_0.mcap", ())
    write_metadata(
        root,
        metadata_payload((("robot_0.mcap", 100, 0, 0),)),
    )

    _, _, evidence = inspect_rosbag2_mcap_recording(plan(), root)

    assert evidence.disposition is Rosbag2RecordingDisposition.REVIEW_REQUIRED
    assert any(
        item.kind is Rosbag2RecordingIssueKind.TIME_MISMATCH for item in evidence.issues
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["rosbag2_bagfile_information"].update(
                storage_identifier="sqlite3"
            ),
            "storage",
        ),
        (
            lambda payload: payload["rosbag2_bagfile_information"].update(
                compression_mode="file",
                compression_format="zstd",
            ),
            "compressed",
        ),
        (
            lambda payload: payload["rosbag2_bagfile_information"].update(version=10),
            "version",
        ),
    ],
)
def test_unsupported_metadata_contracts_are_rejected(
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    """Reject formats that cannot be interpreted through the MCAP adapter."""
    root = tmp_path / message
    root.mkdir()
    write_mcap(root / "robot.mcap", (100,))
    payload = metadata_payload((("robot.mcap", 100, 0, 1),))
    mutator(payload)
    write_metadata(root, payload)

    with pytest.raises(DatasetIngestionError, match=message):
        inspect_rosbag2_mcap_recording(plan(), root)


def test_metadata_file_lists_must_match_exactly(tmp_path: Path) -> None:
    """Reject disagreement between legacy and detailed file declarations."""
    root = tmp_path / "bag"
    root.mkdir()
    write_mcap(root / "robot.mcap", (100,))
    payload = metadata_payload((("robot.mcap", 100, 0, 1),))
    payload["rosbag2_bagfile_information"]["relative_file_paths"] = ["other.mcap"]
    write_metadata(root, payload)

    with pytest.raises(DatasetIngestionError, match="exactly match"):
        inspect_rosbag2_mcap_recording(plan(), root)


def test_recording_path_cannot_escape_its_root(tmp_path: Path) -> None:
    """Reject traversal even when the target MCAP exists."""
    outside = write_mcap(tmp_path / "outside.mcap", (100,))
    root = tmp_path / "bag"
    root.mkdir()
    assert outside.is_file()
    write_metadata(root, metadata_payload((("../outside.mcap", 100, 0, 1),)))

    with pytest.raises(DatasetIngestionError, match="unsafe"):
        inspect_rosbag2_mcap_recording(plan(), root)


def test_metadata_symlink_cannot_escape_recording_root(tmp_path: Path) -> None:
    """Keep the authoritative manifest within the supplied recording."""
    outside = tmp_path / "metadata.yaml"
    outside.write_text("{}", encoding="utf-8")
    root = tmp_path / "bag"
    root.mkdir()
    (root / "metadata.yaml").symlink_to(outside)

    with pytest.raises(DatasetIngestionError, match="escapes"):
        inspect_rosbag2_mcap_recording(plan(), root)


def test_duplicate_yaml_keys_and_aliases_are_rejected(tmp_path: Path) -> None:
    """Reject YAML features that make manifest identity ambiguous."""
    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    (duplicate_root / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n" "  version: 9\n" "  version: 9\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetIngestionError, match="duplicate"):
        inspect_rosbag2_mcap_recording(plan(), duplicate_root)

    alias_root = tmp_path / "alias"
    alias_root.mkdir()
    (alias_root / "metadata.yaml").write_text(
        "rosbag2_bagfile_information: &info\n" "  version: 9\n" "copy: *info\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetIngestionError, match="aliases"):
        inspect_rosbag2_mcap_recording(plan(), alias_root)


def test_missing_yaml_dependency_has_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name the install extra when PyYAML is unavailable."""
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text("{}", encoding="utf-8")
    original = importlib.import_module

    def unavailable(name: str, package: str | None = None) -> Any:
        """Hide only the optional YAML dependency."""
        if name == "yaml":
            raise ImportError(name)
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable)

    with pytest.raises(DatasetReaderUnavailableError, match=r"iso-obs\[rosbag2\]"):
        inspect_rosbag2_mcap_recording(plan(), metadata_path)


def test_recording_digest_detects_source_mutation(tmp_path: Path) -> None:
    """Re-hashing a reader identifies post-inspection file mutation."""
    metadata_path = split_recording(tmp_path / "bag")
    root, _, evidence = inspect_rosbag2_mcap_recording(plan(), metadata_path)
    reader = Rosbag2McapDatasetRowReader(
        root,
        metadata_path,
        plan().mcap_plan,
        evidence,
    )
    reader.content_digest()

    with (root / "robot_1.mcap").open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(DatasetIngestionError, match="changed"):
        reader.content_digest()


def test_plan_digest_changes_with_nested_mcap_contract() -> None:
    """Bind recording evidence to every nested topic-mapping decision."""
    original = plan()
    changed = replace(
        original,
        mcap_plan=replace(
            original.mcap_plan,
            plan_version="2",
        ),
    )

    assert original.content_digest() != changed.content_digest()
