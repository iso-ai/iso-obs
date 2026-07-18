"""End-to-end CLI tests for split rosbag2 MCAP ingestion."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

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
    Rosbag2McapIngestionPlan,
)
from iso_obs.evidence import sha256_digest
from iso_obs_cli.main import app


def rosbag2_plan() -> Rosbag2McapIngestionPlan:
    """Build the rosbag2 wrapper selected by schema version in the CLI."""
    mcap_plan = McapDatasetIngestionPlan(
        schema_version=MCAP_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="cli-rosbag2-ingestion",
        plan_version="1",
        dataset_id="robot-rosbag2",
        dataset_version="1",
        source_uri="file:///datasets/robot-bag",
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
                clock_id="recorder-clock",
                payload_mode=McapPayloadMode.JSON,
                timestamp_source=McapTimestampSource.LOG_TIME,
                timestamp_uncertainty_seconds=1e-6,
                expected_message_encoding="json",
                expected_schema_name="robot-state",
                expected_schema_encoding="jsonschema",
            ),
        ),
        episode_windows=(
            McapEpisodeWindow(
                episode_id="episode-1",
                independence_unit_id="run-1",
            ),
        ),
        episode_window_timestamp_source=McapTimestampSource.LOG_TIME,
    )
    return Rosbag2McapIngestionPlan(
        schema_version=ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION,
        mcap_plan=mcap_plan,
    )


def write_recording(root: Path) -> Path:
    """Write one valid rosbag2 recording split across two MCAP files."""
    pytest.importorskip("mcap")
    writer_module = importlib.import_module("mcap.writer")
    root.mkdir()
    files = (
        ("robot_0.mcap", 100),
        ("robot_1.mcap", 200),
    )
    for name, timestamp in files:
        with (root / name).open("wb") as stream:
            writer = writer_module.Writer(
                stream,
                compression=writer_module.CompressionType.NONE,
                enable_crcs=True,
                enable_data_crcs=True,
            )
            writer.start(profile="ros2", library="iso-obs-cli-test")
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
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp,
                publish_time=timestamp,
                data=f'{{"timestamp":{timestamp}}}'.encode(),
            )
            writer.finish()
    yaml = importlib.import_module("yaml")
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 9,
            "storage_identifier": "mcap",
            "duration": {"nanoseconds": 100},
            "starting_time": {"nanoseconds_since_epoch": 100},
            "message_count": 2,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/robot/state",
                        "type": "robot-state",
                        "serialization_format": "json",
                        "offered_qos_profiles": [],
                        "type_description_hash": "",
                    },
                    "message_count": 2,
                }
            ],
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": [item[0] for item in files],
            "files": [
                {
                    "path": name,
                    "starting_time": {"nanoseconds_since_epoch": timestamp},
                    "duration": {"nanoseconds": 0},
                    "message_count": 1,
                }
                for name, timestamp in files
            ],
            "custom_data": {},
            "ros_distro": "jazzy",
        }
    }
    (root / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    return root


def test_cli_writes_dataset_and_recording_evidence(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Route a rosbag2 plan and create all three linked artifacts."""
    source = write_recording(tmp_path / "bag")
    plan_path = tmp_path / "plan.json"
    bundle_path = tmp_path / "bundle.json"
    report_path = tmp_path / "report.json"
    recording_path = tmp_path / "recording.json"
    plan_path.write_text(rosbag2_plan().to_json(), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            str(source),
            "--plan",
            str(plan_path),
            "--output",
            str(bundle_path),
            "--report",
            str(report_path),
            "--recording-report",
            str(recording_path),
        ],
    )

    assert result.exit_code == 0
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recording = json.loads(recording_path.read_text(encoding="utf-8"))
    assert recording["disposition"] == "verified"
    assert len(recording["files"]) == 2
    assert recording["recording_content_digest"] == report["source_content_digest"]
    assert bundle["manifest"]["content_digest"] == report["source_content_digest"]
    assert report["plan_content_digest"] == rosbag2_plan().content_digest()


def test_cli_requires_recording_evidence_output(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Do not permit rosbag2 ingestion to discard file-level provenance."""
    source = write_recording(tmp_path / "bag")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(rosbag2_plan().to_json(), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            str(source),
            "--plan",
            str(plan_path),
            "--output",
            str(tmp_path / "bundle.json"),
            "--report",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == 2
    assert "--recording-report is required" in result.output


def test_cli_rejects_colliding_output_paths_without_partial_writes(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Validate all output identities before creating any artifact."""
    source = write_recording(tmp_path / "bag")
    plan_path = tmp_path / "plan.json"
    shared_path = tmp_path / "shared.json"
    report_path = tmp_path / "report.json"
    plan_path.write_text(rosbag2_plan().to_json(), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            str(source),
            "--plan",
            str(plan_path),
            "--output",
            str(shared_path),
            "--report",
            str(report_path),
            "--recording-report",
            str(shared_path),
        ],
    )

    assert result.exit_code == 2
    assert "output paths must differ" in result.output
    assert not shared_path.exists()
    assert not report_path.exists()
