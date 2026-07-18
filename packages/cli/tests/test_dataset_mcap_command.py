"""End-to-end CLI test for an explicit MCAP ingestion plan."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

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
from iso_obs.evidence import sha256_digest
from iso_obs_cli.main import app


def writer_module() -> Any:
    """Load the optional official MCAP writer or skip the integration test."""
    pytest.importorskip("mcap")
    return importlib.import_module("mcap.writer")


def write_mcap(path: Path) -> Path:
    """Write one JSON state message to a valid CRC-protected MCAP."""
    module = writer_module()
    with path.open("wb") as stream:
        writer = module.Writer(
            stream,
            compression=module.CompressionType.NONE,
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
            log_time=1_000,
            publish_time=900,
            data=b'{"position":[0.1]}',
        )
        writer.finish()
    return path


def mcap_plan() -> McapDatasetIngestionPlan:
    """Build the MCAP plan selected by schema version in the CLI."""
    return McapDatasetIngestionPlan(
        schema_version=MCAP_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="cli-mcap-ingestion",
        plan_version="1",
        dataset_id="robot-mcap",
        dataset_version="1",
        source_uri="file:///datasets/robot.mcap",
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
            ),
        ),
        episode_window_timestamp_source=McapTimestampSource.LOG_TIME,
    )


def test_cli_routes_mcap_plan_and_writes_linked_artifacts(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """The existing ingest command selects the MCAP contract by schema."""
    source = write_mcap(tmp_path / "source.mcap")
    plan_path = tmp_path / "plan.json"
    bundle_path = tmp_path / "bundle.json"
    report_path = tmp_path / "report.json"
    plan_path.write_text(mcap_plan().to_json(), encoding="utf-8")

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
        ],
    )

    assert result.exit_code == 0
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert bundle["manifest"]["content_digest"] == report["source_content_digest"]
    assert report["source_format"] == "mcap"
    assert report["plan_content_digest"] == mcap_plan().content_digest()
