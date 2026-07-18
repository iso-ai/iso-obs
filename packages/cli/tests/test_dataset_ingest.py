"""End-to-end tests for source dataset ingestion commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner, Result

from iso_obs.dataset_ingestion import (
    DATASET_INGESTION_PLAN_SCHEMA_VERSION,
    ChannelIngestionRule,
    DatasetIngestionPlan,
    DatasetSourceFormat,
    SourceField,
    TimestampUnit,
)
from iso_obs.dataset_reliability import DatasetEvidenceRole, DatasetScope
from iso_obs.evidence import sha256_digest
from iso_obs_cli.main import app


def ingestion_plan() -> DatasetIngestionPlan:
    """Build a compact trajectory ingestion plan."""
    return DatasetIngestionPlan(
        schema_version=DATASET_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="trajectory-ingestion",
        plan_version="1",
        source_format=DatasetSourceFormat.JSONL,
        dataset_id="robot-trajectories",
        dataset_version="1",
        source_uri="file:///datasets/trajectories.jsonl",
        license_id="Apache-2.0",
        scope=DatasetScope(
            domain_namespace="robotics",
            target_population_digest=sha256_digest("population"),
            collection_protocol_digest=sha256_digest("protocol"),
            modality_ids=("camera", "state"),
            evidence_roles=(DatasetEvidenceRole.REPRESENTATION_LEARNING,),
        ),
        episode_id_field=SourceField(("episode_id",)),
        independence_unit_id_field=SourceField(("run_id",)),
        timestamp_field=SourceField(("timestamp_ns",)),
        timestamp_unit=TimestampUnit.NANOSECONDS,
        channel_rules=(
            ChannelIngestionRule(
                channel_id="camera-front",
                modality_id="camera",
                clock_id="sensor-clock",
                payload_field=SourceField(("camera_uri",)),
            ),
            ChannelIngestionRule(
                channel_id="state",
                modality_id="state",
                clock_id="controller",
                payload_field=SourceField(("state",)),
            ),
        ),
    )


def write_inputs(tmp_path: Path, source_text: str) -> tuple[Path, Path]:
    """Write a source file and matching canonical plan."""
    source_path = tmp_path / "source.jsonl"
    plan_path = tmp_path / "plan.json"
    source_path.write_text(source_text, encoding="utf-8")
    plan_path.write_text(ingestion_plan().to_json(), encoding="utf-8")
    return source_path, plan_path


def invoke_ingest(
    runner: CliRunner,
    source_path: Path,
    plan_path: Path,
    bundle_path: Path,
    report_path: Path,
) -> Result:
    """Invoke the complete local ingestion command."""
    return runner.invoke(
        app,
        [
            "dataset",
            "ingest",
            str(source_path),
            "--plan",
            str(plan_path),
            "--output",
            str(bundle_path),
            "--report",
            str(report_path),
        ],
    )


def test_ingest_writes_linked_bundle_and_report(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Successful ingestion creates both content-addressed artifacts."""
    source, plan = write_inputs(
        tmp_path,
        '{"episode_id":"ep-1","run_id":"run-1","timestamp_ns":1000000000,'
        '"camera_uri":"frames/1.png","state":[0.1,0.2]}\n',
    )
    bundle_path = tmp_path / "bundle.json"
    report_path = tmp_path / "report.json"

    result = invoke_ingest(
        runner,
        source,
        plan,
        bundle_path,
        report_path,
    )

    assert result.exit_code == 0
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "iso-obs.dataset-bundle.v1"
    assert report["disposition"] == "ingested_within_declared_mapping"
    assert report["bundle_content_digest"].startswith("sha256:")


def test_ingest_surfaces_required_channel_review(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """A missing required payload creates artifacts and exits for review."""
    source, plan = write_inputs(
        tmp_path,
        '{"episode_id":"ep-1","run_id":"run-1","timestamp_ns":0,'
        '"camera_uri":null,"state":[0.0]}\n',
    )
    bundle_path = tmp_path / "bundle.json"
    report_path = tmp_path / "report.json"

    result = invoke_ingest(
        runner,
        source,
        plan,
        bundle_path,
        report_path,
    )

    assert result.exit_code == 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["disposition"] == "review_required"
    assert report["issues"][0]["channel_id"] == "camera-front"


def test_ingest_preflight_preserves_existing_output(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """An existing target prevents either output artifact from changing."""
    source, plan = write_inputs(
        tmp_path,
        '{"episode_id":"ep-1","run_id":"run-1","timestamp_ns":0,'
        '"camera_uri":"frame.png","state":[0.0]}\n',
    )
    bundle_path = tmp_path / "bundle.json"
    report_path = tmp_path / "report.json"
    bundle_path.write_text("owner-data", encoding="utf-8")

    result = invoke_ingest(
        runner,
        source,
        plan,
        bundle_path,
        report_path,
    )

    assert result.exit_code == 2
    assert bundle_path.read_text(encoding="utf-8") == "owner-data"
    assert not report_path.exists()
