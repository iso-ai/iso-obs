"""Ingest a multimodal robot trajectory into Reliability Studio artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from iso_obs.dataset_ingestion import (
    DATASET_INGESTION_PLAN_SCHEMA_VERSION,
    ChannelIngestionRule,
    DatasetIngestionPlan,
    DatasetSourceFormat,
    SourceField,
    TimestampUnit,
    ingest_dataset_source,
)
from iso_obs.dataset_reliability import DatasetEvidenceRole, DatasetScope
from iso_obs.evidence import sha256_digest


def build_plan(source_uri: str) -> DatasetIngestionPlan:
    """Build an explicit mapping for the accompanying robot trajectory."""
    return DatasetIngestionPlan(
        schema_version=DATASET_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="warehouse-pick-ingestion",
        plan_version="1",
        source_format=DatasetSourceFormat.JSONL,
        dataset_id="warehouse-pick-trajectories",
        dataset_version="2026-07",
        source_uri=source_uri,
        license_id="LicenseRef-example",
        scope=DatasetScope(
            domain_namespace="robot-manipulation",
            target_population_digest=sha256_digest(
                "warehouse-picks-across-three-sites"
            ),
            collection_protocol_digest=sha256_digest("synchronized-robot-capture-v2"),
            modality_ids=("camera", "force-torque", "robot-state"),
            evidence_roles=(
                DatasetEvidenceRole.REPRESENTATION_LEARNING,
                DatasetEvidenceRole.FAILURE_LABEL_LEARNING,
            ),
            limitations=(
                "Example covers one robot embodiment and does not include labels.",
            ),
        ),
        episode_id_field=SourceField(("episode_id",)),
        independence_unit_id_field=SourceField(("physical_run_id",)),
        timestamp_field=SourceField(("controller_timestamp_ns",)),
        timestamp_unit=TimestampUnit.NANOSECONDS,
        channel_rules=(
            ChannelIngestionRule(
                channel_id="camera-front",
                modality_id="camera",
                clock_id="camera-clock",
                payload_field=SourceField(("observation", "camera_uri")),
                timestamp_field=SourceField(("camera_timestamp_ns",)),
                timestamp_uncertainty_seconds=0.0005,
            ),
            ChannelIngestionRule(
                channel_id="joint-state",
                modality_id="robot-state",
                clock_id="controller",
                payload_field=SourceField(("observation", "joint_state")),
                timestamp_uncertainty_seconds=0.0001,
            ),
            ChannelIngestionRule(
                channel_id="wrench",
                modality_id="force-torque",
                clock_id="force-clock",
                payload_field=SourceField(("observation", "wrench")),
                timestamp_field=SourceField(("force_timestamp_ns",)),
                timestamp_uncertainty_seconds=0.0002,
            ),
        ),
        source_split_field=SourceField(("source_split",)),
        limitations=(
            "Clock alignment requires a separate synchronization plan.",
            "Failure labels require independently attributable evidence.",
        ),
    )


def main() -> None:
    """Parse paths, ingest the example, and create canonical artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--bundle", type=Path, default=Path("dataset-bundle.json"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset-ingestion-report.json"),
    )
    arguments = parser.parse_args()
    if arguments.bundle.resolve() == arguments.report.resolve():
        raise ValueError("bundle and report paths must differ")
    for path in (arguments.bundle, arguments.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    result = ingest_dataset_source(
        build_plan(arguments.source.resolve().as_uri()),
        arguments.source,
    )
    with arguments.bundle.open("x", encoding="utf-8") as stream:
        stream.write(result.bundle.to_json())
        stream.write("\n")
    with arguments.report.open("x", encoding="utf-8") as stream:
        stream.write(result.report.to_json())
        stream.write("\n")
    print(result.report.disposition.value)


if __name__ == "__main__":
    main()
