"""Ingest a ROS 2 MCAP bag with explicit topics, clocks, and episode scope."""

from __future__ import annotations

import argparse
from pathlib import Path

from iso_obs.dataset_mcap import (
    MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    McapDatasetIngestionPlan,
    McapEpisodeWindow,
    McapPayloadMode,
    McapTimestampSource,
    McapTopicRule,
    ingest_mcap_source,
)
from iso_obs.dataset_reliability import DatasetEvidenceRole, DatasetScope
from iso_obs.evidence import sha256_digest


def build_plan(
    source_uri: str,
    episode_id: str,
    independence_unit_id: str,
) -> McapDatasetIngestionPlan:
    """Build a one-bag, one-episode ROS 2 ingestion plan."""
    return McapDatasetIngestionPlan(
        schema_version=MCAP_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="warehouse-ros2-mcap-ingestion",
        plan_version="1",
        dataset_id="warehouse-robot-bags",
        dataset_version="2026-07",
        source_uri=source_uri,
        license_id="LicenseRef-example",
        scope=DatasetScope(
            domain_namespace="robot-manipulation",
            target_population_digest=sha256_digest(
                "warehouse-picks-across-three-sites"
            ),
            collection_protocol_digest=sha256_digest("ros2-mcap-capture-v1"),
            modality_ids=("camera", "force-torque", "robot-state"),
            evidence_roles=(
                DatasetEvidenceRole.REPRESENTATION_LEARNING,
                DatasetEvidenceRole.FAILURE_LABEL_LEARNING,
            ),
            limitations=(
                "Example assumes one physical run per MCAP file.",
                "Camera timing uses recorder time; other topics use ROS headers.",
            ),
        ),
        topic_rules=(
            McapTopicRule(
                topic="/camera/front/image_raw",
                channel_id="camera-front",
                modality_id="camera",
                clock_id="recorder-clock",
                payload_mode=McapPayloadMode.RAW_BYTES,
                timestamp_source=McapTimestampSource.LOG_TIME,
                timestamp_uncertainty_seconds=0.0005,
                expected_message_encoding="cdr",
                expected_schema_name="sensor_msgs/msg/Image",
                expected_schema_encoding="ros2msg",
            ),
            McapTopicRule(
                topic="/joint_states",
                channel_id="joint-state",
                modality_id="robot-state",
                clock_id="ros-header-clock",
                payload_mode=McapPayloadMode.ROS2_DECODED,
                timestamp_source=McapTimestampSource.ROS_HEADER,
                timestamp_uncertainty_seconds=0.0001,
                expected_message_encoding="cdr",
                expected_schema_name="sensor_msgs/msg/JointState",
                expected_schema_encoding="ros2msg",
            ),
            McapTopicRule(
                topic="/wrench",
                channel_id="wrench",
                modality_id="force-torque",
                clock_id="ros-header-clock",
                payload_mode=McapPayloadMode.ROS2_DECODED,
                timestamp_source=McapTimestampSource.ROS_HEADER,
                timestamp_uncertainty_seconds=0.0002,
                expected_message_encoding="cdr",
                expected_schema_name="geometry_msgs/msg/WrenchStamped",
                expected_schema_encoding="ros2msg",
            ),
        ),
        episode_windows=(
            McapEpisodeWindow(
                episode_id=episode_id,
                independence_unit_id=independence_unit_id,
                source_split="unassigned",
            ),
        ),
        episode_window_timestamp_source=McapTimestampSource.LOG_TIME,
        limitations=(
            "Clock alignment requires a separate synchronization plan.",
            "Failure labels require independently attributable evidence.",
        ),
    )


def main() -> None:
    """Parse paths and create linked dataset and ingestion artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--independence-unit-id", required=True)
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
    result = ingest_mcap_source(
        build_plan(
            arguments.source.resolve().as_uri(),
            arguments.episode_id,
            arguments.independence_unit_id,
        ),
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
