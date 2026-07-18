"""Ingest a split rosbag2 MCAP recording with manifest verification."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcap_ros2_ingestion import build_plan

from iso_obs.dataset_rosbag2 import (
    ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    Rosbag2McapIngestionPlan,
    ingest_rosbag2_mcap_source,
)


def main() -> None:
    """Create dataset artifacts plus independently checked recording evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "recording",
        type=Path,
        help="Rosbag2 directory containing metadata.yaml and MCAP files.",
    )
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--independence-unit-id", required=True)
    parser.add_argument("--bundle", type=Path, default=Path("dataset-bundle.json"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("dataset-ingestion-report.json"),
    )
    parser.add_argument(
        "--recording-evidence",
        type=Path,
        default=Path("rosbag2-recording-evidence.json"),
    )
    arguments = parser.parse_args()

    outputs = (
        arguments.bundle,
        arguments.report,
        arguments.recording_evidence,
    )
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("output paths must differ")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    recording = arguments.recording.resolve()
    mcap_plan = build_plan(
        recording.as_uri(),
        arguments.episode_id,
        arguments.independence_unit_id,
    )
    plan = Rosbag2McapIngestionPlan(
        schema_version=ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION,
        mcap_plan=mcap_plan,
        limitations=(
            "Recorder host clock synchronization was not independently measured.",
        ),
    )
    result = ingest_rosbag2_mcap_source(plan, recording)

    artifacts = (
        (arguments.bundle, result.bundle.to_json()),
        (arguments.report, result.report.to_json()),
        (
            arguments.recording_evidence,
            result.recording_evidence.to_json(),
        ),
    )
    created: list[Path] = []
    try:
        for path, payload in artifacts:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                created.append(path)
                stream.write(payload)
                stream.write("\n")
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise

    print(f"dataset: {result.report.disposition.value}")
    print(f"recording: {result.recording_evidence.disposition.value}")
    print(
        "verified files: "
        f"{len(result.recording_evidence.files)}, "
        f"topics: {len(result.recording_evidence.topics)}"
    )
    for issue in result.recording_evidence.issues:
        location = issue.relative_file_path or "recording"
        print(f"review [{issue.kind.value}] {location}: {issue.description}")


if __name__ == "__main__":
    main()
