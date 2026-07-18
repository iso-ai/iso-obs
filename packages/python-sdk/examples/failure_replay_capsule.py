"""Compile recorded failure evidence into a simulator replay capsule."""

from __future__ import annotations

import argparse
from pathlib import Path

from iso_obs.dataset_io import DatasetBundle, load_json_artifact
from iso_obs.dataset_rosbag2 import Rosbag2RecordingEvidence
from iso_obs.dataset_synchronization import DatasetSynchronizationReport
from iso_obs.replay_capsule import (
    FailureClaimQualification,
    FailureReplayRequest,
    ReplayFidelity,
    compile_failure_replay_capsule,
)
from iso_obs.simulation import SimulationAdapterManifest


def _output_path(value: str) -> Path:
    """Return a new output path without allowing an existing file."""
    path = Path(value)
    if path.exists():
        raise argparse.ArgumentTypeError(f"refusing to overwrite {path}")
    return path


def main() -> None:
    """Compile one content-addressed evidence chain and explain its claims."""
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--synchronization", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--recording-evidence", type=Path)
    parser.add_argument(
        "--output",
        type=_output_path,
        default=Path("failure-replay-capsule.json"),
    )
    arguments = parser.parse_args()

    request = load_json_artifact(arguments.request, FailureReplayRequest)
    bundle = load_json_artifact(arguments.bundle, DatasetBundle)
    synchronization = load_json_artifact(
        arguments.synchronization,
        DatasetSynchronizationReport,
    )
    adapter = load_json_artifact(arguments.adapter, SimulationAdapterManifest)
    recording = (
        load_json_artifact(
            arguments.recording_evidence,
            Rosbag2RecordingEvidence,
        )
        if arguments.recording_evidence is not None
        else None
    )

    capsule = compile_failure_replay_capsule(
        request,
        bundle=bundle,
        synchronization=synchronization,
        adapter=adapter,
        recording_evidence=recording,
    )
    with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(capsule.to_json())
        stream.write("\n")

    print(f"reconstruction: {capsule.fidelity.value}")
    print(f"failure claim: {capsule.failure_claim_qualification.value}")
    print(f"capsule digest: {capsule.content_digest()}")
    for issue in capsule.issues:
        impact = "reconstruction" if issue.affects_replay_fidelity else "failure claim"
        print(
            f"{issue.severity.value} [{impact}/{issue.kind.value}] "
            f"{issue.subject}: {issue.description}"
        )

    if (
        capsule.fidelity is ReplayFidelity.EXACT_REPLAY_READY
        and capsule.failure_claim_qualification
        is FailureClaimQualification.SUPPORTED_BY_SELECTED_EVIDENCE
    ):
        print("promotion: eligible for canonical regression-case construction")
    else:
        print("promotion: withheld until the findings above are resolved")


if __name__ == "__main__":
    main()
