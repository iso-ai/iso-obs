"""End-to-end CLI tests for failure replay capsule compilation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from iso_obs.dataset_io import DATASET_BUNDLE_SCHEMA_VERSION, DatasetBundle
from iso_obs.dataset_reliability import (
    DatasetEvidenceRole,
    DatasetLabelAssertion,
    DatasetManifest,
    DatasetScope,
    EpisodeManifest,
    LabelStatus,
    ModeFamily,
)
from iso_obs.dataset_synchronization import (
    DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
    ChannelSynchronizationSummary,
    ChannelTimingTrace,
    DatasetSynchronizationReport,
    InterpolationPolicy,
    SynchronizationDisposition,
)
from iso_obs.evidence import NamedDigest, sha256_digest
from iso_obs.replay_capsule import (
    FAILURE_REPLAY_REQUEST_SCHEMA_VERSION,
    FailureReplayRequest,
    InitialStateCompleteness,
    InitialStateEvidence,
    ReplayChannelBinding,
    ReplayChannelRole,
    ReplayInterval,
    StochasticReplayControl,
)
from iso_obs.simulation import (
    BackendType,
    EvidenceUse,
    SimulationAdapterManifest,
    SimulationCapability,
)
from iso_obs_cli.main import app


def replay_artifacts(
    *,
    label_status: LabelStatus = LabelStatus.CORROBORATED,
) -> tuple[
    DatasetBundle,
    DatasetSynchronizationReport,
    SimulationAdapterManifest,
    FailureReplayRequest,
]:
    """Build a minimal exact replay evidence chain."""
    scope = DatasetScope(
        domain_namespace="robotics",
        target_population_digest=sha256_digest("population"),
        collection_protocol_digest=sha256_digest("protocol"),
        modality_ids=("state",),
        evidence_roles=(DatasetEvidenceRole.FAILURE_LABEL_LEARNING,),
    )
    manifest = DatasetManifest(
        dataset_id="failures",
        dataset_version="1",
        content_digest=sha256_digest("source"),
        source_uri="s3://evidence/run",
        license_id="LicenseRef-internal",
        scope=scope,
    )
    episode = EpisodeManifest(
        dataset_manifest_digest=manifest.manifest_digest(),
        episode_id="run-1",
        independence_unit_id="physical-run-1",
        content_digest=sha256_digest("episode"),
        start_seconds=0.0,
        end_seconds=10.0,
        modality_digests=(NamedDigest("state", sha256_digest("state")),),
    )
    source = DatasetBundle(
        schema_version=DATASET_BUNDLE_SCHEMA_VERSION,
        manifest=manifest,
        episodes=(episode,),
        label_assertions=(
            DatasetLabelAssertion(
                assertion_id="failure-1",
                episode_id="run-1",
                label_namespace="reliability/failure_state/v1",
                mode_family=ModeFamily.FAILURE_STATE,
                start_seconds=5.0,
                end_seconds=6.0,
                candidate_values=("unsafe-stop",),
                status=label_status,
                evidence=(),
                method_digest=(
                    sha256_digest("detector")
                    if label_status is LabelStatus.SUGGESTED
                    else None
                ),
            ),
        ),
        timing_traces=(
            ChannelTimingTrace(
                episode_id="run-1",
                channel_id="state",
                modality_id="state",
                clock_id="sim-clock",
                content_digest=sha256_digest("state-trace"),
                timestamps_seconds=(4.0, 5.0, 6.0, 7.0),
                timestamp_uncertainty_seconds=0.0,
            ),
        ),
    )
    synchronization = DatasetSynchronizationReport(
        schema_version=DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
        plan_content_digest=sha256_digest("sync-plan"),
        dataset_manifest_digest=manifest.manifest_digest(),
        episode_id="run-1",
        disposition=SynchronizationDisposition.ALIGNED_WITHIN_SCOPE,
        reference_clock_id="sim-clock",
        channel_summaries=(
            ChannelSynchronizationSummary(
                channel_id="state",
                modality_id="state",
                clock_id="sim-clock",
                sample_count=4,
                coverage_start_seconds=4.0,
                coverage_end_seconds=7.0,
                maximum_observed_gap_seconds=1.0,
                alignment_uncertainty_upper_seconds=0.0,
                interpolation_policy=InterpolationPolicy.NONE,
            ),
        ),
        issues=(),
        scope=scope,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=(),
    )
    target = SimulationAdapterManifest(
        adapter_name="replay",
        adapter_version="1",
        backend_type=BackendType.PHYSICS_SIMULATOR,
        simulator_name="twin",
        simulator_version="2",
        simulation_evidence_manifest_digest=sha256_digest("simulation"),
        capabilities=(
            SimulationCapability.ARTIFACT_LOADING,
            SimulationCapability.DETERMINISTIC_TIME_CONTROL,
            SimulationCapability.SNAPSHOT_RESTORE,
        ),
        supported_parameters=(),
        supported_artifacts=("state-playback",),
    )
    request = FailureReplayRequest(
        schema_version=FAILURE_REPLAY_REQUEST_SCHEMA_VERSION,
        capsule_id="run-1-replay",
        capsule_version="1",
        dataset_bundle_digest=source.content_digest(),
        episode_id="run-1",
        failure_assertion_ids=("failure-1",),
        interval=ReplayInterval(5.0, 6.0, 1.0, 1.0),
        reference_clock_id="sim-clock",
        scenario_id="unsafe-stop",
        scenario_version="1",
        environment_id="twin",
        environment_version="2",
        system_versions=("policy-1",),
        simulation_manifest_digest=target.simulation_evidence_manifest_digest,
        adapter_manifest_digest=target.content_digest(),
        recording_evidence_digest=None,
        channel_bindings=(
            ReplayChannelBinding(
                channel_id="state",
                role=ReplayChannelRole.SYSTEM_OBSERVATION,
                adapter_artifact_name="state-playback",
                transformation_digest=sha256_digest("transform"),
            ),
        ),
        initial_state=InitialStateEvidence(
            artifact_digest=sha256_digest("initial"),
            capture_time_seconds=4.0,
            reference_clock_id="sim-clock",
            completeness=InitialStateCompleteness.COMPLETE,
            method="snapshot",
        ),
        maximum_initial_state_age_seconds=0.0,
        stochastic_control=StochasticReplayControl(stochastic=False),
        perturbation_realization_digest=None,
        sdk_version="0.1.0",
        command="run-replay",
        configuration_digests=(),
        required_adapter_capabilities=(
            SimulationCapability.ARTIFACT_LOADING,
            SimulationCapability.DETERMINISTIC_TIME_CONTROL,
            SimulationCapability.SNAPSHOT_RESTORE,
        ),
    )
    return source, synchronization, target, request


def write_inputs(
    tmp_path: Path,
    *,
    request_override: FailureReplayRequest | None = None,
    label_status: LabelStatus = LabelStatus.CORROBORATED,
    context_outside_episode: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """Write strict JSON inputs for the CLI command."""
    source, synchronization, target, request = replay_artifacts(
        label_status=label_status
    )
    if context_outside_episode:
        request = replace(
            request,
            interval=ReplayInterval(5.0, 6.0, 6.0, 1.0),
        )
    paths = (
        tmp_path / "request.json",
        tmp_path / "bundle.json",
        tmp_path / "synchronization.json",
        tmp_path / "adapter.json",
    )
    payloads = (
        (request_override or request).to_json(),
        source.to_json(),
        synchronization.to_json(),
        target.to_json(),
    )
    for path, payload in zip(paths, payloads, strict=True):
        path.write_text(payload, encoding="utf-8")
    return paths


def invoke(
    runner: CliRunner,
    paths: tuple[Path, Path, Path, Path],
    output: Path,
) -> object:
    """Invoke the replay compiler with one complete evidence chain."""
    request_path, bundle_path, synchronization_path, adapter_path = paths
    return runner.invoke(
        app,
        [
            "dataset",
            "compile-replay",
            str(request_path),
            "--bundle",
            str(bundle_path),
            "--synchronization",
            str(synchronization_path),
            "--adapter",
            str(adapter_path),
            "--output",
            str(output),
        ],
    )


def test_cli_compiles_exact_replay_capsule(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Write an exact capsule and return the supported exit code."""
    output = tmp_path / "capsule.json"

    result = invoke(runner, write_inputs(tmp_path), output)

    assert result.exit_code == 0
    capsule = json.loads(output.read_text(encoding="utf-8"))
    assert capsule["fidelity"] == "exact_replay_ready"
    assert capsule["reproduction"]["scenario_id"] == "unsafe-stop"


def test_cli_emits_blocked_capsule_with_insufficient_exit(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Retain a blocked artifact for automation instead of discarding it."""
    source, _, _, replay_request = replay_artifacts()
    blocked = replace(
        replay_request,
        interval=ReplayInterval(5.0, 6.0, 6.0, 1.0),
        dataset_bundle_digest=source.content_digest(),
    )
    output = tmp_path / "capsule.json"

    result = invoke(
        runner,
        write_inputs(tmp_path, request_override=blocked),
        output,
    )

    assert result.exit_code == 5
    capsule = json.loads(output.read_text(encoding="utf-8"))
    assert capsule["fidelity"] == "insufficient_evidence"


def test_cli_keeps_replay_fidelity_separate_from_label_review(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Return review while preserving an exact reconstruction claim."""
    output = tmp_path / "capsule.json"

    result = invoke(
        runner,
        write_inputs(tmp_path, label_status=LabelStatus.SUGGESTED),
        output,
    )

    assert result.exit_code == 3
    capsule = json.loads(output.read_text(encoding="utf-8"))
    assert capsule["fidelity"] == "exact_replay_ready"
    assert capsule["failure_claim_qualification"] == "review_required"


def test_cli_blocking_reconstruction_takes_precedence_over_label_review(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Use the strongest automation signal when both axes have findings."""
    output = tmp_path / "capsule.json"

    result = invoke(
        runner,
        write_inputs(
            tmp_path,
            label_status=LabelStatus.SUGGESTED,
            context_outside_episode=True,
        ),
        output,
    )

    assert result.exit_code == 5
    capsule = json.loads(output.read_text(encoding="utf-8"))
    assert capsule["fidelity"] == "insufficient_evidence"
    assert capsule["failure_claim_qualification"] == "review_required"


def test_cli_rejects_mixed_artifact_identity(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Treat content-address mismatch as invalid input, not uncertainty."""
    _, _, _, replay_request = replay_artifacts()
    invalid = replace(
        replay_request,
        dataset_bundle_digest=sha256_digest("other"),
    )
    output = tmp_path / "capsule.json"

    result = invoke(
        runner,
        write_inputs(tmp_path, request_override=invalid),
        output,
    )

    assert result.exit_code == 2
    assert "different dataset bundle" in result.output
    assert not output.exists()
