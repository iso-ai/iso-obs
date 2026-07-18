"""Tests for recorded-failure replay capsule compilation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.dataset_io import (
    DATASET_BUNDLE_SCHEMA_VERSION,
    DatasetBundle,
    parse_json_artifact,
)
from iso_obs.dataset_reliability import (
    DatasetEvidenceRole,
    DatasetLabelAssertion,
    DatasetManifest,
    DatasetScope,
    EpisodeManifest,
    EvidenceRelation,
    LabelEvidence,
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
    SynchronizationIssue,
    SynchronizationIssueKind,
    SynchronizationIssueSeverity,
)
from iso_obs.evidence import NamedDigest, sha256_digest
from iso_obs.regression import (
    CaseRole,
    ComparisonOperator,
    ExpectedOutcome,
    InvariantOracle,
    ScenarioParameter,
)
from iso_obs.replay_capsule import (
    FAILURE_REPLAY_CAPSULE_SCHEMA_VERSION,
    FAILURE_REPLAY_REQUEST_SCHEMA_VERSION,
    FailureClaimQualification,
    FailureReplayCapsule,
    FailureReplayRequest,
    InitialStateCompleteness,
    InitialStateEvidence,
    ReplayChannelBinding,
    ReplayChannelRole,
    ReplayFidelity,
    ReplayInterval,
    ReplayIssueKind,
    ReplayIssueSeverity,
    StochasticReplayControl,
    build_canonical_regression_case,
    compile_failure_replay_capsule,
)
from iso_obs.simulation import (
    BackendType,
    EvidenceUse,
    SimulationAdapterManifest,
    SimulationCapability,
)


def bundle(
    *,
    label_status: LabelStatus = LabelStatus.CORROBORATED,
    include_trace: bool = True,
) -> DatasetBundle:
    """Build one labeled episode with a replayable observation trace."""
    scope = DatasetScope(
        domain_namespace="robot-manipulation",
        target_population_digest=sha256_digest("warehouse-picks"),
        collection_protocol_digest=sha256_digest("recording-protocol"),
        modality_ids=("camera",),
        evidence_roles=(DatasetEvidenceRole.FAILURE_LABEL_LEARNING,),
    )
    manifest = DatasetManifest(
        dataset_id="warehouse-failures",
        dataset_version="1",
        content_digest=sha256_digest("recording"),
        source_uri="s3://evidence/run-42",
        license_id="LicenseRef-internal",
        scope=scope,
    )
    episode = EpisodeManifest(
        dataset_manifest_digest=manifest.manifest_digest(),
        episode_id="pick-42",
        independence_unit_id="physical-run-42",
        content_digest=sha256_digest("episode-42"),
        start_seconds=0.0,
        end_seconds=10.0,
        modality_digests=(NamedDigest("camera", sha256_digest("camera-evidence")),),
        source_split="unassigned",
    )
    assertion = DatasetLabelAssertion(
        assertion_id="failure-42",
        episode_id=episode.episode_id,
        label_namespace="reliability/failure_state/v1",
        mode_family=ModeFamily.FAILURE_STATE,
        start_seconds=5.0,
        end_seconds=6.0,
        candidate_values=("unsafe-contact",),
        status=label_status,
        evidence=(
            LabelEvidence(
                evidence_id="review-42",
                evidence_digest=sha256_digest("review"),
                independent_source_id="reviewer-a",
                relation=EvidenceRelation.OBSERVED_OUTCOME,
                description="Unsafe contact is visible in synchronized evidence.",
            ),
        ),
        method_digest=(
            sha256_digest("detector") if label_status is LabelStatus.SUGGESTED else None
        ),
    )
    traces = (
        ChannelTimingTrace(
            episode_id=episode.episode_id,
            channel_id="camera-front",
            modality_id="camera",
            clock_id="controller-clock",
            content_digest=sha256_digest("camera-trace"),
            timestamps_seconds=(4.0, 5.0, 6.0, 7.0),
            timestamp_uncertainty_seconds=0.0,
        ),
    )
    return DatasetBundle(
        schema_version=DATASET_BUNDLE_SCHEMA_VERSION,
        manifest=manifest,
        episodes=(episode,),
        label_assertions=(assertion,),
        timing_traces=traces if include_trace else (),
    )


def synchronization(
    source: DatasetBundle,
    *,
    disposition: SynchronizationDisposition = (
        SynchronizationDisposition.ALIGNED_WITHIN_SCOPE
    ),
    interpolation: InterpolationPolicy = InterpolationPolicy.NONE,
    uncertainty: float = 0.0,
) -> DatasetSynchronizationReport:
    """Build timing evidence for the selected replay channel."""
    issues: tuple[SynchronizationIssue, ...] = ()
    if disposition is SynchronizationDisposition.REVIEW_REQUIRED:
        issues = (
            SynchronizationIssue(
                kind=SynchronizationIssueKind.DUPLICATE_TIMESTAMPS,
                severity=SynchronizationIssueSeverity.REVIEW,
                channel_id="camera-front",
                description="One duplicate timestamp requires review.",
            ),
        )
    elif disposition is SynchronizationDisposition.INSUFFICIENT_EVIDENCE:
        issues = (
            SynchronizationIssue(
                kind=SynchronizationIssueKind.MISSING_CLOCK_ALIGNMENT,
                severity=SynchronizationIssueSeverity.BLOCKING,
                channel_id="camera-front",
                description="Clock alignment is absent.",
            ),
        )
    return DatasetSynchronizationReport(
        schema_version=DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
        plan_content_digest=sha256_digest("sync-plan"),
        dataset_manifest_digest=source.manifest.manifest_digest(),
        episode_id="pick-42",
        disposition=disposition,
        reference_clock_id="controller-clock",
        channel_summaries=(
            ChannelSynchronizationSummary(
                channel_id="camera-front",
                modality_id="camera",
                clock_id="controller-clock",
                sample_count=4,
                coverage_start_seconds=4.0,
                coverage_end_seconds=7.0,
                maximum_observed_gap_seconds=1.0,
                alignment_uncertainty_upper_seconds=uncertainty,
                interpolation_policy=interpolation,
            ),
        ),
        issues=issues,
        scope=source.manifest.scope,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=(),
    )


def adapter(
    *,
    supported_artifacts: tuple[str, ...] = ("camera-playback",),
    capabilities: tuple[SimulationCapability, ...] | None = None,
    backend_type: BackendType = BackendType.PHYSICS_SIMULATOR,
) -> SimulationAdapterManifest:
    """Build a replay-capable physics simulator adapter."""
    return SimulationAdapterManifest(
        adapter_name="warehouse-replay",
        adapter_version="2",
        backend_type=backend_type,
        simulator_name="warehouse-twin",
        simulator_version="5",
        simulation_evidence_manifest_digest=sha256_digest("simulation"),
        capabilities=capabilities
        or (
            SimulationCapability.ARTIFACT_LOADING,
            SimulationCapability.DETERMINISTIC_TIME_CONTROL,
            SimulationCapability.SNAPSHOT_RESTORE,
        ),
        supported_parameters=("object_friction",),
        supported_artifacts=supported_artifacts,
    )


def request(
    source: DatasetBundle,
    target: SimulationAdapterManifest,
    *,
    initial_state: InitialStateEvidence | None = None,
    stochastic_control: StochasticReplayControl | None = None,
) -> FailureReplayRequest:
    """Build an exact replay request for one failure interval."""
    state = initial_state or InitialStateEvidence(
        artifact_digest=sha256_digest("state-at-4"),
        capture_time_seconds=4.0,
        reference_clock_id="controller-clock",
        completeness=InitialStateCompleteness.COMPLETE,
        method="simulator snapshot",
    )
    return FailureReplayRequest(
        schema_version=FAILURE_REPLAY_REQUEST_SCHEMA_VERSION,
        capsule_id="pick-42-replay",
        capsule_version="1",
        dataset_bundle_digest=source.content_digest(),
        episode_id="pick-42",
        failure_assertion_ids=("failure-42",),
        interval=ReplayInterval(
            failure_start_seconds=5.0,
            failure_end_seconds=6.0,
            pre_context_seconds=1.0,
            post_context_seconds=1.0,
        ),
        reference_clock_id="controller-clock",
        scenario_id="obstructed-pick",
        scenario_version="4",
        environment_id="warehouse",
        environment_version="5",
        system_versions=("policy-17",),
        simulation_manifest_digest=target.simulation_evidence_manifest_digest,
        adapter_manifest_digest=target.content_digest(),
        recording_evidence_digest=None,
        channel_bindings=(
            ReplayChannelBinding(
                channel_id="camera-front",
                role=ReplayChannelRole.SYSTEM_OBSERVATION,
                adapter_artifact_name="camera-playback",
                transformation_digest=sha256_digest("camera-transform"),
            ),
        ),
        initial_state=state,
        maximum_initial_state_age_seconds=0.0,
        stochastic_control=stochastic_control
        or StochasticReplayControl(stochastic=False),
        perturbation_realization_digest=None,
        sdk_version="0.1.0",
        command="iso replay execute pick-42-replay.json",
        configuration_digests=(
            NamedDigest("controller configuration", sha256_digest("controller")),
        ),
        required_adapter_capabilities=(
            SimulationCapability.ARTIFACT_LOADING,
            SimulationCapability.DETERMINISTIC_TIME_CONTROL,
            SimulationCapability.SNAPSHOT_RESTORE,
        ),
        limitations=("Contact material parameters are facility-specific.",),
    )


def compile_exact() -> FailureReplayCapsule:
    """Compile the reusable exact replay fixture."""
    source = bundle()
    target = adapter()
    return compile_failure_replay_capsule(
        request(source, target),
        bundle=source,
        synchronization=synchronization(source),
        adapter=target,
    )


def test_exact_capsule_binds_reproduction_and_builds_regression_case() -> None:
    """Compile a fully evidenced interval into downstream SDK contracts."""
    capsule = compile_exact()

    assert capsule.schema_version == FAILURE_REPLAY_CAPSULE_SCHEMA_VERSION
    assert capsule.fidelity is ReplayFidelity.EXACT_REPLAY_READY
    assert capsule.issues == ()
    assert capsule.extraction_start_seconds == 4.0
    assert capsule.extraction_end_seconds == 7.0
    assert capsule.reproduction.trace_digest.startswith("sha256:")
    assert capsule.reproduction.environment_version == "5"

    case = build_canonical_regression_case(
        capsule,
        case_id="canonical-pick-42",
        parameters=(ScenarioParameter("object_friction", 0.4),),
        oracles=(
            InvariantOracle(
                invariant_id="contact-force",
                invariant_version="2",
                signal="peak_contact_force_n",
                operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                threshold=18.0,
            ),
        ),
        expected_outcome=ExpectedOutcome.SATISFY_ALL,
        source_evidence_event_ids=("evt-contact-42",),
        rationale="Prevent recurrence of the observed unsafe contact.",
    )

    assert case.role is CaseRole.CANONICAL_REPRODUCTION
    assert case.expected_outcome is ExpectedOutcome.SATISFY_ALL
    assert any(item.name == "failure replay capsule" for item in case.artifact_digests)


def test_synchronization_review_limits_replay_to_approximate() -> None:
    """Keep a usable capsule while withholding an exact replay claim."""
    source = bundle()
    target = adapter()
    capsule = compile_failure_replay_capsule(
        request(source, target),
        bundle=source,
        synchronization=synchronization(
            source,
            disposition=SynchronizationDisposition.REVIEW_REQUIRED,
        ),
        adapter=target,
    )

    assert capsule.fidelity is ReplayFidelity.APPROXIMATE_REPLAY_ONLY
    assert any(
        item.kind is ReplayIssueKind.SYNCHRONIZATION_REVIEW for item in capsule.issues
    )
    with pytest.raises(ValueError, match="exact replay-ready"):
        build_canonical_regression_case(
            capsule,
            case_id="invalid",
            parameters=(),
            oracles=(
                InvariantOracle(
                    invariant_id="force",
                    invariant_version="1",
                    signal="force",
                    operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                    threshold=1.0,
                ),
            ),
            expected_outcome=ExpectedOutcome.SATISFY_ALL,
            source_evidence_event_ids=(),
            rationale="Approximate evidence is not gate-ready.",
        )


def test_missing_required_channel_blocks_replay() -> None:
    """A binding cannot be satisfied by a summary without payload provenance."""
    source = bundle(include_trace=False)
    target = adapter()

    capsule = compile_failure_replay_capsule(
        request(source, target),
        bundle=source,
        synchronization=synchronization(source),
        adapter=target,
    )

    assert capsule.fidelity is ReplayFidelity.INSUFFICIENT_EVIDENCE
    assert any(
        item.kind is ReplayIssueKind.MISSING_REQUIRED_CHANNEL
        and item.severity is ReplayIssueSeverity.BLOCKING
        for item in capsule.issues
    )


@pytest.mark.parametrize(
    ("change", "expected_kind", "expected_fidelity"),
    [
        (
            "weak_label",
            ReplayIssueKind.WEAK_FAILURE_LABEL,
            ReplayFidelity.EXACT_REPLAY_READY,
        ),
        (
            "context",
            ReplayIssueKind.CONTEXT_OUTSIDE_EPISODE,
            ReplayFidelity.INSUFFICIENT_EVIDENCE,
        ),
        (
            "unsupported_artifact",
            ReplayIssueKind.UNSUPPORTED_ADAPTER_ARTIFACT,
            ReplayFidelity.INSUFFICIENT_EVIDENCE,
        ),
        (
            "missing_state",
            ReplayIssueKind.MISSING_INITIAL_STATE,
            ReplayFidelity.APPROXIMATE_REPLAY_ONLY,
        ),
        (
            "stochastic",
            ReplayIssueKind.MISSING_STOCHASTIC_CONTROL,
            ReplayFidelity.APPROXIMATE_REPLAY_ONLY,
        ),
    ],
)
def test_readiness_findings_are_typed(
    change: str,
    expected_kind: ReplayIssueKind,
    expected_fidelity: ReplayFidelity,
) -> None:
    """Distinguish approximate replay limitations from blocking gaps."""
    source = bundle(
        label_status=(
            LabelStatus.SUGGESTED
            if change == "weak_label"
            else LabelStatus.CORROBORATED
        )
    )
    target = adapter(
        supported_artifacts=(
            () if change == "unsupported_artifact" else ("camera-playback",)
        )
    )
    replay_request = request(
        source,
        target,
        initial_state=(
            None
            if change == "missing_state"
            else InitialStateEvidence(
                artifact_digest=sha256_digest("state-at-4"),
                capture_time_seconds=4.0,
                reference_clock_id="controller-clock",
                completeness=InitialStateCompleteness.COMPLETE,
                method="simulator snapshot",
            )
        ),
        stochastic_control=(
            StochasticReplayControl(stochastic=True)
            if change == "stochastic"
            else StochasticReplayControl(stochastic=False)
        ),
    )
    if change == "missing_state":
        replay_request = replace(replay_request, initial_state=None)
    if change == "context":
        replay_request = replace(
            replay_request,
            interval=ReplayInterval(
                failure_start_seconds=5.0,
                failure_end_seconds=6.0,
                pre_context_seconds=6.0,
                post_context_seconds=1.0,
            ),
        )

    capsule = compile_failure_replay_capsule(
        replay_request,
        bundle=source,
        synchronization=synchronization(source),
        adapter=target,
    )

    assert capsule.fidelity is expected_fidelity
    assert any(item.kind is expected_kind for item in capsule.issues)
    if change == "weak_label":
        assert (
            capsule.failure_claim_qualification
            is FailureClaimQualification.REVIEW_REQUIRED
        )


def test_recording_evidence_identity_without_artifact_blocks_replay() -> None:
    """A named rosbag2 sidecar must be supplied, not trusted by digest alone."""
    source = bundle()
    target = adapter()
    replay_request = replace(
        request(source, target),
        recording_evidence_digest=sha256_digest("recording-evidence"),
    )

    capsule = compile_failure_replay_capsule(
        replay_request,
        bundle=source,
        synchronization=synchronization(source),
        adapter=target,
    )

    assert capsule.fidelity is ReplayFidelity.INSUFFICIENT_EVIDENCE
    assert any(
        item.kind is ReplayIssueKind.RECORDING_EVIDENCE_MISSING
        for item in capsule.issues
    )


def test_weak_claim_cannot_be_promoted_despite_exact_reconstruction() -> None:
    """Do not turn a reproducible but unqualified label into a release gate."""
    source = bundle(label_status=LabelStatus.SUGGESTED)
    target = adapter()
    capsule = compile_failure_replay_capsule(
        request(source, target),
        bundle=source,
        synchronization=synchronization(source),
        adapter=target,
    )

    assert capsule.fidelity is ReplayFidelity.EXACT_REPLAY_READY
    with pytest.raises(ValueError, match="supported failure claim"):
        build_canonical_regression_case(
            capsule,
            case_id="unqualified",
            parameters=(),
            oracles=(
                InvariantOracle(
                    invariant_id="force",
                    invariant_version="1",
                    signal="force",
                    operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                    threshold=1.0,
                ),
            ),
            expected_outcome=ExpectedOutcome.SATISFY_ALL,
            source_evidence_event_ids=(),
            rationale="A suggested label is not gate-ready.",
        )


def test_replay_artifacts_round_trip_through_strict_json() -> None:
    """Keep request and capsule contracts usable across CLI boundaries."""
    source = bundle()
    target = adapter()
    replay_request = request(source, target)
    capsule = compile_failure_replay_capsule(
        replay_request,
        bundle=source,
        synchronization=synchronization(source),
        adapter=target,
    )

    restored_request = parse_json_artifact(
        replay_request.to_json(),
        FailureReplayRequest,
    )
    restored_capsule = parse_json_artifact(
        capsule.to_json(),
        FailureReplayCapsule,
    )

    assert restored_request == replay_request
    assert restored_capsule == capsule
    assert restored_capsule.content_digest() == capsule.content_digest()


def test_artifact_identity_mismatch_is_invalid_not_approximate() -> None:
    """Reject mixed evidence chains before scientific readiness assessment."""
    source = bundle()
    target = adapter()
    replay_request = replace(
        request(source, target),
        dataset_bundle_digest=sha256_digest("other-bundle"),
    )

    with pytest.raises(ValueError, match="different dataset bundle"):
        compile_failure_replay_capsule(
            replay_request,
            bundle=source,
            synchronization=synchronization(source),
            adapter=target,
        )


def test_input_binding_requires_explicit_transformation() -> None:
    """Never assume recorded payload semantics match simulator inputs."""
    with pytest.raises(ValueError, match="transformation digest"):
        ReplayChannelBinding(
            channel_id="camera-front",
            role=ReplayChannelRole.SYSTEM_OBSERVATION,
            adapter_artifact_name="camera-playback",
            transformation_digest=None,
        )


def test_capsule_is_deterministic_and_content_addressed() -> None:
    """Equivalent compilation produces byte-identical replay artifacts."""
    first = compile_exact()
    second = compile_exact()

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.content_digest() == second.content_digest()
