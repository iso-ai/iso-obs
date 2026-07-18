"""Compile recorded failure evidence into simulator-neutral replay capsules.

The compiler binds an explicit dataset interval, failure assertions, timing
evidence, source-channel mappings, initial state, stochastic controls, and
simulator adapter capabilities. It never infers clock relationships, labels,
state completeness, transformations, or replay fidelity.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .dataset_io import DatasetBundle
from .dataset_reliability import (
    DatasetLabelAssertion,
    EpisodeManifest,
    LabelStatus,
)
from .dataset_rosbag2 import (
    Rosbag2RecordingDisposition,
    Rosbag2RecordingEvidence,
)
from .dataset_synchronization import (
    DatasetSynchronizationReport,
    InterpolationPolicy,
    SynchronizationDisposition,
)
from .evidence import (
    NamedDigest,
    ReproductionManifest,
    _canonical_json,
    sha256_digest,
)
from .regression import (
    CaseRole,
    ExpectedOutcome,
    InvariantOracle,
    RegressionCase,
    ScenarioParameter,
)
from .simulation import (
    BackendType,
    SimulationAdapterManifest,
    SimulationCapability,
)

FAILURE_REPLAY_REQUEST_SCHEMA_VERSION = "iso-obs.failure-replay-request.v1"
FAILURE_REPLAY_CAPSULE_SCHEMA_VERSION = "iso-obs.failure-replay-capsule.v1"


class ReplayFidelity(StrEnum):
    """Strongest reconstruction claim supported by a replay capsule."""

    EXACT_REPLAY_READY = "exact_replay_ready"
    APPROXIMATE_REPLAY_ONLY = "approximate_replay_only"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class FailureClaimQualification(StrEnum):
    """Strength of the failure claim selected for replay."""

    SUPPORTED_BY_SELECTED_EVIDENCE = "supported_by_selected_evidence"
    REVIEW_REQUIRED = "review_required"


class ReplayChannelRole(StrEnum):
    """Scientific role of one recorded channel during replay."""

    INITIAL_STATE = "initial_state"
    EXOGENOUS_INPUT = "exogenous_input"
    SYSTEM_OBSERVATION = "system_observation"
    SYSTEM_OUTPUT = "system_output"
    ORACLE_SIGNAL = "oracle_signal"


class InitialStateCompleteness(StrEnum):
    """Declared completeness of a captured simulator initial state."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ReplayIssueSeverity(StrEnum):
    """Effect of one finding on replay execution fidelity."""

    REVIEW = "review"
    BLOCKING = "blocking"


class ReplayIssueKind(StrEnum):
    """Machine-readable replay evidence or compatibility limitation."""

    CONTEXT_OUTSIDE_EPISODE = "context_outside_episode"
    FAILURE_INTERVAL_UNSUPPORTED = "failure_interval_unsupported"
    WEAK_FAILURE_LABEL = "weak_failure_label"
    SYNCHRONIZATION_REVIEW = "synchronization_review"
    SYNCHRONIZATION_BLOCKING = "synchronization_blocking"
    MISSING_REQUIRED_CHANNEL = "missing_required_channel"
    INCOMPLETE_CHANNEL_COVERAGE = "incomplete_channel_coverage"
    INTERPOLATED_CHANNEL = "interpolated_channel"
    TIMING_UNCERTAINTY = "timing_uncertainty"
    MISSING_INITIAL_STATE = "missing_initial_state"
    INCOMPLETE_INITIAL_STATE = "incomplete_initial_state"
    STALE_INITIAL_STATE = "stale_initial_state"
    MISSING_STOCHASTIC_CONTROL = "missing_stochastic_control"
    MISSING_ADAPTER_CAPABILITY = "missing_adapter_capability"
    UNSUPPORTED_ADAPTER_ARTIFACT = "unsupported_adapter_artifact"
    RECORDING_EVIDENCE_MISSING = "recording_evidence_missing"
    RECORDING_EVIDENCE_REVIEW = "recording_evidence_review"


@dataclass(frozen=True, slots=True)
class ReplayInterval:
    """Failure interval and explicitly requested surrounding context."""

    failure_start_seconds: float
    failure_end_seconds: float
    pre_context_seconds: float
    post_context_seconds: float

    def __post_init__(self) -> None:
        """Validate a closed failure interval and nonnegative context."""
        start = _finite(self.failure_start_seconds, "failure interval start")
        end = _finite(self.failure_end_seconds, "failure interval end")
        if start > end:
            raise ValueError("failure interval start must not exceed its end")
        pre = _nonnegative_finite(self.pre_context_seconds, "pre-failure context")
        post = _nonnegative_finite(
            self.post_context_seconds,
            "post-failure context",
        )
        object.__setattr__(self, "failure_start_seconds", start)
        object.__setattr__(self, "failure_end_seconds", end)
        object.__setattr__(self, "pre_context_seconds", pre)
        object.__setattr__(self, "post_context_seconds", post)

    @property
    def extraction_start_seconds(self) -> float:
        """Return the requested replay extraction start."""
        return self.failure_start_seconds - self.pre_context_seconds

    @property
    def extraction_end_seconds(self) -> float:
        """Return the requested replay extraction end."""
        return self.failure_end_seconds + self.post_context_seconds


@dataclass(frozen=True, slots=True)
class ReplayChannelBinding:
    """Bind one recorded channel to an adapter artifact input or evidence role."""

    channel_id: str
    role: ReplayChannelRole
    adapter_artifact_name: str
    transformation_digest: str | None
    required: bool = True

    def __post_init__(self) -> None:
        """Validate channel identity, target, transformation, and role."""
        _require_text(self.channel_id, "replay channel ID")
        _require_text(self.adapter_artifact_name, "adapter artifact name")
        role = ReplayChannelRole(self.role)
        if not isinstance(self.required, bool):
            raise ValueError("replay channel required must be boolean")
        if self.transformation_digest is not None:
            NamedDigest("replay channel transformation", self.transformation_digest)
        if role in _INPUT_ROLES and self.transformation_digest is None:
            raise ValueError(
                f"{role.value} channel bindings require a transformation digest"
            )
        object.__setattr__(self, "role", role)


@dataclass(frozen=True, slots=True)
class InitialStateEvidence:
    """Immutable evidence used to initialize a replay backend."""

    artifact_digest: str
    capture_time_seconds: float
    reference_clock_id: str
    completeness: InitialStateCompleteness
    method: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate initial-state identity, time, completeness, and method."""
        NamedDigest("initial state", self.artifact_digest)
        capture = _finite(self.capture_time_seconds, "initial-state capture time")
        _require_text(self.reference_clock_id, "initial-state reference clock ID")
        _require_text(self.method, "initial-state capture method")
        limitations = _unique_text(
            self.limitations,
            "initial-state limitations",
            required=False,
        )
        object.__setattr__(self, "capture_time_seconds", capture)
        object.__setattr__(
            self,
            "completeness",
            InitialStateCompleteness(self.completeness),
        )
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class StochasticReplayControl:
    """Declared random controls for deterministic or stochastic replay."""

    stochastic: bool
    seed: int | None = None
    random_stream_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate consistency between stochasticity, seed, and random stream."""
        if not isinstance(self.stochastic, bool):
            raise ValueError("stochastic replay flag must be boolean")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("replay seed must be an integer")
        if self.random_stream_digest is not None:
            NamedDigest("replay random stream", self.random_stream_digest)
        if not self.stochastic and (
            self.seed is not None or self.random_stream_digest is not None
        ):
            raise ValueError(
                "deterministic replay must not declare stochastic controls"
            )


@dataclass(frozen=True, slots=True)
class FailureReplayRequest:
    """Versioned request to compile a recorded interval for simulation replay."""

    schema_version: str
    capsule_id: str
    capsule_version: str
    dataset_bundle_digest: str
    episode_id: str
    failure_assertion_ids: tuple[str, ...]
    interval: ReplayInterval
    reference_clock_id: str
    scenario_id: str
    scenario_version: str
    environment_id: str
    environment_version: str
    system_versions: tuple[str, ...]
    simulation_manifest_digest: str
    adapter_manifest_digest: str
    recording_evidence_digest: str | None
    channel_bindings: tuple[ReplayChannelBinding, ...]
    initial_state: InitialStateEvidence | None
    maximum_initial_state_age_seconds: float
    stochastic_control: StochasticReplayControl
    perturbation_realization_digest: str | None
    sdk_version: str
    command: str
    configuration_digests: tuple[NamedDigest, ...]
    required_adapter_capabilities: tuple[SimulationCapability, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate replay identity, mappings, controls, and target references."""
        if self.schema_version != FAILURE_REPLAY_REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported failure replay request schema version")
        for value, label in (
            (self.capsule_id, "replay capsule ID"),
            (self.capsule_version, "replay capsule version"),
            (self.episode_id, "replay episode ID"),
            (self.reference_clock_id, "replay reference clock ID"),
            (self.scenario_id, "replay scenario ID"),
            (self.scenario_version, "replay scenario version"),
            (self.environment_id, "replay environment ID"),
            (self.environment_version, "replay environment version"),
            (self.sdk_version, "replay SDK version"),
            (self.command, "replay command"),
        ):
            _require_text(value, label)
        for value, label in (
            (self.dataset_bundle_digest, "dataset bundle"),
            (self.simulation_manifest_digest, "simulation manifest"),
            (self.adapter_manifest_digest, "simulation adapter manifest"),
        ):
            NamedDigest(label, value)
        if self.recording_evidence_digest is not None:
            NamedDigest("recording evidence", self.recording_evidence_digest)
        if self.perturbation_realization_digest is not None:
            NamedDigest(
                "perturbation realization",
                self.perturbation_realization_digest,
            )
        assertions = _unique_text(
            self.failure_assertion_ids,
            "failure assertion IDs",
        )
        system_versions = _unique_text(self.system_versions, "system versions")
        bindings = tuple(
            sorted(self.channel_bindings, key=lambda item: item.channel_id)
        )
        if not bindings:
            raise ValueError("failure replay requires channel bindings")
        _require_unique(
            (item.channel_id for item in bindings),
            "replay channel IDs",
        )
        _require_unique(
            (item.adapter_artifact_name for item in bindings),
            "replay adapter artifact names",
        )
        _require_unique(
            (item.name for item in self.configuration_digests),
            "replay configuration digest names",
        )
        reserved_names = _RESERVED_CONFIGURATION_NAMES & {
            item.name for item in self.configuration_digests
        }
        if reserved_names:
            raise ValueError(
                "configuration digest names are reserved: "
                + ", ".join(sorted(reserved_names))
            )
        configurations = tuple(
            sorted(self.configuration_digests, key=lambda item: item.name)
        )
        capabilities = tuple(
            sorted(
                {
                    SimulationCapability(item)
                    for item in self.required_adapter_capabilities
                },
                key=lambda item: item.value,
            )
        )
        maximum_age = _nonnegative_finite(
            self.maximum_initial_state_age_seconds,
            "maximum initial-state age",
        )
        limitations = _unique_text(
            self.limitations,
            "replay request limitations",
            required=False,
        )
        object.__setattr__(self, "failure_assertion_ids", assertions)
        object.__setattr__(self, "system_versions", system_versions)
        object.__setattr__(self, "channel_bindings", bindings)
        object.__setattr__(self, "configuration_digests", configurations)
        object.__setattr__(self, "required_adapter_capabilities", capabilities)
        object.__setattr__(
            self,
            "maximum_initial_state_age_seconds",
            maximum_age,
        )
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the replay request to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact replay-request digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class ReplayFailureAssertion:
    """Content-addressed label evidence selected for the replay interval."""

    assertion_id: str
    assertion_content_digest: str
    status: LabelStatus
    candidate_values: tuple[str, ...]
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        """Validate selected assertion identity, status, values, and interval."""
        _require_text(self.assertion_id, "replay assertion ID")
        NamedDigest("replay assertion", self.assertion_content_digest)
        object.__setattr__(self, "status", LabelStatus(self.status))
        object.__setattr__(
            self,
            "candidate_values",
            _unique_text(
                self.candidate_values,
                "replay assertion candidate values",
                required=False,
            ),
        )
        start = _finite(self.start_seconds, "replay assertion start")
        end = _finite(self.end_seconds, "replay assertion end")
        if start > end:
            raise ValueError("replay assertion interval must be ordered")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True, slots=True)
class ReplayChannelArtifact:
    """Replay binding paired with observed timing and payload provenance."""

    channel_id: str
    modality_id: str | None
    role: ReplayChannelRole
    adapter_artifact_name: str
    transformation_digest: str | None
    required: bool
    trace_content_digest: str | None
    sample_count: int
    coverage_start_seconds: float | None
    coverage_end_seconds: float | None
    alignment_uncertainty_upper_seconds: float | None
    interpolation_policy: InterpolationPolicy | None

    def __post_init__(self) -> None:
        """Validate replay artifact identity and nullable evidence fields."""
        _require_text(self.channel_id, "replay artifact channel ID")
        if self.modality_id is not None:
            _require_text(self.modality_id, "replay artifact modality ID")
        _require_text(self.adapter_artifact_name, "replay adapter artifact name")
        object.__setattr__(self, "role", ReplayChannelRole(self.role))
        if not isinstance(self.required, bool):
            raise ValueError("replay artifact required must be boolean")
        if self.transformation_digest is not None:
            NamedDigest("replay artifact transformation", self.transformation_digest)
        if self.trace_content_digest is not None:
            NamedDigest("replay channel trace", self.trace_content_digest)
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise ValueError("replay channel sample count must be nonnegative")
        if (self.coverage_start_seconds is None) != (self.coverage_end_seconds is None):
            raise ValueError("replay channel coverage bounds must be paired")
        if self.coverage_start_seconds is not None:
            coverage_end = self.coverage_end_seconds
            if coverage_end is None:
                raise AssertionError("replay channel coverage was validated")
            start = _finite(
                self.coverage_start_seconds,
                "replay channel coverage start",
            )
            end = _finite(
                coverage_end,
                "replay channel coverage end",
            )
            if start > end:
                raise ValueError("replay channel coverage must be ordered")
        if self.alignment_uncertainty_upper_seconds is not None:
            _nonnegative_finite(
                self.alignment_uncertainty_upper_seconds,
                "replay channel alignment uncertainty",
            )
        if self.interpolation_policy is not None:
            object.__setattr__(
                self,
                "interpolation_policy",
                InterpolationPolicy(self.interpolation_policy),
            )


@dataclass(frozen=True, slots=True)
class ReplayReadinessIssue:
    """One explicit limitation on replay execution or fidelity."""

    kind: ReplayIssueKind
    severity: ReplayIssueSeverity
    subject: str
    description: str
    affects_replay_fidelity: bool = True

    def __post_init__(self) -> None:
        """Validate issue classification, severity, subject, and description."""
        object.__setattr__(self, "kind", ReplayIssueKind(self.kind))
        object.__setattr__(self, "severity", ReplayIssueSeverity(self.severity))
        _require_text(self.subject, "replay issue subject")
        _require_text(self.description, "replay issue description")
        if not isinstance(self.affects_replay_fidelity, bool):
            raise ValueError("replay issue fidelity impact must be boolean")


@dataclass(frozen=True, slots=True)
class FailureReplayCapsule:
    """Portable replay definition derived from immutable failure evidence."""

    schema_version: str
    request_content_digest: str
    capsule_id: str
    capsule_version: str
    dataset_bundle_digest: str
    dataset_manifest_digest: str
    source_uri: str
    episode_id: str
    episode_content_digest: str
    independence_unit_id: str
    recording_evidence_digest: str | None
    failure_assertions: tuple[ReplayFailureAssertion, ...]
    reference_clock_id: str
    failure_start_seconds: float
    failure_end_seconds: float
    extraction_start_seconds: float
    extraction_end_seconds: float
    channels: tuple[ReplayChannelArtifact, ...]
    initial_state: InitialStateEvidence | None
    simulation_manifest_digest: str
    adapter_manifest_digest: str
    required_adapter_capabilities: tuple[SimulationCapability, ...]
    reproduction: ReproductionManifest
    fidelity: ReplayFidelity
    failure_claim_qualification: FailureClaimQualification
    issues: tuple[ReplayReadinessIssue, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate capsule identity, ordering, readiness, and provenance."""
        if self.schema_version != FAILURE_REPLAY_CAPSULE_SCHEMA_VERSION:
            raise ValueError("unsupported failure replay capsule schema version")
        for value, label in (
            (self.request_content_digest, "replay request"),
            (self.dataset_bundle_digest, "dataset bundle"),
            (self.dataset_manifest_digest, "dataset manifest"),
            (self.episode_content_digest, "dataset episode"),
            (self.simulation_manifest_digest, "simulation manifest"),
            (self.adapter_manifest_digest, "simulation adapter manifest"),
        ):
            NamedDigest(label, value)
        if self.recording_evidence_digest is not None:
            NamedDigest("recording evidence", self.recording_evidence_digest)
        for value, label in (
            (self.capsule_id, "replay capsule ID"),
            (self.capsule_version, "replay capsule version"),
            (self.source_uri, "replay source URI"),
            (self.episode_id, "replay episode ID"),
            (self.independence_unit_id, "replay independence unit ID"),
            (self.reference_clock_id, "replay reference clock ID"),
        ):
            _require_text(value, label)
        failure_start = _finite(
            self.failure_start_seconds,
            "capsule failure start",
        )
        failure_end = _finite(self.failure_end_seconds, "capsule failure end")
        extraction_start = _finite(
            self.extraction_start_seconds,
            "capsule extraction start",
        )
        extraction_end = _finite(
            self.extraction_end_seconds,
            "capsule extraction end",
        )
        if not extraction_start <= failure_start <= failure_end <= extraction_end:
            raise ValueError("capsule extraction must contain its failure interval")
        assertions = tuple(
            sorted(self.failure_assertions, key=lambda item: item.assertion_id)
        )
        _require_unique(
            (item.assertion_id for item in assertions),
            "capsule failure assertion IDs",
        )
        channels = tuple(sorted(self.channels, key=lambda item: item.channel_id))
        _require_unique(
            (item.channel_id for item in channels),
            "capsule replay channel IDs",
        )
        capabilities = tuple(
            sorted(
                {
                    SimulationCapability(item)
                    for item in self.required_adapter_capabilities
                },
                key=lambda item: item.value,
            )
        )
        issues = tuple(
            sorted(
                self.issues,
                key=lambda item: (
                    item.severity.value,
                    item.kind.value,
                    item.subject,
                ),
            )
        )
        fidelity = ReplayFidelity(self.fidelity)
        claim_qualification = FailureClaimQualification(
            self.failure_claim_qualification
        )
        fidelity_issues = tuple(item for item in issues if item.affects_replay_fidelity)
        blocking = any(
            item.severity is ReplayIssueSeverity.BLOCKING for item in fidelity_issues
        )
        if fidelity is ReplayFidelity.EXACT_REPLAY_READY and fidelity_issues:
            raise ValueError(
                "exact replay capsule cannot contain fidelity-affecting issues"
            )
        if fidelity is ReplayFidelity.APPROXIMATE_REPLAY_ONLY and (
            not fidelity_issues or blocking
        ):
            raise ValueError(
                "approximate replay requires fidelity review issues and no "
                "fidelity blocking issues"
            )
        if fidelity is ReplayFidelity.INSUFFICIENT_EVIDENCE and not blocking:
            raise ValueError(
                "insufficient replay evidence requires a fidelity blocking issue"
            )
        claim_issues = {
            ReplayIssueKind.FAILURE_INTERVAL_UNSUPPORTED,
            ReplayIssueKind.WEAK_FAILURE_LABEL,
        }
        claim_needs_review = any(item.kind in claim_issues for item in issues)
        if (
            claim_qualification
            is FailureClaimQualification.SUPPORTED_BY_SELECTED_EVIDENCE
            and claim_needs_review
        ):
            raise ValueError(
                "supported failure claim cannot contain claim qualification issues"
            )
        if (
            claim_qualification is FailureClaimQualification.REVIEW_REQUIRED
            and not claim_needs_review
        ):
            raise ValueError(
                "failure claim review requires a claim qualification issue"
            )
        limitations = _unique_text(
            self.limitations,
            "replay capsule limitations",
        )
        object.__setattr__(self, "failure_start_seconds", failure_start)
        object.__setattr__(self, "failure_end_seconds", failure_end)
        object.__setattr__(self, "extraction_start_seconds", extraction_start)
        object.__setattr__(self, "extraction_end_seconds", extraction_end)
        object.__setattr__(self, "failure_assertions", assertions)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "required_adapter_capabilities", capabilities)
        object.__setattr__(self, "fidelity", fidelity)
        object.__setattr__(
            self,
            "failure_claim_qualification",
            claim_qualification,
        )
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the replay capsule to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact replay-capsule digest."""
        return sha256_digest(self.to_json())


def compile_failure_replay_capsule(
    request: FailureReplayRequest,
    *,
    bundle: DatasetBundle,
    synchronization: DatasetSynchronizationReport,
    adapter: SimulationAdapterManifest,
    recording_evidence: Rosbag2RecordingEvidence | None = None,
) -> FailureReplayCapsule:
    """Compile immutable dataset evidence into a replay-readiness artifact.

    Args:
        request: Explicit interval, mappings, controls, and replay target.
        bundle: Content-addressed dataset bundle containing the source episode.
        synchronization: Timing audit for the selected episode and channels.
        adapter: Simulator adapter capabilities and artifact vocabulary.
        recording_evidence: Optional verified rosbag2 source-file evidence.

    Returns:
        Portable replay capsule with exact, approximate, or blocked fidelity.

    Raises:
        ValueError: If supplied artifacts do not match request identities.
    """
    _verify_input_identities(
        request,
        bundle,
        synchronization,
        adapter,
        recording_evidence,
    )
    episode = _episode(bundle, request.episode_id)
    assertions = _selected_assertions(bundle, request)
    issues: list[ReplayReadinessIssue] = []
    _interval_issues(request, episode.start_seconds, episode.end_seconds, issues)
    replay_assertions = _assertion_artifacts(request, assertions, issues)
    _synchronization_issues(synchronization, issues)
    channels = _channel_artifacts(
        request,
        bundle,
        synchronization,
        adapter,
        issues,
    )
    _initial_state_issues(request, adapter, issues)
    _stochastic_control_issues(request, adapter, issues)
    _adapter_capability_issues(request, adapter, issues)
    _recording_evidence_issues(request, recording_evidence, issues)
    fidelity = _fidelity(issues)
    claim_qualification = _failure_claim_qualification(issues)
    trace_digest = _replay_trace_digest(request, episode.content_digest, channels)
    reproduction = _reproduction_manifest(
        request,
        bundle,
        episode.content_digest,
        channels,
        trace_digest,
    )
    return FailureReplayCapsule(
        schema_version=FAILURE_REPLAY_CAPSULE_SCHEMA_VERSION,
        request_content_digest=request.content_digest(),
        capsule_id=request.capsule_id,
        capsule_version=request.capsule_version,
        dataset_bundle_digest=bundle.content_digest(),
        dataset_manifest_digest=bundle.manifest.manifest_digest(),
        source_uri=bundle.manifest.source_uri,
        episode_id=episode.episode_id,
        episode_content_digest=episode.content_digest,
        independence_unit_id=episode.independence_unit_id,
        recording_evidence_digest=request.recording_evidence_digest,
        failure_assertions=replay_assertions,
        reference_clock_id=request.reference_clock_id,
        failure_start_seconds=request.interval.failure_start_seconds,
        failure_end_seconds=request.interval.failure_end_seconds,
        extraction_start_seconds=request.interval.extraction_start_seconds,
        extraction_end_seconds=request.interval.extraction_end_seconds,
        channels=channels,
        initial_state=request.initial_state,
        simulation_manifest_digest=request.simulation_manifest_digest,
        adapter_manifest_digest=request.adapter_manifest_digest,
        required_adapter_capabilities=request.required_adapter_capabilities,
        reproduction=reproduction,
        fidelity=fidelity,
        failure_claim_qualification=claim_qualification,
        issues=tuple(issues),
        limitations=(
            *request.limitations,
            "Replay fidelity describes reconstructability of declared inputs; it "
            "does not establish that a selected label is causally correct.",
            "A replay capsule packages one observed interval and does not establish "
            "robustness throughout its neighborhood or the real world.",
            "Approximate replay remains discovery-only until blocking uncertainty "
            "in state, timing, stochastic controls, or transformations is resolved.",
        ),
    )


def build_canonical_regression_case(
    capsule: FailureReplayCapsule,
    *,
    case_id: str,
    parameters: Sequence[ScenarioParameter],
    oracles: Sequence[InvariantOracle],
    expected_outcome: ExpectedOutcome,
    source_evidence_event_ids: Sequence[str],
    rationale: str,
) -> RegressionCase:
    """Build a canonical regression case from an exact replay capsule.

    Args:
        capsule: Exact replay-ready source evidence.
        case_id: Stable regression-case identity.
        parameters: Explicit simulator parameter assignments.
        oracles: Versioned invariant decisions for the replay.
        expected_outcome: Expected result for the system version under test.
        source_evidence_event_ids: Optional normalized event provenance.
        rationale: Human-readable reason for retaining this regression.

    Returns:
        Canonical regression case linked to every replay artifact digest.

    Raises:
        ValueError: If the capsule is not exact replay-ready.
    """
    if capsule.fidelity is not ReplayFidelity.EXACT_REPLAY_READY:
        raise ValueError(
            "canonical regression cases require an exact replay-ready capsule"
        )
    if (
        capsule.failure_claim_qualification
        is not FailureClaimQualification.SUPPORTED_BY_SELECTED_EVIDENCE
    ):
        raise ValueError("canonical regression cases require a supported failure claim")
    artifacts = [
        NamedDigest("failure replay capsule", capsule.content_digest()),
        NamedDigest("dataset bundle", capsule.dataset_bundle_digest),
        NamedDigest("dataset episode", capsule.episode_content_digest),
        NamedDigest("simulation manifest", capsule.simulation_manifest_digest),
        NamedDigest("simulation adapter", capsule.adapter_manifest_digest),
    ]
    if capsule.recording_evidence_digest is not None:
        artifacts.append(
            NamedDigest(
                "recording evidence",
                capsule.recording_evidence_digest,
            )
        )
    if capsule.initial_state is not None:
        artifacts.append(
            NamedDigest("initial state", capsule.initial_state.artifact_digest)
        )
    return RegressionCase(
        case_id=case_id,
        role=CaseRole.CANONICAL_REPRODUCTION,
        scenario_id=capsule.reproduction.scenario_id,
        scenario_version=capsule.reproduction.scenario_version,
        parameters=tuple(parameters),
        oracles=tuple(oracles),
        expected_outcome=expected_outcome,
        artifact_digests=tuple(artifacts),
        source_evidence_event_ids=tuple(source_evidence_event_ids),
        rationale=rationale,
    )


def _verify_input_identities(
    request: FailureReplayRequest,
    bundle: DatasetBundle,
    synchronization: DatasetSynchronizationReport,
    adapter: SimulationAdapterManifest,
    recording_evidence: Rosbag2RecordingEvidence | None,
) -> None:
    """Require supplied artifacts to match all request identities."""
    if request.dataset_bundle_digest != bundle.content_digest():
        raise ValueError("replay request identifies a different dataset bundle")
    manifest_digest = bundle.manifest.manifest_digest()
    if synchronization.dataset_manifest_digest != manifest_digest:
        raise ValueError("synchronization report identifies a different dataset")
    if synchronization.episode_id != request.episode_id:
        raise ValueError("synchronization report identifies a different episode")
    if synchronization.reference_clock_id != request.reference_clock_id:
        raise ValueError("synchronization report uses a different reference clock")
    if request.adapter_manifest_digest != adapter.content_digest():
        raise ValueError("replay request identifies a different adapter manifest")
    if (
        request.simulation_manifest_digest
        != adapter.simulation_evidence_manifest_digest
    ):
        raise ValueError("adapter identifies a different simulation manifest")
    if recording_evidence is not None:
        if request.recording_evidence_digest is None:
            raise ValueError("unexpected rosbag2 recording evidence was supplied")
        if request.recording_evidence_digest != recording_evidence.content_digest():
            raise ValueError("replay request identifies different recording evidence")


def _episode(bundle: DatasetBundle, episode_id: str) -> EpisodeManifest:
    """Return the exact selected episode or reject a missing identity."""
    for episode in bundle.episodes:
        if episode.episode_id == episode_id:
            return episode
    raise ValueError(f"dataset bundle has no episode {episode_id!r}")


def _selected_assertions(
    bundle: DatasetBundle,
    request: FailureReplayRequest,
) -> tuple[DatasetLabelAssertion, ...]:
    """Resolve selected failure labels and require episode consistency."""
    by_id = {item.assertion_id: item for item in bundle.label_assertions}
    missing = sorted(set(request.failure_assertion_ids) - set(by_id))
    if missing:
        raise ValueError(
            "replay request references unknown assertions: " + ", ".join(missing)
        )
    selected = tuple(by_id[item] for item in request.failure_assertion_ids)
    if any(item.episode_id != request.episode_id for item in selected):
        raise ValueError("replay assertions must belong to the selected episode")
    return selected


def _interval_issues(
    request: FailureReplayRequest,
    episode_start: float,
    episode_end: float,
    issues: list[ReplayReadinessIssue],
) -> None:
    """Surface requested context outside immutable episode evidence."""
    if (
        request.interval.extraction_start_seconds < episode_start
        or request.interval.extraction_end_seconds > episode_end
    ):
        issues.append(
            _issue(
                ReplayIssueKind.CONTEXT_OUTSIDE_EPISODE,
                ReplayIssueSeverity.BLOCKING,
                request.episode_id,
                "Requested replay context extends outside the selected episode.",
            )
        )


def _assertion_artifacts(
    request: FailureReplayRequest,
    assertions: Sequence[DatasetLabelAssertion],
    issues: list[ReplayReadinessIssue],
) -> tuple[ReplayFailureAssertion, ...]:
    """Bind selected assertions and assess interval and status support."""
    artifacts: list[ReplayFailureAssertion] = []
    for assertion in assertions:
        overlaps = (
            assertion.start_seconds <= request.interval.failure_end_seconds
            and assertion.end_seconds >= request.interval.failure_start_seconds
        )
        if not overlaps:
            issues.append(
                _issue(
                    ReplayIssueKind.FAILURE_INTERVAL_UNSUPPORTED,
                    ReplayIssueSeverity.BLOCKING,
                    assertion.assertion_id,
                    "Selected assertion does not overlap the requested failure "
                    "interval.",
                )
            )
        if assertion.status not in {
            LabelStatus.CORROBORATED,
            LabelStatus.ADJUDICATED,
        }:
            issues.append(
                _issue(
                    ReplayIssueKind.WEAK_FAILURE_LABEL,
                    ReplayIssueSeverity.REVIEW,
                    assertion.assertion_id,
                    f"Assertion status {assertion.status.value!r} does not establish "
                    "a corroborated failure label.",
                    affects_replay_fidelity=False,
                )
            )
        artifacts.append(
            ReplayFailureAssertion(
                assertion_id=assertion.assertion_id,
                assertion_content_digest=sha256_digest(_canonical_json(assertion)),
                status=assertion.status,
                candidate_values=assertion.candidate_values,
                start_seconds=assertion.start_seconds,
                end_seconds=assertion.end_seconds,
            )
        )
    return tuple(artifacts)


def _synchronization_issues(
    report: DatasetSynchronizationReport,
    issues: list[ReplayReadinessIssue],
) -> None:
    """Translate synchronization disposition into replay fidelity."""
    if report.disposition is SynchronizationDisposition.REVIEW_REQUIRED:
        issues.append(
            _issue(
                ReplayIssueKind.SYNCHRONIZATION_REVIEW,
                ReplayIssueSeverity.REVIEW,
                report.episode_id,
                "Temporal synchronization contains nonblocking review findings.",
            )
        )
    elif report.disposition is SynchronizationDisposition.INSUFFICIENT_EVIDENCE:
        issues.append(
            _issue(
                ReplayIssueKind.SYNCHRONIZATION_BLOCKING,
                ReplayIssueSeverity.BLOCKING,
                report.episode_id,
                "Temporal synchronization contains blocking evidence gaps.",
            )
        )


def _channel_artifacts(
    request: FailureReplayRequest,
    bundle: DatasetBundle,
    synchronization: DatasetSynchronizationReport,
    adapter: SimulationAdapterManifest,
    issues: list[ReplayReadinessIssue],
) -> tuple[ReplayChannelArtifact, ...]:
    """Bind timing traces to adapter artifact inputs without interpolation."""
    traces = {
        item.channel_id: item
        for item in bundle.timing_traces
        if item.episode_id == request.episode_id
    }
    summaries = {item.channel_id: item for item in synchronization.channel_summaries}
    artifacts: list[ReplayChannelArtifact] = []
    for binding in request.channel_bindings:
        trace = traces.get(binding.channel_id)
        summary = summaries.get(binding.channel_id)
        if trace is None or summary is None or summary.sample_count == 0:
            if binding.required:
                issues.append(
                    _issue(
                        ReplayIssueKind.MISSING_REQUIRED_CHANNEL,
                        ReplayIssueSeverity.BLOCKING,
                        binding.channel_id,
                        "Required replay channel has no timing and payload evidence.",
                    )
                )
            artifacts.append(
                ReplayChannelArtifact(
                    channel_id=binding.channel_id,
                    modality_id=(summary.modality_id if summary else None),
                    role=binding.role,
                    adapter_artifact_name=binding.adapter_artifact_name,
                    transformation_digest=binding.transformation_digest,
                    required=binding.required,
                    trace_content_digest=None,
                    sample_count=0,
                    coverage_start_seconds=None,
                    coverage_end_seconds=None,
                    alignment_uncertainty_upper_seconds=None,
                    interpolation_policy=None,
                )
            )
            continue
        coverage_complete = (
            summary.coverage_start_seconds is not None
            and summary.coverage_end_seconds is not None
            and summary.coverage_start_seconds
            <= request.interval.extraction_start_seconds
            and summary.coverage_end_seconds >= request.interval.extraction_end_seconds
        )
        if binding.required and not coverage_complete:
            severity = (
                ReplayIssueSeverity.BLOCKING
                if binding.role in _INPUT_ROLES
                else ReplayIssueSeverity.REVIEW
            )
            issues.append(
                _issue(
                    ReplayIssueKind.INCOMPLETE_CHANNEL_COVERAGE,
                    severity,
                    binding.channel_id,
                    "Channel timing evidence does not cover the full replay "
                    "extraction interval.",
                )
            )
        if summary.interpolation_policy is not InterpolationPolicy.NONE:
            issues.append(
                _issue(
                    ReplayIssueKind.INTERPOLATED_CHANNEL,
                    ReplayIssueSeverity.REVIEW,
                    binding.channel_id,
                    "Replay channel requires an interpolation policy and cannot "
                    "support byte-for-byte exact temporal reconstruction.",
                )
            )
        if (
            summary.alignment_uncertainty_upper_seconds is not None
            and summary.alignment_uncertainty_upper_seconds > 0.0
        ):
            issues.append(
                _issue(
                    ReplayIssueKind.TIMING_UNCERTAINTY,
                    ReplayIssueSeverity.REVIEW,
                    binding.channel_id,
                    "Replay timing retains nonzero alignment uncertainty.",
                )
            )
        if (
            binding.role in _INPUT_ROLES
            and not adapter.accepts_arbitrary_artifacts
            and binding.adapter_artifact_name not in adapter.supported_artifacts
        ):
            issues.append(
                _issue(
                    ReplayIssueKind.UNSUPPORTED_ADAPTER_ARTIFACT,
                    ReplayIssueSeverity.BLOCKING,
                    binding.adapter_artifact_name,
                    "Simulation adapter does not declare the bound replay artifact.",
                )
            )
        artifacts.append(
            ReplayChannelArtifact(
                channel_id=binding.channel_id,
                modality_id=trace.modality_id,
                role=binding.role,
                adapter_artifact_name=binding.adapter_artifact_name,
                transformation_digest=binding.transformation_digest,
                required=binding.required,
                trace_content_digest=trace.content_digest,
                sample_count=summary.sample_count,
                coverage_start_seconds=summary.coverage_start_seconds,
                coverage_end_seconds=summary.coverage_end_seconds,
                alignment_uncertainty_upper_seconds=(
                    summary.alignment_uncertainty_upper_seconds
                ),
                interpolation_policy=summary.interpolation_policy,
            )
        )
    return tuple(artifacts)


def _initial_state_issues(
    request: FailureReplayRequest,
    adapter: SimulationAdapterManifest,
    issues: list[ReplayReadinessIssue],
) -> None:
    """Assess initial-state completeness, time, and adapter restoration."""
    if adapter.backend_type is BackendType.LOG_REPLAY:
        return
    state = request.initial_state
    if state is None:
        issues.append(
            _issue(
                ReplayIssueKind.MISSING_INITIAL_STATE,
                ReplayIssueSeverity.REVIEW,
                request.environment_id,
                "Non-log replay has no captured initial-state artifact.",
            )
        )
        return
    if state.reference_clock_id != request.reference_clock_id:
        raise ValueError("initial state uses a different reference clock")
    if state.completeness is not InitialStateCompleteness.COMPLETE:
        issues.append(
            _issue(
                ReplayIssueKind.INCOMPLETE_INITIAL_STATE,
                ReplayIssueSeverity.REVIEW,
                request.environment_id,
                f"Initial-state completeness is {state.completeness.value!r}.",
            )
        )
    age = request.interval.extraction_start_seconds - state.capture_time_seconds
    if age < 0.0 or age > request.maximum_initial_state_age_seconds:
        issues.append(
            _issue(
                ReplayIssueKind.STALE_INITIAL_STATE,
                ReplayIssueSeverity.REVIEW,
                request.environment_id,
                "Initial-state capture is after replay start or older than the "
                "declared maximum age.",
            )
        )
    if SimulationCapability.SNAPSHOT_RESTORE not in adapter.capabilities:
        issues.append(
            _issue(
                ReplayIssueKind.MISSING_ADAPTER_CAPABILITY,
                ReplayIssueSeverity.BLOCKING,
                SimulationCapability.SNAPSHOT_RESTORE.value,
                "Adapter cannot restore the declared initial-state artifact.",
            )
        )


def _stochastic_control_issues(
    request: FailureReplayRequest,
    adapter: SimulationAdapterManifest,
    issues: list[ReplayReadinessIssue],
) -> None:
    """Assess seed and random-stream controls for stochastic replay."""
    control = request.stochastic_control
    if not control.stochastic:
        return
    if control.seed is None or control.random_stream_digest is None:
        issues.append(
            _issue(
                ReplayIssueKind.MISSING_STOCHASTIC_CONTROL,
                ReplayIssueSeverity.REVIEW,
                request.scenario_id,
                "Stochastic replay lacks a seed or random-stream identity.",
            )
        )
    if (
        control.seed is not None
        and SimulationCapability.SEEDED_RESET not in adapter.capabilities
    ):
        issues.append(
            _issue(
                ReplayIssueKind.MISSING_ADAPTER_CAPABILITY,
                ReplayIssueSeverity.BLOCKING,
                SimulationCapability.SEEDED_RESET.value,
                "Adapter cannot apply the declared replay seed.",
            )
        )
    if (
        control.random_stream_digest is not None
        and SimulationCapability.RANDOM_STREAM_CONTROL not in adapter.capabilities
    ):
        issues.append(
            _issue(
                ReplayIssueKind.MISSING_ADAPTER_CAPABILITY,
                ReplayIssueSeverity.BLOCKING,
                SimulationCapability.RANDOM_STREAM_CONTROL.value,
                "Adapter cannot apply the declared random stream.",
            )
        )


def _adapter_capability_issues(
    request: FailureReplayRequest,
    adapter: SimulationAdapterManifest,
    issues: list[ReplayReadinessIssue],
) -> None:
    """Require every user-declared adapter capability."""
    missing = sorted(
        set(request.required_adapter_capabilities) - set(adapter.capabilities),
        key=lambda item: item.value,
    )
    for capability in missing:
        issues.append(
            _issue(
                ReplayIssueKind.MISSING_ADAPTER_CAPABILITY,
                ReplayIssueSeverity.BLOCKING,
                capability.value,
                "Simulation adapter lacks a required replay capability.",
            )
        )
    if SimulationCapability.DETERMINISTIC_TIME_CONTROL not in adapter.capabilities:
        issues.append(
            _issue(
                ReplayIssueKind.MISSING_ADAPTER_CAPABILITY,
                ReplayIssueSeverity.REVIEW,
                SimulationCapability.DETERMINISTIC_TIME_CONTROL.value,
                "Adapter does not guarantee deterministic replay time control.",
            )
        )


def _recording_evidence_issues(
    request: FailureReplayRequest,
    recording: Rosbag2RecordingEvidence | None,
    issues: list[ReplayReadinessIssue],
) -> None:
    """Assess optional recording-level provenance and integrity."""
    if request.recording_evidence_digest is None:
        return
    if recording is None:
        issues.append(
            _issue(
                ReplayIssueKind.RECORDING_EVIDENCE_MISSING,
                ReplayIssueSeverity.BLOCKING,
                request.episode_id,
                "Replay request names recording evidence that was not supplied.",
            )
        )
    elif recording.disposition is Rosbag2RecordingDisposition.REVIEW_REQUIRED:
        issues.append(
            _issue(
                ReplayIssueKind.RECORDING_EVIDENCE_REVIEW,
                ReplayIssueSeverity.REVIEW,
                request.episode_id,
                "Rosbag2 recording evidence contains unresolved integrity findings.",
            )
        )


def _fidelity(issues: Sequence[ReplayReadinessIssue]) -> ReplayFidelity:
    """Resolve replay fidelity from explicit blocking and review findings."""
    fidelity_issues = tuple(item for item in issues if item.affects_replay_fidelity)
    if any(item.severity is ReplayIssueSeverity.BLOCKING for item in fidelity_issues):
        return ReplayFidelity.INSUFFICIENT_EVIDENCE
    if fidelity_issues:
        return ReplayFidelity.APPROXIMATE_REPLAY_ONLY
    return ReplayFidelity.EXACT_REPLAY_READY


def _failure_claim_qualification(
    issues: Sequence[ReplayReadinessIssue],
) -> FailureClaimQualification:
    """Resolve label support independently from reconstruction fidelity."""
    claim_issues = {
        ReplayIssueKind.FAILURE_INTERVAL_UNSUPPORTED,
        ReplayIssueKind.WEAK_FAILURE_LABEL,
    }
    if any(item.kind in claim_issues for item in issues):
        return FailureClaimQualification.REVIEW_REQUIRED
    return FailureClaimQualification.SUPPORTED_BY_SELECTED_EVIDENCE


def _replay_trace_digest(
    request: FailureReplayRequest,
    episode_digest: str,
    channels: Sequence[ReplayChannelArtifact],
) -> str:
    """Bind the selected interval and ordered channel evidence into one trace."""
    return sha256_digest(
        _canonical_json(
            {
                "episode_content_digest": episode_digest,
                "extraction_end_seconds": (request.interval.extraction_end_seconds),
                "extraction_start_seconds": (request.interval.extraction_start_seconds),
                "reference_clock_id": request.reference_clock_id,
                "channels": channels,
            }
        )
    )


def _reproduction_manifest(
    request: FailureReplayRequest,
    bundle: DatasetBundle,
    episode_digest: str,
    channels: Sequence[ReplayChannelArtifact],
    trace_digest: str,
) -> ReproductionManifest:
    """Build the existing SDK reproduction contract from capsule evidence."""
    configurations = [
        *request.configuration_digests,
        NamedDigest("dataset bundle", bundle.content_digest()),
        NamedDigest("dataset episode", episode_digest),
        NamedDigest("failure replay request", request.content_digest()),
        NamedDigest("simulation adapter", request.adapter_manifest_digest),
        NamedDigest("simulation manifest", request.simulation_manifest_digest),
    ]
    if request.recording_evidence_digest is not None:
        configurations.append(
            NamedDigest("recording evidence", request.recording_evidence_digest)
        )
    if request.initial_state is not None:
        configurations.append(
            NamedDigest("initial state", request.initial_state.artifact_digest)
        )
    for channel in channels:
        if channel.trace_content_digest is not None:
            configurations.append(
                NamedDigest(
                    f"replay channel {channel.channel_id}",
                    channel.trace_content_digest,
                )
            )
    return ReproductionManifest(
        scenario_id=request.scenario_id,
        scenario_version=request.scenario_version,
        environment_id=request.environment_id,
        environment_version=request.environment_version,
        system_versions=request.system_versions,
        seed=request.stochastic_control.seed,
        perturbation_realization_digest=(request.perturbation_realization_digest),
        sdk_version=request.sdk_version,
        schema_version=FAILURE_REPLAY_CAPSULE_SCHEMA_VERSION,
        configuration_digests=tuple(configurations),
        trace_digest=trace_digest,
        command=request.command,
    )


def _issue(
    kind: ReplayIssueKind,
    severity: ReplayIssueSeverity,
    subject: str,
    description: str,
    *,
    affects_replay_fidelity: bool = True,
) -> ReplayReadinessIssue:
    """Construct one typed replay-readiness issue."""
    return ReplayReadinessIssue(
        kind=kind,
        severity=severity,
        subject=subject,
        description=description,
        affects_replay_fidelity=affects_replay_fidelity,
    )


def _finite(value: float, label: str) -> float:
    """Require a finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _nonnegative_finite(value: float, label: str) -> float:
    """Require a finite nonnegative numeric value."""
    resolved = _finite(value, label)
    if resolved < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return resolved


def _require_text(value: str, label: str) -> None:
    """Require nonempty text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _require_unique(values: Iterable[object], label: str) -> None:
    """Require a collection to contain no duplicate values."""
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be unique")


def _unique_text(
    values: Sequence[str],
    label: str,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    """Validate, deduplicate, and sort a text collection."""
    if required and not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_text(value, label)
    _require_unique(values, label)
    return tuple(sorted(value.strip() for value in values))


_INPUT_ROLES = {
    ReplayChannelRole.INITIAL_STATE,
    ReplayChannelRole.EXOGENOUS_INPUT,
    ReplayChannelRole.SYSTEM_OBSERVATION,
}

_RESERVED_CONFIGURATION_NAMES = {
    "dataset bundle",
    "dataset episode",
    "failure replay request",
    "initial state",
    "recording evidence",
    "simulation adapter",
    "simulation manifest",
}
