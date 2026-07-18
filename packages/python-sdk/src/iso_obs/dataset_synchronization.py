"""Evidence-aware ingestion and temporal synchronization for datasets.

Adapters expose immutable manifests and channel timing traces. The audit
determines whether required modalities can be aligned within preregistered
tolerances while preserving clock uncertainty, gaps, and ordering defects.
It never silently sorts samples, imputes a clock model, or performs payload
interpolation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .dataset_reliability import DatasetManifest, DatasetScope, EpisodeManifest
from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse

DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION = (
    "iso-obs.dataset-synchronization-report.v1"
)


class ClockBasis(StrEnum):
    """Declared basis of a channel timestamp clock."""

    DEVICE_MONOTONIC = "device_monotonic"
    HOST_MONOTONIC = "host_monotonic"
    UTC = "utc"
    SIMULATOR_TIME = "simulator_time"
    UNKNOWN = "unknown"


class InterpolationPolicy(StrEnum):
    """Declared payload-alignment policy for a channel."""

    NONE = "none"
    NEAREST = "nearest"
    LINEAR = "linear"
    ZERO_ORDER_HOLD = "zero_order_hold"
    EVENT_WINDOW = "event_window"


class SynchronizationIssueSeverity(StrEnum):
    """Effect of a synchronization issue on downstream use."""

    REVIEW = "review"
    BLOCKING = "blocking"


class SynchronizationIssueKind(StrEnum):
    """Kind of temporal-integrity problem found in a channel."""

    MISSING_REQUIRED_CHANNEL = "missing_required_channel"
    EMPTY_REQUIRED_CHANNEL = "empty_required_channel"
    UNKNOWN_CLOCK_BASIS = "unknown_clock_basis"
    MISSING_CLOCK_ALIGNMENT = "missing_clock_alignment"
    ALIGNMENT_OUTSIDE_VALIDITY_INTERVAL = "alignment_outside_validity_interval"
    ALIGNMENT_UNCERTAINTY_EXCEEDS_TOLERANCE = "alignment_uncertainty_exceeds_tolerance"
    NON_MONOTONIC_TIMESTAMPS = "non_monotonic_timestamps"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    INCOMPLETE_EPISODE_COVERAGE = "incomplete_episode_coverage"
    SAMPLE_GAP_EXCEEDS_TOLERANCE = "sample_gap_exceeds_tolerance"


class SynchronizationDisposition(StrEnum):
    """Strongest temporal-alignment claim supported by an audit."""

    ALIGNED_WITHIN_SCOPE = "aligned_within_scope"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class ClockDeclaration:
    """Identity and basis of one acquisition clock."""

    clock_id: str
    basis: ClockBasis
    implementation_digest: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate clock identity, implementation, and limitations."""
        _require_text(self.clock_id, "clock ID")
        NamedDigest("clock implementation", self.implementation_digest)
        limitations = _unique_text(
            self.limitations,
            "clock limitations",
            required=False,
        )
        object.__setattr__(self, "basis", ClockBasis(self.basis))
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class ClockAlignmentEvidence:
    """Bounded residual timing error relative to a reference clock."""

    clock_id: str
    reference_clock_id: str
    anchor_clock_seconds: float
    anchor_reference_seconds: float
    rate_correction_ppm: float
    valid_start_seconds: float
    valid_end_seconds: float
    residual_offset_error_lower_seconds: float
    residual_offset_error_upper_seconds: float
    residual_drift_error_lower_ppm: float
    residual_drift_error_upper_ppm: float
    evidence_digest: str
    method: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate clock mapping, validity interval, and error bounds."""
        _require_text(self.clock_id, "aligned clock ID")
        _require_text(self.reference_clock_id, "reference clock ID")
        if self.clock_id == self.reference_clock_id:
            raise ValueError("reference clock must not align to itself")
        anchor_clock = _finite(
            self.anchor_clock_seconds,
            "clock alignment channel anchor",
        )
        anchor_reference = _finite(
            self.anchor_reference_seconds,
            "clock alignment reference anchor",
        )
        rate_correction = _finite(
            self.rate_correction_ppm,
            "clock alignment rate correction",
        )
        if rate_correction <= -1_000_000.0:
            raise ValueError("clock rate correction must preserve temporal order")
        valid_start, valid_end = _closed_interval(
            self.valid_start_seconds,
            self.valid_end_seconds,
            "clock alignment validity",
        )
        offset_lower, offset_upper = _closed_interval(
            self.residual_offset_error_lower_seconds,
            self.residual_offset_error_upper_seconds,
            "residual offset error",
        )
        drift_lower, drift_upper = _closed_interval(
            self.residual_drift_error_lower_ppm,
            self.residual_drift_error_upper_ppm,
            "residual drift error",
        )
        NamedDigest("clock alignment evidence", self.evidence_digest)
        _require_text(self.method, "clock alignment method")
        limitations = _unique_text(
            self.limitations,
            "clock alignment limitations",
            required=False,
        )
        object.__setattr__(self, "anchor_clock_seconds", anchor_clock)
        object.__setattr__(self, "anchor_reference_seconds", anchor_reference)
        object.__setattr__(self, "rate_correction_ppm", rate_correction)
        object.__setattr__(self, "valid_start_seconds", valid_start)
        object.__setattr__(self, "valid_end_seconds", valid_end)
        object.__setattr__(
            self,
            "residual_offset_error_lower_seconds",
            offset_lower,
        )
        object.__setattr__(
            self,
            "residual_offset_error_upper_seconds",
            offset_upper,
        )
        object.__setattr__(
            self,
            "residual_drift_error_lower_ppm",
            drift_lower,
        )
        object.__setattr__(
            self,
            "residual_drift_error_upper_ppm",
            drift_upper,
        )
        object.__setattr__(self, "limitations", limitations)

    def uncertainty_upper_seconds(self, duration_seconds: float) -> float:
        """Calculate a conservative absolute timing-error upper bound.

        Args:
            duration_seconds: Episode duration over which drift accumulates.

        Returns:
            Sum of the largest residual offset and drift error magnitudes.
        """
        duration = _nonnegative_finite(
            duration_seconds,
            "clock alignment duration",
        )
        offset = max(
            abs(self.residual_offset_error_lower_seconds),
            abs(self.residual_offset_error_upper_seconds),
        )
        drift_ppm = max(
            abs(self.residual_drift_error_lower_ppm),
            abs(self.residual_drift_error_upper_ppm),
        )
        return offset + duration * drift_ppm * 1e-6

    def to_reference_seconds(self, timestamp_seconds: float) -> float:
        """Map one source timestamp into nominal reference-clock coordinates.

        Args:
            timestamp_seconds: Source timestamp expressed on ``clock_id``.

        Returns:
            Nominal timestamp on ``reference_clock_id``. Residual uncertainty
            remains described by :meth:`uncertainty_upper_seconds`.
        """
        timestamp = _finite(timestamp_seconds, "source clock timestamp")
        scale = 1.0 + self.rate_correction_ppm * 1e-6
        return (
            self.anchor_reference_seconds
            + (timestamp - self.anchor_clock_seconds) * scale
        )


@dataclass(frozen=True, slots=True)
class ChannelTimingTrace:
    """Timestamp evidence for one modality channel in one episode."""

    episode_id: str
    channel_id: str
    modality_id: str
    clock_id: str
    content_digest: str
    timestamps_seconds: tuple[float, ...]
    timestamp_uncertainty_seconds: float

    def __post_init__(self) -> None:
        """Validate channel identity and preserve timestamp source ordering."""
        _require_text(self.episode_id, "timing trace episode ID")
        _require_text(self.channel_id, "timing trace channel ID")
        _require_text(self.modality_id, "timing trace modality ID")
        _require_text(self.clock_id, "timing trace clock ID")
        NamedDigest("channel timing content", self.content_digest)
        timestamps = tuple(
            _finite(value, "channel timestamp") for value in self.timestamps_seconds
        )
        uncertainty = _nonnegative_finite(
            self.timestamp_uncertainty_seconds,
            "channel timestamp uncertainty",
        )
        object.__setattr__(self, "timestamps_seconds", timestamps)
        object.__setattr__(
            self,
            "timestamp_uncertainty_seconds",
            uncertainty,
        )


@dataclass(frozen=True, slots=True)
class ChannelSynchronizationRule:
    """Preregistered timing requirements for one modality channel."""

    channel_id: str
    modality_id: str
    required: bool
    interpolation_policy: InterpolationPolicy
    maximum_alignment_error_seconds: float
    maximum_sample_gap_seconds: float
    allow_extrapolation: bool = False

    def __post_init__(self) -> None:
        """Validate required-channel and timing-tolerance declarations."""
        _require_text(self.channel_id, "synchronization rule channel ID")
        _require_text(self.modality_id, "synchronization rule modality ID")
        if not isinstance(self.required, bool):
            raise ValueError("synchronization rule required must be boolean")
        if not isinstance(self.allow_extrapolation, bool):
            raise ValueError("allow_extrapolation must be boolean")
        alignment_error = _positive_finite(
            self.maximum_alignment_error_seconds,
            "maximum alignment error",
        )
        sample_gap = _positive_finite(
            self.maximum_sample_gap_seconds,
            "maximum sample gap",
        )
        object.__setattr__(
            self,
            "interpolation_policy",
            InterpolationPolicy(self.interpolation_policy),
        )
        object.__setattr__(
            self,
            "maximum_alignment_error_seconds",
            alignment_error,
        )
        object.__setattr__(self, "maximum_sample_gap_seconds", sample_gap)


@dataclass(frozen=True, slots=True)
class DatasetSynchronizationPlan:
    """Content-addressed temporal-alignment design for one episode."""

    plan_id: str
    plan_version: str
    dataset_manifest_digest: str
    episode_id: str
    reference_clock_id: str
    clocks: tuple[ClockDeclaration, ...]
    clock_alignments: tuple[ClockAlignmentEvidence, ...]
    channel_rules: tuple[ChannelSynchronizationRule, ...]
    limitations: tuple[str, ...] = ()
    evidence_use: EvidenceUse = EvidenceUse.DISCOVERY_ONLY

    def __post_init__(self) -> None:
        """Validate reference clock, alignment graph, and channel rules."""
        _require_text(self.plan_id, "synchronization plan ID")
        _require_text(self.plan_version, "synchronization plan version")
        NamedDigest("dataset manifest", self.dataset_manifest_digest)
        _require_text(self.episode_id, "synchronization plan episode ID")
        _require_text(self.reference_clock_id, "reference clock ID")
        clocks = tuple(sorted(self.clocks, key=lambda item: item.clock_id))
        _require_unique((item.clock_id for item in clocks), "clock IDs")
        if self.reference_clock_id not in {item.clock_id for item in clocks}:
            raise ValueError("reference clock must be declared")
        alignments = tuple(
            sorted(self.clock_alignments, key=lambda item: item.clock_id)
        )
        _require_unique(
            (item.clock_id for item in alignments),
            "clock alignment source IDs",
        )
        declared_clock_ids = {item.clock_id for item in clocks}
        for alignment in alignments:
            if alignment.clock_id not in declared_clock_ids:
                raise ValueError("clock alignment references an undeclared clock")
            if alignment.reference_clock_id != self.reference_clock_id:
                raise ValueError("clock alignment uses a different reference clock")
        rules = tuple(sorted(self.channel_rules, key=lambda item: item.channel_id))
        if not rules:
            raise ValueError("synchronization plan requires channel rules")
        _require_unique((item.channel_id for item in rules), "channel rule IDs")
        limitations = _unique_text(
            self.limitations,
            "synchronization plan limitations",
            required=False,
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("dataset synchronization audit must be discovery-only")
        object.__setattr__(self, "clocks", clocks)
        object.__setattr__(self, "clock_alignments", alignments)
        object.__setattr__(self, "channel_rules", rules)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_use", evidence_use)

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact plan content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class ChannelSynchronizationSummary:
    """Audited timing coverage and uncertainty for one channel."""

    channel_id: str
    modality_id: str
    clock_id: str | None
    sample_count: int
    coverage_start_seconds: float | None
    coverage_end_seconds: float | None
    maximum_observed_gap_seconds: float | None
    alignment_uncertainty_upper_seconds: float | None
    interpolation_policy: InterpolationPolicy

    def __post_init__(self) -> None:
        """Validate nullable-until-supported channel timing results."""
        _require_text(self.channel_id, "channel summary ID")
        _require_text(self.modality_id, "channel summary modality ID")
        if self.clock_id is not None:
            _require_text(self.clock_id, "channel summary clock ID")
        if self.sample_count < 0:
            raise ValueError("channel summary sample count must be nonnegative")
        for value, label in (
            (self.coverage_start_seconds, "channel coverage start"),
            (self.coverage_end_seconds, "channel coverage end"),
            (self.maximum_observed_gap_seconds, "maximum observed gap"),
            (
                self.alignment_uncertainty_upper_seconds,
                "alignment uncertainty upper bound",
            ),
        ):
            if value is not None:
                _finite(value, label)
        if (self.coverage_start_seconds is None) != (self.coverage_end_seconds is None):
            raise ValueError("channel coverage bounds must be supplied together")
        if (
            self.coverage_start_seconds is not None
            and self.coverage_end_seconds is not None
            and self.coverage_start_seconds > self.coverage_end_seconds
        ):
            raise ValueError("channel coverage bounds must be ordered")
        if (
            self.maximum_observed_gap_seconds is not None
            and self.maximum_observed_gap_seconds < 0.0
        ):
            raise ValueError("maximum observed gap must be nonnegative")
        if (
            self.alignment_uncertainty_upper_seconds is not None
            and self.alignment_uncertainty_upper_seconds < 0.0
        ):
            raise ValueError("alignment uncertainty must be nonnegative")
        object.__setattr__(
            self,
            "interpolation_policy",
            InterpolationPolicy(self.interpolation_policy),
        )


@dataclass(frozen=True, slots=True)
class SynchronizationIssue:
    """One channel-specific temporal-integrity limitation."""

    kind: SynchronizationIssueKind
    severity: SynchronizationIssueSeverity
    channel_id: str
    description: str

    def __post_init__(self) -> None:
        """Validate issue classification and affected channel."""
        _require_text(self.channel_id, "synchronization issue channel ID")
        _require_text(self.description, "synchronization issue description")
        object.__setattr__(self, "kind", SynchronizationIssueKind(self.kind))
        object.__setattr__(
            self,
            "severity",
            SynchronizationIssueSeverity(self.severity),
        )


@dataclass(frozen=True, slots=True)
class DatasetSynchronizationReport:
    """Versioned artifact describing support for temporal alignment."""

    schema_version: str
    plan_content_digest: str
    dataset_manifest_digest: str
    episode_id: str
    disposition: SynchronizationDisposition
    reference_clock_id: str
    channel_summaries: tuple[ChannelSynchronizationSummary, ...]
    issues: tuple[SynchronizationIssue, ...]
    scope: DatasetScope
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate report identity, canonical order, and scientific scope."""
        if self.schema_version != DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported dataset synchronization schema version")
        NamedDigest("synchronization plan", self.plan_content_digest)
        NamedDigest("dataset manifest", self.dataset_manifest_digest)
        _require_text(self.episode_id, "synchronization report episode ID")
        _require_text(self.reference_clock_id, "report reference clock ID")
        summaries = tuple(
            sorted(self.channel_summaries, key=lambda item: item.channel_id)
        )
        _require_unique(
            (item.channel_id for item in summaries),
            "channel summary IDs",
        )
        issues = tuple(
            sorted(
                self.issues,
                key=lambda item: (
                    item.severity.value,
                    item.channel_id,
                    item.kind.value,
                ),
            )
        )
        disposition = SynchronizationDisposition(self.disposition)
        has_blocking_issue = any(
            item.severity is SynchronizationIssueSeverity.BLOCKING for item in issues
        )
        if disposition is SynchronizationDisposition.ALIGNED_WITHIN_SCOPE:
            if not summaries or issues:
                raise ValueError(
                    "aligned synchronization report requires summaries and no issues"
                )
        elif disposition is SynchronizationDisposition.REVIEW_REQUIRED:
            if not issues or has_blocking_issue:
                raise ValueError(
                    "review-required synchronization report needs nonblocking issues"
                )
        elif not has_blocking_issue:
            raise ValueError(
                "insufficient synchronization report requires a blocking issue"
            )
        limitations = _unique_text(
            self.limitations,
            "synchronization report limitations",
            required=False,
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("synchronization report must be discovery-only")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "channel_summaries", summaries)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact report content digest."""
        return sha256_digest(self.to_json())


class DatasetAdapter(ABC):
    """Adapter exposing dataset provenance and timing evidence to the SDK."""

    @abstractmethod
    def dataset_manifest(self) -> DatasetManifest:
        """Return the immutable source dataset manifest."""

    @abstractmethod
    def episode_manifests(self) -> Sequence[EpisodeManifest]:
        """Return independently identified source episodes."""

    @abstractmethod
    def timing_traces(self, episode_id: str) -> Sequence[ChannelTimingTrace]:
        """Return channel timestamps in unmodified source order.

        Args:
            episode_id: Episode whose acquisition timing should be audited.

        Returns:
            Timestamp traces without implicit sorting or interpolation.
        """


def audit_dataset_synchronization(
    plan: DatasetSynchronizationPlan,
    manifest: DatasetManifest,
    episode: EpisodeManifest,
    traces: Sequence[ChannelTimingTrace],
) -> DatasetSynchronizationReport:
    """Audit whether episode channels can be aligned within declared bounds.

    Args:
        plan: Preregistered clocks, alignments, channel rules, and tolerances.
        manifest: Immutable identity and scientific scope of the source dataset.
        episode: Temporal episode being aligned.
        traces: Per-channel timestamps in original acquisition order.

    Returns:
        A scope-bound report with nullable results when evidence is insufficient.

    Raises:
        ValueError: If dataset, episode, channel, or modality identities conflict.
    """
    manifest_digest = manifest.manifest_digest()
    if plan.dataset_manifest_digest != manifest_digest:
        raise ValueError("synchronization plan targets a different dataset manifest")
    if episode.dataset_manifest_digest != manifest_digest:
        raise ValueError("episode references a different dataset manifest")
    if plan.episode_id != episode.episode_id:
        raise ValueError("synchronization plan targets a different episode")

    ordered_traces = tuple(sorted(traces, key=lambda item: item.channel_id))
    _require_unique(
        (trace.channel_id for trace in ordered_traces),
        "channel timing trace IDs",
    )
    trace_by_id = {trace.channel_id: trace for trace in ordered_traces}
    declared_modalities = set(manifest.scope.modality_ids)
    for trace in ordered_traces:
        if trace.episode_id != episode.episode_id:
            raise ValueError("channel timing trace references a different episode")
        if trace.modality_id not in declared_modalities:
            raise ValueError("channel timing trace uses an undeclared modality")

    clocks = {clock.clock_id: clock for clock in plan.clocks}
    alignments = {alignment.clock_id: alignment for alignment in plan.clock_alignments}
    summaries: list[ChannelSynchronizationSummary] = []
    issues: list[SynchronizationIssue] = []
    duration = episode.end_seconds - episode.start_seconds

    for rule in plan.channel_rules:
        if rule.modality_id not in declared_modalities:
            raise ValueError("channel rule uses an undeclared modality")
        current_trace = trace_by_id.get(rule.channel_id)
        if current_trace is None:
            summaries.append(_missing_summary(rule))
            if rule.required:
                issues.append(
                    _issue(
                        SynchronizationIssueKind.MISSING_REQUIRED_CHANNEL,
                        SynchronizationIssueSeverity.BLOCKING,
                        rule.channel_id,
                        "Required channel has no timing trace.",
                    )
                )
            continue
        if current_trace.modality_id != rule.modality_id:
            raise ValueError("channel rule and timing trace modalities differ")
        summary, trace_issues = _audit_channel(
            plan,
            episode,
            rule,
            current_trace,
            clocks,
            alignments,
            duration,
        )
        summaries.append(summary)
        issues.extend(trace_issues)

    blocking = any(
        issue.severity is SynchronizationIssueSeverity.BLOCKING for issue in issues
    )
    if blocking:
        disposition = SynchronizationDisposition.INSUFFICIENT_EVIDENCE
    elif issues:
        disposition = SynchronizationDisposition.REVIEW_REQUIRED
    else:
        disposition = SynchronizationDisposition.ALIGNED_WITHIN_SCOPE
    limitations = _unique_text(
        (
            *manifest.scope.limitations,
            *plan.limitations,
            "Synchronization support does not establish causal ordering below "
            "the reported timing-uncertainty bounds.",
            "The report audits timestamps only and does not interpolate payloads.",
        ),
        "synchronization report limitations",
    )
    return DatasetSynchronizationReport(
        schema_version=DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
        plan_content_digest=plan.content_digest(),
        dataset_manifest_digest=manifest_digest,
        episode_id=episode.episode_id,
        disposition=disposition,
        reference_clock_id=plan.reference_clock_id,
        channel_summaries=tuple(summaries),
        issues=tuple(issues),
        scope=manifest.scope,
        evidence_use=plan.evidence_use,
        limitations=limitations,
    )


def _audit_channel(
    plan: DatasetSynchronizationPlan,
    episode: EpisodeManifest,
    rule: ChannelSynchronizationRule,
    trace: ChannelTimingTrace,
    clocks: dict[str, ClockDeclaration],
    alignments: dict[str, ClockAlignmentEvidence],
    duration_seconds: float,
) -> tuple[ChannelSynchronizationSummary, tuple[SynchronizationIssue, ...]]:
    """Audit ordering, coverage, gaps, and clock evidence for one channel."""
    issues: list[SynchronizationIssue] = []
    timestamps = trace.timestamps_seconds
    if not timestamps:
        if rule.required:
            issues.append(
                _issue(
                    SynchronizationIssueKind.EMPTY_REQUIRED_CHANNEL,
                    SynchronizationIssueSeverity.BLOCKING,
                    rule.channel_id,
                    "Required channel contains no timestamps.",
                )
            )
        return _empty_summary(rule, trace.clock_id), tuple(issues)

    strictly_increasing = all(
        right > left for left, right in zip(timestamps, timestamps[1:], strict=False)
    )
    nondecreasing = all(
        right >= left for left, right in zip(timestamps, timestamps[1:], strict=False)
    )
    if not nondecreasing:
        issues.append(
            _issue(
                SynchronizationIssueKind.NON_MONOTONIC_TIMESTAMPS,
                SynchronizationIssueSeverity.BLOCKING,
                rule.channel_id,
                "Source timestamps decrease; the SDK did not reorder them.",
            )
        )
    elif not strictly_increasing:
        issues.append(
            _issue(
                SynchronizationIssueKind.DUPLICATE_TIMESTAMPS,
                SynchronizationIssueSeverity.REVIEW,
                rule.channel_id,
                "Source timestamps contain duplicates.",
            )
        )

    aligned_timestamps, alignment_uncertainty = _alignment_result(
        plan,
        episode,
        rule,
        trace,
        clocks,
        alignments,
        duration_seconds,
        issues,
    )
    coverage_start = min(aligned_timestamps) if aligned_timestamps is not None else None
    coverage_end = max(aligned_timestamps) if aligned_timestamps is not None else None
    maximum_gap = (
        max(
            (
                right - left
                for left, right in zip(
                    aligned_timestamps,
                    aligned_timestamps[1:],
                    strict=False,
                )
            ),
            default=0.0,
        )
        if aligned_timestamps is not None and nondecreasing
        else None
    )
    if maximum_gap is not None and maximum_gap > rule.maximum_sample_gap_seconds:
        issues.append(
            _issue(
                SynchronizationIssueKind.SAMPLE_GAP_EXCEEDS_TOLERANCE,
                SynchronizationIssueSeverity.REVIEW,
                rule.channel_id,
                "Observed sample gap exceeds the preregistered channel tolerance.",
            )
        )
    if (
        coverage_start is not None
        and coverage_end is not None
        and not rule.allow_extrapolation
        and (
            coverage_start > episode.start_seconds or coverage_end < episode.end_seconds
        )
    ):
        issues.append(
            _issue(
                SynchronizationIssueKind.INCOMPLETE_EPISODE_COVERAGE,
                SynchronizationIssueSeverity.REVIEW,
                rule.channel_id,
                "Aligned timestamps do not cover the complete episode interval.",
            )
        )
    if (
        alignment_uncertainty is not None
        and alignment_uncertainty > rule.maximum_alignment_error_seconds
    ):
        issues.append(
            _issue(
                SynchronizationIssueKind.ALIGNMENT_UNCERTAINTY_EXCEEDS_TOLERANCE,
                SynchronizationIssueSeverity.REVIEW,
                rule.channel_id,
                "Clock and timestamp uncertainty exceed the alignment tolerance.",
            )
        )

    return (
        ChannelSynchronizationSummary(
            channel_id=rule.channel_id,
            modality_id=rule.modality_id,
            clock_id=trace.clock_id,
            sample_count=len(timestamps),
            coverage_start_seconds=coverage_start,
            coverage_end_seconds=coverage_end,
            maximum_observed_gap_seconds=maximum_gap,
            alignment_uncertainty_upper_seconds=alignment_uncertainty,
            interpolation_policy=rule.interpolation_policy,
        ),
        tuple(issues),
    )


def _alignment_result(
    plan: DatasetSynchronizationPlan,
    episode: EpisodeManifest,
    rule: ChannelSynchronizationRule,
    trace: ChannelTimingTrace,
    clocks: dict[str, ClockDeclaration],
    alignments: dict[str, ClockAlignmentEvidence],
    duration_seconds: float,
    issues: list[SynchronizationIssue],
) -> tuple[tuple[float, ...] | None, float | None]:
    """Return mapped timestamps and timing uncertainty or append blockers."""
    clock = clocks.get(trace.clock_id)
    if clock is None or clock.basis is ClockBasis.UNKNOWN:
        issues.append(
            _issue(
                SynchronizationIssueKind.UNKNOWN_CLOCK_BASIS,
                SynchronizationIssueSeverity.BLOCKING,
                rule.channel_id,
                "Channel clock is undeclared or has an unknown basis.",
            )
        )
        return None, None
    if trace.clock_id == plan.reference_clock_id:
        return trace.timestamps_seconds, trace.timestamp_uncertainty_seconds
    alignment = alignments.get(trace.clock_id)
    if alignment is None:
        issues.append(
            _issue(
                SynchronizationIssueKind.MISSING_CLOCK_ALIGNMENT,
                SynchronizationIssueSeverity.BLOCKING,
                rule.channel_id,
                "No evidence maps this channel clock to the reference clock.",
            )
        )
        return None, None
    if (
        alignment.valid_start_seconds > episode.start_seconds
        or alignment.valid_end_seconds < episode.end_seconds
    ):
        issues.append(
            _issue(
                SynchronizationIssueKind.ALIGNMENT_OUTSIDE_VALIDITY_INTERVAL,
                SynchronizationIssueSeverity.BLOCKING,
                rule.channel_id,
                "Clock alignment evidence does not cover the complete episode.",
            )
        )
        return None, None
    return (
        tuple(
            alignment.to_reference_seconds(timestamp)
            for timestamp in trace.timestamps_seconds
        ),
        trace.timestamp_uncertainty_seconds
        + alignment.uncertainty_upper_seconds(duration_seconds),
    )


def _missing_summary(
    rule: ChannelSynchronizationRule,
) -> ChannelSynchronizationSummary:
    """Build a semantically withheld summary for an absent channel."""
    return ChannelSynchronizationSummary(
        channel_id=rule.channel_id,
        modality_id=rule.modality_id,
        clock_id=None,
        sample_count=0,
        coverage_start_seconds=None,
        coverage_end_seconds=None,
        maximum_observed_gap_seconds=None,
        alignment_uncertainty_upper_seconds=None,
        interpolation_policy=rule.interpolation_policy,
    )


def _empty_summary(
    rule: ChannelSynchronizationRule,
    clock_id: str,
) -> ChannelSynchronizationSummary:
    """Build a semantically withheld summary for an empty timing trace."""
    return ChannelSynchronizationSummary(
        channel_id=rule.channel_id,
        modality_id=rule.modality_id,
        clock_id=clock_id,
        sample_count=0,
        coverage_start_seconds=None,
        coverage_end_seconds=None,
        maximum_observed_gap_seconds=None,
        alignment_uncertainty_upper_seconds=None,
        interpolation_policy=rule.interpolation_policy,
    )


def _issue(
    kind: SynchronizationIssueKind,
    severity: SynchronizationIssueSeverity,
    channel_id: str,
    description: str,
) -> SynchronizationIssue:
    """Build one validated synchronization issue."""
    return SynchronizationIssue(
        kind=kind,
        severity=severity,
        channel_id=channel_id,
        description=description,
    )


def _closed_interval(
    lower: float,
    upper: float,
    label: str,
) -> tuple[float, float]:
    """Validate a finite closed interval."""
    lower_value = _finite(lower, f"{label} lower bound")
    upper_value = _finite(upper, f"{label} upper bound")
    if lower_value > upper_value:
        raise ValueError(f"{label} lower bound must not exceed upper")
    return lower_value, upper_value


def _finite(value: float, label: str) -> float:
    """Validate and return a finite number."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_finite(value: float, label: str) -> float:
    """Validate and return a finite positive number."""
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_finite(value: float, label: str) -> float:
    """Validate and return a finite nonnegative number."""
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _require_text(value: str, label: str) -> str:
    """Validate a required nonempty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _unique_text(
    values: Iterable[str],
    label: str,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    """Validate, deduplicate, and sort a collection of strings."""
    normalized = tuple(_require_text(value, label) for value in values)
    if required and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(normalized))


def _require_unique(values: Iterable[str], label: str) -> None:
    """Require a collection of identifiers to contain no duplicates."""
    collected = tuple(values)
    if len(set(collected)) != len(collected):
        raise ValueError(f"{label} must be unique")
