"""Tests for evidence-aware dataset temporal synchronization."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.dataset_reliability import (
    DatasetEvidenceRole,
    DatasetManifest,
    DatasetScope,
    EpisodeManifest,
)
from iso_obs.dataset_synchronization import (
    DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
    ChannelSynchronizationRule,
    ChannelTimingTrace,
    ClockAlignmentEvidence,
    ClockBasis,
    ClockDeclaration,
    DatasetSynchronizationPlan,
    InterpolationPolicy,
    SynchronizationDisposition,
    SynchronizationIssue,
    SynchronizationIssueKind,
    SynchronizationIssueSeverity,
    audit_dataset_synchronization,
)
from iso_obs.evidence import NamedDigest, sha256_digest


def manifest() -> DatasetManifest:
    """Build a multimodal robot dataset manifest."""
    return DatasetManifest(
        dataset_id="robot-manipulation",
        dataset_version="1",
        content_digest=sha256_digest("raw-robot-data"),
        source_uri="https://example.test/robot-data",
        license_id="Apache-2.0",
        scope=DatasetScope(
            domain_namespace="robot-manipulation",
            target_population_digest=sha256_digest("warehouse-picks"),
            collection_protocol_digest=sha256_digest("capture-protocol"),
            modality_ids=("camera", "force-torque"),
            evidence_roles=(DatasetEvidenceRole.FAILURE_LABEL_LEARNING,),
            limitations=("One robot embodiment.",),
        ),
    )


def episode(dataset: DatasetManifest) -> EpisodeManifest:
    """Build a ten-second source episode."""
    return EpisodeManifest(
        dataset_manifest_digest=dataset.manifest_digest(),
        episode_id="episode-1",
        independence_unit_id="run-1",
        content_digest=sha256_digest("episode"),
        start_seconds=0.0,
        end_seconds=10.0,
        modality_digests=(
            NamedDigest("camera", sha256_digest("camera")),
            NamedDigest("force-torque", sha256_digest("force")),
        ),
    )


def plan(dataset: DatasetManifest) -> DatasetSynchronizationPlan:
    """Build bounded camera-to-controller synchronization rules."""
    return DatasetSynchronizationPlan(
        plan_id="robot-sync",
        plan_version="1",
        dataset_manifest_digest=dataset.manifest_digest(),
        episode_id="episode-1",
        reference_clock_id="controller",
        clocks=(
            ClockDeclaration(
                clock_id="camera-clock",
                basis=ClockBasis.DEVICE_MONOTONIC,
                implementation_digest=sha256_digest("camera-clock"),
            ),
            ClockDeclaration(
                clock_id="controller",
                basis=ClockBasis.HOST_MONOTONIC,
                implementation_digest=sha256_digest("controller-clock"),
            ),
        ),
        clock_alignments=(
            ClockAlignmentEvidence(
                clock_id="camera-clock",
                reference_clock_id="controller",
                anchor_clock_seconds=0.0,
                anchor_reference_seconds=0.0,
                rate_correction_ppm=0.0,
                valid_start_seconds=0.0,
                valid_end_seconds=10.0,
                residual_offset_error_lower_seconds=-0.001,
                residual_offset_error_upper_seconds=0.001,
                residual_drift_error_lower_ppm=-10.0,
                residual_drift_error_upper_ppm=10.0,
                evidence_digest=sha256_digest("clock-calibration"),
                method="hardware synchronization pulse",
            ),
        ),
        channel_rules=(
            ChannelSynchronizationRule(
                channel_id="camera-front",
                modality_id="camera",
                required=True,
                interpolation_policy=InterpolationPolicy.NEAREST,
                maximum_alignment_error_seconds=0.01,
                maximum_sample_gap_seconds=5.1,
            ),
            ChannelSynchronizationRule(
                channel_id="wrench",
                modality_id="force-torque",
                required=True,
                interpolation_policy=InterpolationPolicy.LINEAR,
                maximum_alignment_error_seconds=0.001,
                maximum_sample_gap_seconds=5.1,
            ),
        ),
        limitations=("Payload interpolation requires a separate transform.",),
    )


def traces() -> tuple[ChannelTimingTrace, ...]:
    """Build aligned channel timing evidence."""
    return (
        ChannelTimingTrace(
            episode_id="episode-1",
            channel_id="camera-front",
            modality_id="camera",
            clock_id="camera-clock",
            content_digest=sha256_digest("camera-timing"),
            timestamps_seconds=(0.0, 5.0, 10.0),
            timestamp_uncertainty_seconds=0.0005,
        ),
        ChannelTimingTrace(
            episode_id="episode-1",
            channel_id="wrench",
            modality_id="force-torque",
            clock_id="controller",
            content_digest=sha256_digest("wrench-timing"),
            timestamps_seconds=(0.0, 5.0, 10.0),
            timestamp_uncertainty_seconds=0.0005,
        ),
    )


def test_aligned_report_is_order_invariant_and_content_addressed() -> None:
    """Trace order cannot change the supported temporal-alignment claim."""
    dataset = manifest()
    synchronization_plan = plan(dataset)
    source_episode = episode(dataset)
    timing_traces = traces()

    first = audit_dataset_synchronization(
        synchronization_plan,
        dataset,
        source_episode,
        timing_traces,
    )
    second = audit_dataset_synchronization(
        synchronization_plan,
        dataset,
        source_episode,
        tuple(reversed(timing_traces)),
    )

    assert first == second
    assert first.content_digest() == second.content_digest()
    assert first.schema_version == DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION
    assert first.disposition is SynchronizationDisposition.ALIGNED_WITHIN_SCOPE
    assert first.issues == ()


def test_missing_required_channel_abstains_and_withholds_bounds() -> None:
    """An absent required modality cannot produce zero-valued timing results."""
    dataset = manifest()

    report = audit_dataset_synchronization(
        plan(dataset),
        dataset,
        episode(dataset),
        traces()[:1],
    )

    assert report.disposition is SynchronizationDisposition.INSUFFICIENT_EVIDENCE
    wrench = next(
        item for item in report.channel_summaries if item.channel_id == "wrench"
    )
    assert wrench.sample_count == 0
    assert wrench.coverage_start_seconds is None
    assert wrench.alignment_uncertainty_upper_seconds is None


def test_non_monotonic_timestamps_are_not_silently_sorted() -> None:
    """Decreasing source timestamps remain a blocking integrity defect."""
    dataset = manifest()
    camera, wrench = traces()
    broken_camera = ChannelTimingTrace(
        episode_id=camera.episode_id,
        channel_id=camera.channel_id,
        modality_id=camera.modality_id,
        clock_id=camera.clock_id,
        content_digest=camera.content_digest,
        timestamps_seconds=(0.0, 10.0, 5.0),
        timestamp_uncertainty_seconds=camera.timestamp_uncertainty_seconds,
    )

    report = audit_dataset_synchronization(
        plan(dataset),
        dataset,
        episode(dataset),
        (broken_camera, wrench),
    )

    assert report.disposition is SynchronizationDisposition.INSUFFICIENT_EVIDENCE
    assert SynchronizationIssueKind.NON_MONOTONIC_TIMESTAMPS in {
        issue.kind for issue in report.issues
    }
    camera_summary = next(
        item for item in report.channel_summaries if item.channel_id == "camera-front"
    )
    assert camera_summary.maximum_observed_gap_seconds is None


def test_clock_uncertainty_is_propagated_and_compared_to_tolerance() -> None:
    """Offset, drift, and capture uncertainty jointly govern review."""
    dataset = manifest()
    synchronization_plan = plan(dataset)
    strict_camera = ChannelSynchronizationRule(
        channel_id="camera-front",
        modality_id="camera",
        required=True,
        interpolation_policy=InterpolationPolicy.NEAREST,
        maximum_alignment_error_seconds=0.001,
        maximum_sample_gap_seconds=5.1,
    )
    synchronization_plan = DatasetSynchronizationPlan(
        plan_id=synchronization_plan.plan_id,
        plan_version=synchronization_plan.plan_version,
        dataset_manifest_digest=synchronization_plan.dataset_manifest_digest,
        episode_id=synchronization_plan.episode_id,
        reference_clock_id=synchronization_plan.reference_clock_id,
        clocks=synchronization_plan.clocks,
        clock_alignments=synchronization_plan.clock_alignments,
        channel_rules=(strict_camera, synchronization_plan.channel_rules[1]),
        limitations=synchronization_plan.limitations,
    )

    report = audit_dataset_synchronization(
        synchronization_plan,
        dataset,
        episode(dataset),
        traces(),
    )

    assert report.disposition is SynchronizationDisposition.REVIEW_REQUIRED
    assert SynchronizationIssueKind.ALIGNMENT_UNCERTAINTY_EXCEEDS_TOLERANCE in {
        issue.kind for issue in report.issues
    }
    camera = next(
        item for item in report.channel_summaries if item.channel_id == "camera-front"
    )
    assert camera.alignment_uncertainty_upper_seconds == 0.0016


def test_unknown_clock_basis_forces_abstention() -> None:
    """A named but scientifically unknown clock cannot anchor precedence."""
    dataset = manifest()
    synchronization_plan = plan(dataset)
    unknown_camera = ClockDeclaration(
        clock_id="camera-clock",
        basis=ClockBasis.UNKNOWN,
        implementation_digest=sha256_digest("camera-clock"),
    )
    synchronization_plan = DatasetSynchronizationPlan(
        plan_id=synchronization_plan.plan_id,
        plan_version=synchronization_plan.plan_version,
        dataset_manifest_digest=synchronization_plan.dataset_manifest_digest,
        episode_id=synchronization_plan.episode_id,
        reference_clock_id=synchronization_plan.reference_clock_id,
        clocks=(unknown_camera, synchronization_plan.clocks[1]),
        clock_alignments=synchronization_plan.clock_alignments,
        channel_rules=synchronization_plan.channel_rules,
        limitations=synchronization_plan.limitations,
    )

    report = audit_dataset_synchronization(
        synchronization_plan,
        dataset,
        episode(dataset),
        traces(),
    )

    assert report.disposition is SynchronizationDisposition.INSUFFICIENT_EVIDENCE
    assert SynchronizationIssueKind.UNKNOWN_CLOCK_BASIS in {
        issue.kind for issue in report.issues
    }


def test_incomplete_coverage_and_large_gap_require_review() -> None:
    """Sparse partial observations cannot appear fully synchronized."""
    dataset = manifest()
    camera, wrench = traces()
    sparse_camera = ChannelTimingTrace(
        episode_id=camera.episode_id,
        channel_id=camera.channel_id,
        modality_id=camera.modality_id,
        clock_id=camera.clock_id,
        content_digest=camera.content_digest,
        timestamps_seconds=(1.0, 9.0),
        timestamp_uncertainty_seconds=camera.timestamp_uncertainty_seconds,
    )

    report = audit_dataset_synchronization(
        plan(dataset),
        dataset,
        episode(dataset),
        (sparse_camera, wrench),
    )

    assert report.disposition is SynchronizationDisposition.REVIEW_REQUIRED
    kinds = {issue.kind for issue in report.issues}
    assert SynchronizationIssueKind.INCOMPLETE_EPISODE_COVERAGE in kinds
    assert SynchronizationIssueKind.SAMPLE_GAP_EXCEEDS_TOLERANCE in kinds


def test_clock_alignment_validity_must_cover_episode() -> None:
    """Calibration evidence outside part of the episode cannot be extrapolated."""
    dataset = manifest()
    synchronization_plan = plan(dataset)
    short_alignment = ClockAlignmentEvidence(
        clock_id="camera-clock",
        reference_clock_id="controller",
        anchor_clock_seconds=0.0,
        anchor_reference_seconds=0.0,
        rate_correction_ppm=0.0,
        valid_start_seconds=0.0,
        valid_end_seconds=8.0,
        residual_offset_error_lower_seconds=-0.001,
        residual_offset_error_upper_seconds=0.001,
        residual_drift_error_lower_ppm=-10.0,
        residual_drift_error_upper_ppm=10.0,
        evidence_digest=sha256_digest("short-calibration"),
        method="partial synchronization pulse record",
    )
    synchronization_plan = DatasetSynchronizationPlan(
        plan_id=synchronization_plan.plan_id,
        plan_version=synchronization_plan.plan_version,
        dataset_manifest_digest=synchronization_plan.dataset_manifest_digest,
        episode_id=synchronization_plan.episode_id,
        reference_clock_id=synchronization_plan.reference_clock_id,
        clocks=synchronization_plan.clocks,
        clock_alignments=(short_alignment,),
        channel_rules=synchronization_plan.channel_rules,
        limitations=synchronization_plan.limitations,
    )

    report = audit_dataset_synchronization(
        synchronization_plan,
        dataset,
        episode(dataset),
        traces(),
    )

    assert report.disposition is SynchronizationDisposition.INSUFFICIENT_EVIDENCE
    assert SynchronizationIssueKind.ALIGNMENT_OUTSIDE_VALIDITY_INTERVAL in {
        issue.kind for issue in report.issues
    }


def test_clock_mapping_precedes_coverage_audit() -> None:
    """Coverage is evaluated in reference time, not raw device coordinates."""
    dataset = manifest()
    synchronization_plan = plan(dataset)
    shifted_alignment = ClockAlignmentEvidence(
        clock_id="camera-clock",
        reference_clock_id="controller",
        anchor_clock_seconds=100.0,
        anchor_reference_seconds=0.0,
        rate_correction_ppm=0.0,
        valid_start_seconds=0.0,
        valid_end_seconds=10.0,
        residual_offset_error_lower_seconds=-0.001,
        residual_offset_error_upper_seconds=0.001,
        residual_drift_error_lower_ppm=-10.0,
        residual_drift_error_upper_ppm=10.0,
        evidence_digest=sha256_digest("shifted-clock-calibration"),
        method="hardware synchronization pulse",
    )
    synchronization_plan = DatasetSynchronizationPlan(
        plan_id=synchronization_plan.plan_id,
        plan_version=synchronization_plan.plan_version,
        dataset_manifest_digest=synchronization_plan.dataset_manifest_digest,
        episode_id=synchronization_plan.episode_id,
        reference_clock_id=synchronization_plan.reference_clock_id,
        clocks=synchronization_plan.clocks,
        clock_alignments=(shifted_alignment,),
        channel_rules=synchronization_plan.channel_rules,
        limitations=synchronization_plan.limitations,
    )
    camera, wrench = traces()
    shifted_camera = ChannelTimingTrace(
        episode_id=camera.episode_id,
        channel_id=camera.channel_id,
        modality_id=camera.modality_id,
        clock_id=camera.clock_id,
        content_digest=camera.content_digest,
        timestamps_seconds=(100.0, 105.0, 110.0),
        timestamp_uncertainty_seconds=camera.timestamp_uncertainty_seconds,
    )

    report = audit_dataset_synchronization(
        synchronization_plan,
        dataset,
        episode(dataset),
        (shifted_camera, wrench),
    )

    assert report.disposition is SynchronizationDisposition.ALIGNED_WITHIN_SCOPE
    camera_summary = next(
        item for item in report.channel_summaries if item.channel_id == "camera-front"
    )
    assert camera_summary.coverage_start_seconds == 0.0
    assert camera_summary.coverage_end_seconds == 10.0


def test_report_disposition_must_match_issue_structure() -> None:
    """An aligned artifact cannot be forged without channel evidence."""
    dataset = manifest()
    report = audit_dataset_synchronization(
        plan(dataset),
        dataset,
        episode(dataset),
        traces(),
    )

    with pytest.raises(ValueError, match="summaries and no issues"):
        replace(
            report,
            channel_summaries=(),
        )
    with pytest.raises(ValueError, match="blocking issue"):
        replace(
            report,
            disposition=SynchronizationDisposition.INSUFFICIENT_EVIDENCE,
            issues=(
                SynchronizationIssue(
                    kind=SynchronizationIssueKind.DUPLICATE_TIMESTAMPS,
                    severity=SynchronizationIssueSeverity.REVIEW,
                    channel_id="camera-front",
                    description="Review-only issue.",
                ),
            ),
        )
