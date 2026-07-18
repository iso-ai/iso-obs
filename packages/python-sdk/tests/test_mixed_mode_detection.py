"""Tests for calibrated mixed-mode change-point contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.dataset_reliability import (
    DatasetEvidenceRole,
    DatasetScope,
    LabelStatus,
    ModeFamily,
)
from iso_obs.dataset_synchronization import (
    DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
    ChannelSynchronizationSummary,
    DatasetSynchronizationReport,
    InterpolationPolicy,
    SynchronizationDisposition,
    SynchronizationIssue,
    SynchronizationIssueKind,
    SynchronizationIssueSeverity,
)
from iso_obs.evidence import NamedDigest, sha256_digest
from iso_obs.mixed_mode_detection import (
    MIXED_MODE_DETECTION_REPORT_SCHEMA_VERSION,
    DetectorModelFamily,
    ModeChangePointCandidate,
    ModeDetectionDisposition,
    ModeDetectionWindow,
    ModeDetectorManifest,
    ModeDetectorOutput,
    ModeInterval,
    ModeSupportInterval,
    evaluate_mixed_mode_detection,
    suggested_label_assertions,
)
from iso_obs.simulation import EvidenceUse


def scope() -> DatasetScope:
    """Build a compact source-dataset scope."""
    return DatasetScope(
        domain_namespace="autonomous-driving",
        target_population_digest=sha256_digest("urban-operation"),
        collection_protocol_digest=sha256_digest("capture-protocol"),
        modality_ids=("camera", "control-state"),
        evidence_roles=(DatasetEvidenceRole.FAILURE_LABEL_LEARNING,),
        limitations=("One geographic region.",),
    )


def synchronization_report(
    disposition: SynchronizationDisposition = (
        SynchronizationDisposition.ALIGNED_WITHIN_SCOPE
    ),
) -> DatasetSynchronizationReport:
    """Build a content-addressed synchronization prerequisite."""
    is_aligned = disposition is SynchronizationDisposition.ALIGNED_WITHIN_SCOPE
    return DatasetSynchronizationReport(
        schema_version=DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
        plan_content_digest=sha256_digest("sync-plan"),
        dataset_manifest_digest=sha256_digest("dataset-manifest"),
        episode_id="episode-1",
        disposition=disposition,
        reference_clock_id="vehicle-clock",
        channel_summaries=(
            ChannelSynchronizationSummary(
                channel_id="camera",
                modality_id="camera",
                clock_id="vehicle-clock",
                sample_count=11,
                coverage_start_seconds=0.0,
                coverage_end_seconds=10.0,
                maximum_observed_gap_seconds=1.0,
                alignment_uncertainty_upper_seconds=0.001,
                interpolation_policy=InterpolationPolicy.NEAREST,
            ),
        ),
        issues=(
            ()
            if is_aligned
            else (
                SynchronizationIssue(
                    kind=SynchronizationIssueKind.UNKNOWN_CLOCK_BASIS,
                    severity=SynchronizationIssueSeverity.BLOCKING,
                    channel_id="camera",
                    description="Test fixture with unsupported clock evidence.",
                ),
            )
        ),
        scope=scope(),
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=("Timing uncertainty is bounded.",),
    )


def manifest(*, calibrated: bool = True) -> ModeDetectorManifest:
    """Build a neural operating-mode detector manifest."""
    return ModeDetectorManifest(
        model_id="operating-mode-encoder",
        model_version="1",
        model_family=DetectorModelFamily.NEURAL,
        artifact_digest=sha256_digest("model-weights"),
        architecture_digest=sha256_digest("architecture"),
        feature_extractor_digest=sha256_digest("feature-extractor"),
        training_dataset_digests=(sha256_digest("training-data"),),
        target_population_digest=sha256_digest("urban-operation"),
        mode_family=ModeFamily.OPERATING,
        label_namespace="reliability/operating/v1",
        calibration_evidence_digest=(
            sha256_digest("held-out-calibration") if calibrated else None
        ),
        limitations=("Validated on one sensor configuration.",),
    )


def window(sync: DatasetSynchronizationReport) -> ModeDetectionWindow:
    """Build one synchronized detection window."""
    return ModeDetectionWindow(
        window_id="window-1",
        episode_id="episode-1",
        synchronization_report_digest=sync.content_digest(),
        start_seconds=0.0,
        end_seconds=10.0,
        modality_artifact_digests=(
            NamedDigest("camera", sha256_digest("camera-window")),
            NamedDigest("control-state", sha256_digest("control-window")),
        ),
        feature_artifact_digest=sha256_digest("window-features"),
    )


def candidate() -> ModeChangePointCandidate:
    """Build an uncertain manual-to-autonomous boundary."""
    return ModeChangePointCandidate(
        candidate_id="change-1",
        before_interval=ModeInterval(0.0, 4.5),
        change_interval=ModeInterval(4.5, 5.5),
        after_interval=ModeInterval(5.5, 10.0),
        before_prediction_set=("manual",),
        after_prediction_set=("autonomous",),
        before_support=(
            ModeSupportInterval("autonomous", 0.01, 0.08),
            ModeSupportInterval("manual", 0.91, 0.98),
        ),
        after_support=(
            ModeSupportInterval("autonomous", 0.89, 0.97),
            ModeSupportInterval("manual", 0.02, 0.10),
        ),
        evidence_digest=sha256_digest("change-point-evidence"),
        limitations=("Transition interval is unresolved.",),
    )


def output(
    detector: ModeDetectorManifest,
    detection_window: ModeDetectionWindow,
    *,
    candidates: tuple[ModeChangePointCandidate, ...] | None = None,
) -> ModeDetectorOutput:
    """Build detector output tied to immutable model and window identities."""
    return ModeDetectorOutput(
        model_manifest_digest=detector.content_digest(),
        detection_window_digest=detection_window.content_digest(),
        execution_digest=sha256_digest("detector-execution"),
        candidates=(candidate(),) if candidates is None else candidates,
        limitations=("Candidate sets use held-out calibration.",),
    )


def test_candidate_report_is_ordered_versioned_and_content_addressed() -> None:
    """A calibrated detector emits a stable suggestion artifact."""
    sync = synchronization_report()
    detector = manifest()
    detection_window = window(sync)
    detector_output = output(detector, detection_window)

    report = evaluate_mixed_mode_detection(
        detector,
        detection_window,
        sync,
        detector_output,
    )

    assert report.schema_version == MIXED_MODE_DETECTION_REPORT_SCHEMA_VERSION
    assert report.disposition is ModeDetectionDisposition.CANDIDATES_IDENTIFIED
    assert report.content_digest() == sha256_digest(report.to_json())
    assert report.candidates == (candidate(),)


def test_unaligned_input_forces_abstention() -> None:
    """A detector cannot establish temporal candidates on unsupported timing."""
    sync = synchronization_report(SynchronizationDisposition.INSUFFICIENT_EVIDENCE)
    detector = manifest()
    detection_window = window(sync)

    report = evaluate_mixed_mode_detection(
        detector,
        detection_window,
        sync,
        output(detector, detection_window),
    )

    assert report.disposition is ModeDetectionDisposition.INSUFFICIENT_EVIDENCE


def test_missing_calibration_forces_abstention() -> None:
    """Uncalibrated neural scores cannot be presented as candidate sets."""
    sync = synchronization_report()
    detector = manifest(calibrated=False)
    detection_window = window(sync)

    report = evaluate_mixed_mode_detection(
        detector,
        detection_window,
        sync,
        output(detector, detection_window),
    )

    assert report.disposition is ModeDetectionDisposition.INSUFFICIENT_EVIDENCE


def test_target_population_mismatch_forces_abstention() -> None:
    """Calibration cannot transfer across populations without evidence."""
    sync = synchronization_report()
    detector = replace(
        manifest(),
        target_population_digest=sha256_digest("different-population"),
    )
    detection_window = window(sync)

    report = evaluate_mixed_mode_detection(
        detector,
        detection_window,
        sync,
        output(detector, detection_window),
    )

    assert report.disposition is ModeDetectionDisposition.INSUFFICIENT_EVIDENCE
    assert any("transport is unsupported" in item for item in report.limitations)


def test_empty_output_is_not_a_no_failure_claim() -> None:
    """No candidate is a scope-bound detector result, not system assurance."""
    sync = synchronization_report()
    detector = manifest()
    detection_window = window(sync)

    report = evaluate_mixed_mode_detection(
        detector,
        detection_window,
        sync,
        output(detector, detection_window, candidates=()),
    )

    assert report.disposition is ModeDetectionDisposition.NO_CANDIDATE_WITHIN_SCOPE
    assert report.candidates == ()
    assert any(
        "not evidence" in limitation.lower() for limitation in report.limitations
    )


def test_candidate_sets_become_suggestions_not_ground_truth() -> None:
    """The label-ledger bridge preserves machine-derived epistemic status."""
    sync = synchronization_report()
    detector = manifest()
    detection_window = window(sync)
    report = evaluate_mixed_mode_detection(
        detector,
        detection_window,
        sync,
        output(detector, detection_window),
    )

    assertions = suggested_label_assertions(report, detector)

    assert tuple(item.assertion_id for item in assertions) == (
        "change-1:after",
        "change-1:before",
    )
    assert all(item.status is LabelStatus.SUGGESTED for item in assertions)
    assert assertions[0].candidate_values == ("autonomous",)
    assert assertions[1].candidate_values == ("manual",)
    assert all(item.method_digest == detector.content_digest() for item in assertions)


def test_abstaining_report_cannot_create_label_assertions() -> None:
    """Unsupported model output cannot enter the label ledger."""
    sync = synchronization_report(SynchronizationDisposition.INSUFFICIENT_EVIDENCE)
    detector = manifest()
    detection_window = window(sync)
    report = evaluate_mixed_mode_detection(
        detector,
        detection_window,
        sync,
        output(detector, detection_window),
    )

    with pytest.raises(ValueError, match="insufficient"):
        suggested_label_assertions(report, detector)


def test_candidate_must_remain_inside_detection_window() -> None:
    """A detector cannot claim temporal context it did not receive."""
    sync = synchronization_report()
    detector = manifest()
    detection_window = window(sync)
    outside = replace(
        candidate(),
        after_interval=ModeInterval(5.5, 11.0),
    )

    with pytest.raises(ValueError, match="outside"):
        evaluate_mixed_mode_detection(
            detector,
            detection_window,
            sync,
            output(detector, detection_window, candidates=(outside,)),
        )


def test_prediction_set_requires_matching_support_interval() -> None:
    """A candidate value cannot be emitted without calibrated support."""
    with pytest.raises(ValueError, match="lacks support"):
        replace(
            candidate(),
            before_prediction_set=("remote_assist",),
        )
