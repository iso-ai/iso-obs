"""Detect mixed operating modes without manufacturing ground-truth labels."""

from __future__ import annotations

from iso_obs.dataset_reliability import (
    DatasetEvidenceRole,
    DatasetLabelAssertion,
    DatasetManifest,
    DatasetReliabilityAuditPlan,
    DatasetScope,
    EpisodeManifest,
    EvidenceRelation,
    LabelEvidence,
    LabelStatus,
    ModeFamily,
    audit_dataset_reliability,
)
from iso_obs.dataset_synchronization import (
    DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
    ChannelSynchronizationSummary,
    DatasetSynchronizationReport,
    InterpolationPolicy,
    SynchronizationDisposition,
)
from iso_obs.evidence import NamedDigest, sha256_digest
from iso_obs.mixed_mode_detection import (
    DetectorModelFamily,
    ModeChangePointCandidate,
    ModeDetectionWindow,
    ModeDetectorManifest,
    ModeDetectorOutput,
    ModeInterval,
    ModeSupportInterval,
    evaluate_mixed_mode_detection,
    suggested_label_assertions,
)
from iso_obs.simulation import EvidenceUse


def build_scope() -> DatasetScope:
    """Declare the population and scientific roles of the source dataset."""
    return DatasetScope(
        domain_namespace="autonomous-warehouse-navigation",
        target_population_digest=sha256_digest("indoor low-speed production missions"),
        collection_protocol_digest=sha256_digest(
            "synchronized camera control and safety capture v2"
        ),
        modality_ids=("camera-front", "control-state", "safety-events"),
        evidence_roles=(
            DatasetEvidenceRole.REPRESENTATION_LEARNING,
            DatasetEvidenceRole.FAILURE_LABEL_LEARNING,
        ),
        limitations=(
            "One robot embodiment and one warehouse geometry are represented.",
        ),
    )


def build_dataset(
    scope: DatasetScope,
) -> tuple[DatasetManifest, EpisodeManifest]:
    """Build immutable dataset and episode identities."""
    dataset = DatasetManifest(
        dataset_id="mixed-autonomy-missions",
        dataset_version="2026-07",
        content_digest=sha256_digest("raw-multimodal-dataset-bytes"),
        source_uri="s3://reliability-evidence/mixed-autonomy-missions",
        license_id="LicenseRef-proprietary-evaluation",
        scope=scope,
    )
    episode = EpisodeManifest(
        dataset_manifest_digest=dataset.manifest_digest(),
        episode_id="mission-0042",
        independence_unit_id="physical-run-0042",
        content_digest=sha256_digest("mission-0042-content"),
        start_seconds=0.0,
        end_seconds=30.0,
        modality_digests=(
            NamedDigest("camera-front", sha256_digest("mission-0042-camera")),
            NamedDigest("control-state", sha256_digest("mission-0042-control")),
            NamedDigest("safety-events", sha256_digest("mission-0042-safety")),
        ),
        source_split="unassigned",
    )
    return dataset, episode


def build_synchronization_report(
    dataset: DatasetManifest,
    scope: DatasetScope,
) -> DatasetSynchronizationReport:
    """Record timing support instead of assuming aligned modalities."""
    return DatasetSynchronizationReport(
        schema_version=DATASET_SYNCHRONIZATION_REPORT_SCHEMA_VERSION,
        plan_content_digest=sha256_digest("mission-sync-plan-v1"),
        dataset_manifest_digest=dataset.manifest_digest(),
        episode_id="mission-0042",
        disposition=SynchronizationDisposition.ALIGNED_WITHIN_SCOPE,
        reference_clock_id="robot-monotonic-clock",
        channel_summaries=(
            ChannelSynchronizationSummary(
                channel_id="camera-front",
                modality_id="camera-front",
                clock_id="robot-monotonic-clock",
                sample_count=900,
                coverage_start_seconds=0.0,
                coverage_end_seconds=30.0,
                maximum_observed_gap_seconds=0.040,
                alignment_uncertainty_upper_seconds=0.006,
                interpolation_policy=InterpolationPolicy.NEAREST,
            ),
            ChannelSynchronizationSummary(
                channel_id="control-state",
                modality_id="control-state",
                clock_id="robot-monotonic-clock",
                sample_count=3_000,
                coverage_start_seconds=0.0,
                coverage_end_seconds=30.0,
                maximum_observed_gap_seconds=0.012,
                alignment_uncertainty_upper_seconds=0.002,
                interpolation_policy=InterpolationPolicy.ZERO_ORDER_HOLD,
            ),
        ),
        issues=(),
        scope=scope,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=("Causal ordering below six milliseconds is not supported.",),
    )


def build_detector() -> ModeDetectorManifest:
    """Describe the neural detector, training lineage, and calibration evidence."""
    return ModeDetectorManifest(
        model_id="multimodal-operating-mode-encoder",
        model_version="3",
        model_family=DetectorModelFamily.NEURAL,
        artifact_digest=sha256_digest("encoder-v3-weights"),
        architecture_digest=sha256_digest("temporal-fusion-encoder-v3"),
        feature_extractor_digest=sha256_digest("camera-control-window-features-v2"),
        training_dataset_digests=(
            sha256_digest("representation-dataset-v5"),
            sha256_digest("failure-label-training-ledger-v2"),
        ),
        target_population_digest=sha256_digest("indoor low-speed production missions"),
        mode_family=ModeFamily.OPERATING,
        label_namespace="reliability/operating-mode/v1",
        calibration_evidence_digest=sha256_digest("held-out-conformal-calibration-v3"),
        limitations=(
            "Calibration excludes outdoor lighting and different camera rigs.",
        ),
    )


def build_detector_output(
    detector: ModeDetectorManifest,
    window: ModeDetectionWindow,
) -> ModeDetectorOutput:
    """Create one interval-valued candidate from a calibrated model adapter."""
    candidate = ModeChangePointCandidate(
        candidate_id="manual-to-autonomous-transition",
        before_interval=ModeInterval(0.0, 11.5),
        change_interval=ModeInterval(11.5, 12.5),
        after_interval=ModeInterval(12.5, 30.0),
        before_prediction_set=("remote_assist",),
        after_prediction_set=("autonomous",),
        before_support=(
            ModeSupportInterval("autonomous", 0.02, 0.10),
            ModeSupportInterval("remote_assist", 0.90, 0.98),
        ),
        after_support=(
            ModeSupportInterval("autonomous", 0.88, 0.96),
            ModeSupportInterval("remote_assist", 0.03, 0.12),
        ),
        evidence_digest=sha256_digest("calibrated-transition-evidence-mission-0042"),
        limitations=(
            "The transition is localized to an interval, not an exact frame.",
        ),
    )
    return ModeDetectorOutput(
        model_manifest_digest=detector.content_digest(),
        detection_window_digest=window.content_digest(),
        execution_digest=sha256_digest("detector-execution-mission-0042"),
        candidates=(candidate,),
        limitations=("Prediction sets use held-out calibration.",),
    )


def build_failure_assertion() -> DatasetLabelAssertion:
    """Build an independently observed failure-state assertion."""
    return DatasetLabelAssertion(
        assertion_id="safety-monitor-intervention",
        episode_id="mission-0042",
        label_namespace="reliability/failure-state/v1",
        mode_family=ModeFamily.FAILURE_STATE,
        start_seconds=0.0,
        end_seconds=30.0,
        candidate_values=("safety_monitor_intervened",),
        status=LabelStatus.ADJUDICATED,
        evidence=(
            LabelEvidence(
                evidence_id="safety-controller-event",
                evidence_digest=sha256_digest("safety-event-0042"),
                independent_source_id="independent-safety-controller",
                relation=EvidenceRelation.OBSERVED_OUTCOME,
                description="The independent safety controller recorded intervention.",
            ),
        ),
        limitations=("Intervention identifies an outcome, not the causal mechanism.",),
    )


def main() -> None:
    """Evaluate the model output and audit the provenance-preserving ledger."""
    scope = build_scope()
    dataset, episode = build_dataset(scope)
    synchronization = build_synchronization_report(dataset, scope)
    detector = build_detector()
    window = ModeDetectionWindow(
        window_id="mission-0042-full-window",
        episode_id=episode.episode_id,
        synchronization_report_digest=synchronization.content_digest(),
        start_seconds=episode.start_seconds,
        end_seconds=episode.end_seconds,
        modality_artifact_digests=episode.modality_digests,
        feature_artifact_digest=sha256_digest("mission-0042-features"),
    )
    detection = evaluate_mixed_mode_detection(
        detector,
        window,
        synchronization,
        build_detector_output(detector, window),
    )
    suggestions = suggested_label_assertions(detection, detector)
    failure_assertion = build_failure_assertion()
    audit_plan = DatasetReliabilityAuditPlan(
        plan_id="mixed-mode-ledger-audit",
        plan_version="1",
        dataset_manifest_digest=dataset.manifest_digest(),
        label_taxonomy_digest=sha256_digest("reliability-label-taxonomy-v1"),
        required_mode_families=(
            ModeFamily.OPERATING,
            ModeFamily.FAILURE_STATE,
        ),
        limitations=(
            "Machine suggestions require independent review before training use.",
        ),
    )
    audit = audit_dataset_reliability(
        audit_plan,
        dataset,
        (episode,),
        (*suggestions, failure_assertion),
    )

    candidate = detection.candidates[0]
    print("Mixed-mode dataset curation")
    print(f"  synchronization: {synchronization.disposition.value}")
    print(f"  detector family: {detector.model_family.value}")
    print(f"  detection: {detection.disposition.value}")
    print(
        "  candidate transition: "
        f"{candidate.change_interval.start_seconds:.1f}–"
        f"{candidate.change_interval.end_seconds:.1f} seconds"
    )
    print(
        "  prediction sets: "
        f"{candidate.before_prediction_set} → {candidate.after_prediction_set}"
    )
    print(
        "  ledger statuses: "
        + ", ".join(
            f"{assertion.assertion_id}={assertion.status.value}"
            for assertion in (*suggestions, failure_assertion)
        )
    )
    print(f"  dataset audit: {audit.disposition.value}")
    for issue in audit.issues:
        print(f"    review [{issue.kind.value}]: {issue.description}")
    print(f"  detection digest: {detection.content_digest()}")
    print(f"  audit digest: {audit.content_digest()}")
    print("  claim boundary: neural candidates remain suggestions, never truth")


if __name__ == "__main__":
    main()
