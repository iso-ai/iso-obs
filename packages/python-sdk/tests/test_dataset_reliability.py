"""Tests for provenance-preserving dataset reliability audits."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.dataset_reliability import (
    DATASET_RELIABILITY_REPORT_SCHEMA_VERSION,
    DatasetAuditDisposition,
    DatasetAuditIssueKind,
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
from iso_obs.evidence import NamedDigest, sha256_digest


def manifest() -> DatasetManifest:
    """Build a compact autonomous-driving dataset manifest."""
    return DatasetManifest(
        dataset_id="mixed-driving-modes",
        dataset_version="2026-07",
        content_digest=sha256_digest("raw-dataset"),
        source_uri="https://example.test/datasets/mixed-driving-modes",
        license_id="LicenseRef-research-only",
        scope=DatasetScope(
            domain_namespace="autonomous-driving",
            target_population_digest=sha256_digest("urban-night-driving"),
            collection_protocol_digest=sha256_digest("collection-protocol"),
            modality_ids=("camera-front", "can-bus"),
            evidence_roles=(
                DatasetEvidenceRole.REPRESENTATION_LEARNING,
                DatasetEvidenceRole.FAILURE_LABEL_LEARNING,
            ),
            limitations=("Manual and autonomous operation are mixed.",),
        ),
    )


def episode(dataset: DatasetManifest, episode_id: str = "episode-1") -> EpisodeManifest:
    """Build one independently identified temporal episode."""
    return EpisodeManifest(
        dataset_manifest_digest=dataset.manifest_digest(),
        episode_id=episode_id,
        independence_unit_id=f"drive-{episode_id}",
        content_digest=sha256_digest(f"content-{episode_id}"),
        start_seconds=0.0,
        end_seconds=10.0,
        modality_digests=(
            NamedDigest("camera-front", sha256_digest(f"camera-{episode_id}")),
            NamedDigest("can-bus", sha256_digest(f"can-{episode_id}")),
        ),
        source_split="unassigned",
    )


def plan(dataset: DatasetManifest) -> DatasetReliabilityAuditPlan:
    """Build conservative audit rules for operating and failure modes."""
    return DatasetReliabilityAuditPlan(
        plan_id="mixed-mode-audit",
        plan_version="1",
        dataset_manifest_digest=dataset.manifest_digest(),
        label_taxonomy_digest=sha256_digest("label-taxonomy"),
        required_mode_families=(
            ModeFamily.OPERATING,
            ModeFamily.FAILURE_STATE,
        ),
        limitations=("Candidate modes require expert review.",),
    )


def evidence(evidence_id: str, source_id: str) -> LabelEvidence:
    """Build independently attributable label evidence."""
    return LabelEvidence(
        evidence_id=evidence_id,
        evidence_digest=sha256_digest(evidence_id),
        independent_source_id=source_id,
        relation=EvidenceRelation.OBSERVED_OUTCOME,
        description="Observed in the synchronized episode record.",
    )


def assertion(
    *,
    assertion_id: str,
    family: ModeFamily,
    value: str,
    status: LabelStatus = LabelStatus.ADJUDICATED,
    start: float = 0.0,
    end: float = 10.0,
) -> DatasetLabelAssertion:
    """Build one evidence-qualified temporal label assertion."""
    return DatasetLabelAssertion(
        assertion_id=assertion_id,
        episode_id="episode-1",
        label_namespace=f"reliability/{family.value}/v1",
        mode_family=family,
        start_seconds=start,
        end_seconds=end,
        candidate_values=(value,),
        status=status,
        evidence=(evidence(f"{assertion_id}-evidence", "reviewer-1"),),
        method_digest=(
            sha256_digest("mode-detector") if status is LabelStatus.SUGGESTED else None
        ),
    )


def curated_assertions() -> tuple[DatasetLabelAssertion, ...]:
    """Build complete, adjudicated assertions for one episode."""
    return (
        assertion(
            assertion_id="operating-mode",
            family=ModeFamily.OPERATING,
            value="autonomous",
        ),
        assertion(
            assertion_id="failure-state",
            family=ModeFamily.FAILURE_STATE,
            value="no_observed_failure",
        ),
    )


def test_curated_audit_is_order_invariant_and_content_addressed() -> None:
    """Input order cannot change the report or its artifact identity."""
    dataset = manifest()
    dataset_episode = episode(dataset)
    audit_plan = plan(dataset)
    assertions = curated_assertions()

    first = audit_dataset_reliability(
        audit_plan,
        dataset,
        (dataset_episode,),
        assertions,
    )
    second = audit_dataset_reliability(
        audit_plan,
        dataset,
        (dataset_episode,),
        tuple(reversed(assertions)),
    )

    assert first == second
    assert first.content_digest() == second.content_digest()
    assert first.schema_version == DATASET_RELIABILITY_REPORT_SCHEMA_VERSION
    assert first.disposition is DatasetAuditDisposition.CURATED_WITHIN_SCOPE
    assert first.issues == ()


def test_suggestion_remains_review_required_and_is_not_rewritten() -> None:
    """A model suggestion stays a suggestion and cannot become ground truth."""
    dataset = manifest()
    suggested = assertion(
        assertion_id="operating-mode",
        family=ModeFamily.OPERATING,
        value="autonomous",
        status=LabelStatus.SUGGESTED,
    )
    labels = (
        suggested,
        assertion(
            assertion_id="failure-state",
            family=ModeFamily.FAILURE_STATE,
            value="failure_precursor",
        ),
    )

    report = audit_dataset_reliability(
        plan(dataset),
        dataset,
        (episode(dataset),),
        labels,
    )

    assert suggested.status is LabelStatus.SUGGESTED
    assert report.disposition is DatasetAuditDisposition.REVIEW_REQUIRED
    assert DatasetAuditIssueKind.MACHINE_SUGGESTION in {
        issue.kind for issue in report.issues
    }


def test_reported_source_label_is_not_automatically_curated() -> None:
    """A source label remains review-required until independently qualified."""
    dataset = manifest()
    labels = (
        assertion(
            assertion_id="operating-mode",
            family=ModeFamily.OPERATING,
            value="autonomous",
            status=LabelStatus.REPORTED,
        ),
        assertion(
            assertion_id="failure-state",
            family=ModeFamily.FAILURE_STATE,
            value="no_observed_failure",
        ),
    )

    report = audit_dataset_reliability(
        plan(dataset),
        dataset,
        (episode(dataset),),
        labels,
    )

    assert report.disposition is DatasetAuditDisposition.REVIEW_REQUIRED
    assert DatasetAuditIssueKind.UNVERIFIED_REPORTED_LABEL in {
        issue.kind for issue in report.issues
    }


def test_overlapping_distinct_modes_are_flagged_not_collapsed() -> None:
    """Overlapping candidate modes remain visible as a review obligation."""
    dataset = manifest()
    labels = (
        assertion(
            assertion_id="manual-mode",
            family=ModeFamily.OPERATING,
            value="manual",
            start=0.0,
            end=6.0,
        ),
        assertion(
            assertion_id="autonomous-mode",
            family=ModeFamily.OPERATING,
            value="autonomous",
            start=5.0,
            end=10.0,
        ),
        assertion(
            assertion_id="failure-state",
            family=ModeFamily.FAILURE_STATE,
            value="failure_precursor",
        ),
    )

    report = audit_dataset_reliability(
        plan(dataset),
        dataset,
        (episode(dataset),),
        labels,
    )

    overlap = tuple(
        issue
        for issue in report.issues
        if issue.kind is DatasetAuditIssueKind.OVERLAPPING_MODE_ASSERTIONS
    )
    assert report.disposition is DatasetAuditDisposition.REVIEW_REQUIRED
    assert len(overlap) == 1
    assert overlap[0].assertion_ids == ("autonomous-mode", "manual-mode")


def test_corroborated_status_requires_independent_sources() -> None:
    """Repeated evidence from one source cannot count as corroboration."""
    dataset = manifest()
    weakly_corroborated = replace(
        assertion(
            assertion_id="operating-mode",
            family=ModeFamily.OPERATING,
            value="autonomous",
            status=LabelStatus.CORROBORATED,
        ),
        evidence=(
            evidence("camera-evidence", "detector-1"),
            evidence("telemetry-evidence", "detector-1"),
        ),
    )
    labels = (
        weakly_corroborated,
        assertion(
            assertion_id="failure-state",
            family=ModeFamily.FAILURE_STATE,
            value="no_observed_failure",
        ),
    )

    report = audit_dataset_reliability(
        plan(dataset),
        dataset,
        (episode(dataset),),
        labels,
    )

    assert DatasetAuditIssueKind.INSUFFICIENT_CORROBORATION in {
        issue.kind for issue in report.issues
    }


def test_non_supporting_relations_cannot_supply_corroboration() -> None:
    """Analogy and contradiction do not confirm an asserted label."""
    dataset = manifest()
    weakly_corroborated = replace(
        assertion(
            assertion_id="operating-mode",
            family=ModeFamily.OPERATING,
            value="autonomous",
            status=LabelStatus.CORROBORATED,
        ),
        evidence=(
            replace(
                evidence("analogy", "source-1"),
                relation=EvidenceRelation.ANALOGOUS_TO,
            ),
            replace(
                evidence("contradiction", "source-2"),
                relation=EvidenceRelation.CONTRADICTS,
            ),
        ),
    )
    labels = (
        weakly_corroborated,
        assertion(
            assertion_id="failure-state",
            family=ModeFamily.FAILURE_STATE,
            value="no_observed_failure",
        ),
    )

    report = audit_dataset_reliability(
        plan(dataset),
        dataset,
        (episode(dataset),),
        labels,
    )

    assert DatasetAuditIssueKind.INSUFFICIENT_CORROBORATION in {
        issue.kind for issue in report.issues
    }


def test_partial_mode_label_does_not_count_as_episode_coverage() -> None:
    """A short label interval cannot stand in for an entire episode."""
    dataset = manifest()
    labels = (
        assertion(
            assertion_id="partial-operating-mode",
            family=ModeFamily.OPERATING,
            value="autonomous",
            start=0.0,
            end=4.0,
        ),
        assertion(
            assertion_id="failure-state",
            family=ModeFamily.FAILURE_STATE,
            value="no_observed_failure",
        ),
    )

    report = audit_dataset_reliability(
        plan(dataset),
        dataset,
        (episode(dataset),),
        labels,
    )

    assert DatasetAuditIssueKind.MISSING_MODE_FAMILY in {
        issue.kind for issue in report.issues
    }


def test_empty_dataset_abstains() -> None:
    """An empty audit cannot claim that a dataset is curated."""
    dataset = manifest()

    report = audit_dataset_reliability(plan(dataset), dataset, (), ())

    assert report.disposition is DatasetAuditDisposition.INSUFFICIENT_EVIDENCE


def test_assertion_must_remain_inside_episode_interval() -> None:
    """Temporal evidence outside its episode is rejected."""
    dataset = manifest()
    outside = assertion(
        assertion_id="operating-mode",
        family=ModeFamily.OPERATING,
        value="autonomous",
        end=11.0,
    )

    with pytest.raises(ValueError, match="outside"):
        audit_dataset_reliability(
            plan(dataset),
            dataset,
            (episode(dataset),),
            (outside,),
        )
