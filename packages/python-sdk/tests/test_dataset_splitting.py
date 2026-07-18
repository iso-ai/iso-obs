"""Tests for lineage-aware dataset split contamination audits."""

from __future__ import annotations

from dataclasses import replace

from iso_obs.dataset_reliability import DatasetEvidenceRole, DatasetScope
from iso_obs.dataset_splitting import (
    DATASET_SPLIT_REPORT_SCHEMA_VERSION,
    DatasetSplitAssignment,
    DatasetSplitDisposition,
    DatasetSplitPlan,
    DatasetSplitRole,
    DatasetSplitUnit,
    DatasetUnitOrigin,
    SplitAttribute,
    SplitAttributeKind,
    SplitIssueKind,
    SplitRoleMinimum,
    audit_dataset_split,
)
from iso_obs.evidence import sha256_digest


def scope() -> DatasetScope:
    """Build the population scope for split auditing."""
    return DatasetScope(
        domain_namespace="robot-manipulation",
        target_population_digest=sha256_digest("warehouse-picks"),
        collection_protocol_digest=sha256_digest("multi-site-capture"),
        modality_ids=("camera", "force-torque"),
        evidence_roles=(
            DatasetEvidenceRole.REPRESENTATION_LEARNING,
            DatasetEvidenceRole.FAILURE_LABEL_LEARNING,
            DatasetEvidenceRole.CALIBRATION,
            DatasetEvidenceRole.VALIDATION,
            DatasetEvidenceRole.EVALUATION_ONLY,
        ),
        limitations=("Three robot embodiments.",),
    )


def plan() -> DatasetSplitPlan:
    """Build a compact five-role split plan."""
    roles = tuple(DatasetSplitRole)
    return DatasetSplitPlan(
        plan_id="robot-failure-split",
        plan_version="1",
        dataset_collection_digest=sha256_digest("dataset-collection"),
        scope=scope(),
        required_roles=roles,
        role_minimums=tuple(
            SplitRoleMinimum(role=role, minimum_unit_count=1) for role in roles
        ),
        holdout_roles=(
            DatasetSplitRole.VALIDATION,
            DatasetSplitRole.REAL_WORLD_HOLDOUT,
        ),
        protected_holdout_attributes=(
            SplitAttributeKind.GEOGRAPHY,
            SplitAttributeKind.EMBODIMENT,
        ),
        limitations=("Near-duplicate groups come from a preregistered scanner.",),
    )


def unit(
    unit_id: str,
    *,
    origin: DatasetUnitOrigin = DatasetUnitOrigin.REAL_OBSERVED,
    parent_ids: tuple[str, ...] = (),
    independence_id: str | None = None,
    duplicate_groups: tuple[str, ...] = (),
    geography: str | None = None,
    embodiment: str | None = None,
    source_split: str | None = None,
) -> DatasetSplitUnit:
    """Build one episode split unit."""
    attributes = []
    if geography is not None:
        attributes.append(SplitAttribute(SplitAttributeKind.GEOGRAPHY, geography))
    if embodiment is not None:
        attributes.append(SplitAttribute(SplitAttributeKind.EMBODIMENT, embodiment))
    synthetic = origin in {
        DatasetUnitOrigin.SYNTHETIC_COUNTERFACTUAL,
        DatasetUnitOrigin.SYNTHETIC_GENERATED,
    }
    return DatasetSplitUnit(
        unit_id=unit_id,
        episode_id=f"episode-{unit_id}",
        episode_content_digest=sha256_digest(f"episode-{unit_id}"),
        dataset_manifest_digest=sha256_digest(f"dataset-{unit_id}"),
        independence_unit_id=independence_id or f"independence-{unit_id}",
        origin=origin,
        parent_unit_ids=parent_ids,
        near_duplicate_group_ids=duplicate_groups,
        attributes=tuple(attributes),
        generator_digest=sha256_digest("generator") if synthetic else None,
        transformation_digest=(
            sha256_digest(f"transform-{unit_id}") if synthetic else None
        ),
        source_split=source_split,
    )


def valid_units() -> tuple[DatasetSplitUnit, ...]:
    """Build independent, lineage-complete units for every role."""
    return (
        unit("representation", geography="site-a", embodiment="arm-a"),
        unit("fit", geography="site-a", embodiment="arm-a"),
        unit("calibration", geography="site-a", embodiment="arm-a"),
        unit("validation", geography="site-b", embodiment="arm-b"),
        unit("holdout", geography="site-c", embodiment="arm-c"),
    )


def valid_assignments() -> tuple[DatasetSplitAssignment, ...]:
    """Assign one independent unit to each scientific role."""
    return (
        DatasetSplitAssignment(
            "representation",
            DatasetSplitRole.REPRESENTATION_FIT,
        ),
        DatasetSplitAssignment("fit", DatasetSplitRole.MODEL_FIT),
        DatasetSplitAssignment("calibration", DatasetSplitRole.CALIBRATION),
        DatasetSplitAssignment("validation", DatasetSplitRole.VALIDATION),
        DatasetSplitAssignment("holdout", DatasetSplitRole.REAL_WORLD_HOLDOUT),
    )


def test_valid_split_is_order_invariant_and_content_addressed() -> None:
    """Input ordering cannot change a valid split artifact."""
    units = valid_units()
    assignments = valid_assignments()

    first = audit_dataset_split(plan(), units, assignments)
    second = audit_dataset_split(
        plan(),
        tuple(reversed(units)),
        tuple(reversed(assignments)),
    )

    assert first == second
    assert first.content_digest() == second.content_digest()
    assert first.schema_version == DATASET_SPLIT_REPORT_SCHEMA_VERSION
    assert first.disposition is DatasetSplitDisposition.VALID_WITHIN_SCOPE
    assert first.issues == ()


def test_independence_unit_cannot_cross_fit_and_calibration() -> None:
    """Correlated observations cannot supply fitting and calibration evidence."""
    units = tuple(
        (
            replace(item, independence_unit_id="shared-run")
            if item.unit_id in {"fit", "calibration"}
            else item
        )
        for item in valid_units()
    )

    report = audit_dataset_split(plan(), units, valid_assignments())

    assert report.disposition is DatasetSplitDisposition.CONTAMINATED
    assert SplitIssueKind.INDEPENDENCE_UNIT_LEAKAGE in {
        issue.kind for issue in report.issues
    }


def test_transitive_synthetic_lineage_stays_in_parent_partition() -> None:
    """A grandchild crossing into validation exposes transitive contamination."""
    units = (
        *valid_units(),
        unit(
            "synthetic-child",
            origin=DatasetUnitOrigin.SYNTHETIC_COUNTERFACTUAL,
            parent_ids=("fit",),
            geography="site-a",
            embodiment="arm-a",
        ),
        unit(
            "synthetic-grandchild",
            origin=DatasetUnitOrigin.SYNTHETIC_GENERATED,
            parent_ids=("synthetic-child",),
            geography="site-a",
            embodiment="arm-a",
        ),
    )
    assignments = (
        *valid_assignments(),
        DatasetSplitAssignment("synthetic-child", DatasetSplitRole.MODEL_FIT),
        DatasetSplitAssignment(
            "synthetic-grandchild",
            DatasetSplitRole.VALIDATION,
        ),
    )

    report = audit_dataset_split(plan(), units, assignments)

    assert report.disposition is DatasetSplitDisposition.CONTAMINATED
    lineage_issue = next(
        issue for issue in report.issues if issue.kind is SplitIssueKind.LINEAGE_LEAKAGE
    )
    assert lineage_issue.unit_ids == (
        "fit",
        "synthetic-child",
        "synthetic-grandchild",
    )


def test_missing_synthetic_parent_is_insufficient_not_clean() -> None:
    """Absent ancestry prevents a contamination conclusion."""
    units = (
        *valid_units(),
        unit(
            "synthetic",
            origin=DatasetUnitOrigin.SYNTHETIC_COUNTERFACTUAL,
            parent_ids=("unavailable-real-parent",),
            geography="site-a",
            embodiment="arm-a",
        ),
    )
    assignments = (
        *valid_assignments(),
        DatasetSplitAssignment("synthetic", DatasetSplitRole.MODEL_FIT),
    )

    report = audit_dataset_split(plan(), units, assignments)

    assert report.disposition is DatasetSplitDisposition.INSUFFICIENT_EVIDENCE
    assert SplitIssueKind.MISSING_LINEAGE_PARENT in {
        issue.kind for issue in report.issues
    }


def test_lineage_cycle_blocks_split_validation() -> None:
    """Circular ancestry invalidates lineage-based leakage analysis."""
    units = tuple(
        (
            replace(item, parent_unit_ids=("calibration",))
            if item.unit_id == "fit"
            else (
                replace(item, parent_unit_ids=("fit",))
                if item.unit_id == "calibration"
                else item
            )
        )
        for item in valid_units()
    )

    report = audit_dataset_split(plan(), units, valid_assignments())

    assert report.disposition is DatasetSplitDisposition.INSUFFICIENT_EVIDENCE
    assert SplitIssueKind.LINEAGE_CYCLE in {issue.kind for issue in report.issues}


def test_near_duplicate_group_cannot_cross_roles() -> None:
    """Near duplicates cannot inflate fitting and validation evidence."""
    units = tuple(
        (
            replace(item, near_duplicate_group_ids=("visual-cluster-1",))
            if item.unit_id in {"fit", "validation"}
            else item
        )
        for item in valid_units()
    )

    report = audit_dataset_split(plan(), units, valid_assignments())

    assert report.disposition is DatasetSplitDisposition.CONTAMINATED
    assert SplitIssueKind.NEAR_DUPLICATE_LEAKAGE in {
        issue.kind for issue in report.issues
    }


def test_protected_holdout_attributes_cannot_leak() -> None:
    """Holdout geography and embodiment must remain outside development."""
    units = tuple(
        (
            replace(
                item,
                attributes=(
                    SplitAttribute(SplitAttributeKind.GEOGRAPHY, "site-a"),
                    SplitAttribute(SplitAttributeKind.EMBODIMENT, "arm-a"),
                ),
            )
            if item.unit_id == "validation"
            else item
        )
        for item in valid_units()
    )

    report = audit_dataset_split(plan(), units, valid_assignments())

    assert report.disposition is DatasetSplitDisposition.CONTAMINATED
    holdout_issues = tuple(
        issue
        for issue in report.issues
        if issue.kind is SplitIssueKind.HOLDOUT_ATTRIBUTE_LEAKAGE
    )
    assert len(holdout_issues) == 2


def test_validation_cannot_share_protected_value_with_final_holdout() -> None:
    """Validation-guided choices cannot leak into the final real holdout."""
    units = tuple(
        (
            replace(
                item,
                attributes=(
                    SplitAttribute(SplitAttributeKind.GEOGRAPHY, "site-b"),
                    SplitAttribute(SplitAttributeKind.EMBODIMENT, "arm-b"),
                ),
            )
            if item.unit_id == "holdout"
            else item
        )
        for item in valid_units()
    )

    report = audit_dataset_split(plan(), units, valid_assignments())

    assert report.disposition is DatasetSplitDisposition.CONTAMINATED
    assert SplitIssueKind.HOLDOUT_ATTRIBUTE_LEAKAGE in {
        issue.kind for issue in report.issues
    }


def test_missing_protected_attribute_is_insufficient_evidence() -> None:
    """Unknown holdout metadata cannot be interpreted as unique metadata."""
    units = tuple(
        replace(item, attributes=()) if item.unit_id == "holdout" else item
        for item in valid_units()
    )

    report = audit_dataset_split(plan(), units, valid_assignments())

    assert report.disposition is DatasetSplitDisposition.INSUFFICIENT_EVIDENCE
    missing = tuple(
        issue
        for issue in report.issues
        if issue.kind is SplitIssueKind.MISSING_PROTECTED_ATTRIBUTE
    )
    assert len(missing) == 2


def test_real_world_holdout_rejects_simulated_or_synthetic_units() -> None:
    """A real-world gold set cannot contain simulation evidence."""
    units = tuple(
        (
            replace(item, origin=DatasetUnitOrigin.SIMULATED)
            if item.unit_id == "holdout"
            else item
        )
        for item in valid_units()
    )

    report = audit_dataset_split(plan(), units, valid_assignments())

    assert report.disposition is DatasetSplitDisposition.CONTAMINATED
    assert SplitIssueKind.NON_REAL_HOLDOUT in {issue.kind for issue in report.issues}


def test_incomplete_synthetic_provenance_blocks_validation() -> None:
    """A synthetic child needs generator and transformation provenance."""
    units = (
        *valid_units(),
        replace(
            unit(
                "synthetic",
                origin=DatasetUnitOrigin.SYNTHETIC_GENERATED,
                parent_ids=("fit",),
                geography="site-a",
                embodiment="arm-a",
            ),
            generator_digest=None,
        ),
    )
    assignments = (
        *valid_assignments(),
        DatasetSplitAssignment("synthetic", DatasetSplitRole.MODEL_FIT),
    )

    report = audit_dataset_split(plan(), units, assignments)

    assert report.disposition is DatasetSplitDisposition.INSUFFICIENT_EVIDENCE
    assert SplitIssueKind.SYNTHETIC_PROVENANCE_INCOMPLETE in {
        issue.kind for issue in report.issues
    }


def test_inherited_source_split_disagreement_requires_review() -> None:
    """A source split is preserved as evidence rather than silently replaced."""
    units = tuple(
        replace(item, source_split="validation") if item.unit_id == "fit" else item
        for item in valid_units()
    )

    report = audit_dataset_split(plan(), units, valid_assignments())

    assert report.disposition is DatasetSplitDisposition.REVIEW_REQUIRED
    assert SplitIssueKind.SOURCE_SPLIT_DISAGREEMENT in {
        issue.kind for issue in report.issues
    }


def test_deep_lineage_does_not_depend_on_python_recursion_limit() -> None:
    """Long augmentation chains remain auditable without recursive traversal."""
    depth = 1_100
    chain_units = tuple(
        unit(
            f"chain-{index:04d}",
            origin=(
                DatasetUnitOrigin.REAL_OBSERVED
                if index == 0
                else DatasetUnitOrigin.SYNTHETIC_COUNTERFACTUAL
            ),
            parent_ids=(() if index == 0 else (f"chain-{index - 1:04d}",)),
        )
        for index in range(depth)
    )
    validation_unit = unit("deep-validation")
    deep_plan = DatasetSplitPlan(
        plan_id="deep-lineage",
        plan_version="1",
        dataset_collection_digest=sha256_digest("deep-collection"),
        scope=scope(),
        required_roles=(
            DatasetSplitRole.MODEL_FIT,
            DatasetSplitRole.VALIDATION,
        ),
        role_minimums=(
            SplitRoleMinimum(DatasetSplitRole.MODEL_FIT, 1),
            SplitRoleMinimum(DatasetSplitRole.VALIDATION, 1),
        ),
        holdout_roles=(DatasetSplitRole.VALIDATION,),
        protected_holdout_attributes=(),
    )
    assignments = (
        *(
            DatasetSplitAssignment(item.unit_id, DatasetSplitRole.MODEL_FIT)
            for item in chain_units
        ),
        DatasetSplitAssignment(
            validation_unit.unit_id,
            DatasetSplitRole.VALIDATION,
        ),
    )

    report = audit_dataset_split(
        deep_plan,
        (*chain_units, validation_unit),
        assignments,
    )

    assert report.disposition is DatasetSplitDisposition.VALID_WITHIN_SCOPE
