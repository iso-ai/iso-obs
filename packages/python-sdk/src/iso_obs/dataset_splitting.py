"""Lineage-aware dataset splitting and contamination audits.

The guardian audits proposed fit, calibration, validation, and real-world
holdout assignments. It keeps independence groups, derivation families, and
near-duplicate groups intact; protects declared holdout attributes; and keeps
known contamination separate from missing evidence. It does not create a
composite leakage score or claim that undetected leakage is absent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .dataset_reliability import DatasetScope
from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse

DATASET_SPLIT_REPORT_SCHEMA_VERSION = "iso-obs.dataset-split-report.v1"


class DatasetSplitRole(StrEnum):
    """Scientific role assigned to one independent split unit."""

    REPRESENTATION_FIT = "representation_fit"
    MODEL_FIT = "model_fit"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    REAL_WORLD_HOLDOUT = "real_world_holdout"


class DatasetUnitOrigin(StrEnum):
    """Origin of one dataset split unit."""

    REAL_OBSERVED = "real_observed"
    SIMULATED = "simulated"
    SYNTHETIC_COUNTERFACTUAL = "synthetic_counterfactual"
    SYNTHETIC_GENERATED = "synthetic_generated"


class SplitAttributeKind(StrEnum):
    """Attribute available for protected holdout separation."""

    SOURCE_DATASET = "source_dataset"
    GEOGRAPHY = "geography"
    TIME_BLOCK = "time_block"
    EMBODIMENT = "embodiment"
    SYSTEM_VERSION = "system_version"
    SCENARIO_FAMILY = "scenario_family"


class SplitIssueSeverity(StrEnum):
    """Scientific consequence of one split-audit issue."""

    REVIEW = "review"
    CONTAMINATION = "contamination"
    BLOCKING = "blocking"


class SplitIssueKind(StrEnum):
    """Kind of split-integrity problem found by the guardian."""

    MISSING_REQUIRED_ROLE = "missing_required_role"
    ROLE_BELOW_MINIMUM_SIZE = "role_below_minimum_size"
    UNIT_UNASSIGNED = "unit_unassigned"
    UNDECLARED_ROLE_ASSIGNMENT = "undeclared_role_assignment"
    INDEPENDENCE_UNIT_LEAKAGE = "independence_unit_leakage"
    LINEAGE_LEAKAGE = "lineage_leakage"
    NEAR_DUPLICATE_LEAKAGE = "near_duplicate_leakage"
    HOLDOUT_ATTRIBUTE_LEAKAGE = "holdout_attribute_leakage"
    MISSING_PROTECTED_ATTRIBUTE = "missing_protected_attribute"
    NON_REAL_HOLDOUT = "non_real_holdout"
    MISSING_LINEAGE_PARENT = "missing_lineage_parent"
    LINEAGE_CYCLE = "lineage_cycle"
    SYNTHETIC_PROVENANCE_INCOMPLETE = "synthetic_provenance_incomplete"
    SOURCE_SPLIT_DISAGREEMENT = "source_split_disagreement"


class DatasetSplitDisposition(StrEnum):
    """Strongest split-integrity claim supported by an audit."""

    VALID_WITHIN_SCOPE = "valid_within_scope"
    REVIEW_REQUIRED = "review_required"
    CONTAMINATED = "contaminated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class SplitAttribute:
    """One categorical attribute used for holdout separation."""

    kind: SplitAttributeKind
    value: str

    def __post_init__(self) -> None:
        """Validate attribute kind and stable value."""
        _require_text(self.value, "split attribute value")
        object.__setattr__(self, "kind", SplitAttributeKind(self.kind))


@dataclass(frozen=True, slots=True)
class DatasetSplitUnit:
    """One episode-level unit with ancestry and correlation provenance."""

    unit_id: str
    episode_id: str
    episode_content_digest: str
    dataset_manifest_digest: str
    independence_unit_id: str
    origin: DatasetUnitOrigin
    parent_unit_ids: tuple[str, ...] = ()
    near_duplicate_group_ids: tuple[str, ...] = ()
    attributes: tuple[SplitAttribute, ...] = ()
    generator_digest: str | None = None
    transformation_digest: str | None = None
    source_split: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate identity, ancestry references, and source provenance."""
        _require_text(self.unit_id, "dataset split unit ID")
        _require_text(self.episode_id, "dataset split episode ID")
        NamedDigest("split unit episode content", self.episode_content_digest)
        NamedDigest("split unit dataset manifest", self.dataset_manifest_digest)
        _require_text(
            self.independence_unit_id,
            "dataset split independence unit ID",
        )
        parents = _unique_text(
            self.parent_unit_ids,
            "split unit parent IDs",
            required=False,
        )
        if self.unit_id in parents:
            raise ValueError("dataset split unit cannot be its own parent")
        duplicate_groups = _unique_text(
            self.near_duplicate_group_ids,
            "near-duplicate group IDs",
            required=False,
        )
        attributes = tuple(
            sorted(self.attributes, key=lambda item: (item.kind.value, item.value))
        )
        _require_unique(
            (item.kind.value for item in attributes),
            "split attribute kinds per unit",
        )
        if self.generator_digest is not None:
            NamedDigest("dataset unit generator", self.generator_digest)
        if self.transformation_digest is not None:
            NamedDigest(
                "dataset unit transformation",
                self.transformation_digest,
            )
        if self.source_split is not None:
            _require_text(self.source_split, "source dataset split")
        limitations = _unique_text(
            self.limitations,
            "dataset split unit limitations",
            required=False,
        )
        object.__setattr__(self, "origin", DatasetUnitOrigin(self.origin))
        object.__setattr__(self, "parent_unit_ids", parents)
        object.__setattr__(
            self,
            "near_duplicate_group_ids",
            duplicate_groups,
        )
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class DatasetSplitAssignment:
    """Proposed scientific role for one dataset split unit."""

    unit_id: str
    role: DatasetSplitRole

    def __post_init__(self) -> None:
        """Validate assignment identity and role."""
        _require_text(self.unit_id, "dataset split assignment unit ID")
        object.__setattr__(self, "role", DatasetSplitRole(self.role))


@dataclass(frozen=True, slots=True)
class SplitRoleMinimum:
    """Preregistered minimum independent units for one split role."""

    role: DatasetSplitRole
    minimum_unit_count: int

    def __post_init__(self) -> None:
        """Validate role and positive minimum count."""
        minimum = self.minimum_unit_count
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("split role minimum count must be a positive integer")
        object.__setattr__(self, "role", DatasetSplitRole(self.role))
        object.__setattr__(self, "minimum_unit_count", minimum)


@dataclass(frozen=True, slots=True)
class DatasetSplitPlan:
    """Content-addressed requirements for a defensible dataset split."""

    plan_id: str
    plan_version: str
    dataset_collection_digest: str
    scope: DatasetScope
    required_roles: tuple[DatasetSplitRole, ...]
    role_minimums: tuple[SplitRoleMinimum, ...]
    holdout_roles: tuple[DatasetSplitRole, ...]
    protected_holdout_attributes: tuple[SplitAttributeKind, ...]
    require_complete_lineage: bool = True
    require_real_only_holdout: bool = True
    limitations: tuple[str, ...] = ()
    evidence_use: EvidenceUse = EvidenceUse.DISCOVERY_ONLY

    def __post_init__(self) -> None:
        """Validate split roles, holdout policy, and evidence use."""
        _require_text(self.plan_id, "dataset split plan ID")
        _require_text(self.plan_version, "dataset split plan version")
        NamedDigest("dataset collection", self.dataset_collection_digest)
        roles = tuple(sorted({DatasetSplitRole(role) for role in self.required_roles}))
        if not roles:
            raise ValueError("dataset split plan requires scientific roles")
        minimums = tuple(sorted(self.role_minimums, key=lambda item: item.role.value))
        _require_unique(
            (item.role.value for item in minimums),
            "split role minimum declarations",
        )
        minimum_roles = {item.role for item in minimums}
        if set(roles) != minimum_roles:
            raise ValueError(
                "split role minimums must match the required roles exactly"
            )
        holdout_roles = tuple(
            sorted({DatasetSplitRole(role) for role in self.holdout_roles})
        )
        if not holdout_roles:
            raise ValueError("dataset split plan requires at least one holdout role")
        if not set(holdout_roles).issubset(set(roles)):
            raise ValueError("holdout roles must also be required roles")
        protected_attributes = tuple(
            sorted(
                {
                    SplitAttributeKind(attribute)
                    for attribute in self.protected_holdout_attributes
                }
            )
        )
        if not isinstance(self.require_complete_lineage, bool):
            raise ValueError("require_complete_lineage must be boolean")
        if not isinstance(self.require_real_only_holdout, bool):
            raise ValueError("require_real_only_holdout must be boolean")
        limitations = _unique_text(
            self.limitations,
            "dataset split plan limitations",
            required=False,
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("dataset split audit must be discovery-only")
        object.__setattr__(self, "required_roles", roles)
        object.__setattr__(self, "role_minimums", minimums)
        object.__setattr__(self, "holdout_roles", holdout_roles)
        object.__setattr__(
            self,
            "protected_holdout_attributes",
            protected_attributes,
        )
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_use", evidence_use)

    def to_json(self) -> str:
        """Serialize the split plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact split-plan digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class DatasetSplitIssue:
    """One review, contamination, or evidence blocker."""

    kind: SplitIssueKind
    severity: SplitIssueSeverity
    unit_ids: tuple[str, ...]
    roles: tuple[DatasetSplitRole, ...]
    description: str

    def __post_init__(self) -> None:
        """Validate issue classification, units, roles, and description."""
        unit_ids = _unique_text(
            self.unit_ids,
            "dataset split issue unit IDs",
            required=False,
        )
        roles = tuple(sorted({DatasetSplitRole(role) for role in self.roles}))
        _require_text(self.description, "dataset split issue description")
        object.__setattr__(self, "kind", SplitIssueKind(self.kind))
        object.__setattr__(self, "severity", SplitIssueSeverity(self.severity))
        object.__setattr__(self, "unit_ids", unit_ids)
        object.__setattr__(self, "roles", roles)


@dataclass(frozen=True, slots=True)
class DatasetSplitReport:
    """Versioned, content-addressed dataset split integrity report."""

    schema_version: str
    plan_content_digest: str
    dataset_collection_digest: str
    disposition: DatasetSplitDisposition
    assignments: tuple[DatasetSplitAssignment, ...]
    role_counts: tuple[tuple[str, int], ...]
    issues: tuple[DatasetSplitIssue, ...]
    scope: DatasetScope
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate report identity, canonical members, and disposition."""
        if self.schema_version != DATASET_SPLIT_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported dataset split report schema version")
        NamedDigest("dataset split plan", self.plan_content_digest)
        NamedDigest("dataset collection", self.dataset_collection_digest)
        assignments = tuple(sorted(self.assignments, key=lambda item: item.unit_id))
        _require_unique(
            (item.unit_id for item in assignments),
            "dataset split report assignment unit IDs",
        )
        role_counts = tuple(sorted(self.role_counts))
        _require_unique(
            (role for role, _ in role_counts),
            "dataset split report role counts",
        )
        for role, count in role_counts:
            DatasetSplitRole(role)
            if count < 0:
                raise ValueError("dataset split report role counts must be nonnegative")
        issues = tuple(
            sorted(
                self.issues,
                key=lambda item: (
                    item.severity.value,
                    item.kind.value,
                    item.unit_ids,
                    tuple(role.value for role in item.roles),
                ),
            )
        )
        disposition = DatasetSplitDisposition(self.disposition)
        _validate_disposition(disposition, issues)
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("dataset split report must be discovery-only")
        limitations = _unique_text(
            self.limitations,
            "dataset split report limitations",
            required=False,
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "role_counts", role_counts)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact report content digest."""
        return sha256_digest(self.to_json())


def audit_dataset_split(
    plan: DatasetSplitPlan,
    units: Sequence[DatasetSplitUnit],
    assignments: Sequence[DatasetSplitAssignment],
) -> DatasetSplitReport:
    """Audit a proposed split for lineage and correlation contamination.

    Args:
        plan: Preregistered roles, minimums, and holdout protections.
        units: Episode units with ancestry and correlation provenance.
        assignments: One proposed role per included unit.

    Returns:
        A worst-issue-governed split report with no composite leakage score.

    Raises:
        ValueError: If unit identities or assignment references are malformed.
    """
    ordered_units = tuple(sorted(units, key=lambda item: item.unit_id))
    _require_unique((unit.unit_id for unit in ordered_units), "dataset split unit IDs")
    _require_unique(
        (unit.episode_id for unit in ordered_units),
        "dataset split episode IDs",
    )
    ordered_assignments = tuple(sorted(assignments, key=lambda item: item.unit_id))
    _require_unique(
        (item.unit_id for item in ordered_assignments),
        "dataset split assignment unit IDs",
    )
    unit_by_id = {unit.unit_id: unit for unit in ordered_units}
    assignment_by_id = {
        assignment.unit_id: assignment for assignment in ordered_assignments
    }
    unknown_assignments = set(assignment_by_id) - set(unit_by_id)
    if unknown_assignments:
        raise ValueError("dataset split assignment references an unknown unit")

    issues: list[DatasetSplitIssue] = []
    issues.extend(_assignment_coverage_issues(plan, ordered_units, assignment_by_id))
    issues.extend(_source_split_issues(ordered_units, assignment_by_id))
    issues.extend(_synthetic_provenance_issues(ordered_units, assignment_by_id))
    issues.extend(_lineage_issues(plan, ordered_units, assignment_by_id))
    issues.extend(
        _group_leakage_issues(
            ordered_units,
            assignment_by_id,
            group_name="independence unit",
            kind=SplitIssueKind.INDEPENDENCE_UNIT_LEAKAGE,
            keys=lambda unit: (unit.independence_unit_id,),
        )
    )
    issues.extend(
        _group_leakage_issues(
            ordered_units,
            assignment_by_id,
            group_name="near-duplicate group",
            kind=SplitIssueKind.NEAR_DUPLICATE_LEAKAGE,
            keys=lambda unit: unit.near_duplicate_group_ids,
        )
    )
    issues.extend(_holdout_attribute_issues(plan, ordered_units, assignment_by_id))
    issues.extend(_real_holdout_issues(plan, ordered_units, assignment_by_id))

    if any(issue.severity is SplitIssueSeverity.BLOCKING for issue in issues):
        disposition = DatasetSplitDisposition.INSUFFICIENT_EVIDENCE
    elif any(issue.severity is SplitIssueSeverity.CONTAMINATION for issue in issues):
        disposition = DatasetSplitDisposition.CONTAMINATED
    elif issues:
        disposition = DatasetSplitDisposition.REVIEW_REQUIRED
    else:
        disposition = DatasetSplitDisposition.VALID_WITHIN_SCOPE
    role_counts = Counter(assignment.role.value for assignment in ordered_assignments)
    limitations = _unique_text(
        (
            *plan.scope.limitations,
            *plan.limitations,
            "A valid disposition covers only declared lineage, duplicate groups, "
            "attributes, and independence units.",
            "Undetected semantic or perceptual duplicates may still contaminate "
            "the split.",
            "Split validity does not establish dataset representativeness.",
        ),
        "dataset split report limitations",
    )
    return DatasetSplitReport(
        schema_version=DATASET_SPLIT_REPORT_SCHEMA_VERSION,
        plan_content_digest=plan.content_digest(),
        dataset_collection_digest=plan.dataset_collection_digest,
        disposition=disposition,
        assignments=ordered_assignments,
        role_counts=tuple(role_counts.items()),
        issues=tuple(issues),
        scope=plan.scope,
        evidence_use=plan.evidence_use,
        limitations=limitations,
    )


def _assignment_coverage_issues(
    plan: DatasetSplitPlan,
    units: Sequence[DatasetSplitUnit],
    assignments: dict[str, DatasetSplitAssignment],
) -> tuple[DatasetSplitIssue, ...]:
    """Check assignment completeness, required roles, and minimum sizes."""
    issues: list[DatasetSplitIssue] = []
    for unit in units:
        if unit.unit_id not in assignments:
            issues.append(
                _issue(
                    SplitIssueKind.UNIT_UNASSIGNED,
                    SplitIssueSeverity.BLOCKING,
                    (unit.unit_id,),
                    (),
                    "Dataset unit has no proposed scientific role.",
                )
            )
    declared_roles = set(plan.required_roles)
    for assignment in assignments.values():
        if assignment.role not in declared_roles:
            issues.append(
                _issue(
                    SplitIssueKind.UNDECLARED_ROLE_ASSIGNMENT,
                    SplitIssueSeverity.BLOCKING,
                    (assignment.unit_id,),
                    (assignment.role,),
                    "Assignment uses a scientific role absent from the split plan.",
                )
            )
    counts = Counter(item.role for item in assignments.values())
    for role in plan.required_roles:
        if counts[role] == 0:
            issues.append(
                _issue(
                    SplitIssueKind.MISSING_REQUIRED_ROLE,
                    SplitIssueSeverity.BLOCKING,
                    (),
                    (role,),
                    "A preregistered split role has no assigned units.",
                )
            )
    for minimum in plan.role_minimums:
        count = counts[minimum.role]
        if 0 < count < minimum.minimum_unit_count:
            issues.append(
                _issue(
                    SplitIssueKind.ROLE_BELOW_MINIMUM_SIZE,
                    SplitIssueSeverity.BLOCKING,
                    tuple(
                        sorted(
                            assignment.unit_id
                            for assignment in assignments.values()
                            if assignment.role is minimum.role
                        )
                    ),
                    (minimum.role,),
                    "Assigned units do not meet the preregistered role minimum.",
                )
            )
    return tuple(issues)


def _source_split_issues(
    units: Sequence[DatasetSplitUnit],
    assignments: dict[str, DatasetSplitAssignment],
) -> tuple[DatasetSplitIssue, ...]:
    """Surface disagreements with inherited source split declarations."""
    issues: list[DatasetSplitIssue] = []
    for unit in units:
        assignment = assignments.get(unit.unit_id)
        if (
            assignment is not None
            and unit.source_split is not None
            and unit.source_split != assignment.role.value
        ):
            issues.append(
                _issue(
                    SplitIssueKind.SOURCE_SPLIT_DISAGREEMENT,
                    SplitIssueSeverity.REVIEW,
                    (unit.unit_id,),
                    (assignment.role,),
                    "Proposed role differs from the inherited source split.",
                )
            )
    return tuple(issues)


def _synthetic_provenance_issues(
    units: Sequence[DatasetSplitUnit],
    assignments: dict[str, DatasetSplitAssignment],
) -> tuple[DatasetSplitIssue, ...]:
    """Require synthetic units to retain parents, generator, and transform."""
    synthetic_origins = {
        DatasetUnitOrigin.SYNTHETIC_COUNTERFACTUAL,
        DatasetUnitOrigin.SYNTHETIC_GENERATED,
    }
    issues: list[DatasetSplitIssue] = []
    for unit in units:
        if unit.origin not in synthetic_origins:
            continue
        if (
            not unit.parent_unit_ids
            or unit.generator_digest is None
            or unit.transformation_digest is None
        ):
            assignment = assignments.get(unit.unit_id)
            issues.append(
                _issue(
                    SplitIssueKind.SYNTHETIC_PROVENANCE_INCOMPLETE,
                    SplitIssueSeverity.BLOCKING,
                    (unit.unit_id,),
                    () if assignment is None else (assignment.role,),
                    "Synthetic unit lacks parent, generator, or transformation "
                    "provenance.",
                )
            )
    return tuple(issues)


def _lineage_issues(
    plan: DatasetSplitPlan,
    units: Sequence[DatasetSplitUnit],
    assignments: dict[str, DatasetSplitAssignment],
) -> tuple[DatasetSplitIssue, ...]:
    """Detect missing parents, cycles, and transitive lineage leakage."""
    unit_by_id = {unit.unit_id: unit for unit in units}
    issues: list[DatasetSplitIssue] = []
    missing_edges: list[tuple[str, str]] = []
    for unit in units:
        for parent_id in unit.parent_unit_ids:
            if parent_id not in unit_by_id:
                missing_edges.append((unit.unit_id, parent_id))
    if plan.require_complete_lineage:
        for child_id, parent_id in sorted(missing_edges):
            assignment = assignments.get(child_id)
            issues.append(
                _issue(
                    SplitIssueKind.MISSING_LINEAGE_PARENT,
                    SplitIssueSeverity.BLOCKING,
                    (child_id,),
                    () if assignment is None else (assignment.role,),
                    f"Declared lineage parent '{parent_id}' is absent.",
                )
            )

    cycle_units = _find_cycle_units(unit_by_id)
    if cycle_units:
        issues.append(
            _issue(
                SplitIssueKind.LINEAGE_CYCLE,
                SplitIssueSeverity.BLOCKING,
                cycle_units,
                _roles_for_units(cycle_units, assignments),
                "Dataset derivation lineage contains a cycle.",
            )
        )

    for component in _lineage_components(unit_by_id):
        roles = _roles_for_units(component, assignments)
        if len(roles) > 1:
            issues.append(
                _issue(
                    SplitIssueKind.LINEAGE_LEAKAGE,
                    SplitIssueSeverity.CONTAMINATION,
                    component,
                    roles,
                    "A transitive derivation family crosses scientific roles.",
                )
            )
    return tuple(issues)


def _group_leakage_issues(
    units: Sequence[DatasetSplitUnit],
    assignments: dict[str, DatasetSplitAssignment],
    *,
    group_name: str,
    kind: SplitIssueKind,
    keys: Callable[[DatasetSplitUnit], Sequence[str]],
) -> tuple[DatasetSplitIssue, ...]:
    """Detect a correlation group assigned to more than one role."""
    group_members: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        for key in keys(unit):
            group_members[key].append(unit.unit_id)
    issues: list[DatasetSplitIssue] = []
    for group_id, member_ids in sorted(group_members.items()):
        unit_ids = tuple(sorted(member_ids))
        roles = _roles_for_units(unit_ids, assignments)
        if len(roles) > 1:
            issues.append(
                _issue(
                    kind,
                    SplitIssueSeverity.CONTAMINATION,
                    unit_ids,
                    roles,
                    f"{group_name} '{group_id}' crosses scientific roles.",
                )
            )
    return tuple(issues)


def _holdout_attribute_issues(
    plan: DatasetSplitPlan,
    units: Sequence[DatasetSplitUnit],
    assignments: dict[str, DatasetSplitAssignment],
) -> tuple[DatasetSplitIssue, ...]:
    """Detect protected attribute values shared across holdout boundaries."""
    holdout_roles = set(plan.holdout_roles)
    members: dict[tuple[SplitAttributeKind, str], list[str]] = defaultdict(list)
    issues: list[DatasetSplitIssue] = []
    for unit in units:
        present_kinds = {attribute.kind for attribute in unit.attributes}
        for protected_kind in plan.protected_holdout_attributes:
            if protected_kind not in present_kinds:
                assignment = assignments.get(unit.unit_id)
                issues.append(
                    _issue(
                        SplitIssueKind.MISSING_PROTECTED_ATTRIBUTE,
                        SplitIssueSeverity.BLOCKING,
                        (unit.unit_id,),
                        () if assignment is None else (assignment.role,),
                        f"Unit lacks protected {protected_kind.value} metadata.",
                    )
                )
        for attribute in unit.attributes:
            if attribute.kind in plan.protected_holdout_attributes:
                members[(attribute.kind, attribute.value)].append(unit.unit_id)

    for (kind, value), unit_ids_list in sorted(
        members.items(),
        key=lambda item: (item[0][0].value, item[0][1]),
    ):
        unit_ids = tuple(sorted(unit_ids_list))
        roles = _roles_for_units(unit_ids, assignments)
        represented_holdouts = set(roles) & holdout_roles
        leaks_across_holdout_boundary = any(
            any(role is not holdout_role for role in roles)
            for holdout_role in represented_holdouts
        )
        if leaks_across_holdout_boundary:
            issues.append(
                _issue(
                    SplitIssueKind.HOLDOUT_ATTRIBUTE_LEAKAGE,
                    SplitIssueSeverity.CONTAMINATION,
                    unit_ids,
                    roles,
                    f"Protected {kind.value} value '{value}' appears in both "
                    "development and holdout roles.",
                )
            )
    return tuple(issues)


def _real_holdout_issues(
    plan: DatasetSplitPlan,
    units: Sequence[DatasetSplitUnit],
    assignments: dict[str, DatasetSplitAssignment],
) -> tuple[DatasetSplitIssue, ...]:
    """Prevent non-real evidence from entering protected real-world holdouts."""
    if not plan.require_real_only_holdout:
        return ()
    issues: list[DatasetSplitIssue] = []
    for unit in units:
        assignment = assignments.get(unit.unit_id)
        if (
            assignment is not None
            and assignment.role is DatasetSplitRole.REAL_WORLD_HOLDOUT
            and unit.origin is not DatasetUnitOrigin.REAL_OBSERVED
        ):
            issues.append(
                _issue(
                    SplitIssueKind.NON_REAL_HOLDOUT,
                    SplitIssueSeverity.CONTAMINATION,
                    (unit.unit_id,),
                    (assignment.role,),
                    "Real-world holdout contains simulated or synthetic evidence.",
                )
            )
    return tuple(issues)


def _find_cycle_units(
    units: dict[str, DatasetSplitUnit],
) -> tuple[str, ...]:
    """Return cycle members without recursion-depth dependence."""
    state: dict[str, int] = {}
    cycle_units: set[str] = set()

    for start_id in sorted(units):
        if state.get(start_id, 0) != 0:
            continue
        stack: list[tuple[str, int]] = [(start_id, 0)]
        path: list[str] = []
        path_positions: dict[str, int] = {}
        while stack:
            unit_id, parent_index = stack[-1]
            if state.get(unit_id, 0) == 0:
                state[unit_id] = 1
                path_positions[unit_id] = len(path)
                path.append(unit_id)
            parents = tuple(
                parent_id
                for parent_id in units[unit_id].parent_unit_ids
                if parent_id in units
            )
            if parent_index < len(parents):
                parent_id = parents[parent_index]
                stack[-1] = (unit_id, parent_index + 1)
                if state.get(parent_id, 0) == 0:
                    stack.append((parent_id, 0))
                elif state.get(parent_id) == 1:
                    cycle_units.update(path[path_positions[parent_id] :])
                continue
            stack.pop()
            state[unit_id] = 2
            path_positions.pop(unit_id)
            if not path or path[-1] != unit_id:
                raise RuntimeError("lineage traversal path invariant violated")
            path.pop()
    return tuple(sorted(cycle_units))


def _lineage_components(
    units: dict[str, DatasetSplitUnit],
) -> tuple[tuple[str, ...], ...]:
    """Return undirected connected components of declared lineage."""
    neighbors: dict[str, set[str]] = {unit_id: set() for unit_id in units}
    for unit in units.values():
        for parent_id in unit.parent_unit_ids:
            if parent_id in units:
                neighbors[unit.unit_id].add(parent_id)
                neighbors[parent_id].add(unit.unit_id)
    visited: set[str] = set()
    components: list[tuple[str, ...]] = []
    for unit_id in sorted(units):
        if unit_id in visited:
            continue
        pending = [unit_id]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(neighbors[current] - component)
        visited.update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _roles_for_units(
    unit_ids: Sequence[str],
    assignments: dict[str, DatasetSplitAssignment],
) -> tuple[DatasetSplitRole, ...]:
    """Return canonical roles represented by assigned units."""
    return tuple(
        sorted(
            {
                assignments[unit_id].role
                for unit_id in unit_ids
                if unit_id in assignments
            }
        )
    )


def _issue(
    kind: SplitIssueKind,
    severity: SplitIssueSeverity,
    unit_ids: tuple[str, ...],
    roles: tuple[DatasetSplitRole, ...],
    description: str,
) -> DatasetSplitIssue:
    """Build one validated split issue."""
    return DatasetSplitIssue(
        kind=kind,
        severity=severity,
        unit_ids=unit_ids,
        roles=roles,
        description=description,
    )


def _validate_disposition(
    disposition: DatasetSplitDisposition,
    issues: Sequence[DatasetSplitIssue],
) -> None:
    """Require report disposition to match its worst issue."""
    has_blocking = any(
        issue.severity is SplitIssueSeverity.BLOCKING for issue in issues
    )
    has_contamination = any(
        issue.severity is SplitIssueSeverity.CONTAMINATION for issue in issues
    )
    has_review = any(issue.severity is SplitIssueSeverity.REVIEW for issue in issues)
    expected = (
        DatasetSplitDisposition.INSUFFICIENT_EVIDENCE
        if has_blocking
        else (
            DatasetSplitDisposition.CONTAMINATED
            if has_contamination
            else (
                DatasetSplitDisposition.REVIEW_REQUIRED
                if has_review
                else DatasetSplitDisposition.VALID_WITHIN_SCOPE
            )
        )
    )
    if disposition is not expected:
        raise ValueError("dataset split disposition does not match worst issue")


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
