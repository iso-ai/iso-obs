"""Provenance-preserving contracts for reliable dataset curation.

The module records source data, temporally scoped label assertions, and typed
evidence without treating model suggestions as ground truth. The deterministic
audit is deliberately conservative: it identifies review obligations and may
abstain, but it does not infer causal mechanisms or rewrite source labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse

DATASET_RELIABILITY_REPORT_SCHEMA_VERSION = "iso-obs.dataset-reliability-report.v1"


class DatasetEvidenceRole(StrEnum):
    """Permitted scientific role for a dataset."""

    REPRESENTATION_LEARNING = "representation_learning"
    FAILURE_LABEL_LEARNING = "failure_label_learning"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    EVALUATION_ONLY = "evaluation_only"
    SYNTHETIC_SEED = "synthetic_seed"


class ModeFamily(StrEnum):
    """Family of candidate regimes kept distinct during curation."""

    OPERATING = "operating"
    BEHAVIORAL = "behavioral"
    ENVIRONMENTAL = "environmental"
    SYSTEM = "system"
    DATA_QUALITY = "data_quality"
    FAILURE_STATE = "failure_state"
    FAILURE_MECHANISM = "failure_mechanism"


class LabelStatus(StrEnum):
    """Epistemic status of a dataset label assertion."""

    REPORTED = "reported"
    SUGGESTED = "suggested"
    CORROBORATED = "corroborated"
    ADJUDICATED = "adjudicated"
    DISPUTED = "disputed"
    UNKNOWN = "unknown"


class EvidenceRelation(StrEnum):
    """Typed relationship between evidence and a label assertion."""

    OBSERVED_OUTCOME = "observed_outcome"
    CONDITION_PRESENT = "condition_present"
    MECHANISM_HYPOTHESIZED = "mechanism_hypothesized"
    MECHANISM_CONFIRMED = "mechanism_confirmed"
    CAUSED_BY_INTERVENTION = "caused_by_intervention"
    REPRODUCED_IN_SIM = "reproduced_in_sim"
    NOT_REPRODUCED = "not_reproduced"
    ANALOGOUS_TO = "analogous_to"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"


class DatasetAuditIssueKind(StrEnum):
    """Kind of review obligation discovered by a dataset audit."""

    MISSING_MODE_FAMILY = "missing_mode_family"
    UNVERIFIED_REPORTED_LABEL = "unverified_reported_label"
    MACHINE_SUGGESTION = "machine_suggestion"
    AMBIGUOUS_CANDIDATE_SET = "ambiguous_candidate_set"
    DISPUTED_LABEL = "disputed_label"
    UNKNOWN_LABEL = "unknown_label"
    INSUFFICIENT_CORROBORATION = "insufficient_corroboration"
    OVERLAPPING_MODE_ASSERTIONS = "overlapping_mode_assertions"


class DatasetAuditDisposition(StrEnum):
    """Strongest dataset-curation claim supported by an audit."""

    CURATED_WITHIN_SCOPE = "curated_within_scope"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class DatasetScope:
    """Declared population, acquisition process, and scientific uses."""

    domain_namespace: str
    target_population_digest: str
    collection_protocol_digest: str
    modality_ids: tuple[str, ...]
    evidence_roles: tuple[DatasetEvidenceRole, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and canonicalize dataset scope."""
        _require_text(self.domain_namespace, "dataset domain namespace")
        NamedDigest("target population", self.target_population_digest)
        NamedDigest("collection protocol", self.collection_protocol_digest)
        modalities = _unique_text(self.modality_ids, "dataset modality IDs")
        roles = tuple(
            sorted({DatasetEvidenceRole(role) for role in self.evidence_roles})
        )
        if not roles:
            raise ValueError("dataset scope requires at least one evidence role")
        limitations = _unique_text(
            self.limitations,
            "dataset scope limitations",
            required=False,
        )
        object.__setattr__(self, "modality_ids", modalities)
        object.__setattr__(self, "evidence_roles", roles)
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable identity and provenance for one source dataset version."""

    dataset_id: str
    dataset_version: str
    content_digest: str
    source_uri: str
    license_id: str
    scope: DatasetScope
    parent_content_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate source identity, license, and lineage."""
        _require_text(self.dataset_id, "dataset ID")
        _require_text(self.dataset_version, "dataset version")
        NamedDigest("dataset content", self.content_digest)
        _require_text(self.source_uri, "dataset source URI")
        _require_text(self.license_id, "dataset license ID")
        parents = tuple(sorted(set(self.parent_content_digests)))
        for digest in parents:
            NamedDigest("parent dataset content", digest)
        if self.content_digest in parents:
            raise ValueError("dataset cannot derive from its own content digest")
        object.__setattr__(self, "parent_content_digests", parents)

    def to_json(self) -> str:
        """Serialize the manifest to canonical JSON."""
        return _canonical_json(self)

    def manifest_digest(self) -> str:
        """Calculate the exact manifest content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    """Immutable temporal episode within a source dataset."""

    dataset_manifest_digest: str
    episode_id: str
    independence_unit_id: str
    content_digest: str
    start_seconds: float
    end_seconds: float
    modality_digests: tuple[NamedDigest, ...]
    source_split: str | None = None

    def __post_init__(self) -> None:
        """Validate episode identity, interval, and modality provenance."""
        NamedDigest("dataset manifest", self.dataset_manifest_digest)
        _require_text(self.episode_id, "dataset episode ID")
        _require_text(self.independence_unit_id, "episode independence unit ID")
        NamedDigest("episode content", self.content_digest)
        start, end = _closed_interval(
            self.start_seconds,
            self.end_seconds,
            "episode",
        )
        if not self.modality_digests:
            raise ValueError("episode requires at least one modality digest")
        modalities = tuple(sorted(self.modality_digests, key=lambda item: item.name))
        _require_unique(
            (item.name for item in modalities),
            "episode modality names",
        )
        if self.source_split is not None:
            _require_text(self.source_split, "episode source split")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "modality_digests", modalities)


@dataclass(frozen=True, slots=True)
class LabelEvidence:
    """One independently attributable item supporting or opposing a label."""

    evidence_id: str
    evidence_digest: str
    independent_source_id: str
    relation: EvidenceRelation
    description: str

    def __post_init__(self) -> None:
        """Validate evidence identity, source, relation, and description."""
        _require_text(self.evidence_id, "label evidence ID")
        NamedDigest("label evidence", self.evidence_digest)
        _require_text(
            self.independent_source_id,
            "label evidence independent source ID",
        )
        _require_text(self.description, "label evidence description")
        object.__setattr__(self, "relation", EvidenceRelation(self.relation))


@dataclass(frozen=True, slots=True)
class DatasetLabelAssertion:
    """Evidence-qualified candidate label over a temporal interval."""

    assertion_id: str
    episode_id: str
    label_namespace: str
    mode_family: ModeFamily
    start_seconds: float
    end_seconds: float
    candidate_values: tuple[str, ...]
    status: LabelStatus
    evidence: tuple[LabelEvidence, ...]
    method_digest: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate candidate set, temporal scope, and epistemic status."""
        _require_text(self.assertion_id, "dataset label assertion ID")
        _require_text(self.episode_id, "assertion episode ID")
        _require_text(self.label_namespace, "label namespace")
        start, end = _closed_interval(
            self.start_seconds,
            self.end_seconds,
            "label assertion",
        )
        candidates = _unique_text(
            self.candidate_values,
            "label candidate values",
            required=False,
        )
        status = LabelStatus(self.status)
        if status is LabelStatus.UNKNOWN and candidates:
            raise ValueError("unknown label assertion must have an empty candidate set")
        if status is LabelStatus.DISPUTED and len(candidates) < 2:
            raise ValueError("disputed label assertion requires multiple candidates")
        if (
            status
            in {
                LabelStatus.REPORTED,
                LabelStatus.CORROBORATED,
                LabelStatus.ADJUDICATED,
            }
            and not candidates
        ):
            raise ValueError(f"{status.value} label assertion requires a candidate")
        if status is LabelStatus.SUGGESTED and self.method_digest is None:
            raise ValueError("suggested label assertion requires a method digest")
        if self.method_digest is not None:
            NamedDigest("labeling method", self.method_digest)
        evidence = tuple(sorted(self.evidence, key=lambda item: item.evidence_id))
        _require_unique(
            (item.evidence_id for item in evidence),
            "label evidence IDs",
        )
        limitations = _unique_text(
            self.limitations,
            "label assertion limitations",
            required=False,
        )
        object.__setattr__(self, "mode_family", ModeFamily(self.mode_family))
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "candidate_values", candidates)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class DatasetReliabilityAuditPlan:
    """Content-addressed rules for auditing a dataset version."""

    plan_id: str
    plan_version: str
    dataset_manifest_digest: str
    label_taxonomy_digest: str
    required_mode_families: tuple[ModeFamily, ...]
    minimum_independent_corroborating_sources: int = 2
    limitations: tuple[str, ...] = ()
    evidence_use: EvidenceUse = EvidenceUse.DISCOVERY_ONLY

    def __post_init__(self) -> None:
        """Validate the audit target and conservative review rules."""
        _require_text(self.plan_id, "dataset reliability audit plan ID")
        _require_text(self.plan_version, "dataset reliability audit plan version")
        NamedDigest("dataset manifest", self.dataset_manifest_digest)
        NamedDigest("label taxonomy", self.label_taxonomy_digest)
        families = tuple(
            sorted({ModeFamily(family) for family in self.required_mode_families})
        )
        if not families:
            raise ValueError("audit plan requires at least one mode family")
        minimum_sources = self.minimum_independent_corroborating_sources
        if (
            isinstance(minimum_sources, bool)
            or not isinstance(minimum_sources, int)
            or minimum_sources < 2
        ):
            raise ValueError(
                "minimum independent corroborating sources must be at least two"
            )
        limitations = _unique_text(
            self.limitations,
            "dataset reliability audit plan limitations",
            required=False,
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("dataset reliability audit must be discovery-only")
        object.__setattr__(self, "required_mode_families", families)
        object.__setattr__(
            self,
            "minimum_independent_corroborating_sources",
            minimum_sources,
        )
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_use", evidence_use)

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact plan content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class DatasetAuditIssue:
    """One review obligation discovered without rewriting source evidence."""

    kind: DatasetAuditIssueKind
    episode_id: str
    assertion_ids: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        """Validate issue classification and affected assertions."""
        _require_text(self.episode_id, "dataset audit issue episode ID")
        assertion_ids = _unique_text(
            self.assertion_ids,
            "dataset audit issue assertion IDs",
            required=False,
        )
        _require_text(self.description, "dataset audit issue description")
        object.__setattr__(self, "kind", DatasetAuditIssueKind(self.kind))
        object.__setattr__(self, "assertion_ids", assertion_ids)


@dataclass(frozen=True, slots=True)
class DatasetReliabilityReport:
    """Versioned, content-addressed result of a dataset reliability audit."""

    schema_version: str
    plan_content_digest: str
    dataset_manifest_digest: str
    disposition: DatasetAuditDisposition
    episode_count: int
    assertion_count: int
    status_counts: tuple[tuple[str, int], ...]
    issues: tuple[DatasetAuditIssue, ...]
    scope: DatasetScope
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate report identity, counts, ordering, and limitations."""
        if self.schema_version != DATASET_RELIABILITY_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported dataset reliability report schema version")
        NamedDigest("dataset audit plan", self.plan_content_digest)
        NamedDigest("dataset manifest", self.dataset_manifest_digest)
        if self.episode_count < 0 or self.assertion_count < 0:
            raise ValueError("dataset reliability counts must be nonnegative")
        status_counts = tuple(sorted(self.status_counts))
        _require_unique(
            (status for status, _ in status_counts),
            "dataset reliability status count labels",
        )
        for status, count in status_counts:
            LabelStatus(status)
            if count < 0:
                raise ValueError(
                    "dataset reliability status counts must be nonnegative"
                )
        issues = tuple(
            sorted(
                self.issues,
                key=lambda item: (
                    item.episode_id,
                    item.kind.value,
                    item.assertion_ids,
                ),
            )
        )
        limitations = _unique_text(
            self.limitations,
            "dataset reliability report limitations",
            required=False,
        )
        object.__setattr__(
            self,
            "disposition",
            DatasetAuditDisposition(self.disposition),
        )
        object.__setattr__(self, "status_counts", status_counts)
        object.__setattr__(self, "issues", issues)
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("dataset reliability report must be discovery-only")
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact report content digest."""
        return sha256_digest(self.to_json())


def audit_dataset_reliability(
    plan: DatasetReliabilityAuditPlan,
    manifest: DatasetManifest,
    episodes: Sequence[EpisodeManifest],
    assertions: Sequence[DatasetLabelAssertion],
) -> DatasetReliabilityReport:
    """Audit evidence-qualified labels without changing their source status.

    Args:
        plan: Preregistered audit rules and required mode families.
        manifest: Immutable source dataset identity and scope.
        episodes: Independently identified temporal episodes.
        assertions: Reported, suggested, reviewed, or abstaining labels.

    Returns:
        A deterministic review report whose worst issue governs disposition.

    Raises:
        ValueError: If identities, temporal extents, or references do not align.
    """
    if plan.dataset_manifest_digest != manifest.manifest_digest():
        raise ValueError("audit plan does not target the supplied dataset manifest")
    ordered_episodes = tuple(sorted(episodes, key=lambda item: item.episode_id))
    _require_unique(
        (episode.episode_id for episode in ordered_episodes),
        "dataset episode IDs",
    )
    _require_unique(
        (episode.independence_unit_id for episode in ordered_episodes),
        "dataset episode independence unit IDs",
    )
    for episode in ordered_episodes:
        if episode.dataset_manifest_digest != manifest.manifest_digest():
            raise ValueError("episode references a different dataset manifest")

    ordered_assertions = tuple(sorted(assertions, key=lambda item: item.assertion_id))
    _require_unique(
        (assertion.assertion_id for assertion in ordered_assertions),
        "dataset label assertion IDs",
    )
    episode_by_id = {episode.episode_id: episode for episode in ordered_episodes}
    for assertion in ordered_assertions:
        referenced_episode = episode_by_id.get(assertion.episode_id)
        if referenced_episode is None:
            raise ValueError("label assertion references an unknown episode")
        if (
            assertion.start_seconds < referenced_episode.start_seconds
            or assertion.end_seconds > referenced_episode.end_seconds
        ):
            raise ValueError("label assertion falls outside its episode interval")

    issues = _audit_assertions(plan, ordered_episodes, ordered_assertions)
    if not ordered_episodes or not ordered_assertions:
        disposition = DatasetAuditDisposition.INSUFFICIENT_EVIDENCE
    elif issues:
        disposition = DatasetAuditDisposition.REVIEW_REQUIRED
    else:
        disposition = DatasetAuditDisposition.CURATED_WITHIN_SCOPE

    counts = Counter(assertion.status.value for assertion in ordered_assertions)
    limitations = _unique_text(
        (
            *manifest.scope.limitations,
            *plan.limitations,
            "The audit detects review obligations but does not establish causality.",
            "A curated disposition applies only to the declared dataset scope.",
        ),
        "dataset reliability report limitations",
    )
    return DatasetReliabilityReport(
        schema_version=DATASET_RELIABILITY_REPORT_SCHEMA_VERSION,
        plan_content_digest=plan.content_digest(),
        dataset_manifest_digest=manifest.manifest_digest(),
        disposition=disposition,
        episode_count=len(ordered_episodes),
        assertion_count=len(ordered_assertions),
        status_counts=tuple(counts.items()),
        issues=issues,
        scope=manifest.scope,
        evidence_use=plan.evidence_use,
        limitations=limitations,
    )


def _audit_assertions(
    plan: DatasetReliabilityAuditPlan,
    episodes: Sequence[EpisodeManifest],
    assertions: Sequence[DatasetLabelAssertion],
) -> tuple[DatasetAuditIssue, ...]:
    """Identify conservative review obligations in canonical order."""
    issues: list[DatasetAuditIssue] = []
    by_episode: dict[str, list[DatasetLabelAssertion]] = defaultdict(list)
    for assertion in assertions:
        by_episode[assertion.episode_id].append(assertion)
        issues.extend(_assertion_issues(plan, assertion))

    for episode in episodes:
        episode_assertions = by_episode[episode.episode_id]
        for family in plan.required_mode_families:
            family_assertions = tuple(
                item for item in episode_assertions if item.mode_family is family
            )
            if not _covers_episode(episode, family_assertions):
                issues.append(
                    DatasetAuditIssue(
                        kind=DatasetAuditIssueKind.MISSING_MODE_FAMILY,
                        episode_id=episode.episode_id,
                        assertion_ids=(),
                        description=(
                            f"{family.value} assertions do not cover the complete "
                            "episode interval."
                        ),
                    )
                )
        issues.extend(_overlapping_mode_issues(episode.episode_id, episode_assertions))
    return tuple(issues)


def _assertion_issues(
    plan: DatasetReliabilityAuditPlan,
    assertion: DatasetLabelAssertion,
) -> tuple[DatasetAuditIssue, ...]:
    """Return all review obligations attached to one assertion."""
    issues: list[DatasetAuditIssue] = []
    mapping = {
        LabelStatus.REPORTED: (
            DatasetAuditIssueKind.UNVERIFIED_REPORTED_LABEL,
            "A source-reported label remains unverified.",
        ),
        LabelStatus.SUGGESTED: (
            DatasetAuditIssueKind.MACHINE_SUGGESTION,
            "Machine-suggested labels require independent review.",
        ),
        LabelStatus.DISPUTED: (
            DatasetAuditIssueKind.DISPUTED_LABEL,
            "Disputed labels must preserve alternatives until adjudication.",
        ),
        LabelStatus.UNKNOWN: (
            DatasetAuditIssueKind.UNKNOWN_LABEL,
            "The label remains explicitly unknown.",
        ),
    }
    if assertion.status in mapping:
        kind, description = mapping[assertion.status]
        issues.append(
            DatasetAuditIssue(
                kind=kind,
                episode_id=assertion.episode_id,
                assertion_ids=(assertion.assertion_id,),
                description=description,
            )
        )
    if len(assertion.candidate_values) > 1:
        issues.append(
            DatasetAuditIssue(
                kind=DatasetAuditIssueKind.AMBIGUOUS_CANDIDATE_SET,
                episode_id=assertion.episode_id,
                assertion_ids=(assertion.assertion_id,),
                description="Multiple candidate values remain supported.",
            )
        )
    if assertion.status is LabelStatus.CORROBORATED:
        supporting_relations = {
            EvidenceRelation.OBSERVED_OUTCOME,
            EvidenceRelation.CONDITION_PRESENT,
            EvidenceRelation.MECHANISM_CONFIRMED,
            EvidenceRelation.CAUSED_BY_INTERVENTION,
            EvidenceRelation.REPRODUCED_IN_SIM,
        }
        source_count = len(
            {
                item.independent_source_id
                for item in assertion.evidence
                if item.relation in supporting_relations
            }
        )
        if source_count < plan.minimum_independent_corroborating_sources:
            issues.append(
                DatasetAuditIssue(
                    kind=DatasetAuditIssueKind.INSUFFICIENT_CORROBORATION,
                    episode_id=assertion.episode_id,
                    assertion_ids=(assertion.assertion_id,),
                    description=(
                        "Corroborated status lacks the required number of "
                        "independent evidence sources."
                    ),
                )
            )
    return tuple(issues)


def _covers_episode(
    episode: EpisodeManifest,
    assertions: Sequence[DatasetLabelAssertion],
) -> bool:
    """Return whether assertion intervals cover an entire episode."""
    if not assertions:
        return False
    intervals = sorted((item.start_seconds, item.end_seconds) for item in assertions)
    covered_until = episode.start_seconds
    for start, end in intervals:
        if start > covered_until:
            return False
        covered_until = max(covered_until, end)
        if covered_until >= episode.end_seconds:
            return True
    return covered_until >= episode.end_seconds


def _overlapping_mode_issues(
    episode_id: str,
    assertions: Sequence[DatasetLabelAssertion],
) -> tuple[DatasetAuditIssue, ...]:
    """Flag incompatible singleton candidates over overlapping intervals."""
    issues: list[DatasetAuditIssue] = []
    ordered = sorted(
        assertions,
        key=lambda item: (
            item.mode_family.value,
            item.start_seconds,
            item.end_seconds,
            item.assertion_id,
        ),
    )
    for index, left in enumerate(ordered):
        if len(left.candidate_values) != 1:
            continue
        for right in ordered[index + 1 :]:
            if right.mode_family is not left.mode_family:
                continue
            if right.start_seconds > left.end_seconds:
                break
            if (
                len(right.candidate_values) == 1
                and right.candidate_values != left.candidate_values
                and _intervals_overlap(left, right)
            ):
                issues.append(
                    DatasetAuditIssue(
                        kind=DatasetAuditIssueKind.OVERLAPPING_MODE_ASSERTIONS,
                        episode_id=episode_id,
                        assertion_ids=(
                            left.assertion_id,
                            right.assertion_id,
                        ),
                        description=(
                            "Distinct candidate modes overlap within one mode "
                            "family; this is a review candidate, not an "
                            "automatically resolved conflict."
                        ),
                    )
                )
    return tuple(issues)


def _intervals_overlap(
    left: DatasetLabelAssertion,
    right: DatasetLabelAssertion,
) -> bool:
    """Return whether two temporal intervals overlap beyond a shared boundary."""
    return max(left.start_seconds, right.start_seconds) < min(
        left.end_seconds,
        right.end_seconds,
    )


def _closed_interval(
    lower: float,
    upper: float,
    label: str,
) -> tuple[float, float]:
    """Validate a finite closed interval."""
    lower_value = float(lower)
    upper_value = float(upper)
    if not (
        lower_value == lower_value
        and upper_value == upper_value
        and abs(lower_value) != float("inf")
        and abs(upper_value) != float("inf")
    ):
        raise ValueError(f"{label} interval must be finite")
    if lower_value > upper_value:
        raise ValueError(f"{label} interval lower bound must not exceed upper")
    return lower_value, upper_value


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
