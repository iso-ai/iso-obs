"""Scoped release assurance without composite confidence scores.

This module assembles qualified regression evidence, completed simulation
campaigns, independent challenge evidence, and explicit defeaters. Evidence
cannot compensate numerically for a failed required campaign or an unresolved
critical defeater. The resulting disposition is technical decision support,
not an authorization to deploy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .evidence import NamedDigest, _canonical_json, sha256_digest
from .execution import SimulationCampaignReport
from .promotion import (
    PromotionStatus,
    RegressionPromotionReview,
)


class _ContentAddressed(Protocol):
    """Structural contract for immutable evidence artifacts."""

    def content_digest(self) -> str:
        """Return the artifact's content digest."""


class ChallengeKind(StrEnum):
    """Independent or complementary source of challenge evidence."""

    PHYSICAL_VALIDATION = "physical_validation"
    CROSS_BACKEND_VALIDATION = "cross_backend_validation"
    RED_TEAM = "red_team"
    FIELD_HISTORY = "field_history"
    FORMAL_ANALYSIS = "formal_analysis"


class ChallengeOutcome(StrEnum):
    """Outcome of one scoped challenge activity."""

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class DefeaterSeverity(StrEnum):
    """Consequence of an unresolved counterargument to release assurance."""

    CRITICAL = "critical"
    MATERIAL = "material"
    MONITOR = "monitor"


class DefeaterStatus(StrEnum):
    """Resolution state of an assurance defeater."""

    OPEN = "open"
    RESOLVED = "resolved"


class AssuranceDisposition(StrEnum):
    """Bounded technical disposition for the declared release scope."""

    BLOCKED = "blocked"
    RESTRICTED = "restricted"
    SUPPORTED_WITHIN_SCOPE = "supported_within_scope"


class AssuranceReasonCode(StrEnum):
    """Machine-readable reason affecting an assurance disposition."""

    PROMOTION_NOT_QUALIFIED = "promotion_not_qualified"
    CAMPAIGN_NOT_RELEASE_ELIGIBLE = "campaign_not_release_eligible"
    REGRESSION_GATE_MISSING = "regression_gate_missing"
    REGRESSION_GATE_FAILED = "regression_gate_failed"
    CHALLENGE_FAILED = "challenge_failed"
    INDEPENDENT_CHALLENGE_INSUFFICIENT = "independent_challenge_insufficient"
    CHALLENGE_INCONCLUSIVE = "challenge_inconclusive"
    CRITICAL_DEFEATER_OPEN = "critical_defeater_open"
    MATERIAL_DEFEATER_OPEN = "material_defeater_open"


@dataclass(frozen=True, slots=True)
class ReleaseScope:
    """Exact system and operational scope of an assurance claim."""

    scope_id: str
    scope_version: str
    intended_use: str
    system_artifacts: tuple[NamedDigest, ...]
    operational_domain_digest: str
    validity_envelope_digest: str
    risk_acceptance_policy_digest: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and canonicalize release-scope identity."""
        for value, label in (
            (self.scope_id, "release scope ID"),
            (self.scope_version, "release scope version"),
            (self.intended_use, "release intended use"),
        ):
            _require_text(value, label)
        if not self.system_artifacts:
            raise ValueError("release scope requires system artifacts")
        _require_unique(
            (item.name for item in self.system_artifacts),
            "release system artifact names",
        )
        for value, label in (
            (self.operational_domain_digest, "operational domain"),
            (self.validity_envelope_digest, "validity envelope"),
            (self.risk_acceptance_policy_digest, "risk acceptance policy"),
        ):
            NamedDigest(label, value)
        _require_unique_text(self.limitations, "release scope limitations")
        object.__setattr__(
            self,
            "system_artifacts",
            tuple(sorted(self.system_artifacts, key=lambda item: item.name)),
        )
        object.__setattr__(self, "limitations", tuple(sorted(self.limitations)))

    def to_json(self) -> str:
        """Serialize the scope to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact release-scope content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class AssurancePolicy:
    """Pre-specified evidence requirements for one assurance dossier."""

    policy_id: str
    policy_version: str
    required_failure_fingerprints: tuple[str, ...]
    minimum_independent_challenges: int = 1
    required_independent_challenge_kinds: tuple[ChallengeKind, ...] = ()

    def __post_init__(self) -> None:
        """Validate required coverage and independent-challenge criteria."""
        _require_text(self.policy_id, "assurance policy ID")
        _require_text(self.policy_version, "assurance policy version")
        _require_unique_text(
            self.required_failure_fingerprints,
            "required failure fingerprints",
        )
        for fingerprint in self.required_failure_fingerprints:
            NamedDigest("required failure fingerprint", fingerprint)
        if (
            isinstance(self.minimum_independent_challenges, bool)
            or not isinstance(self.minimum_independent_challenges, int)
            or self.minimum_independent_challenges < 0
        ):
            raise ValueError(
                "minimum_independent_challenges must be a non-negative integer"
            )
        kinds = tuple(
            ChallengeKind(item) for item in self.required_independent_challenge_kinds
        )
        _require_unique(kinds, "required independent challenge kinds")
        object.__setattr__(
            self,
            "required_failure_fingerprints",
            tuple(sorted(self.required_failure_fingerprints)),
        )
        object.__setattr__(
            self,
            "required_independent_challenge_kinds",
            tuple(sorted(kinds, key=lambda item: item.value)),
        )

    def to_json(self) -> str:
        """Serialize the policy to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact policy content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class RegressionEvidenceLink:
    """Link one required failure to promotion and campaign evidence."""

    failure_fingerprint: str
    promotion_review_digest: str
    campaign_report_digest: str

    def __post_init__(self) -> None:
        """Validate linked artifact identities."""
        for value, label in (
            (self.failure_fingerprint, "linked failure fingerprint"),
            (self.promotion_review_digest, "linked promotion review"),
            (self.campaign_report_digest, "linked campaign report"),
        ):
            NamedDigest(label, value)


@dataclass(frozen=True, slots=True)
class ChallengeEvidence:
    """Scoped result from an independent or complementary challenge."""

    evidence_id: str
    kind: ChallengeKind
    outcome: ChallengeOutcome
    artifact_digest: str
    release_scope_digest: str
    independent_from_development: bool
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate challenge identity, scope, and epistemic limitations."""
        _require_text(self.evidence_id, "challenge evidence ID")
        object.__setattr__(self, "kind", ChallengeKind(self.kind))
        object.__setattr__(self, "outcome", ChallengeOutcome(self.outcome))
        NamedDigest("challenge artifact", self.artifact_digest)
        NamedDigest("challenge release scope", self.release_scope_digest)
        if not isinstance(self.independent_from_development, bool):
            raise ValueError("independent_from_development must be a boolean")
        _require_unique_text(
            self.limitations,
            "challenge evidence limitations",
        )
        object.__setattr__(self, "limitations", tuple(sorted(self.limitations)))


@dataclass(frozen=True, slots=True)
class AssuranceDefeater:
    """Explicit counterargument that may weaken or block assurance."""

    defeater_id: str
    description: str
    severity: DefeaterSeverity
    status: DefeaterStatus
    release_scope_digest: str
    evidence_digest: str
    resolution_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate defeater scope, evidence, and resolution state."""
        _require_text(self.defeater_id, "assurance defeater ID")
        _require_text(self.description, "assurance defeater description")
        object.__setattr__(self, "severity", DefeaterSeverity(self.severity))
        object.__setattr__(self, "status", DefeaterStatus(self.status))
        NamedDigest("defeater release scope", self.release_scope_digest)
        NamedDigest("defeater evidence", self.evidence_digest)
        if self.status is DefeaterStatus.RESOLVED:
            if self.resolution_digest is None:
                raise ValueError("resolved defeater requires a resolution digest")
            NamedDigest("defeater resolution", self.resolution_digest)
        elif self.resolution_digest is not None:
            raise ValueError("open defeater must not declare a resolution digest")


@dataclass(frozen=True, slots=True)
class ReleaseAssuranceDossier:
    """Content-addressed evidence plan for a scoped release claim."""

    dossier_id: str
    dossier_version: str
    release_scope: ReleaseScope
    policy: AssurancePolicy
    regression_evidence: tuple[RegressionEvidenceLink, ...]
    challenge_evidence: tuple[ChallengeEvidence, ...]
    defeaters: tuple[AssuranceDefeater, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate coverage, scope identity, and canonical ordering."""
        _require_text(self.dossier_id, "assurance dossier ID")
        _require_text(self.dossier_version, "assurance dossier version")
        _require_unique(
            (item.failure_fingerprint for item in self.regression_evidence),
            "dossier regression failure fingerprints",
        )
        linked_fingerprints = {
            item.failure_fingerprint for item in self.regression_evidence
        }
        required_fingerprints = set(self.policy.required_failure_fingerprints)
        if linked_fingerprints != required_fingerprints:
            missing = sorted(required_fingerprints - linked_fingerprints)
            extra = sorted(linked_fingerprints - required_fingerprints)
            raise ValueError(
                "regression evidence must exactly cover policy fingerprints "
                f"(missing={missing}, extra={extra})"
            )
        _require_unique(
            (item.promotion_review_digest for item in self.regression_evidence),
            "dossier promotion review digests",
        )
        _require_unique(
            (item.campaign_report_digest for item in self.regression_evidence),
            "dossier campaign report digests",
        )
        _require_unique(
            (item.evidence_id for item in self.challenge_evidence),
            "challenge evidence IDs",
        )
        _require_unique(
            (item.defeater_id for item in self.defeaters),
            "assurance defeater IDs",
        )
        scope_digest = self.release_scope.content_digest()
        for challenge in self.challenge_evidence:
            if challenge.release_scope_digest != scope_digest:
                raise ValueError(
                    f"challenge '{challenge.evidence_id}' release scope "
                    "digest mismatch"
                )
        for defeater in self.defeaters:
            if defeater.release_scope_digest != scope_digest:
                raise ValueError(
                    f"defeater '{defeater.defeater_id}' release scope "
                    "digest mismatch"
                )
        _require_unique_text(self.limitations, "assurance dossier limitations")
        object.__setattr__(
            self,
            "regression_evidence",
            tuple(
                sorted(
                    self.regression_evidence,
                    key=lambda item: item.failure_fingerprint,
                )
            ),
        )
        object.__setattr__(
            self,
            "challenge_evidence",
            tuple(
                sorted(
                    self.challenge_evidence,
                    key=lambda item: item.evidence_id,
                )
            ),
        )
        object.__setattr__(
            self,
            "defeaters",
            tuple(sorted(self.defeaters, key=lambda item: item.defeater_id)),
        )
        object.__setattr__(self, "limitations", tuple(sorted(self.limitations)))

    def to_json(self) -> str:
        """Serialize the dossier to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact dossier content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class RegressionAssuranceResult:
    """Qualification and execution outcome for one known failure."""

    failure_fingerprint: str
    regression_pack_digest: str
    promotion_qualified: bool
    campaign_release_eligible: bool
    regression_gate_passed: bool
    reason_codes: tuple[AssuranceReasonCode, ...]


@dataclass(frozen=True, slots=True)
class ReleaseAssuranceReport:
    """Deterministic technical disposition for one release dossier."""

    dossier_content_digest: str
    release_scope_digest: str
    policy_content_digest: str
    regression_results: tuple[RegressionAssuranceResult, ...]
    independent_passing_challenge_count: int
    independent_passing_challenge_kind_counts: tuple[tuple[ChallengeKind, int], ...]
    open_defeater_ids: tuple[str, ...]
    blocking_reasons: tuple[AssuranceReasonCode, ...]
    restriction_reasons: tuple[AssuranceReasonCode, ...]
    disposition: AssuranceDisposition
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def evaluate_release_assurance(
    dossier: ReleaseAssuranceDossier,
    *,
    promotion_reviews: Sequence[RegressionPromotionReview],
    campaign_reports: Sequence[SimulationCampaignReport],
) -> ReleaseAssuranceReport:
    """Evaluate a scoped assurance dossier without evidence averaging.

    Args:
        dossier: Pre-specified scope, policy, evidence links, and defeaters.
        promotion_reviews: Exact promotion artifacts linked by the dossier.
        campaign_reports: Exact campaign artifacts linked by the dossier.

    Returns:
        Per-failure results, explicit reasons, and bounded disposition.

    Raises:
        ValueError: If supplied artifacts are duplicate, missing, extra, or
            inconsistent with their declared evidence links.
    """
    promotions_by_digest = _index_artifacts(
        promotion_reviews,
        "promotion reviews",
    )
    campaigns_by_digest = _index_artifacts(
        campaign_reports,
        "campaign reports",
    )
    expected_promotions = {
        item.promotion_review_digest for item in dossier.regression_evidence
    }
    expected_campaigns = {
        item.campaign_report_digest for item in dossier.regression_evidence
    }
    _require_exact_artifact_set(
        set(promotions_by_digest),
        expected_promotions,
        "promotion reviews",
    )
    _require_exact_artifact_set(
        set(campaigns_by_digest),
        expected_campaigns,
        "campaign reports",
    )

    regression_results = []
    blocking: set[AssuranceReasonCode] = set()
    for link in dossier.regression_evidence:
        promotion = promotions_by_digest[link.promotion_review_digest]
        campaign = campaigns_by_digest[link.campaign_report_digest]
        if promotion.source_failure_fingerprint != link.failure_fingerprint:
            raise ValueError(
                "promotion review failure fingerprint does not match "
                f"evidence link '{link.failure_fingerprint}'"
            )
        if campaign.pack_content_digest != promotion.regression_pack_digest:
            raise ValueError(
                "campaign regression pack does not match its promotion review"
            )
        reasons = []
        promotion_qualified = (
            promotion.status is PromotionStatus.RELEASE_GATE_CANDIDATE_READY
            and not promotion.unmet_requirements
        )
        if not promotion_qualified:
            reasons.append(AssuranceReasonCode.PROMOTION_NOT_QUALIFIED)
        if not campaign.release_evidence_eligible:
            reasons.append(AssuranceReasonCode.CAMPAIGN_NOT_RELEASE_ELIGIBLE)
        gate = campaign.regression_gate_report
        if gate is None:
            reasons.append(AssuranceReasonCode.REGRESSION_GATE_MISSING)
            gate_passed = False
        else:
            gate_passed = gate.passed
            if not gate_passed:
                reasons.append(AssuranceReasonCode.REGRESSION_GATE_FAILED)
        blocking.update(reasons)
        regression_results.append(
            RegressionAssuranceResult(
                failure_fingerprint=link.failure_fingerprint,
                regression_pack_digest=promotion.regression_pack_digest,
                promotion_qualified=promotion_qualified,
                campaign_release_eligible=(campaign.release_evidence_eligible),
                regression_gate_passed=gate_passed,
                reason_codes=tuple(reasons),
            )
        )

    independent_passing = tuple(
        item
        for item in dossier.challenge_evidence
        if item.independent_from_development and item.outcome is ChallengeOutcome.PASSED
    )
    passing_kind_counts = Counter(item.kind for item in independent_passing)
    if len(independent_passing) < dossier.policy.minimum_independent_challenges or any(
        passing_kind_counts[kind] == 0
        for kind in dossier.policy.required_independent_challenge_kinds
    ):
        blocking.add(AssuranceReasonCode.INDEPENDENT_CHALLENGE_INSUFFICIENT)
    if any(
        item.outcome is ChallengeOutcome.FAILED for item in dossier.challenge_evidence
    ):
        blocking.add(AssuranceReasonCode.CHALLENGE_FAILED)

    restrictions: set[AssuranceReasonCode] = set()
    if any(
        item.outcome is ChallengeOutcome.INCONCLUSIVE
        for item in dossier.challenge_evidence
    ):
        restrictions.add(AssuranceReasonCode.CHALLENGE_INCONCLUSIVE)
    open_defeaters = tuple(
        item for item in dossier.defeaters if item.status is DefeaterStatus.OPEN
    )
    if any(item.severity is DefeaterSeverity.CRITICAL for item in open_defeaters):
        blocking.add(AssuranceReasonCode.CRITICAL_DEFEATER_OPEN)
    if any(item.severity is DefeaterSeverity.MATERIAL for item in open_defeaters):
        restrictions.add(AssuranceReasonCode.MATERIAL_DEFEATER_OPEN)

    if blocking:
        disposition = AssuranceDisposition.BLOCKED
    elif restrictions:
        disposition = AssuranceDisposition.RESTRICTED
    else:
        disposition = AssuranceDisposition.SUPPORTED_WITHIN_SCOPE
    limitations = tuple(
        sorted(
            dossier.limitations
            + dossier.release_scope.limitations
            + (
                "This report is technical decision support, not authorization "
                "to deploy or a guarantee of failure-free operation.",
                "The disposition applies only to the exact system artifacts, "
                "operational domain, validity envelope, evidence, and risk "
                "policy content-addressed by this dossier.",
                "Independent challenge evidence reduces common-mode bias but "
                "does not eliminate model-form, measurement, selection, or "
                "sim-to-real uncertainty.",
                "No composite confidence score is calculated; failed required "
                "evidence and critical defeaters cannot be averaged away.",
            )
        )
    )
    return ReleaseAssuranceReport(
        dossier_content_digest=dossier.content_digest(),
        release_scope_digest=dossier.release_scope.content_digest(),
        policy_content_digest=dossier.policy.content_digest(),
        regression_results=tuple(regression_results),
        independent_passing_challenge_count=len(independent_passing),
        independent_passing_challenge_kind_counts=tuple(
            (kind, passing_kind_counts[kind])
            for kind in ChallengeKind
            if passing_kind_counts[kind]
        ),
        open_defeater_ids=tuple(sorted(item.defeater_id for item in open_defeaters)),
        blocking_reasons=tuple(
            reason for reason in AssuranceReasonCode if reason in blocking
        ),
        restriction_reasons=tuple(
            reason for reason in AssuranceReasonCode if reason in restrictions
        ),
        disposition=disposition,
        limitations=limitations,
    )


def _index_artifacts[T: _ContentAddressed](
    artifacts: Sequence[T],
    label: str,
) -> dict[str, T]:
    """Index content-addressed artifacts and reject duplicates.

    Args:
        artifacts: Objects exposing a ``content_digest`` method.
        label: Human-readable artifact collection label.

    Returns:
        Mapping from content digest to artifact.

    Raises:
        ValueError: If a duplicate digest is supplied.
    """
    indexed: dict[str, T] = {}
    for artifact in artifacts:
        digest = artifact.content_digest()
        if digest in indexed:
            raise ValueError(f"{label} contain duplicate artifact digests")
        indexed[digest] = artifact
    return indexed


def _require_exact_artifact_set(
    observed: set[str],
    expected: set[str],
    label: str,
) -> None:
    """Require supplied artifact identities to match dossier links exactly.

    Args:
        observed: Supplied content digests.
        expected: Digests linked by the dossier.
        label: Human-readable artifact collection label.

    Raises:
        ValueError: If artifacts are missing or unlinked.
    """
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} must exactly match dossier links "
            f"(missing={missing}, extra={extra})"
        )


def _require_text(value: str, label: str) -> None:
    """Require non-empty text.

    Args:
        value: Candidate text.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_unique(values: Iterable[object], label: str) -> None:
    """Require unique values.

    Args:
        values: Candidate values.
        label: Human-readable collection label.
    """
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} must be unique")


def _require_unique_text(values: Sequence[str], label: str) -> None:
    """Require unique non-empty text values.

    Args:
        values: Candidate text values.
        label: Human-readable collection label.
    """
    if not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_text(value, label)
    _require_unique(values, label)
