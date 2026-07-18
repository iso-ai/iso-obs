"""Evidence-qualified promotion of failures into regression assets.

Promotion verifies that failure, intervention, minimization, and regression
artifacts refer to the same evidence chain. A release-gate-ready result means
the pack is scientifically eligible to be executed as a gate; it is not a
release decision and does not imply that the gate has passed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .counterfactual import (
    CounterfactualDesign,
    CounterfactualEvidenceLevel,
    CounterfactualEvidenceReport,
    MinimalCounterexampleReport,
)
from .evidence import (
    FailureEvidenceBundle,
    NamedDigest,
    _canonical_json,
    sha256_digest,
)
from .regression import (
    CaseRole,
    ExpectedOutcome,
    RegressionCase,
    RegressionPack,
)


class PromotionIntent(StrEnum):
    """Intended evidentiary use of a promoted regression pack."""

    DISCOVERY_REGRESSION = "discovery_regression"
    RELEASE_GATE_CANDIDATE = "release_gate_candidate"


class PromotionStatus(StrEnum):
    """Outcome of an evidence-qualification review."""

    DISCOVERY_READY = "discovery_ready"
    RELEASE_GATE_CANDIDATE_READY = "release_gate_candidate_ready"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"


class PromotionRequirement(StrEnum):
    """Scientific requirement evaluated for release-gate candidacy."""

    REPLICATED_INTERVENTION_ASSOCIATION = "replicated_intervention_association"
    ONE_MINIMAL_COUNTEREXAMPLE = "one_minimal_counterexample"
    SIMULATOR_VALIDATION_EVIDENCE = "simulator_validation_evidence"
    REAL_WORLD_ANCHOR = "real_world_anchor"
    CANONICAL_CASE_EXPECTS_RESOLUTION = "canonical_case_expects_resolution"


@dataclass(frozen=True, slots=True)
class ConditionParameterBinding:
    """Bind one retained failure condition to a canonical case parameter."""

    condition_id: str
    scenario_parameter_name: str

    def __post_init__(self) -> None:
        """Validate condition and parameter identities."""
        _require_text(self.condition_id, "condition binding ID")
        _require_text(
            self.scenario_parameter_name,
            "condition binding scenario parameter",
        )


@dataclass(frozen=True, slots=True)
class RegressionPromotionRequest:
    """Content-addressed request to qualify one regression pack."""

    request_id: str
    request_version: str
    intent: PromotionIntent
    source_failure_content_digest: str
    counterfactual_design_digest: str
    counterfactual_report_digest: str
    minimal_counterexample_report_digest: str
    regression_pack_digest: str
    condition_bindings: tuple[ConditionParameterBinding, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and canonicalize promotion identity and bindings."""
        _require_text(self.request_id, "promotion request ID")
        _require_text(self.request_version, "promotion request version")
        object.__setattr__(self, "intent", PromotionIntent(self.intent))
        for value, label in (
            (self.source_failure_content_digest, "source failure"),
            (self.counterfactual_design_digest, "counterfactual design"),
            (self.counterfactual_report_digest, "counterfactual report"),
            (
                self.minimal_counterexample_report_digest,
                "minimal counterexample report",
            ),
            (self.regression_pack_digest, "regression pack"),
        ):
            NamedDigest(label, value)
        _require_unique(
            (item.condition_id for item in self.condition_bindings),
            "bound condition IDs",
        )
        _require_unique(
            (item.scenario_parameter_name for item in self.condition_bindings),
            "bound scenario parameter names",
        )
        _require_unique_text(self.limitations, "promotion request limitations")
        object.__setattr__(
            self,
            "condition_bindings",
            tuple(
                sorted(
                    self.condition_bindings,
                    key=lambda item: item.condition_id,
                )
            ),
        )
        object.__setattr__(self, "limitations", tuple(sorted(self.limitations)))

    def to_json(self) -> str:
        """Serialize the request to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact request content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class PromotionRequirementResult:
    """Result of one release-gate qualification requirement."""

    requirement: PromotionRequirement
    satisfied: bool
    evidence: str


@dataclass(frozen=True, slots=True)
class RegressionPromotionReview:
    """Auditable decision about regression-asset promotion eligibility."""

    request_content_digest: str
    source_failure_content_digest: str
    source_failure_fingerprint: str
    regression_pack_digest: str
    status: PromotionStatus
    requirement_results: tuple[PromotionRequirementResult, ...]
    unmet_requirements: tuple[PromotionRequirement, ...]
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the review to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact review content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def review_regression_promotion(
    request: RegressionPromotionRequest,
    *,
    source_failure: FailureEvidenceBundle,
    counterfactual_design: CounterfactualDesign,
    counterfactual_report: CounterfactualEvidenceReport,
    minimal_counterexample: MinimalCounterexampleReport,
    regression_pack: RegressionPack,
) -> RegressionPromotionReview:
    """Verify provenance and assess regression-promotion eligibility.

    Artifact identity and lineage mismatches raise errors because they make the
    review invalid. Scientifically incomplete evidence instead produces an
    ``evidence_insufficient`` review with explicit unmet requirements.

    Args:
        request: Content-addressed promotion request.
        source_failure: Failure evidence from which the asset is derived.
        counterfactual_design: Paired intervention design.
        counterfactual_report: Analyzed intervention result.
        minimal_counterexample: Condition-minimization result.
        regression_pack: Proposed regression definition.

    Returns:
        Deterministic promotion review and requirement results.

    Raises:
        ValueError: If artifact identities or lineage do not agree.
    """
    _verify_artifact_digests(
        request,
        source_failure=source_failure,
        counterfactual_design=counterfactual_design,
        counterfactual_report=counterfactual_report,
        minimal_counterexample=minimal_counterexample,
        regression_pack=regression_pack,
    )
    canonical_case = _verify_lineage(
        request,
        source_failure=source_failure,
        counterfactual_design=counterfactual_design,
        counterfactual_report=counterfactual_report,
        minimal_counterexample=minimal_counterexample,
        regression_pack=regression_pack,
    )
    results = _requirement_results(
        source_failure,
        counterfactual_report,
        minimal_counterexample,
        canonical_case,
    )
    unmet = tuple(item.requirement for item in results if not item.satisfied)
    if request.intent is PromotionIntent.DISCOVERY_REGRESSION:
        status = PromotionStatus.DISCOVERY_READY
    elif unmet:
        status = PromotionStatus.EVIDENCE_INSUFFICIENT
    else:
        status = PromotionStatus.RELEASE_GATE_CANDIDATE_READY
    limitations = tuple(
        sorted(
            request.limitations
            + (
                "Promotion verifies evidence lineage and eligibility; it does "
                "not execute the regression pack or approve a system release.",
                "Release decisions still require complete, identity-verified "
                "campaign results that pass every declared regression "
                "criterion.",
                "Simulator validity, uncertainty, known omissions, and "
                "sim-to-real limits remain attached to every promoted "
                "regression result.",
                "Intervention evidence remains mechanism-supporting rather "
                "than proof of a real-world causal mechanism.",
            )
        )
    )
    return RegressionPromotionReview(
        request_content_digest=request.content_digest(),
        source_failure_content_digest=source_failure.content_digest(),
        source_failure_fingerprint=source_failure.failure_fingerprint(),
        regression_pack_digest=regression_pack.content_digest(),
        status=status,
        requirement_results=results,
        unmet_requirements=unmet,
        limitations=limitations,
    )


def _verify_artifact_digests(
    request: RegressionPromotionRequest,
    *,
    source_failure: FailureEvidenceBundle,
    counterfactual_design: CounterfactualDesign,
    counterfactual_report: CounterfactualEvidenceReport,
    minimal_counterexample: MinimalCounterexampleReport,
    regression_pack: RegressionPack,
) -> None:
    """Verify every request digest against the supplied artifact.

    Args:
        request: Promotion request with expected identities.
        source_failure: Supplied failure bundle.
        counterfactual_design: Supplied intervention design.
        counterfactual_report: Supplied intervention report.
        minimal_counterexample: Supplied minimization report.
        regression_pack: Supplied regression pack.

    Raises:
        ValueError: If any content digest differs.
    """
    expected_and_observed = (
        (
            "source failure",
            request.source_failure_content_digest,
            source_failure.content_digest(),
        ),
        (
            "counterfactual design",
            request.counterfactual_design_digest,
            counterfactual_design.content_digest(),
        ),
        (
            "counterfactual report",
            request.counterfactual_report_digest,
            counterfactual_report.content_digest(),
        ),
        (
            "minimal counterexample report",
            request.minimal_counterexample_report_digest,
            minimal_counterexample.content_digest(),
        ),
        (
            "regression pack",
            request.regression_pack_digest,
            regression_pack.content_digest(),
        ),
    )
    for label, expected, observed in expected_and_observed:
        if expected != observed:
            raise ValueError(f"{label} content digest mismatch")


def _verify_lineage(
    request: RegressionPromotionRequest,
    *,
    source_failure: FailureEvidenceBundle,
    counterfactual_design: CounterfactualDesign,
    counterfactual_report: CounterfactualEvidenceReport,
    minimal_counterexample: MinimalCounterexampleReport,
    regression_pack: RegressionPack,
) -> RegressionCase:
    """Verify cross-artifact lineage and canonical condition bindings.

    Args:
        request: Promotion request and condition bindings.
        source_failure: Source evidence bundle.
        counterfactual_design: Intervention design derived from the failure.
        counterfactual_report: Intervention analysis result.
        minimal_counterexample: Failure-condition minimization result.
        regression_pack: Proposed regression definition.

    Returns:
        Identity-verified canonical reproduction case.

    Raises:
        ValueError: If any lineage relationship is inconsistent.
    """
    failure_digest = source_failure.content_digest()
    failure_fingerprint = source_failure.failure_fingerprint()
    simulation_digest = source_failure.simulation.content_digest()
    if regression_pack.source_failure_content_digest != failure_digest:
        raise ValueError("regression pack source failure digest mismatch")
    if regression_pack.source_failure_fingerprint != failure_fingerprint:
        raise ValueError("regression pack failure fingerprint mismatch")
    if regression_pack.simulation_manifest_digest != simulation_digest:
        raise ValueError("regression pack simulation manifest digest mismatch")
    if counterfactual_design.simulation_evidence_manifest_digest != simulation_digest:
        raise ValueError("counterfactual design simulation manifest digest mismatch")
    if (
        counterfactual_report.design_content_digest
        != counterfactual_design.content_digest()
    ):
        raise ValueError("counterfactual report design digest mismatch")
    if (
        counterfactual_report.intervention_id
        != counterfactual_design.intervention.intervention_id
        or counterfactual_report.mechanism_hypothesis_id
        != counterfactual_design.intervention.mechanism_hypothesis_id
    ):
        raise ValueError("counterfactual report intervention identity mismatch")
    source_events = set(source_failure.evidence_event_ids)
    for pair in counterfactual_design.pairs:
        if not set(pair.source_failure_event_ids) <= source_events:
            raise ValueError(
                f"counterfactual pair '{pair.pair_id}' references events "
                "outside the source failure"
            )

    canonical = next(
        item
        for item in regression_pack.cases
        if item.role is CaseRole.CANONICAL_REPRODUCTION
    )
    if (
        canonical.scenario_id != source_failure.reproduction.scenario_id
        or canonical.scenario_version != source_failure.reproduction.scenario_version
    ):
        raise ValueError(
            "canonical regression case does not match failure reproduction"
        )
    canonical_events = set(canonical.source_evidence_event_ids)
    if not canonical_events or not canonical_events <= source_events:
        raise ValueError(
            "canonical regression case must reference only source failure events"
        )
    retained = set(minimal_counterexample.retained_condition_ids)
    bound = {item.condition_id for item in request.condition_bindings}
    if bound != retained:
        missing = sorted(retained - bound)
        extra = sorted(bound - retained)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"extra: {', '.join(extra)}")
        raise ValueError(
            "condition bindings must exactly match retained conditions ("
            + "; ".join(details)
            + ")"
        )
    canonical_parameters = {item.name for item in canonical.parameters}
    unknown_parameters = sorted(
        {item.scenario_parameter_name for item in request.condition_bindings}
        - canonical_parameters
    )
    if unknown_parameters:
        raise ValueError(
            "condition bindings reference unknown canonical parameters: "
            + ", ".join(unknown_parameters)
        )
    return canonical


def _requirement_results(
    source_failure: FailureEvidenceBundle,
    counterfactual_report: CounterfactualEvidenceReport,
    minimal_counterexample: MinimalCounterexampleReport,
    canonical_case: RegressionCase,
) -> tuple[PromotionRequirementResult, ...]:
    """Evaluate release-gate-candidate scientific requirements.

    Args:
        source_failure: Source failure and simulator credibility evidence.
        counterfactual_report: Paired intervention result.
        minimal_counterexample: Condition-minimization result.
        canonical_case: Proposed canonical regression case.

    Returns:
        Requirement results in stable enum order.
    """
    simulation = source_failure.simulation
    values = {
        PromotionRequirement.REPLICATED_INTERVENTION_ASSOCIATION: (
            counterfactual_report.evidence_level
            is CounterfactualEvidenceLevel.REPLICATED_INTERVENTION_ASSOCIATION,
            "Counterfactual report must show a practically meaningful, "
            "replicated paired intervention association.",
        ),
        PromotionRequirement.ONE_MINIMAL_COUNTEREXAMPLE: (
            minimal_counterexample.one_minimal,
            "Every single-condition removal from the selected counterexample "
            "must have been executed and must stop reproduction.",
        ),
        PromotionRequirement.SIMULATOR_VALIDATION_EVIDENCE: (
            bool(simulation.validation_evidence),
            "The simulation manifest must include validation evidence for its "
            "declared intended use and validity envelope.",
        ),
        PromotionRequirement.REAL_WORLD_ANCHOR: (
            bool(simulation.real_world_anchors),
            "The simulation manifest must include at least one real-world "
            "anchor; the anchor does not erase sim-to-real uncertainty.",
        ),
        PromotionRequirement.CANONICAL_CASE_EXPECTS_RESOLUTION: (
            canonical_case.expected_outcome is ExpectedOutcome.SATISFY_ALL,
            "The canonical reproduction must expect the proposed system "
            "version to satisfy every declared invariant.",
        ),
    }
    return tuple(
        PromotionRequirementResult(
            requirement=requirement,
            satisfied=values[requirement][0],
            evidence=values[requirement][1],
        )
        for requirement in PromotionRequirement
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
    """Require unique values from an iterable.

    Args:
        values: Candidate iterable.
        label: Human-readable collection label.
    """
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} must be unique")


def _require_unique_text(values: tuple[str, ...], label: str) -> None:
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
