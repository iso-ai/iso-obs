"""Paired causal perturbation-response assessment.

The model estimates failure-probability changes from predeclared controlled
interventions applied to matched replay contexts. Simultaneous paired
Hoeffding bounds distinguish material increases, material decreases,
equivalence, inconclusive evidence, and insufficient evidence.

Causal interpretation remains conditional on correct intervention isolation,
consistency, no interference, representative contexts, and valid matching.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse


class CausalEffectDisposition(StrEnum):
    """Conclusion for one paired perturbation contrast."""

    CAUSAL_INCREASE_SUPPORTED = "causal_increase_supported"
    CAUSAL_DECREASE_SUPPORTED = "causal_decrease_supported"
    EQUIVALENCE_SUPPORTED = "equivalence_supported"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CausalCampaignDisposition(StrEnum):
    """Non-compensatory conclusion across all perturbation contrasts."""

    HARMFUL_EFFECT_IDENTIFIED = "harmful_effect_identified"
    NO_HARMFUL_EFFECT_SUPPORTED = "no_harmful_effect_supported"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class PerturbationContrast:
    """One predeclared intervention compared with matched baseline replays."""

    contrast_id: str
    perturbation_spec_digest: str
    factor_id: str
    magnitude: float
    unit: str
    minimum_material_failure_rate_change: float
    minimum_pair_count: int

    def __post_init__(self) -> None:
        """Validate contrast identity, intervention, threshold, and sample size."""
        _require_text(self.contrast_id, "perturbation contrast ID")
        NamedDigest("perturbation specification", self.perturbation_spec_digest)
        _require_text(self.factor_id, "perturbation factor ID")
        magnitude = _finite(self.magnitude, "perturbation magnitude")
        if magnitude == 0.0:
            raise ValueError("perturbation magnitude must be non-zero")
        _require_text(self.unit, "perturbation magnitude unit")
        material_change = _closed_open_unit_interval(
            self.minimum_material_failure_rate_change,
            "minimum material failure-rate change",
        )
        minimum_pairs = _positive_integer(
            self.minimum_pair_count,
            "minimum perturbation pair count",
        )
        object.__setattr__(self, "magnitude", magnitude)
        object.__setattr__(
            self,
            "minimum_material_failure_rate_change",
            material_change,
        )
        object.__setattr__(self, "minimum_pair_count", minimum_pairs)


@dataclass(frozen=True, slots=True)
class CausalPerturbationPlan:
    """Content-addressed confirmatory paired-intervention design."""

    plan_id: str
    plan_version: str
    target_population_digest: str
    simulation_evidence_digest: str
    system_artifact_digest: str
    failure_outcome_definition_digest: str
    intervention_protocol_digest: str
    pairing_protocol_digest: str
    contrasts: tuple[PerturbationContrast, ...]
    familywise_error_rate: float
    limitations: tuple[str, ...]
    evidence_use: EvidenceUse = EvidenceUse.CONFIRMATORY

    def __post_init__(self) -> None:
        """Validate design identity, intervention family, and evidence use."""
        _require_text(self.plan_id, "causal perturbation plan ID")
        _require_text(self.plan_version, "causal perturbation plan version")
        for value, label in (
            (self.target_population_digest, "target population"),
            (self.simulation_evidence_digest, "simulation evidence"),
            (self.system_artifact_digest, "system artifact"),
            (self.failure_outcome_definition_digest, "failure outcome definition"),
            (self.intervention_protocol_digest, "intervention protocol"),
            (self.pairing_protocol_digest, "pairing protocol"),
        ):
            NamedDigest(label, value)
        if not self.contrasts:
            raise ValueError("causal perturbation plan requires contrasts")
        contrasts = tuple(sorted(self.contrasts, key=lambda item: item.contrast_id))
        _require_unique(
            (item.contrast_id for item in contrasts),
            "perturbation contrast IDs",
        )
        _require_unique(
            (item.perturbation_spec_digest for item in contrasts),
            "perturbation specifications",
        )
        familywise_error = _open_unit_interval(
            self.familywise_error_rate,
            "causal perturbation familywise error rate",
        )
        limitations = _unique_text(
            self.limitations,
            "causal perturbation plan limitations",
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.CONFIRMATORY:
            raise ValueError("causal perturbation plan must be confirmatory")
        object.__setattr__(self, "contrasts", contrasts)
        object.__setattr__(self, "familywise_error_rate", familywise_error)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_use", evidence_use)

    def to_json(self) -> str:
        """Serialize the causal plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact causal-plan content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class PairedPerturbationObservation:
    """One matched baseline and perturbed binary failure outcome."""

    plan_content_digest: str
    contrast_id: str
    pair_id: str
    matched_context_digest: str
    baseline_evidence_digest: str
    perturbed_evidence_digest: str
    baseline_failed: bool
    perturbed_failed: bool

    def __post_init__(self) -> None:
        """Validate observation identity, matching evidence, and outcomes."""
        NamedDigest("causal perturbation plan", self.plan_content_digest)
        _require_text(self.contrast_id, "paired perturbation contrast ID")
        _require_text(self.pair_id, "paired perturbation pair ID")
        NamedDigest("matched replay context", self.matched_context_digest)
        NamedDigest("baseline execution evidence", self.baseline_evidence_digest)
        NamedDigest("perturbed execution evidence", self.perturbed_evidence_digest)
        if not isinstance(self.baseline_failed, bool):
            raise ValueError("paired baseline failed must be a boolean")
        if not isinstance(self.perturbed_failed, bool):
            raise ValueError("paired perturbed failed must be a boolean")


@dataclass(frozen=True, slots=True)
class PairedFailureEffect:
    """Simultaneous confidence assessment for one paired causal effect."""

    pair_count: int
    baseline_failure_count: int
    perturbed_failure_count: int
    harm_transition_count: int
    benefit_transition_count: int
    failure_rate_change_estimate: float
    confidence_radius: float
    effect_lower_bound: float
    effect_upper_bound: float
    minimum_material_change: float
    allocated_error_rate: float
    disposition: CausalEffectDisposition

    def __post_init__(self) -> None:
        """Validate paired counts, effect interval, and causal disposition."""
        pair_count = _positive_integer(self.pair_count, "paired effect pair count")
        counts = tuple(
            _nonnegative_integer(value, label)
            for value, label in (
                (self.baseline_failure_count, "baseline failure count"),
                (self.perturbed_failure_count, "perturbed failure count"),
                (self.harm_transition_count, "harm transition count"),
                (self.benefit_transition_count, "benefit transition count"),
            )
        )
        if any(value > pair_count for value in counts):
            raise ValueError("paired effect counts cannot exceed pair count")
        both_failures_from_baseline = (
            self.baseline_failure_count - self.benefit_transition_count
        )
        both_failures_from_perturbed = (
            self.perturbed_failure_count - self.harm_transition_count
        )
        if (
            both_failures_from_baseline < 0
            or both_failures_from_baseline != both_failures_from_perturbed
            or self.harm_transition_count
            + self.benefit_transition_count
            + both_failures_from_baseline
            > pair_count
        ):
            raise ValueError("paired failure totals are inconsistent with transitions")
        estimate = _closed_interval(
            self.failure_rate_change_estimate,
            -1.0,
            1.0,
            "paired failure-rate change estimate",
        )
        expected_estimate = (
            self.harm_transition_count - self.benefit_transition_count
        ) / pair_count
        if not math.isclose(
            estimate,
            expected_estimate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("paired effect estimate is inconsistent with transitions")
        radius = _nonnegative_finite(
            self.confidence_radius,
            "paired effect confidence radius",
        )
        lower = _closed_interval(
            self.effect_lower_bound,
            -1.0,
            1.0,
            "paired effect lower bound",
        )
        upper = _closed_interval(
            self.effect_upper_bound,
            -1.0,
            1.0,
            "paired effect upper bound",
        )
        if lower > upper:
            raise ValueError("paired effect lower bound must not exceed upper bound")
        material_change = _closed_open_unit_interval(
            self.minimum_material_change,
            "paired minimum material change",
        )
        allocated_error = _open_unit_interval(
            self.allocated_error_rate,
            "paired effect allocated error rate",
        )
        expected_radius = math.sqrt(2.0 * math.log(2.0 / allocated_error) / pair_count)
        if not math.isclose(
            radius,
            expected_radius,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("paired confidence radius is inconsistent")
        expected_lower = max(-1.0, estimate - radius)
        expected_upper = min(1.0, estimate + radius)
        if not (
            math.isclose(lower, expected_lower, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(upper, expected_upper, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("paired effect bounds are inconsistent")
        disposition = CausalEffectDisposition(self.disposition)
        expected_disposition = _effect_disposition(
            lower,
            upper,
            material_change,
        )
        if disposition is not expected_disposition:
            raise ValueError("paired effect disposition is inconsistent with bounds")
        object.__setattr__(self, "pair_count", pair_count)
        object.__setattr__(self, "baseline_failure_count", counts[0])
        object.__setattr__(self, "perturbed_failure_count", counts[1])
        object.__setattr__(self, "harm_transition_count", counts[2])
        object.__setattr__(self, "benefit_transition_count", counts[3])
        object.__setattr__(self, "failure_rate_change_estimate", estimate)
        object.__setattr__(self, "confidence_radius", radius)
        object.__setattr__(self, "effect_lower_bound", lower)
        object.__setattr__(self, "effect_upper_bound", upper)
        object.__setattr__(self, "minimum_material_change", material_change)
        object.__setattr__(self, "allocated_error_rate", allocated_error)
        object.__setattr__(self, "disposition", disposition)


@dataclass(frozen=True, slots=True)
class CausalContrastAssessment:
    """Evidence status and optional effect for one perturbation contrast."""

    contrast_id: str
    pair_count: int
    minimum_pair_count: int
    disposition: CausalEffectDisposition
    effect: PairedFailureEffect | None
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate sample sufficiency and effect availability."""
        _require_text(self.contrast_id, "causal contrast assessment ID")
        pair_count = _nonnegative_integer(
            self.pair_count,
            "causal contrast pair count",
        )
        minimum_pairs = _positive_integer(
            self.minimum_pair_count,
            "causal contrast minimum pair count",
        )
        disposition = CausalEffectDisposition(self.disposition)
        if pair_count < minimum_pairs:
            if (
                disposition is not CausalEffectDisposition.INSUFFICIENT_EVIDENCE
                or self.effect is not None
            ):
                raise ValueError(
                    "under-sampled contrast must report insufficient evidence"
                )
        elif (
            self.effect is None
            or disposition is not self.effect.disposition
            or self.effect.pair_count != pair_count
        ):
            raise ValueError("sufficient contrast must include its paired effect")
        limitations = _unique_text(
            self.limitations,
            "causal contrast assessment limitations",
        )
        object.__setattr__(self, "pair_count", pair_count)
        object.__setattr__(self, "minimum_pair_count", minimum_pairs)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class CausalPerturbationReport:
    """Content-addressed simultaneous paired-intervention report."""

    report_schema_version: str
    plan_content_digest: str
    observation_data_digest: str
    familywise_error_rate: float
    allocated_error_rate_per_contrast: float
    disposition: CausalCampaignDisposition
    contrast_assessments: tuple[CausalContrastAssessment, ...]
    harm_priority_ranking: tuple[str, ...]
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate multiplicity, aggregation, priority, and evidence use."""
        _require_text(self.report_schema_version, "causal report schema version")
        NamedDigest("causal perturbation plan", self.plan_content_digest)
        NamedDigest("causal perturbation observations", self.observation_data_digest)
        familywise_error = _open_unit_interval(
            self.familywise_error_rate,
            "causal report familywise error rate",
        )
        assessments = tuple(
            sorted(self.contrast_assessments, key=lambda item: item.contrast_id)
        )
        if not assessments:
            raise ValueError("causal report requires contrast assessments")
        _require_unique(
            (item.contrast_id for item in assessments),
            "causal report contrast IDs",
        )
        allocated_error = _open_unit_interval(
            self.allocated_error_rate_per_contrast,
            "causal report allocated contrast error rate",
        )
        if not math.isclose(
            allocated_error * len(assessments),
            familywise_error,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("contrast error allocation must equal familywise error")
        if any(
            item.effect is not None
            and not math.isclose(
                item.effect.allocated_error_rate,
                allocated_error,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for item in assessments
        ):
            raise ValueError("contrast effects must use the allocated error rate")
        disposition = CausalCampaignDisposition(self.disposition)
        expected_disposition = _campaign_disposition(assessments)
        if disposition is not expected_disposition:
            raise ValueError("causal campaign disposition is inconsistent")
        expected_ranking = _harm_priority_ranking(assessments)
        if self.harm_priority_ranking != expected_ranking:
            raise ValueError("harm priority ranking is inconsistent with effects")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.CONFIRMATORY:
            raise ValueError("causal perturbation report must be confirmatory")
        limitations = _unique_text(
            self.limitations,
            "causal perturbation report limitations",
        )
        object.__setattr__(self, "familywise_error_rate", familywise_error)
        object.__setattr__(
            self,
            "allocated_error_rate_per_contrast",
            allocated_error,
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "contrast_assessments", assessments)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the causal report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact causal-report content digest."""
        return sha256_digest(self.to_json())


def assess_causal_perturbations(
    plan: CausalPerturbationPlan,
    observations: Sequence[PairedPerturbationObservation],
) -> CausalPerturbationReport:
    """Estimate simultaneous paired effects for controlled perturbations.

    Args:
        plan: Frozen intervention, pairing, threshold, and error-rate design.
        observations: Matched baseline and perturbed binary outcomes.

    Returns:
        A deterministic causal-effect or abstention report.

    Raises:
        ValueError: If observation identity, matching, evidence reuse, or
            baseline consistency violates the declared design.
    """
    plan_digest = plan.content_digest()
    contrasts = {item.contrast_id: item for item in plan.contrasts}
    grouped: dict[str, list[PairedPerturbationObservation]] = {
        contrast_id: [] for contrast_id in contrasts
    }
    seen_contrast_pairs: set[tuple[str, str]] = set()
    pair_baselines: dict[str, tuple[str, str, bool]] = {}
    context_pairs: dict[str, str] = {}
    baseline_evidence_pairs: dict[str, str] = {}
    perturbed_evidence: set[str] = set()
    for observation in observations:
        if observation.plan_content_digest != plan_digest:
            raise ValueError(
                f"paired observation '{observation.pair_id}' plan mismatch"
            )
        if observation.contrast_id not in contrasts:
            raise ValueError(
                f"unknown perturbation contrast: {observation.contrast_id}"
            )
        contrast_pair = (observation.contrast_id, observation.pair_id)
        if contrast_pair in seen_contrast_pairs:
            raise ValueError("pair IDs must be unique within each contrast")
        baseline_state = (
            observation.matched_context_digest,
            observation.baseline_evidence_digest,
            observation.baseline_failed,
        )
        prior_baseline = pair_baselines.get(observation.pair_id)
        if prior_baseline is not None and prior_baseline != baseline_state:
            raise ValueError(
                "reused pair IDs must preserve matched context and baseline"
            )
        prior_context_pair = context_pairs.get(observation.matched_context_digest)
        if prior_context_pair is not None and prior_context_pair != observation.pair_id:
            raise ValueError(
                "matched context digests cannot represent multiple independent pairs"
            )
        prior_evidence_pair = baseline_evidence_pairs.get(
            observation.baseline_evidence_digest
        )
        if (
            prior_evidence_pair is not None
            and prior_evidence_pair != observation.pair_id
        ):
            raise ValueError(
                "baseline evidence cannot represent multiple independent pairs"
            )
        if observation.perturbed_evidence_digest in perturbed_evidence:
            raise ValueError("perturbed evidence digests must be unique")
        seen_contrast_pairs.add(contrast_pair)
        pair_baselines[observation.pair_id] = baseline_state
        context_pairs[observation.matched_context_digest] = observation.pair_id
        baseline_evidence_pairs[observation.baseline_evidence_digest] = (
            observation.pair_id
        )
        perturbed_evidence.add(observation.perturbed_evidence_digest)
        grouped[observation.contrast_id].append(observation)

    allocated_error = plan.familywise_error_rate / len(plan.contrasts)
    assessments: list[CausalContrastAssessment] = []
    for contrast in plan.contrasts:
        contrast_observations = grouped[contrast.contrast_id]
        if len(contrast_observations) < contrast.minimum_pair_count:
            assessments.append(
                CausalContrastAssessment(
                    contrast_id=contrast.contrast_id,
                    pair_count=len(contrast_observations),
                    minimum_pair_count=contrast.minimum_pair_count,
                    disposition=CausalEffectDisposition.INSUFFICIENT_EVIDENCE,
                    effect=None,
                    limitations=(
                        "Predeclared minimum matched-pair count was not met.",
                    ),
                )
            )
            continue
        effect = _paired_failure_effect(
            contrast,
            contrast_observations,
            allocated_error,
        )
        assessments.append(
            CausalContrastAssessment(
                contrast_id=contrast.contrast_id,
                pair_count=len(contrast_observations),
                minimum_pair_count=contrast.minimum_pair_count,
                disposition=effect.disposition,
                effect=effect,
                limitations=(
                    "Causal interpretation assumes the declared intervention "
                    "was the only systematic within-pair change.",
                ),
            )
        )

    canonical_observations = tuple(
        sorted(
            observations,
            key=lambda item: (item.contrast_id, item.pair_id),
        )
    )
    assessment_tuple = tuple(assessments)
    return CausalPerturbationReport(
        report_schema_version="iso-obs.causal-perturbation-report.v1",
        plan_content_digest=plan_digest,
        observation_data_digest=sha256_digest(_canonical_json(canonical_observations)),
        familywise_error_rate=plan.familywise_error_rate,
        allocated_error_rate_per_contrast=allocated_error,
        disposition=_campaign_disposition(assessment_tuple),
        contrast_assessments=assessment_tuple,
        harm_priority_ranking=_harm_priority_ranking(assessment_tuple),
        evidence_use=EvidenceUse.CONFIRMATORY,
        limitations=(
            "Causal interpretation requires intervention isolation, consistency, "
            "no interference, and representative matched contexts.",
            "Familywise coverage uses conservative Bonferroni-allocated paired "
            "Hoeffding bounds.",
            "Content addressing does not prove that the plan preceded outcomes.",
            "A supported effect does not identify the downstream mechanism.",
        ),
    )


def _paired_failure_effect(
    contrast: PerturbationContrast,
    observations: Sequence[PairedPerturbationObservation],
    allocated_error_rate: float,
) -> PairedFailureEffect:
    """Calculate one paired binary effect and simultaneous Hoeffding bound."""
    pair_count = len(observations)
    baseline_failures = sum(item.baseline_failed for item in observations)
    perturbed_failures = sum(item.perturbed_failed for item in observations)
    harm_transitions = sum(
        not item.baseline_failed and item.perturbed_failed for item in observations
    )
    benefit_transitions = sum(
        item.baseline_failed and not item.perturbed_failed for item in observations
    )
    estimate = (harm_transitions - benefit_transitions) / pair_count
    radius = math.sqrt(2.0 * math.log(2.0 / allocated_error_rate) / pair_count)
    lower = max(-1.0, estimate - radius)
    upper = min(1.0, estimate + radius)
    material_change = contrast.minimum_material_failure_rate_change
    return PairedFailureEffect(
        pair_count=pair_count,
        baseline_failure_count=baseline_failures,
        perturbed_failure_count=perturbed_failures,
        harm_transition_count=harm_transitions,
        benefit_transition_count=benefit_transitions,
        failure_rate_change_estimate=estimate,
        confidence_radius=radius,
        effect_lower_bound=lower,
        effect_upper_bound=upper,
        minimum_material_change=material_change,
        allocated_error_rate=allocated_error_rate,
        disposition=_effect_disposition(lower, upper, material_change),
    )


def _effect_disposition(
    lower_bound: float,
    upper_bound: float,
    material_change: float,
) -> CausalEffectDisposition:
    """Map a paired-effect interval to a causal conclusion."""
    if lower_bound > material_change:
        return CausalEffectDisposition.CAUSAL_INCREASE_SUPPORTED
    if upper_bound < -material_change:
        return CausalEffectDisposition.CAUSAL_DECREASE_SUPPORTED
    if lower_bound >= -material_change and upper_bound <= material_change:
        return CausalEffectDisposition.EQUIVALENCE_SUPPORTED
    return CausalEffectDisposition.INCONCLUSIVE


def _campaign_disposition(
    assessments: Sequence[CausalContrastAssessment],
) -> CausalCampaignDisposition:
    """Aggregate contrasts without allowing benign effects to offset harm."""
    dispositions = {item.disposition for item in assessments}
    if CausalEffectDisposition.CAUSAL_INCREASE_SUPPORTED in dispositions:
        return CausalCampaignDisposition.HARMFUL_EFFECT_IDENTIFIED
    if CausalEffectDisposition.INSUFFICIENT_EVIDENCE in dispositions:
        return CausalCampaignDisposition.INSUFFICIENT_EVIDENCE
    if CausalEffectDisposition.INCONCLUSIVE in dispositions:
        return CausalCampaignDisposition.INCONCLUSIVE
    return CausalCampaignDisposition.NO_HARMFUL_EFFECT_SUPPORTED


def _harm_priority_ranking(
    assessments: Sequence[CausalContrastAssessment],
) -> tuple[str, ...]:
    """Rank sufficiently sampled contrasts by conservative harmful effect."""
    sufficient = tuple(item for item in assessments if item.effect is not None)
    return tuple(
        item.contrast_id
        for item in sorted(
            sufficient,
            key=lambda item: (
                -_require_effect(item).effect_lower_bound,
                -_require_effect(item).failure_rate_change_estimate,
                item.contrast_id,
            ),
        )
    )


def _require_effect(assessment: CausalContrastAssessment) -> PairedFailureEffect:
    """Return a present effect for an internally sufficient assessment."""
    if assessment.effect is None:
        raise ValueError("causal contrast effect is required")
    return assessment.effect


def _finite(value: float, label: str) -> float:
    """Require a finite numeric value."""
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _nonnegative_finite(value: float, label: str) -> float:
    """Require a finite number greater than or equal to zero."""
    resolved = _finite(value, label)
    if resolved < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return resolved


def _open_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``(0, 1)``."""
    resolved = _finite(value, label)
    if resolved <= 0.0 or resolved >= 1.0:
        raise ValueError(f"{label} must be in (0, 1)")
    return resolved


def _closed_open_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``[0, 1)``."""
    resolved = _finite(value, label)
    if resolved < 0.0 or resolved >= 1.0:
        raise ValueError(f"{label} must be in [0, 1)")
    return resolved


def _closed_interval(
    value: float,
    lower: float,
    upper: float,
    label: str,
) -> float:
    """Require a finite number within a closed interval."""
    resolved = _finite(value, label)
    if resolved < lower or resolved > upper:
        raise ValueError(f"{label} must be in [{lower}, {upper}]")
    return resolved


def _positive_integer(value: int, label: str) -> int:
    """Require an integer greater than zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: int, label: str) -> int:
    """Require a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_text(value: str, label: str) -> None:
    """Require a non-empty, trimmed string."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be non-empty trimmed text")


def _require_unique(values: Iterable[object], label: str) -> None:
    """Require a sequence of unique values."""
    resolved = tuple(values)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} must be unique")


def _unique_text(values: Iterable[str], label: str) -> tuple[str, ...]:
    """Validate, deduplicate, and sort non-empty text values."""
    resolved = tuple(values)
    for value in resolved:
        _require_text(value, label)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(resolved))
