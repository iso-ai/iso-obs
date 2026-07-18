"""Multiplicity control and adaptive-search provenance for failure discovery.

This module separates exploratory signals, preregistered associations, and
held-out confirmations. It implements standard FDR and FWER adjustments while
preserving effect estimates and selection provenance. Rejection labels remain
associational; no multiple-testing procedure establishes causality.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest


class MultiplicityMethod(StrEnum):
    """Supported family-level multiplicity adjustment."""

    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BENJAMINI_YEKUTIELI = "benjamini_yekutieli"
    HOLM = "holm"
    BONFERRONI = "bonferroni"


class ControlledErrorRate(StrEnum):
    """Family-level error criterion controlled by a method."""

    FALSE_DISCOVERY_RATE = "false_discovery_rate"
    FAMILY_WISE_ERROR_RATE = "family_wise_error_rate"


class AnalysisMode(StrEnum):
    """Evidentiary relationship between hypothesis selection and analysis."""

    EXPLORATORY = "exploratory"
    PREREGISTERED = "preregistered"
    HELD_OUT_CONFIRMATION = "held_out_confirmation"


class AlternativeHypothesis(StrEnum):
    """Direction declared for an individual hypothesis test."""

    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"


class MultiplicityEvidenceLabel(StrEnum):
    """Bounded evidence label assigned after adjustment."""

    NOT_SELECTED = "not_selected"
    EXPLORATORY_SIGNAL = "exploratory_signal"
    PREREGISTERED_ASSOCIATION = "preregistered_association"
    HELD_OUT_CONFIRMED_ASSOCIATION = "held_out_confirmed_association"


@dataclass(frozen=True, slots=True)
class AdaptiveSearchTrial:
    """One auditable proposal and outcome in adaptive scenario search."""

    iteration: int
    candidate_id: str
    scenario_digest: str
    proposal_policy_digest: str
    information_available_digest: str
    outcome_digest: str
    proposal_score: float | None = None

    def __post_init__(self) -> None:
        """Validate trial ordering coordinates and content identities."""
        if (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or self.iteration < 0
        ):
            raise ValueError("adaptive trial iteration must be a non-negative integer")
        _require_text(self.candidate_id, "adaptive candidate ID")
        for value, label in (
            (self.scenario_digest, "adaptive scenario"),
            (self.proposal_policy_digest, "proposal policy"),
            (self.information_available_digest, "available information"),
            (self.outcome_digest, "adaptive trial outcome"),
        ):
            NamedDigest(label, value)
        if self.proposal_score is not None:
            score = float(self.proposal_score)
            if not math.isfinite(score):
                raise ValueError("adaptive proposal score must be finite")
            object.__setattr__(self, "proposal_score", score)


@dataclass(frozen=True, slots=True)
class AdaptiveSearchLedger:
    """Content-addressed history of adaptively selected scenario trials."""

    search_id: str
    algorithm: str
    algorithm_version: str
    objective: str
    trials: tuple[AdaptiveSearchTrial, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate a complete, contiguous adaptive-search history."""
        for value, label in (
            (self.search_id, "adaptive search ID"),
            (self.algorithm, "adaptive search algorithm"),
            (self.algorithm_version, "adaptive search algorithm version"),
            (self.objective, "adaptive search objective"),
        ):
            _require_text(value, label)
        if not self.trials:
            raise ValueError("adaptive search ledger requires at least one trial")
        _require_unique(
            (item.candidate_id for item in self.trials),
            "adaptive candidate IDs",
        )
        ordered = tuple(sorted(self.trials, key=lambda item: item.iteration))
        if tuple(item.iteration for item in ordered) != tuple(range(len(ordered))):
            raise ValueError("adaptive trial iterations must be contiguous from zero")
        _require_unique_text(self.limitations, "adaptive search limitations")
        object.__setattr__(self, "trials", ordered)
        object.__setattr__(self, "limitations", tuple(sorted(self.limitations)))

    def to_json(self) -> str:
        """Serialize the search ledger to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact search-ledger content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class MultiplicityPlan:
    """Pre-analysis definition of one hypothesis family."""

    family_id: str
    mode: AnalysisMode
    method: MultiplicityMethod
    target_error_rate: float
    analysis_dataset_digest: str
    planned_hypothesis_ids: tuple[str, ...] = ()
    preregistration_digest: str | None = None
    selection_dataset_digest: str | None = None
    adaptive_search_ledger_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate family definition, data separation, and preregistration."""
        _require_text(self.family_id, "hypothesis family ID")
        object.__setattr__(self, "mode", AnalysisMode(self.mode))
        object.__setattr__(self, "method", MultiplicityMethod(self.method))
        rate = float(self.target_error_rate)
        if not math.isfinite(rate) or not 0.0 < rate < 1.0:
            raise ValueError(
                "target_error_rate must be finite and between zero and one"
            )
        object.__setattr__(self, "target_error_rate", rate)
        NamedDigest("analysis dataset", self.analysis_dataset_digest)
        _require_unique_text(
            self.planned_hypothesis_ids,
            "planned hypothesis IDs",
            required=False,
        )
        object.__setattr__(
            self,
            "planned_hypothesis_ids",
            tuple(sorted(self.planned_hypothesis_ids)),
        )
        for digest, label in (
            (self.preregistration_digest, "preregistration"),
            (self.selection_dataset_digest, "selection dataset"),
            (self.adaptive_search_ledger_digest, "adaptive search ledger"),
        ):
            if digest is not None:
                NamedDigest(label, digest)

        if self.mode is AnalysisMode.EXPLORATORY:
            return
        if self.preregistration_digest is None:
            raise ValueError(
                f"{self.mode.value} analysis requires a preregistration digest"
            )
        if not self.planned_hypothesis_ids:
            raise ValueError(
                f"{self.mode.value} analysis requires planned hypothesis IDs"
            )
        if self.adaptive_search_ledger_digest is not None:
            raise ValueError(
                "confirmatory analysis cannot adapt hypotheses on analysis data"
            )
        if self.mode is AnalysisMode.PREREGISTERED:
            if self.selection_dataset_digest is not None:
                raise ValueError(
                    "use held_out_confirmation when hypotheses came from "
                    "selection data"
                )
            return
        if self.selection_dataset_digest is None:
            raise ValueError(
                "held-out confirmation requires a selection dataset digest"
            )
        if self.selection_dataset_digest == self.analysis_dataset_digest:
            raise ValueError(
                "held-out selection and analysis datasets must be different"
            )

    @property
    def controlled_error_rate(self) -> ControlledErrorRate:
        """Return the error criterion associated with the adjustment.

        Returns:
            FDR for Benjamini methods and FWER for Holm or Bonferroni.
        """
        if self.method in {
            MultiplicityMethod.BENJAMINI_HOCHBERG,
            MultiplicityMethod.BENJAMINI_YEKUTIELI,
        }:
            return ControlledErrorRate.FALSE_DISCOVERY_RATE
        return ControlledErrorRate.FAMILY_WISE_ERROR_RATE

    def to_json(self) -> str:
        """Serialize the multiplicity plan to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact plan content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class HypothesisTestResult:
    """One effect estimate and valid raw p-value in a declared family."""

    hypothesis_id: str
    p_value: float
    effect_estimate: float
    sample_size: int
    alternative: AlternativeHypothesis
    analysis_digest: str
    standard_error: float | None = None
    source_discovery_id: str | None = None

    def __post_init__(self) -> None:
        """Validate numeric inference results and analysis provenance."""
        _require_text(self.hypothesis_id, "hypothesis ID")
        p_value = float(self.p_value)
        effect = float(self.effect_estimate)
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError("p_value must be finite and between zero and one")
        if not math.isfinite(effect):
            raise ValueError("effect_estimate must be finite")
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "effect_estimate", effect)
        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size < 2
        ):
            raise ValueError("sample_size must be an integer of at least two")
        object.__setattr__(
            self,
            "alternative",
            AlternativeHypothesis(self.alternative),
        )
        NamedDigest("hypothesis analysis", self.analysis_digest)
        if self.standard_error is not None:
            standard_error = float(self.standard_error)
            if not math.isfinite(standard_error) or standard_error < 0.0:
                raise ValueError("standard_error must be finite and non-negative")
            object.__setattr__(self, "standard_error", standard_error)
        if self.source_discovery_id is not None:
            _require_text(self.source_discovery_id, "source discovery ID")


@dataclass(frozen=True, slots=True)
class AdjustedHypothesis:
    """Multiplicity-adjusted result with a bounded evidence label."""

    hypothesis_id: str
    raw_p_value: float
    adjusted_value: float
    effect_estimate: float
    standard_error: float | None
    sample_size: int
    alternative: AlternativeHypothesis
    rejected: bool
    evidence_label: MultiplicityEvidenceLabel
    source_discovery_id: str | None
    analysis_digest: str


@dataclass(frozen=True, slots=True)
class MultiplicityReport:
    """Deterministic family-level adjustment and interpretive limitations."""

    family_id: str
    plan_content_digest: str
    mode: AnalysisMode
    method: MultiplicityMethod
    controlled_error_rate: ControlledErrorRate
    target_error_rate: float
    hypothesis_count: int
    rejected_count: int
    results: tuple[AdjustedHypothesis, ...]
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the multiplicity report to canonical JSON.

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


def adjust_hypothesis_family(
    plan: MultiplicityPlan,
    tests: tuple[HypothesisTestResult, ...],
) -> MultiplicityReport:
    """Adjust one complete family of hypotheses under a declared plan.

    Args:
        plan: Family definition, analysis mode, and adjustment method.
        tests: Complete raw test results for the family.

    Returns:
        Adjusted values, bounded evidence labels, and limitations.

    Raises:
        ValueError: If tests are missing, duplicated, undeclared, or reuse
            selection data in a held-out confirmation.
    """
    if not tests:
        raise ValueError("hypothesis family requires at least one test")
    _require_unique(
        (item.hypothesis_id for item in tests),
        "hypothesis test IDs",
    )
    test_ids = {item.hypothesis_id for item in tests}
    planned_ids = set(plan.planned_hypothesis_ids)
    if planned_ids and test_ids != planned_ids:
        missing = sorted(planned_ids - test_ids)
        extra = sorted(test_ids - planned_ids)
        raise ValueError(
            f"hypothesis family differs from plan; missing={missing}, extra={extra}"
        )
    if plan.mode is AnalysisMode.HELD_OUT_CONFIRMATION:
        missing_sources = sorted(
            item.hypothesis_id for item in tests if item.source_discovery_id is None
        )
        if missing_sources:
            raise ValueError(
                "held-out tests require source discovery IDs: "
                + ", ".join(missing_sources)
            )

    ordered = tuple(sorted(tests, key=lambda item: (item.p_value, item.hypothesis_id)))
    adjusted = _adjusted_values(
        tuple(item.p_value for item in ordered),
        plan.method,
    )
    adjusted_by_id = {
        item.hypothesis_id: value for item, value in zip(ordered, adjusted, strict=True)
    }
    results = tuple(
        _adjusted_hypothesis(
            item,
            adjusted_by_id[item.hypothesis_id],
            plan,
        )
        for item in sorted(tests, key=lambda value: value.hypothesis_id)
    )
    rejected_count = sum(item.rejected for item in results)
    return MultiplicityReport(
        family_id=plan.family_id,
        plan_content_digest=plan.content_digest(),
        mode=plan.mode,
        method=plan.method,
        controlled_error_rate=plan.controlled_error_rate,
        target_error_rate=plan.target_error_rate,
        hypothesis_count=len(results),
        rejected_count=rejected_count,
        results=results,
        limitations=_limitations(plan),
    )


def _adjusted_values(
    ordered_p_values: tuple[float, ...],
    method: MultiplicityMethod,
) -> tuple[float, ...]:
    """Calculate monotone adjusted p-values or q-values in p-value order.

    Args:
        ordered_p_values: Raw p-values sorted ascending.
        method: Declared family adjustment.

    Returns:
        Adjusted values in the same order.
    """
    count = len(ordered_p_values)
    if method is MultiplicityMethod.BONFERRONI:
        return tuple(min(1.0, count * value) for value in ordered_p_values)
    if method is MultiplicityMethod.HOLM:
        raw = tuple(
            min(1.0, (count - index) * value)
            for index, value in enumerate(ordered_p_values)
        )
        running = 0.0
        adjusted = []
        for value in raw:
            running = max(running, value)
            adjusted.append(running)
        return tuple(adjusted)

    dependence_factor = (
        sum(1.0 / rank for rank in range(1, count + 1))
        if method is MultiplicityMethod.BENJAMINI_YEKUTIELI
        else 1.0
    )
    raw = tuple(
        min(1.0, value * count * dependence_factor / rank)
        for rank, value in enumerate(ordered_p_values, start=1)
    )
    adjusted = [0.0] * count
    running = 1.0
    for index in range(count - 1, -1, -1):
        running = min(running, raw[index])
        adjusted[index] = running
    return tuple(adjusted)


def _adjusted_hypothesis(
    test: HypothesisTestResult,
    adjusted_value: float,
    plan: MultiplicityPlan,
) -> AdjustedHypothesis:
    """Apply rejection and evidence-label semantics to one test.

    Args:
        test: Raw hypothesis result.
        adjusted_value: Family-adjusted p-value or q-value.
        plan: Analysis-mode and error-rate definition.

    Returns:
        Adjusted hypothesis with a bounded evidence label.
    """
    rejected = adjusted_value <= plan.target_error_rate
    label = MultiplicityEvidenceLabel.NOT_SELECTED
    if rejected:
        label = {
            AnalysisMode.EXPLORATORY: (MultiplicityEvidenceLabel.EXPLORATORY_SIGNAL),
            AnalysisMode.PREREGISTERED: (
                MultiplicityEvidenceLabel.PREREGISTERED_ASSOCIATION
            ),
            AnalysisMode.HELD_OUT_CONFIRMATION: (
                MultiplicityEvidenceLabel.HELD_OUT_CONFIRMED_ASSOCIATION
            ),
        }[plan.mode]
    return AdjustedHypothesis(
        hypothesis_id=test.hypothesis_id,
        raw_p_value=test.p_value,
        adjusted_value=adjusted_value,
        effect_estimate=test.effect_estimate,
        standard_error=test.standard_error,
        sample_size=test.sample_size,
        alternative=test.alternative,
        rejected=rejected,
        evidence_label=label,
        source_discovery_id=test.source_discovery_id,
        analysis_digest=test.analysis_digest,
    )


def _limitations(plan: MultiplicityPlan) -> tuple[str, ...]:
    """Return method- and mode-specific interpretive limitations.

    Args:
        plan: Multiplicity and evidence-mode definition.

    Returns:
        Explicit assumptions and bounded-claim language.
    """
    limitations = [
        "Adjusted rejection controls a family-level error criterion only when "
        "the raw p-values are valid for their sampling and selection process.",
        "Multiplicity adjustment does not correct simulator, measurement, "
        "model-form, outcome-definition, or effect-estimation bias.",
        "Rejected hypotheses support association under the declared design; "
        "they do not establish a causal mechanism.",
    ]
    if plan.method is MultiplicityMethod.BENJAMINI_HOCHBERG:
        limitations.append(
            "Benjamini-Hochberg FDR control assumes independent or suitable "
            "positive dependence among tests."
        )
    elif plan.method is MultiplicityMethod.BENJAMINI_YEKUTIELI:
        limitations.append(
            "Benjamini-Yekutieli is valid under arbitrary dependence but can "
            "be substantially conservative."
        )
    else:
        limitations.append(
            f"{plan.method.value} controls family-wise error when each raw "
            "p-value is valid."
        )
    if plan.mode is AnalysisMode.EXPLORATORY:
        limitations.append(
            "Exploratory selections require preregistered confirmation on "
            "independent held-out data before use as confirmatory evidence."
        )
    elif plan.mode is AnalysisMode.HELD_OUT_CONFIRMATION:
        limitations.append(
            "Held-out confirmation is valid only if selection and analysis "
            "datasets remained strictly separated."
        )
    return tuple(limitations)


def _require_text(value: str, label: str) -> None:
    """Require a non-empty string.

    Args:
        value: Candidate text.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_unique(values: Iterable[str], label: str) -> None:
    """Require unique values from a sequence or generator.

    Args:
        values: Candidate values.
        label: Human-readable collection label.
    """
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} must be unique")


def _require_unique_text(
    values: tuple[str, ...],
    label: str,
    *,
    required: bool = True,
) -> None:
    """Require non-empty, unique strings in a tuple.

    Args:
        values: Candidate string tuple.
        label: Human-readable collection label.
        required: Whether at least one value is required.
    """
    if required and not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_text(value, label)
    _require_unique(values, label)
