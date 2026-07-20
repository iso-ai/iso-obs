"""Fixed-design stratified estimation of an operational failure surface.

The estimator supports deliberate oversampling of rare operating conditions
while recovering a target-population failure probability through declared
stratum masses. Simultaneous finite-sample Hoeffding intervals preserve narrow
high-risk regions rather than allowing an aggregate average to hide them.
Inferential conclusions are withheld until every fixed stratum sample is
complete.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest


class StratumConclusion(StrEnum):
    """Bounded conclusion for one operating-condition stratum."""

    INCOMPLETE = "incomplete"
    WITHIN_LIMIT = "within_limit"
    EXCEEDS_LIMIT = "exceeds_limit"
    INCONCLUSIVE = "inconclusive"


class FailureSurfaceConclusion(StrEnum):
    """Non-compensatory conclusion across the complete failure surface."""

    INCOMPLETE = "incomplete"
    WITHIN_LIMITS = "within_limits"
    EXCEEDS_LIMITS = "exceeds_limits"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class FailureSurfaceStratum:
    """One disjoint operating-condition cell in a fixed sampling design."""

    stratum_id: str
    condition_definition_digest: str
    target_population_mass: float
    planned_sample_count: int
    maximum_acceptable_failure_rate: float
    allocated_error_rate: float

    def __post_init__(self) -> None:
        """Validate stratum identity, target mass, sample size, and limits."""
        _require_text(self.stratum_id, "failure-surface stratum ID")
        NamedDigest(
            "stratum condition definition",
            self.condition_definition_digest,
        )
        mass = _positive_unit_interval(
            self.target_population_mass,
            "stratum target population mass",
        )
        maximum = _closed_unit_interval(
            self.maximum_acceptable_failure_rate,
            "stratum maximum acceptable failure rate",
        )
        allocated = _open_unit_interval(
            self.allocated_error_rate,
            "stratum allocated error rate",
        )
        if (
            isinstance(self.planned_sample_count, bool)
            or not isinstance(self.planned_sample_count, int)
            or self.planned_sample_count < 1
        ):
            raise ValueError("planned_sample_count must be a positive integer")
        object.__setattr__(self, "target_population_mass", mass)
        object.__setattr__(
            self,
            "maximum_acceptable_failure_rate",
            maximum,
        )
        object.__setattr__(self, "allocated_error_rate", allocated)


@dataclass(frozen=True, slots=True)
class FailureSurfacePlan:
    """Content-addressed fixed design for stratified failure estimation."""

    plan_id: str
    plan_version: str
    target_population_digest: str
    partition_definition_digest: str
    simulation_manifest_digest: str
    system_artifact_digest: str
    outcome_definition_digest: str
    maximum_acceptable_overall_failure_rate: float
    family_error_rate: float
    strata: tuple[FailureSurfaceStratum, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate identities, partition mass, and simultaneous error budget."""
        _require_text(self.plan_id, "failure-surface plan ID")
        _require_text(self.plan_version, "failure-surface plan version")
        for value, label in (
            (self.target_population_digest, "target population"),
            (self.partition_definition_digest, "partition definition"),
            (self.simulation_manifest_digest, "simulation manifest"),
            (self.system_artifact_digest, "system artifact"),
            (self.outcome_definition_digest, "failure outcome definition"),
        ):
            NamedDigest(label, value)
        overall_limit = _closed_unit_interval(
            self.maximum_acceptable_overall_failure_rate,
            "maximum acceptable overall failure rate",
        )
        family_rate = _open_unit_interval(
            self.family_error_rate,
            "failure-surface family error rate",
        )
        if not self.strata:
            raise ValueError("failure-surface plan requires strata")
        _require_unique(
            (item.stratum_id for item in self.strata),
            "failure-surface stratum IDs",
        )
        _require_unique(
            (item.condition_definition_digest for item in self.strata),
            "stratum condition definitions",
        )
        target_mass = math.fsum(item.target_population_mass for item in self.strata)
        if not math.isclose(target_mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("stratum target population masses must sum to one")
        allocated = math.fsum(item.allocated_error_rate for item in self.strata)
        if allocated > family_rate and not math.isclose(
            allocated,
            family_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "stratum error allocations must not exceed the family error rate"
            )
        _require_unique_text(
            self.limitations,
            "failure-surface plan limitations",
        )
        object.__setattr__(
            self,
            "maximum_acceptable_overall_failure_rate",
            overall_limit,
        )
        object.__setattr__(self, "family_error_rate", family_rate)
        object.__setattr__(
            self,
            "strata",
            tuple(sorted(self.strata, key=lambda item: item.stratum_id)),
        )
        object.__setattr__(self, "limitations", tuple(sorted(self.limitations)))

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact failure-surface plan digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class FailureSurfaceObservation:
    """One fixed-design binary outcome within a declared stratum."""

    plan_content_digest: str
    stratum_id: str
    sample_index: int
    trial_id: str
    failed: bool
    evidence_digest: str

    def __post_init__(self) -> None:
        """Validate observation provenance, ordering coordinate, and outcome."""
        NamedDigest("failure-surface plan", self.plan_content_digest)
        _require_text(self.stratum_id, "observation stratum ID")
        _require_text(self.trial_id, "failure-surface trial ID")
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise ValueError(
                "failure-surface sample_index must be a non-negative integer"
            )
        if not isinstance(self.failed, bool):
            raise ValueError("failure-surface failed must be a boolean")
        NamedDigest("failure-surface observation evidence", self.evidence_digest)


@dataclass(frozen=True, slots=True)
class StratumFailureEstimate:
    """Descriptive or confirmatory estimate for one failure-surface cell."""

    stratum_id: str
    target_population_mass: float
    planned_sample_count: int
    observation_count: int
    failure_count: int
    empirical_failure_rate: float | None
    confidence_lower: float | None
    confidence_upper: float | None
    weighted_point_contribution: float | None
    weighted_lower_contribution: float | None
    weighted_upper_contribution: float | None
    maximum_acceptable_failure_rate: float
    conclusion: StratumConclusion


@dataclass(frozen=True, slots=True)
class FailureSurfaceReport:
    """Simultaneous target-weighted estimate of a complete failure surface."""

    plan_content_digest: str
    confidence_level: float
    completed_target_population_mass: float
    design_complete: bool
    stratum_estimates: tuple[StratumFailureEstimate, ...]
    overall_failure_probability: float | None
    overall_confidence_lower: float | None
    overall_confidence_upper: float | None
    maximum_acceptable_overall_failure_rate: float
    conclusion: FailureSurfaceConclusion
    limiting_stratum_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    report_schema_version: str = "iso-obs.failure-surface-report.v1"

    def to_json(self) -> str:
        """Serialize the report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact failure-surface report digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def estimate_failure_surface(
    plan: FailureSurfacePlan,
    observations: Sequence[FailureSurfaceObservation],
) -> FailureSurfaceReport:
    """Estimate stratified failure probability after a fixed sampling design.

    The confirmatory interval is emitted only when every stratum contains its
    exact planned number of observations. Per-stratum two-sided Hoeffding
    intervals use their allocated error rates. By the union bound, all stratum
    intervals—and therefore their target-mass-weighted sum—cover
    simultaneously with probability at least ``1 - family_error_rate``.

    Args:
        plan: Fixed target partition, samples, limits, and error allocations.
        observations: Binary fixed-design results in any input order.

    Returns:
        Descriptive coverage or a simultaneous failure-surface estimate.

    Raises:
        ValueError: If observations have wrong plan identity, unknown strata,
            duplicate trials, excess samples, or non-contiguous sample indices.
    """
    plan_digest = plan.content_digest()
    by_stratum: dict[str, list[FailureSurfaceObservation]] = {
        item.stratum_id: [] for item in plan.strata
    }
    trial_ids: set[str] = set()
    for observation in observations:
        if observation.plan_content_digest != plan_digest:
            raise ValueError(
                f"trial '{observation.trial_id}' plan content digest mismatch"
            )
        if observation.stratum_id not in by_stratum:
            raise ValueError(
                f"unknown failure-surface stratum: {observation.stratum_id}"
            )
        if observation.trial_id in trial_ids:
            raise ValueError(
                f"duplicate failure-surface trial ID: {observation.trial_id}"
            )
        trial_ids.add(observation.trial_id)
        by_stratum[observation.stratum_id].append(observation)

    estimates = []
    for stratum in plan.strata:
        ordered = tuple(
            sorted(
                by_stratum[stratum.stratum_id],
                key=lambda item: item.sample_index,
            )
        )
        if len(ordered) > stratum.planned_sample_count:
            raise ValueError(
                f"stratum '{stratum.stratum_id}' has more observations than "
                "its fixed plan"
            )
        observed_indices = tuple(item.sample_index for item in ordered)
        if observed_indices != tuple(range(len(ordered))):
            raise ValueError(
                f"stratum '{stratum.stratum_id}' sample indices must be "
                "contiguous from zero"
            )
        estimates.append(_estimate_stratum(stratum, ordered))

    completed_cells = tuple(
        item.observation_count == item.planned_sample_count for item in estimates
    )
    completed_mass = math.fsum(
        item.target_population_mass
        for item, complete in zip(estimates, completed_cells, strict=True)
        if complete
    )
    design_complete = all(completed_cells)
    if not design_complete:
        estimates = [
            replace(
                item,
                confidence_lower=None,
                confidence_upper=None,
                weighted_point_contribution=None,
                weighted_lower_contribution=None,
                weighted_upper_contribution=None,
                conclusion=StratumConclusion.INCOMPLETE,
            )
            for item in estimates
        ]
    (
        overall,
        lower,
        upper,
        conclusion,
        limiting_strata,
    ) = _surface_conclusion(plan, estimates, design_complete=design_complete)
    limitations = tuple(
        sorted(
            plan.limitations
            + (
                "Simultaneous Hoeffding bounds assume the fixed sample count "
                "and independent bounded outcomes representative of each "
                "declared stratum; adaptive stopping or outcome-dependent "
                "sampling invalidates the confirmatory interpretation.",
                "Target-population weights are treated as fixed and correct; "
                "population drift or partition misspecification is not "
                "included in the confidence interval.",
                "The overall average cannot compensate for a stratum whose "
                "simultaneous lower bound exceeds its own declared limit.",
                "A within-limits conclusion applies only to the content-"
                "addressed system, simulator, outcome, target population, "
                "partition, and validity limitations in this plan.",
                "Incomplete designs provide only sampling coverage, counts, "
                "and raw rates; all confidence bounds and limit conclusions "
                "are withheld to prevent selective reporting.",
            )
        )
    )
    return FailureSurfaceReport(
        plan_content_digest=plan_digest,
        confidence_level=1.0 - plan.family_error_rate,
        completed_target_population_mass=completed_mass,
        design_complete=design_complete,
        stratum_estimates=tuple(estimates),
        overall_failure_probability=overall,
        overall_confidence_lower=lower,
        overall_confidence_upper=upper,
        maximum_acceptable_overall_failure_rate=(
            plan.maximum_acceptable_overall_failure_rate
        ),
        conclusion=conclusion,
        limiting_stratum_ids=limiting_strata,
        limitations=limitations,
    )


def _estimate_stratum(
    stratum: FailureSurfaceStratum,
    observations: Sequence[FailureSurfaceObservation],
) -> StratumFailureEstimate:
    """Estimate one stratum when its fixed sample is complete.

    Args:
        stratum: Planned population mass, sample count, and failure limit.
        observations: Contiguous observations for the stratum.

    Returns:
        Descriptive incomplete result or confirmatory bounded estimate.
    """
    count = len(observations)
    failures = sum(item.failed for item in observations)
    empirical = failures / count if count else None
    if count != stratum.planned_sample_count:
        return StratumFailureEstimate(
            stratum_id=stratum.stratum_id,
            target_population_mass=stratum.target_population_mass,
            planned_sample_count=stratum.planned_sample_count,
            observation_count=count,
            failure_count=failures,
            empirical_failure_rate=empirical,
            confidence_lower=None,
            confidence_upper=None,
            weighted_point_contribution=None,
            weighted_lower_contribution=None,
            weighted_upper_contribution=None,
            maximum_acceptable_failure_rate=(stratum.maximum_acceptable_failure_rate),
            conclusion=StratumConclusion.INCOMPLETE,
        )

    assert empirical is not None
    radius = math.sqrt(math.log(2.0 / stratum.allocated_error_rate) / (2.0 * count))
    lower = max(0.0, empirical - radius)
    upper = min(1.0, empirical + radius)
    if lower > stratum.maximum_acceptable_failure_rate:
        conclusion = StratumConclusion.EXCEEDS_LIMIT
    elif upper <= stratum.maximum_acceptable_failure_rate:
        conclusion = StratumConclusion.WITHIN_LIMIT
    else:
        conclusion = StratumConclusion.INCONCLUSIVE
    mass = stratum.target_population_mass
    return StratumFailureEstimate(
        stratum_id=stratum.stratum_id,
        target_population_mass=mass,
        planned_sample_count=stratum.planned_sample_count,
        observation_count=count,
        failure_count=failures,
        empirical_failure_rate=empirical,
        confidence_lower=lower,
        confidence_upper=upper,
        weighted_point_contribution=mass * empirical,
        weighted_lower_contribution=mass * lower,
        weighted_upper_contribution=mass * upper,
        maximum_acceptable_failure_rate=(stratum.maximum_acceptable_failure_rate),
        conclusion=conclusion,
    )


def _surface_conclusion(
    plan: FailureSurfacePlan,
    estimates: Sequence[StratumFailureEstimate],
    *,
    design_complete: bool,
) -> tuple[
    float | None,
    float | None,
    float | None,
    FailureSurfaceConclusion,
    tuple[str, ...],
]:
    """Combine complete stratum estimates without masking local failures.

    Args:
        plan: Overall limit and target population design.
        estimates: Ordered stratum estimates.
        design_complete: Whether every fixed sample is complete.

    Returns:
        Overall point, lower, upper, conclusion, and limiting stratum IDs.
    """
    if not design_complete:
        incomplete = tuple(
            item.stratum_id
            for item in estimates
            if item.conclusion is StratumConclusion.INCOMPLETE
        )
        return (
            None,
            None,
            None,
            FailureSurfaceConclusion.INCOMPLETE,
            incomplete,
        )
    overall = math.fsum(
        _required(item.weighted_point_contribution) for item in estimates
    )
    lower = math.fsum(_required(item.weighted_lower_contribution) for item in estimates)
    upper = math.fsum(_required(item.weighted_upper_contribution) for item in estimates)
    exceeded_strata = tuple(
        item.stratum_id
        for item in estimates
        if item.conclusion is StratumConclusion.EXCEEDS_LIMIT
    )
    inconclusive_strata = tuple(
        item.stratum_id
        for item in estimates
        if item.conclusion is StratumConclusion.INCONCLUSIVE
    )
    global_exceeded = lower > plan.maximum_acceptable_overall_failure_rate
    global_within = upper <= plan.maximum_acceptable_overall_failure_rate
    if global_exceeded or exceeded_strata:
        conclusion = FailureSurfaceConclusion.EXCEEDS_LIMITS
        limiting = exceeded_strata
    elif global_within and not inconclusive_strata:
        conclusion = FailureSurfaceConclusion.WITHIN_LIMITS
        limiting = ()
    else:
        conclusion = FailureSurfaceConclusion.INCONCLUSIVE
        limiting = inconclusive_strata
    return overall, lower, upper, conclusion, limiting


def _required(value: float | None) -> float:
    """Return a value known to exist for a complete design.

    Args:
        value: Optional contribution from a stratum estimate.

    Returns:
        Required float contribution.

    Raises:
        RuntimeError: If a complete-design invariant is violated.
    """
    if value is None:
        raise RuntimeError("complete failure-surface estimate is missing a value")
    return value


def _open_unit_interval(value: float, label: str) -> float:
    """Validate a finite value strictly between zero and one.

    Args:
        value: Candidate probability.
        label: Human-readable field label.

    Returns:
        Canonical float probability.
    """
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ValueError(f"{label} must be finite and between zero and one")
    return resolved


def _positive_unit_interval(value: float, label: str) -> float:
    """Validate a finite value greater than zero and at most one.

    Args:
        value: Candidate probability or population mass.
        label: Human-readable field label.

    Returns:
        Canonical float value.
    """
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved <= 1.0:
        raise ValueError(f"{label} must be finite and between zero and one")
    return resolved


def _closed_unit_interval(value: float, label: str) -> float:
    """Validate a finite value between zero and one, inclusive.

    Args:
        value: Candidate probability.
        label: Human-readable field label.

    Returns:
        Canonical float probability.
    """
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{label} must be finite and between zero and one")
    return resolved


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
