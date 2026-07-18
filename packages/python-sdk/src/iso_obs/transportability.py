"""Fixed-design validation of simulator-to-real transportability.

Transport evidence is defined relative to an intended metric, paired anchor
protocol, and explicit operating envelope. The confirmatory outcome is a
conservative expanded absolute discrepancy that includes declared simulation
and real-measurement uncertainty. It is not a universal simulator score or a
license to extrapolate beyond anchored conditions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from .evidence import (
    NamedDigest,
    ValidityRange,
    _canonical_json,
    sha256_digest,
)


class DiscrepancyKind(StrEnum):
    """Source of possible simulator-to-real discrepancy."""

    PHYSICS_MODEL_FORM = "physics_model_form"
    NUMERICAL = "numerical"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    ENVIRONMENT = "environment"
    MEASUREMENT = "measurement"
    HUMAN_INTERACTION = "human_interaction"
    LEARNED_WORLD_MODEL = "learned_world_model"


class TransportStratumConclusion(StrEnum):
    """Bounded conclusion for one paired anchor stratum."""

    INCOMPLETE = "incomplete"
    WITHIN_LIMIT = "within_limit"
    EXCEEDS_LIMIT = "exceeds_limit"
    INCONCLUSIVE = "inconclusive"


class TransportConclusion(StrEnum):
    """Non-compensatory simulator-to-real validation conclusion."""

    INCOMPLETE = "incomplete"
    ASSUMPTIONS_VIOLATED = "assumptions_violated"
    SUPPORTED_WITHIN_ANCHOR_ENVELOPE = "supported_within_anchor_envelope"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"


class TransportApplicability(StrEnum):
    """Applicability of transport evidence to one operating point."""

    APPLICABLE_WITHIN_ANCHOR_ENVELOPE = "applicable_within_anchor_envelope"
    OUTSIDE_ANCHOR_ENVELOPE = "outside_anchor_envelope"
    EVIDENCE_NOT_SUPPORTIVE = "evidence_not_supportive"


@dataclass(frozen=True, slots=True)
class DiscrepancySource:
    """One declared source of simulator-to-real discrepancy."""

    name: str
    kind: DiscrepancyKind
    characterization: str
    evidence_digest: str
    quantified: bool

    def __post_init__(self) -> None:
        """Validate discrepancy identity, evidence, and quantification state."""
        _require_text(self.name, "transport discrepancy source name")
        _require_text(
            self.characterization,
            "transport discrepancy characterization",
        )
        object.__setattr__(self, "kind", DiscrepancyKind(self.kind))
        NamedDigest("transport discrepancy evidence", self.evidence_digest)
        if not isinstance(self.quantified, bool):
            raise ValueError("discrepancy source quantified must be a boolean")


@dataclass(frozen=True, slots=True)
class TransportStratum:
    """One disjoint matched-condition cell in a paired anchor design."""

    stratum_id: str
    condition_definition_digest: str
    target_population_mass: float
    planned_pair_count: int
    maximum_acceptable_mean_expanded_discrepancy: float
    maximum_possible_expanded_discrepancy: float
    allocated_error_rate: float

    def __post_init__(self) -> None:
        """Validate stratum identity, fixed sample, bounds, and error rate."""
        _require_text(self.stratum_id, "transport stratum ID")
        NamedDigest(
            "transport stratum condition definition",
            self.condition_definition_digest,
        )
        mass = _positive_unit_interval(
            self.target_population_mass,
            "transport stratum target population mass",
        )
        if (
            isinstance(self.planned_pair_count, bool)
            or not isinstance(self.planned_pair_count, int)
            or self.planned_pair_count < 1
        ):
            raise ValueError("planned_pair_count must be a positive integer")
        acceptable = _nonnegative_finite(
            self.maximum_acceptable_mean_expanded_discrepancy,
            "maximum acceptable mean expanded discrepancy",
        )
        possible = _positive_finite(
            self.maximum_possible_expanded_discrepancy,
            "maximum possible expanded discrepancy",
        )
        if acceptable > possible:
            raise ValueError(
                "maximum acceptable mean discrepancy must not exceed the "
                "maximum possible expanded discrepancy"
            )
        allocated = _open_unit_interval(
            self.allocated_error_rate,
            "transport stratum allocated error rate",
        )
        object.__setattr__(self, "target_population_mass", mass)
        object.__setattr__(
            self,
            "maximum_acceptable_mean_expanded_discrepancy",
            acceptable,
        )
        object.__setattr__(
            self,
            "maximum_possible_expanded_discrepancy",
            possible,
        )
        object.__setattr__(self, "allocated_error_rate", allocated)


@dataclass(frozen=True, slots=True)
class TransportValidationPlan:
    """Content-addressed fixed design for paired sim-to-real validation."""

    plan_id: str
    plan_version: str
    simulator_manifest_digest: str
    real_anchor_dataset_digest: str
    pairing_protocol_digest: str
    target_population_digest: str
    partition_definition_digest: str
    metric_name: str
    metric_unit: str
    outcome_definition_digest: str
    maximum_acceptable_overall_mean_expanded_discrepancy: float
    family_error_rate: float
    anchor_envelope: tuple[ValidityRange, ...]
    discrepancy_sources: tuple[DiscrepancySource, ...]
    physics_evidence: tuple[NamedDigest, ...]
    strata: tuple[TransportStratum, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate provenance, scope, partition, and simultaneous design."""
        for value, label in (
            (self.plan_id, "transport validation plan ID"),
            (self.plan_version, "transport validation plan version"),
            (self.metric_name, "transport metric name"),
            (self.metric_unit, "transport metric unit"),
        ):
            _require_text(value, label)
        for value, label in (
            (self.simulator_manifest_digest, "simulator manifest"),
            (self.real_anchor_dataset_digest, "real anchor dataset"),
            (self.pairing_protocol_digest, "paired anchor protocol"),
            (self.target_population_digest, "transport target population"),
            (self.partition_definition_digest, "transport partition definition"),
            (self.outcome_definition_digest, "transport outcome definition"),
        ):
            NamedDigest(label, value)
        overall_limit = _nonnegative_finite(
            self.maximum_acceptable_overall_mean_expanded_discrepancy,
            "maximum acceptable overall mean expanded discrepancy",
        )
        family_rate = _open_unit_interval(
            self.family_error_rate,
            "transport family error rate",
        )
        if not self.anchor_envelope:
            raise ValueError("transport plan requires an anchor envelope")
        _require_unique(
            (item.parameter for item in self.anchor_envelope),
            "transport anchor-envelope parameters",
        )
        if not self.discrepancy_sources:
            raise ValueError("transport plan requires discrepancy sources")
        _require_unique(
            (item.name for item in self.discrepancy_sources),
            "transport discrepancy source names",
        )
        _require_unique(
            (item.name for item in self.physics_evidence),
            "transport physics evidence names",
        )
        if not self.strata:
            raise ValueError("transport validation plan requires strata")
        _require_unique(
            (item.stratum_id for item in self.strata),
            "transport stratum IDs",
        )
        _require_unique(
            (item.condition_definition_digest for item in self.strata),
            "transport stratum condition definitions",
        )
        mass = math.fsum(item.target_population_mass for item in self.strata)
        if not math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "transport stratum target population masses must sum to one"
            )
        allocated = math.fsum(item.allocated_error_rate for item in self.strata)
        if allocated > family_rate and not math.isclose(
            allocated,
            family_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "transport stratum error allocations must not exceed the "
                "family error rate"
            )
        _require_unique_text(
            self.limitations,
            "transport validation limitations",
        )
        object.__setattr__(
            self,
            "maximum_acceptable_overall_mean_expanded_discrepancy",
            overall_limit,
        )
        object.__setattr__(self, "family_error_rate", family_rate)
        object.__setattr__(
            self,
            "anchor_envelope",
            tuple(
                sorted(
                    self.anchor_envelope,
                    key=lambda item: item.parameter,
                )
            ),
        )
        object.__setattr__(
            self,
            "discrepancy_sources",
            tuple(
                sorted(
                    self.discrepancy_sources,
                    key=lambda item: item.name,
                )
            ),
        )
        object.__setattr__(
            self,
            "physics_evidence",
            tuple(sorted(self.physics_evidence, key=lambda item: item.name)),
        )
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
        """Calculate the exact transport-validation plan digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class PairedAnchorObservation:
    """One matched simulator and real-world metric observation."""

    plan_content_digest: str
    stratum_id: str
    pair_index: int
    anchor_id: str
    simulation_run_id: str
    real_run_id: str
    simulation_value: float
    real_value: float
    simulation_uncertainty_bound: float
    measurement_uncertainty_bound: float
    evidence_digest: str

    def __post_init__(self) -> None:
        """Validate matched-pair identity, values, and uncertainty bounds."""
        NamedDigest("transport validation plan", self.plan_content_digest)
        for value, label in (
            (self.stratum_id, "paired anchor stratum ID"),
            (self.anchor_id, "paired anchor ID"),
            (self.simulation_run_id, "paired simulation run ID"),
            (self.real_run_id, "paired real run ID"),
        ):
            _require_text(value, label)
        if self.simulation_run_id == self.real_run_id:
            raise ValueError("simulation and real run IDs must be different")
        if (
            isinstance(self.pair_index, bool)
            or not isinstance(self.pair_index, int)
            or self.pair_index < 0
        ):
            raise ValueError("paired anchor pair_index must be a non-negative integer")
        for field_name in ("simulation_value", "real_value"):
            resolved = float(getattr(self, field_name))
            if not math.isfinite(resolved):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, resolved)
        for field_name in (
            "simulation_uncertainty_bound",
            "measurement_uncertainty_bound",
        ):
            resolved = _nonnegative_finite(
                getattr(self, field_name),
                field_name,
            )
            object.__setattr__(self, field_name, resolved)
        NamedDigest("paired anchor evidence", self.evidence_digest)

    def signed_residual(self) -> float:
        """Calculate real-minus-simulation residual.

        Returns:
            Signed metric residual in the plan's declared unit.
        """
        return self.real_value - self.simulation_value

    def expanded_absolute_discrepancy(self) -> float:
        """Calculate conservative discrepancy including uncertainty bounds.

        Returns:
            Absolute residual plus simulation and measurement uncertainty.
        """
        return (
            abs(self.signed_residual())
            + self.simulation_uncertainty_bound
            + self.measurement_uncertainty_bound
        )


@dataclass(frozen=True, slots=True)
class TransportStratumEstimate:
    """Descriptive or confirmatory paired discrepancy for one stratum."""

    stratum_id: str
    target_population_mass: float
    planned_pair_count: int
    observed_pair_count: int
    mean_signed_residual: float | None
    mean_expanded_discrepancy: float | None
    confidence_lower: float | None
    confidence_upper: float | None
    weighted_point_contribution: float | None
    weighted_lower_contribution: float | None
    weighted_upper_contribution: float | None
    maximum_acceptable_mean_expanded_discrepancy: float
    conclusion: TransportStratumConclusion


@dataclass(frozen=True, slots=True)
class TransportValidationReport:
    """Target-weighted, scope-bounded simulator transport evidence."""

    plan_content_digest: str
    confidence_level: float
    completed_target_population_mass: float
    design_complete: bool
    assumptions_satisfied: bool
    stratum_estimates: tuple[TransportStratumEstimate, ...]
    overall_mean_expanded_discrepancy: float | None
    overall_confidence_lower: float | None
    overall_confidence_upper: float | None
    maximum_acceptable_overall_mean_expanded_discrepancy: float
    conclusion: TransportConclusion
    limiting_stratum_ids: tuple[str, ...]
    violating_anchor_ids: tuple[str, ...]
    unquantified_discrepancy_source_names: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact transport-validation report digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def validate_transportability(
    plan: TransportValidationPlan,
    observations: Sequence[PairedAnchorObservation],
) -> TransportValidationReport:
    """Validate paired simulator discrepancy within an anchor envelope.

    Expanded absolute discrepancies are bounded in ``[0, B]`` by each
    stratum's preregistered maximum. Complete strata use two-sided Hoeffding
    radii ``B * sqrt(log(2 / alpha) / (2n))``. Allocated stratum error rates
    and the union bound provide simultaneous coverage without assuming strata
    are independent. All inference is withheld if the full fixed design is
    incomplete or any declared discrepancy bound is violated.

    Args:
        plan: Fixed pairing, envelope, discrepancy sources, and limits.
        observations: Matched simulator and real outcomes in any input order.

    Returns:
        Descriptive coverage or simultaneous transport-validation evidence.

    Raises:
        ValueError: If plan identity, strata, anchor identity, pair counts, or
            pair ordering are invalid.
    """
    plan_digest = plan.content_digest()
    by_stratum: dict[str, list[PairedAnchorObservation]] = {
        item.stratum_id: [] for item in plan.strata
    }
    anchor_ids: set[str] = set()
    for observation in observations:
        if observation.plan_content_digest != plan_digest:
            raise ValueError(
                f"anchor '{observation.anchor_id}' plan content digest mismatch"
            )
        if observation.stratum_id not in by_stratum:
            raise ValueError(f"unknown transport stratum: {observation.stratum_id}")
        if observation.anchor_id in anchor_ids:
            raise ValueError(f"duplicate paired anchor ID: {observation.anchor_id}")
        anchor_ids.add(observation.anchor_id)
        by_stratum[observation.stratum_id].append(observation)

    estimates = []
    violating_anchor_ids: list[str] = []
    completed_cells = []
    for stratum in plan.strata:
        ordered = tuple(
            sorted(
                by_stratum[stratum.stratum_id],
                key=lambda item: item.pair_index,
            )
        )
        if len(ordered) > stratum.planned_pair_count:
            raise ValueError(
                f"stratum '{stratum.stratum_id}' has more pairs than its " "fixed plan"
            )
        observed_indices = tuple(item.pair_index for item in ordered)
        if observed_indices != tuple(range(len(ordered))):
            raise ValueError(
                f"stratum '{stratum.stratum_id}' pair indices must be "
                "contiguous from zero"
            )
        violating_anchor_ids.extend(
            item.anchor_id
            for item in ordered
            if item.expanded_absolute_discrepancy()
            > stratum.maximum_possible_expanded_discrepancy
        )
        completed_cells.append(len(ordered) == stratum.planned_pair_count)
        estimates.append(_estimate_transport_stratum(stratum, ordered))

    design_complete = all(completed_cells)
    completed_mass = math.fsum(
        item.target_population_mass
        for item, complete in zip(estimates, completed_cells, strict=True)
        if complete
    )
    assumptions_satisfied = not violating_anchor_ids
    if not design_complete or not assumptions_satisfied:
        estimates = [
            replace(
                item,
                confidence_lower=None,
                confidence_upper=None,
                weighted_point_contribution=None,
                weighted_lower_contribution=None,
                weighted_upper_contribution=None,
                conclusion=TransportStratumConclusion.INCOMPLETE,
            )
            for item in estimates
        ]
    unquantified = tuple(
        item.name for item in plan.discrepancy_sources if not item.quantified
    )
    (
        overall,
        lower,
        upper,
        conclusion,
        limiting_strata,
    ) = _transport_conclusion(
        plan,
        estimates,
        design_complete=design_complete,
        assumptions_satisfied=assumptions_satisfied,
        unquantified_sources=unquantified,
    )
    limitations = tuple(
        sorted(
            plan.limitations
            + (
                "Validation is relative to the exact intended metric, paired "
                "anchor protocol, target population, and operating envelope; "
                "it is not universal simulator certification.",
                "Hoeffding bounds require the fixed pair counts, independent "
                "bounded pairs representative of each stratum, and valid "
                "uncertainty bounds; adaptive stopping or outcome-dependent "
                "anchor selection invalidates confirmatory interpretation.",
                "Expanded discrepancy adds absolute residual, declared "
                "simulation uncertainty, and real measurement uncertainty; "
                "omitted or underestimated uncertainty can invalidate support.",
                "Target-population weights are treated as fixed and correct; "
                "population drift and partition misspecification are outside "
                "the statistical interval.",
                "The target-weighted average cannot compensate for a stratum "
                "whose lower bound exceeds its local discrepancy limit.",
                "Support is confined to the declared anchor envelope and does "
                "not establish accuracy, safety, or causality outside it.",
            )
        )
    )
    return TransportValidationReport(
        plan_content_digest=plan_digest,
        confidence_level=1.0 - plan.family_error_rate,
        completed_target_population_mass=completed_mass,
        design_complete=design_complete,
        assumptions_satisfied=assumptions_satisfied,
        stratum_estimates=tuple(estimates),
        overall_mean_expanded_discrepancy=overall,
        overall_confidence_lower=lower,
        overall_confidence_upper=upper,
        maximum_acceptable_overall_mean_expanded_discrepancy=(
            plan.maximum_acceptable_overall_mean_expanded_discrepancy
        ),
        conclusion=conclusion,
        limiting_stratum_ids=limiting_strata,
        violating_anchor_ids=tuple(sorted(violating_anchor_ids)),
        unquantified_discrepancy_source_names=unquantified,
        limitations=limitations,
    )


def _estimate_transport_stratum(
    stratum: TransportStratum,
    observations: Sequence[PairedAnchorObservation],
) -> TransportStratumEstimate:
    """Calculate descriptive and complete-cell discrepancy estimates.

    Args:
        stratum: Planned cell mass, bounds, and local limit.
        observations: Ordered paired anchors for the cell.

    Returns:
        Stratum discrepancy summary.
    """
    count = len(observations)
    mean_signed = (
        math.fsum(item.signed_residual() for item in observations) / count
        if count
        else None
    )
    mean_expanded = (
        math.fsum(item.expanded_absolute_discrepancy() for item in observations) / count
        if count
        else None
    )
    if count != stratum.planned_pair_count or mean_expanded is None:
        return TransportStratumEstimate(
            stratum_id=stratum.stratum_id,
            target_population_mass=stratum.target_population_mass,
            planned_pair_count=stratum.planned_pair_count,
            observed_pair_count=count,
            mean_signed_residual=mean_signed,
            mean_expanded_discrepancy=mean_expanded,
            confidence_lower=None,
            confidence_upper=None,
            weighted_point_contribution=None,
            weighted_lower_contribution=None,
            weighted_upper_contribution=None,
            maximum_acceptable_mean_expanded_discrepancy=(
                stratum.maximum_acceptable_mean_expanded_discrepancy
            ),
            conclusion=TransportStratumConclusion.INCOMPLETE,
        )
    radius = stratum.maximum_possible_expanded_discrepancy * math.sqrt(
        math.log(2.0 / stratum.allocated_error_rate) / (2.0 * count)
    )
    lower = max(0.0, mean_expanded - radius)
    upper = min(
        stratum.maximum_possible_expanded_discrepancy,
        mean_expanded + radius,
    )
    if lower > stratum.maximum_acceptable_mean_expanded_discrepancy:
        conclusion = TransportStratumConclusion.EXCEEDS_LIMIT
    elif upper <= stratum.maximum_acceptable_mean_expanded_discrepancy:
        conclusion = TransportStratumConclusion.WITHIN_LIMIT
    else:
        conclusion = TransportStratumConclusion.INCONCLUSIVE
    mass = stratum.target_population_mass
    return TransportStratumEstimate(
        stratum_id=stratum.stratum_id,
        target_population_mass=mass,
        planned_pair_count=stratum.planned_pair_count,
        observed_pair_count=count,
        mean_signed_residual=mean_signed,
        mean_expanded_discrepancy=mean_expanded,
        confidence_lower=lower,
        confidence_upper=upper,
        weighted_point_contribution=mass * mean_expanded,
        weighted_lower_contribution=mass * lower,
        weighted_upper_contribution=mass * upper,
        maximum_acceptable_mean_expanded_discrepancy=(
            stratum.maximum_acceptable_mean_expanded_discrepancy
        ),
        conclusion=conclusion,
    )


def _transport_conclusion(
    plan: TransportValidationPlan,
    estimates: Sequence[TransportStratumEstimate],
    *,
    design_complete: bool,
    assumptions_satisfied: bool,
    unquantified_sources: tuple[str, ...],
) -> tuple[
    float | None,
    float | None,
    float | None,
    TransportConclusion,
    tuple[str, ...],
]:
    """Combine cell evidence without masking local transport failures.

    Args:
        plan: Overall limit and target-population design.
        estimates: Ordered stratum estimates.
        design_complete: Whether all fixed pair counts are complete.
        assumptions_satisfied: Whether every expanded discrepancy stayed
            within its declared bound.
        unquantified_sources: Known discrepancy sources lacking quantification.

    Returns:
        Overall point, bounds, conclusion, and limiting stratum IDs.
    """
    if not design_complete:
        incomplete = tuple(
            item.stratum_id
            for item in estimates
            if item.conclusion is TransportStratumConclusion.INCOMPLETE
        )
        return (
            None,
            None,
            None,
            TransportConclusion.INCOMPLETE,
            incomplete,
        )
    if not assumptions_satisfied:
        return (
            None,
            None,
            None,
            TransportConclusion.ASSUMPTIONS_VIOLATED,
            (),
        )
    overall = math.fsum(
        _required(item.weighted_point_contribution) for item in estimates
    )
    lower = math.fsum(_required(item.weighted_lower_contribution) for item in estimates)
    upper = math.fsum(_required(item.weighted_upper_contribution) for item in estimates)
    exceeded = tuple(
        item.stratum_id
        for item in estimates
        if item.conclusion is TransportStratumConclusion.EXCEEDS_LIMIT
    )
    inconclusive = tuple(
        item.stratum_id
        for item in estimates
        if item.conclusion is TransportStratumConclusion.INCONCLUSIVE
    )
    if lower > plan.maximum_acceptable_overall_mean_expanded_discrepancy or exceeded:
        conclusion = TransportConclusion.NOT_SUPPORTED
        limiting = exceeded
    elif (
        upper <= plan.maximum_acceptable_overall_mean_expanded_discrepancy
        and not inconclusive
        and not unquantified_sources
    ):
        conclusion = TransportConclusion.SUPPORTED_WITHIN_ANCHOR_ENVELOPE
        limiting = ()
    else:
        conclusion = TransportConclusion.INCONCLUSIVE
        limiting = inconclusive
    return overall, lower, upper, conclusion, limiting


@dataclass(frozen=True, slots=True)
class OperatingCondition:
    """One requested operating-condition value for applicability review."""

    parameter: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        """Validate requested parameter identity, value, and unit."""
        _require_text(self.parameter, "operating condition parameter")
        _require_text(self.unit, "operating condition unit")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("operating condition value must be finite")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class TransportApplicabilityReport:
    """Applicability of validated transport evidence to one operating point."""

    plan_content_digest: str
    validation_report_digest: str
    requested_conditions: tuple[OperatingCondition, ...]
    out_of_range_parameters: tuple[str, ...]
    applicability: TransportApplicability
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the applicability report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact applicability-report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def assess_transport_applicability(
    plan: TransportValidationPlan,
    report: TransportValidationReport,
    conditions: Sequence[OperatingCondition],
) -> TransportApplicabilityReport:
    """Assess whether transport evidence applies to one operating point.

    Args:
        plan: Exact anchor envelope and metric validation plan.
        report: Validation evidence produced from the plan.
        conditions: Exactly one value for every anchor-envelope parameter.

    Returns:
        Scope applicability without extrapolation.

    Raises:
        ValueError: If report identity, parameters, or units are inconsistent.
    """
    if report.plan_content_digest != plan.content_digest():
        raise ValueError("transport validation report plan digest mismatch")
    ordered = tuple(sorted(conditions, key=lambda item: item.parameter))
    _require_unique(
        (item.parameter for item in ordered),
        "requested operating-condition parameters",
    )
    ranges = {item.parameter: item for item in plan.anchor_envelope}
    supplied = {item.parameter for item in ordered}
    expected = set(ranges)
    if supplied != expected:
        raise ValueError(
            "operating conditions must exactly match anchor-envelope "
            f"parameters (missing={sorted(expected - supplied)}, "
            f"extra={sorted(supplied - expected)})"
        )
    out_of_range = []
    for condition in ordered:
        validity_range = ranges[condition.parameter]
        if condition.unit != validity_range.unit:
            raise ValueError(
                f"operating condition '{condition.parameter}' unit mismatch"
            )
        if not validity_range.lower <= condition.value <= validity_range.upper:
            out_of_range.append(condition.parameter)
    if out_of_range:
        applicability = TransportApplicability.OUTSIDE_ANCHOR_ENVELOPE
    elif report.conclusion is TransportConclusion.SUPPORTED_WITHIN_ANCHOR_ENVELOPE:
        applicability = TransportApplicability.APPLICABLE_WITHIN_ANCHOR_ENVELOPE
    else:
        applicability = TransportApplicability.EVIDENCE_NOT_SUPPORTIVE
    limitations = (
        "Applicability means only that this operating point lies inside the "
        "validated anchor envelope and the linked discrepancy evidence was "
        "supportive; it is not a prediction guarantee.",
        "Interpolation within an envelope can still encounter unrepresented "
        "interactions or local model-form error.",
        "Any system, simulator, measurement protocol, target population, "
        "metric, or envelope change requires a new content-addressed review.",
    )
    return TransportApplicabilityReport(
        plan_content_digest=plan.content_digest(),
        validation_report_digest=report.content_digest(),
        requested_conditions=ordered,
        out_of_range_parameters=tuple(sorted(out_of_range)),
        applicability=applicability,
        limitations=limitations,
    )


def _required(value: float | None) -> float:
    """Return a value required by a complete, valid design.

    Args:
        value: Optional weighted contribution.

    Returns:
        Required float.

    Raises:
        RuntimeError: If an internal complete-design invariant is violated.
    """
    if value is None:
        raise RuntimeError("complete transport estimate is missing a value")
    return value


def _positive_unit_interval(value: float, label: str) -> float:
    """Validate a finite value greater than zero and at most one.

    Args:
        value: Candidate population mass.
        label: Human-readable field label.

    Returns:
        Canonical float value.
    """
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved <= 1.0:
        raise ValueError(f"{label} must be finite and between zero and one")
    return resolved


def _open_unit_interval(value: float, label: str) -> float:
    """Validate a finite value strictly between zero and one.

    Args:
        value: Candidate probability.
        label: Human-readable field label.

    Returns:
        Canonical float value.
    """
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ValueError(f"{label} must be finite and between zero and one")
    return resolved


def _nonnegative_finite(value: float, label: str) -> float:
    """Validate a finite nonnegative scalar.

    Args:
        value: Candidate scalar.
        label: Human-readable field label.

    Returns:
        Canonical float value.
    """
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return resolved


def _positive_finite(value: float, label: str) -> float:
    """Validate a finite positive scalar.

    Args:
        value: Candidate scalar.
        label: Human-readable field label.

    Returns:
        Canonical float value.
    """
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
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
