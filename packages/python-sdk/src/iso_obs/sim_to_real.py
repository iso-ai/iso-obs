"""Distribution-free sim-to-real validity assessment.

The model compares bounded feature means and failure rates between simulation
and real-world anchors within predeclared operating regions. Simultaneous
Hoeffding bounds support equivalence, shift, or inconclusive conclusions
without treating a non-significant difference as evidence of equivalence.

Claims require independent sampling units within each domain. Bounds control
the declared familywise error rate by Bonferroni allocation and make no
parametric distribution assumption, but they do not establish causality.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse


class TransferDomain(StrEnum):
    """Evidence domain represented by a validity observation."""

    SIMULATION = "simulation"
    REAL_WORLD = "real_world"


class ValidityClaimKind(StrEnum):
    """Kind of bounded sim-to-real discrepancy claim."""

    FEATURE_MEAN_GAP = "feature_mean_gap"
    FAILURE_RATE_GAP = "failure_rate_gap"


class GapDisposition(StrEnum):
    """Conclusion for one predeclared equivalence claim."""

    SUPPORTED_WITHIN_TOLERANCE = "supported_within_tolerance"
    SHIFT_EXCEEDS_TOLERANCE = "shift_exceeds_tolerance"
    INCONCLUSIVE = "inconclusive"


class TransferDisposition(StrEnum):
    """Non-compensatory validity conclusion for a region or full plan."""

    SUPPORTED_WITHIN_SCOPE = "supported_within_scope"
    SHIFT_DETECTED = "shift_detected"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class TransferFeature:
    """One bounded feature used to compare simulation and reality."""

    feature_id: str
    definition_digest: str
    lower_bound: float
    upper_bound: float
    maximum_acceptable_mean_gap: float

    def __post_init__(self) -> None:
        """Validate feature identity, bounds, and equivalence tolerance."""
        _require_text(self.feature_id, "transfer feature ID")
        NamedDigest("transfer feature definition", self.definition_digest)
        lower = _finite(self.lower_bound, "transfer feature lower bound")
        upper = _finite(self.upper_bound, "transfer feature upper bound")
        if lower >= upper:
            raise ValueError("transfer feature lower bound must be below upper bound")
        tolerance = _nonnegative_finite(
            self.maximum_acceptable_mean_gap,
            "maximum acceptable feature mean gap",
        )
        if tolerance >= upper - lower:
            raise ValueError(
                "maximum acceptable feature mean gap must be below feature range"
            )
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "maximum_acceptable_mean_gap", tolerance)


@dataclass(frozen=True, slots=True)
class TransferRegion:
    """One operating region requiring separate validity support."""

    region_id: str
    condition_definition_digest: str
    target_population_mass: float
    minimum_simulation_samples: int
    minimum_real_world_samples: int

    def __post_init__(self) -> None:
        """Validate region identity, target mass, and sample requirements."""
        _require_text(self.region_id, "transfer region ID")
        NamedDigest("transfer region condition", self.condition_definition_digest)
        mass = _positive_unit_interval(
            self.target_population_mass,
            "transfer region target population mass",
        )
        minimum_simulation = _positive_integer(
            self.minimum_simulation_samples,
            "minimum simulation samples",
        )
        minimum_real = _positive_integer(
            self.minimum_real_world_samples,
            "minimum real-world samples",
        )
        object.__setattr__(self, "target_population_mass", mass)
        object.__setattr__(
            self,
            "minimum_simulation_samples",
            minimum_simulation,
        )
        object.__setattr__(
            self,
            "minimum_real_world_samples",
            minimum_real,
        )


@dataclass(frozen=True, slots=True)
class SimToRealValidityPlan:
    """Content-addressed, predeclared sim-to-real validity design."""

    plan_id: str
    plan_version: str
    target_population_digest: str
    simulation_evidence_digest: str
    real_world_anchor_digest: str
    system_artifact_digest: str
    failure_outcome_definition_digest: str
    features: tuple[TransferFeature, ...]
    regions: tuple[TransferRegion, ...]
    maximum_acceptable_failure_rate_gap: float
    familywise_error_rate: float
    limitations: tuple[str, ...]
    evidence_use: EvidenceUse = EvidenceUse.CONFIRMATORY

    def __post_init__(self) -> None:
        """Validate plan identity, partition, tolerances, and evidence use."""
        _require_text(self.plan_id, "sim-to-real validity plan ID")
        _require_text(self.plan_version, "sim-to-real validity plan version")
        for value, label in (
            (self.target_population_digest, "target population"),
            (self.simulation_evidence_digest, "simulation evidence"),
            (self.real_world_anchor_digest, "real-world anchor"),
            (self.system_artifact_digest, "system artifact"),
            (self.failure_outcome_definition_digest, "failure outcome definition"),
        ):
            NamedDigest(label, value)
        if not self.features:
            raise ValueError("sim-to-real validity plan requires features")
        features = tuple(self.features)
        _require_unique(
            (item.feature_id for item in features),
            "transfer feature IDs",
        )
        _require_unique(
            (item.definition_digest for item in features),
            "transfer feature definitions",
        )
        if not self.regions:
            raise ValueError("sim-to-real validity plan requires regions")
        regions = tuple(sorted(self.regions, key=lambda item: item.region_id))
        _require_unique(
            (item.region_id for item in regions),
            "transfer region IDs",
        )
        _require_unique(
            (item.condition_definition_digest for item in regions),
            "transfer region conditions",
        )
        total_mass = math.fsum(item.target_population_mass for item in regions)
        if not math.isclose(total_mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("transfer region target masses must sum to one")
        failure_tolerance = _closed_open_unit_interval(
            self.maximum_acceptable_failure_rate_gap,
            "maximum acceptable failure-rate gap",
        )
        familywise_error_rate = _open_unit_interval(
            self.familywise_error_rate,
            "sim-to-real familywise error rate",
        )
        limitations = _unique_text(
            self.limitations,
            "sim-to-real validity plan limitations",
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.CONFIRMATORY:
            raise ValueError("sim-to-real validity plan must be confirmatory")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(
            self,
            "maximum_acceptable_failure_rate_gap",
            failure_tolerance,
        )
        object.__setattr__(
            self,
            "familywise_error_rate",
            familywise_error_rate,
        )
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_use", evidence_use)

    def to_json(self) -> str:
        """Serialize the validity plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact validity-plan content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class TransferObservation:
    """One independently sampled simulation or real-world observation."""

    plan_content_digest: str
    evidence_digest: str
    independence_unit_id: str
    domain: TransferDomain
    region_id: str
    feature_values: tuple[float, ...]
    failed: bool

    def __post_init__(self) -> None:
        """Validate observation identity, domain, values, and failure outcome."""
        NamedDigest("sim-to-real validity plan", self.plan_content_digest)
        NamedDigest("transfer observation evidence", self.evidence_digest)
        _require_text(self.independence_unit_id, "transfer independence unit ID")
        domain = TransferDomain(self.domain)
        _require_text(self.region_id, "transfer observation region ID")
        values = _finite_vector(
            self.feature_values,
            "transfer observation feature values",
        )
        if not isinstance(self.failed, bool):
            raise ValueError("transfer observation failed must be a boolean")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "feature_values", values)


@dataclass(frozen=True, slots=True)
class ValidityGapAssessment:
    """Simultaneous confidence assessment for one bounded gap."""

    claim_id: str
    kind: ValidityClaimKind
    simulation_estimate: float
    real_world_estimate: float
    signed_gap_estimate: float
    absolute_gap_estimate: float
    confidence_radius: float
    absolute_gap_lower_bound: float
    absolute_gap_upper_bound: float
    maximum_acceptable_gap: float
    allocated_error_rate: float
    disposition: GapDisposition

    def __post_init__(self) -> None:
        """Validate estimate, confidence interval, tolerance, and disposition."""
        _require_text(self.claim_id, "validity claim ID")
        kind = ValidityClaimKind(self.kind)
        simulation_estimate = _finite(
            self.simulation_estimate,
            "simulation estimate",
        )
        real_estimate = _finite(
            self.real_world_estimate,
            "real-world estimate",
        )
        signed_gap = _finite(self.signed_gap_estimate, "signed gap estimate")
        expected_signed_gap = real_estimate - simulation_estimate
        if not math.isclose(
            signed_gap,
            expected_signed_gap,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("signed gap must equal real minus simulation")
        absolute_gap = _nonnegative_finite(
            self.absolute_gap_estimate,
            "absolute gap estimate",
        )
        if not math.isclose(
            absolute_gap,
            abs(signed_gap),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("absolute gap must equal absolute signed gap")
        radius = _nonnegative_finite(
            self.confidence_radius,
            "gap confidence radius",
        )
        lower = _nonnegative_finite(
            self.absolute_gap_lower_bound,
            "absolute gap lower bound",
        )
        upper = _nonnegative_finite(
            self.absolute_gap_upper_bound,
            "absolute gap upper bound",
        )
        if lower > upper:
            raise ValueError("absolute gap lower bound must not exceed upper bound")
        tolerance = _nonnegative_finite(
            self.maximum_acceptable_gap,
            "maximum acceptable gap",
        )
        allocated_error = _open_unit_interval(
            self.allocated_error_rate,
            "allocated validity claim error rate",
        )
        disposition = GapDisposition(self.disposition)
        expected_disposition = _gap_disposition(lower, upper, tolerance)
        if disposition is not expected_disposition:
            raise ValueError("gap disposition is inconsistent with its bounds")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "simulation_estimate", simulation_estimate)
        object.__setattr__(self, "real_world_estimate", real_estimate)
        object.__setattr__(self, "signed_gap_estimate", signed_gap)
        object.__setattr__(self, "absolute_gap_estimate", absolute_gap)
        object.__setattr__(self, "confidence_radius", radius)
        object.__setattr__(self, "absolute_gap_lower_bound", lower)
        object.__setattr__(self, "absolute_gap_upper_bound", upper)
        object.__setattr__(self, "maximum_acceptable_gap", tolerance)
        object.__setattr__(self, "allocated_error_rate", allocated_error)
        object.__setattr__(self, "disposition", disposition)


@dataclass(frozen=True, slots=True)
class RegionTransferAssessment:
    """Non-compensatory validity conclusion for one operating region."""

    region_id: str
    target_population_mass: float
    simulation_sample_count: int
    real_world_sample_count: int
    disposition: TransferDisposition
    claims: tuple[ValidityGapAssessment, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate regional counts, claims, and aggregate disposition."""
        _require_text(self.region_id, "region transfer assessment ID")
        mass = _positive_unit_interval(
            self.target_population_mass,
            "region transfer target mass",
        )
        simulation_count = _nonnegative_integer(
            self.simulation_sample_count,
            "region simulation sample count",
        )
        real_count = _nonnegative_integer(
            self.real_world_sample_count,
            "region real-world sample count",
        )
        disposition = TransferDisposition(self.disposition)
        claims = tuple(sorted(self.claims, key=lambda item: item.claim_id))
        _require_unique(
            (item.claim_id for item in claims),
            "regional validity claim IDs",
        )
        if disposition is TransferDisposition.INSUFFICIENT_EVIDENCE:
            if claims:
                raise ValueError("insufficient region cannot contain gap claims")
        else:
            expected = _aggregate_claim_disposition(claims)
            if disposition is not expected:
                raise ValueError(
                    "regional transfer disposition is inconsistent with claims"
                )
        limitations = _unique_text(
            self.limitations,
            "region transfer assessment limitations",
        )
        object.__setattr__(self, "target_population_mass", mass)
        object.__setattr__(self, "simulation_sample_count", simulation_count)
        object.__setattr__(self, "real_world_sample_count", real_count)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class SimToRealValidityReport:
    """Content-addressed simultaneous sim-to-real validity report."""

    report_schema_version: str
    plan_content_digest: str
    observation_data_digest: str
    familywise_error_rate: float
    allocated_error_rate_per_claim: float
    planned_claim_count: int
    disposition: TransferDisposition
    supported_target_population_mass: float
    region_assessments: tuple[RegionTransferAssessment, ...]
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate report identity, multiplicity control, and conclusion."""
        _require_text(self.report_schema_version, "transfer report schema version")
        NamedDigest("sim-to-real validity plan", self.plan_content_digest)
        NamedDigest("sim-to-real observation data", self.observation_data_digest)
        familywise_error = _open_unit_interval(
            self.familywise_error_rate,
            "transfer report familywise error rate",
        )
        allocated_error = _open_unit_interval(
            self.allocated_error_rate_per_claim,
            "transfer report allocated claim error rate",
        )
        claim_count = _positive_integer(
            self.planned_claim_count,
            "transfer report planned claim count",
        )
        if not math.isclose(
            allocated_error * claim_count,
            familywise_error,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("allocated claim errors must equal familywise error")
        regions = tuple(
            sorted(self.region_assessments, key=lambda item: item.region_id)
        )
        if not regions:
            raise ValueError("transfer report requires regional assessments")
        _require_unique(
            (item.region_id for item in regions),
            "transfer report region IDs",
        )
        disposition = TransferDisposition(self.disposition)
        expected_disposition = _aggregate_region_disposition(regions)
        if disposition is not expected_disposition:
            raise ValueError(
                "overall transfer disposition is inconsistent with regions"
            )
        supported_mass = _closed_unit_interval(
            self.supported_target_population_mass,
            "supported target population mass",
        )
        expected_mass = math.fsum(
            item.target_population_mass
            for item in regions
            if item.disposition is TransferDisposition.SUPPORTED_WITHIN_SCOPE
        )
        if not math.isclose(
            supported_mass,
            expected_mass,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("supported population mass is inconsistent with regions")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.CONFIRMATORY:
            raise ValueError("sim-to-real validity report must be confirmatory")
        limitations = _unique_text(
            self.limitations,
            "sim-to-real validity report limitations",
        )
        object.__setattr__(self, "familywise_error_rate", familywise_error)
        object.__setattr__(
            self,
            "allocated_error_rate_per_claim",
            allocated_error,
        )
        object.__setattr__(self, "planned_claim_count", claim_count)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "supported_target_population_mass",
            supported_mass,
        )
        object.__setattr__(self, "region_assessments", regions)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the validity report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact validity-report content digest."""
        return sha256_digest(self.to_json())


def assess_sim_to_real_validity(
    plan: SimToRealValidityPlan,
    observations: Sequence[TransferObservation],
) -> SimToRealValidityReport:
    """Assess simultaneous regional equivalence and shift claims.

    Args:
        plan: Frozen feature bounds, tolerances, regions, and error budget.
        observations: Independent simulation and real-world anchor samples.

    Returns:
        A non-compensatory, content-addressed validity report.

    Raises:
        ValueError: If observation identity, labels, feature bounds,
            dimensions, independence units, or evidence provenance are invalid.
    """
    plan_digest = plan.content_digest()
    dimension = len(plan.features)
    region_ids = {item.region_id for item in plan.regions}
    seen_domain_units: set[tuple[TransferDomain, str]] = set()
    unit_regions: dict[str, str] = {}
    evidence_digests: set[str] = set()
    grouped: dict[
        tuple[str, TransferDomain],
        list[TransferObservation],
    ] = {
        (region.region_id, domain): []
        for region in plan.regions
        for domain in TransferDomain
    }
    for observation in observations:
        if observation.plan_content_digest != plan_digest:
            raise ValueError(
                f"transfer observation '{observation.independence_unit_id}' "
                "plan mismatch"
            )
        if observation.region_id not in region_ids:
            raise ValueError(f"unknown transfer region: {observation.region_id}")
        if len(observation.feature_values) != dimension:
            raise ValueError(
                f"transfer observation '{observation.independence_unit_id}' "
                "feature dimension mismatch"
            )
        for feature, value in zip(
            plan.features,
            observation.feature_values,
            strict=True,
        ):
            if value < feature.lower_bound or value > feature.upper_bound:
                raise ValueError(
                    f"feature '{feature.feature_id}' is outside declared bounds"
                )
        domain_unit = (observation.domain, observation.independence_unit_id)
        if domain_unit in seen_domain_units:
            raise ValueError("independence unit IDs must be unique within each domain")
        prior_region = unit_regions.get(observation.independence_unit_id)
        if prior_region is not None and prior_region != observation.region_id:
            raise ValueError(
                "matched cross-domain independence units must share a region"
            )
        if observation.evidence_digest in evidence_digests:
            raise ValueError("transfer observation evidence digests must be unique")
        seen_domain_units.add(domain_unit)
        unit_regions[observation.independence_unit_id] = observation.region_id
        evidence_digests.add(observation.evidence_digest)
        grouped[(observation.region_id, observation.domain)].append(observation)

    planned_claim_count = len(plan.regions) * (len(plan.features) + 1)
    allocated_error_rate = plan.familywise_error_rate / planned_claim_count
    region_assessments: list[RegionTransferAssessment] = []
    for region in plan.regions:
        simulation = grouped[(region.region_id, TransferDomain.SIMULATION)]
        real_world = grouped[(region.region_id, TransferDomain.REAL_WORLD)]
        if (
            len(simulation) < region.minimum_simulation_samples
            or len(real_world) < region.minimum_real_world_samples
        ):
            region_assessments.append(
                RegionTransferAssessment(
                    region_id=region.region_id,
                    target_population_mass=region.target_population_mass,
                    simulation_sample_count=len(simulation),
                    real_world_sample_count=len(real_world),
                    disposition=TransferDisposition.INSUFFICIENT_EVIDENCE,
                    claims=(),
                    limitations=(
                        "Predeclared minimum sample requirements were not met.",
                    ),
                )
            )
            continue

        claims = [
            _feature_gap_assessment(
                feature,
                feature_index,
                simulation,
                real_world,
                allocated_error_rate,
            )
            for feature_index, feature in enumerate(plan.features)
        ]
        claims.append(
            _failure_rate_gap_assessment(
                plan,
                simulation,
                real_world,
                allocated_error_rate,
            )
        )
        region_assessments.append(
            RegionTransferAssessment(
                region_id=region.region_id,
                target_population_mass=region.target_population_mass,
                simulation_sample_count=len(simulation),
                real_world_sample_count=len(real_world),
                disposition=_aggregate_claim_disposition(claims),
                claims=tuple(claims),
                limitations=(
                    "Bounds assume independent sampling units within each domain.",
                    "Regional support does not generalize outside declared bounds.",
                ),
            )
        )

    canonical_observations = tuple(
        sorted(
            observations,
            key=lambda item: (
                item.region_id,
                item.domain.value,
                item.independence_unit_id,
            ),
        )
    )
    disposition = _aggregate_region_disposition(region_assessments)
    supported_mass = math.fsum(
        item.target_population_mass
        for item in region_assessments
        if item.disposition is TransferDisposition.SUPPORTED_WITHIN_SCOPE
    )
    return SimToRealValidityReport(
        report_schema_version="iso-obs.sim-to-real-validity-report.v1",
        plan_content_digest=plan_digest,
        observation_data_digest=sha256_digest(_canonical_json(canonical_observations)),
        familywise_error_rate=plan.familywise_error_rate,
        allocated_error_rate_per_claim=allocated_error_rate,
        planned_claim_count=planned_claim_count,
        disposition=disposition,
        supported_target_population_mass=supported_mass,
        region_assessments=tuple(region_assessments),
        evidence_use=EvidenceUse.CONFIRMATORY,
        limitations=(
            "Familywise coverage uses conservative Bonferroni-allocated "
            "distribution-free Hoeffding bounds.",
            "Supported validity is equivalence within declared mean and "
            "failure-rate tolerances, not proof that simulation is reality.",
            "Content addressing does not prove that the plan was fixed before "
            "outcomes were observed.",
            "The analysis does not establish the cause of a detected shift.",
            "Unmeasured variables and sampling bias remain possible validity threats.",
        ),
    )


def _feature_gap_assessment(
    feature: TransferFeature,
    feature_index: int,
    simulation: Sequence[TransferObservation],
    real_world: Sequence[TransferObservation],
    allocated_error_rate: float,
) -> ValidityGapAssessment:
    """Assess one bounded feature-mean equivalence claim."""
    simulation_mean = math.fsum(
        item.feature_values[feature_index] for item in simulation
    ) / len(simulation)
    real_mean = math.fsum(
        item.feature_values[feature_index] for item in real_world
    ) / len(real_world)
    return _bounded_gap_assessment(
        claim_id=f"feature:{feature.feature_id}",
        kind=ValidityClaimKind.FEATURE_MEAN_GAP,
        simulation_estimate=simulation_mean,
        real_world_estimate=real_mean,
        value_range=feature.upper_bound - feature.lower_bound,
        maximum_acceptable_gap=feature.maximum_acceptable_mean_gap,
        simulation_count=len(simulation),
        real_world_count=len(real_world),
        allocated_error_rate=allocated_error_rate,
    )


def _failure_rate_gap_assessment(
    plan: SimToRealValidityPlan,
    simulation: Sequence[TransferObservation],
    real_world: Sequence[TransferObservation],
    allocated_error_rate: float,
) -> ValidityGapAssessment:
    """Assess the bounded failure-rate equivalence claim."""
    simulation_rate = sum(item.failed for item in simulation) / len(simulation)
    real_rate = sum(item.failed for item in real_world) / len(real_world)
    return _bounded_gap_assessment(
        claim_id="failure-rate",
        kind=ValidityClaimKind.FAILURE_RATE_GAP,
        simulation_estimate=simulation_rate,
        real_world_estimate=real_rate,
        value_range=1.0,
        maximum_acceptable_gap=plan.maximum_acceptable_failure_rate_gap,
        simulation_count=len(simulation),
        real_world_count=len(real_world),
        allocated_error_rate=allocated_error_rate,
    )


def _bounded_gap_assessment(
    *,
    claim_id: str,
    kind: ValidityClaimKind,
    simulation_estimate: float,
    real_world_estimate: float,
    value_range: float,
    maximum_acceptable_gap: float,
    simulation_count: int,
    real_world_count: int,
    allocated_error_rate: float,
) -> ValidityGapAssessment:
    """Construct a simultaneous two-sample Hoeffding gap assessment."""
    simulation_radius = value_range * math.sqrt(
        math.log(4.0 / allocated_error_rate) / (2.0 * simulation_count)
    )
    real_world_radius = value_range * math.sqrt(
        math.log(4.0 / allocated_error_rate) / (2.0 * real_world_count)
    )
    confidence_radius = simulation_radius + real_world_radius
    signed_gap = real_world_estimate - simulation_estimate
    absolute_gap = abs(signed_gap)
    lower = max(0.0, absolute_gap - confidence_radius)
    upper = min(value_range, absolute_gap + confidence_radius)
    return ValidityGapAssessment(
        claim_id=claim_id,
        kind=kind,
        simulation_estimate=simulation_estimate,
        real_world_estimate=real_world_estimate,
        signed_gap_estimate=signed_gap,
        absolute_gap_estimate=absolute_gap,
        confidence_radius=confidence_radius,
        absolute_gap_lower_bound=lower,
        absolute_gap_upper_bound=upper,
        maximum_acceptable_gap=maximum_acceptable_gap,
        allocated_error_rate=allocated_error_rate,
        disposition=_gap_disposition(lower, upper, maximum_acceptable_gap),
    )


def _gap_disposition(
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
) -> GapDisposition:
    """Map an absolute-gap confidence interval to an equivalence conclusion."""
    if upper_bound <= tolerance:
        return GapDisposition.SUPPORTED_WITHIN_TOLERANCE
    if lower_bound > tolerance:
        return GapDisposition.SHIFT_EXCEEDS_TOLERANCE
    return GapDisposition.INCONCLUSIVE


def _aggregate_claim_disposition(
    claims: Sequence[ValidityGapAssessment],
) -> TransferDisposition:
    """Aggregate claims without allowing good metrics to offset a bad one."""
    if not claims:
        raise ValueError("regional transfer assessment requires claims")
    dispositions = {item.disposition for item in claims}
    if GapDisposition.SHIFT_EXCEEDS_TOLERANCE in dispositions:
        return TransferDisposition.SHIFT_DETECTED
    if GapDisposition.INCONCLUSIVE in dispositions:
        return TransferDisposition.INCONCLUSIVE
    return TransferDisposition.SUPPORTED_WITHIN_SCOPE


def _aggregate_region_disposition(
    regions: Sequence[RegionTransferAssessment],
) -> TransferDisposition:
    """Aggregate regions with shift-first, non-compensatory precedence."""
    if not regions:
        raise ValueError("sim-to-real validity assessment requires regions")
    dispositions = {item.disposition for item in regions}
    if TransferDisposition.SHIFT_DETECTED in dispositions:
        return TransferDisposition.SHIFT_DETECTED
    if TransferDisposition.INSUFFICIENT_EVIDENCE in dispositions:
        return TransferDisposition.INSUFFICIENT_EVIDENCE
    if TransferDisposition.INCONCLUSIVE in dispositions:
        return TransferDisposition.INCONCLUSIVE
    return TransferDisposition.SUPPORTED_WITHIN_SCOPE


def _finite_vector(values: Iterable[float], label: str) -> tuple[float, ...]:
    """Require a non-empty vector containing finite numbers."""
    resolved = tuple(_finite(value, label) for value in values)
    if not resolved:
        raise ValueError(f"{label} must not be empty")
    return resolved


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


def _positive_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``(0, 1]``."""
    resolved = _finite(value, label)
    if resolved <= 0.0 or resolved > 1.0:
        raise ValueError(f"{label} must be in (0, 1]")
    return resolved


def _open_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``(0, 1)``."""
    resolved = _finite(value, label)
    if resolved <= 0.0 or resolved >= 1.0:
        raise ValueError(f"{label} must be in (0, 1)")
    return resolved


def _closed_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``[0, 1]``."""
    resolved = _finite(value, label)
    if resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return resolved


def _closed_open_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``[0, 1)``."""
    resolved = _finite(value, label)
    if resolved < 0.0 or resolved >= 1.0:
        raise ValueError(f"{label} must be in [0, 1)")
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
