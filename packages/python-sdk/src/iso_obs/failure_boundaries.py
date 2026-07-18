"""Lipschitz-certified continuous failure-boundary mapping.

The model combines simultaneous failure-probability bounds at predeclared
sampling anchors with a declared Lipschitz constant. It propagates uncertainty
over continuous grid cells and certifies cells as reliable or unreliable only
when a uniform bound clears the failure threshold. Remaining cells form the
unresolved boundary.

The validity of propagated bounds depends on the Lipschitz assumption,
independent Bernoulli trials at anchors, fixed sampling, and correct operating
coordinates. Anchor diagnostics can detect contradictions but cannot prove
global smoothness between anchors.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse


class BoundaryModelDisposition(StrEnum):
    """Readiness of a fitted continuous boundary model."""

    READY = "ready"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DESIGN_VIOLATED = "design_violated"
    ASSUMPTIONS_VIOLATED = "assumptions_violated"


class BoundaryCellDisposition(StrEnum):
    """Uniform failure-probability conclusion over one continuous cell."""

    RELIABLE_CERTIFIED = "reliable_certified"
    UNRELIABLE_CERTIFIED = "unreliable_certified"
    UNRESOLVED_BOUNDARY = "unresolved_boundary"


class BoundaryMapDisposition(StrEnum):
    """Completeness or terminal state of a failure-boundary map."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DESIGN_VIOLATED = "design_violated"
    ASSUMPTIONS_VIOLATED = "assumptions_violated"


@dataclass(frozen=True, slots=True)
class BoundaryDimension:
    """One bounded continuous operating dimension and mapping grid."""

    dimension_id: str
    definition_digest: str
    lower_bound: float
    upper_bound: float
    distance_weight: float
    grid_points: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate coordinate identity, range, metric weight, and grid."""
        _require_text(self.dimension_id, "boundary dimension ID")
        NamedDigest("boundary dimension definition", self.definition_digest)
        lower = _finite(self.lower_bound, "boundary dimension lower bound")
        upper = _finite(self.upper_bound, "boundary dimension upper bound")
        if lower >= upper:
            raise ValueError("boundary dimension lower bound must be below upper")
        weight = _positive_finite(
            self.distance_weight,
            "boundary dimension distance weight",
        )
        points = tuple(
            _finite(item, "boundary dimension grid point") for item in self.grid_points
        )
        if len(points) < 2:
            raise ValueError("boundary dimension requires at least two grid points")
        if points[0] != lower or points[-1] != upper:
            raise ValueError("boundary grid must start and end at dimension bounds")
        if any(left >= right for left, right in itertools.pairwise(points)):
            raise ValueError("boundary grid points must be strictly increasing")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "distance_weight", weight)
        object.__setattr__(self, "grid_points", points)


@dataclass(frozen=True, slots=True)
class BoundaryAnchor:
    """One predeclared operating point with a fixed trial count."""

    anchor_id: str
    coordinates: tuple[float, ...]
    planned_sample_count: int

    def __post_init__(self) -> None:
        """Validate anchor identity, coordinates, and fixed sample count."""
        _require_text(self.anchor_id, "boundary anchor ID")
        coordinates = _finite_vector(
            self.coordinates,
            "boundary anchor coordinates",
        )
        planned_count = _positive_integer(
            self.planned_sample_count,
            "boundary anchor planned sample count",
        )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "planned_sample_count", planned_count)


@dataclass(frozen=True, slots=True)
class FailureBoundaryPlan:
    """Content-addressed confirmatory continuous boundary design."""

    plan_id: str
    plan_version: str
    target_population_digest: str
    simulation_evidence_digest: str
    system_artifact_digest: str
    failure_outcome_definition_digest: str
    coordinate_extraction_digest: str
    dimensions: tuple[BoundaryDimension, ...]
    anchors: tuple[BoundaryAnchor, ...]
    maximum_acceptable_failure_probability: float
    failure_probability_lipschitz_constant: float
    familywise_error_rate: float
    maximum_grid_cell_count: int
    limitations: tuple[str, ...]
    evidence_use: EvidenceUse = EvidenceUse.CONFIRMATORY

    def __post_init__(self) -> None:
        """Validate identity, coordinate space, anchors, grid, and assumptions."""
        _require_text(self.plan_id, "failure boundary plan ID")
        _require_text(self.plan_version, "failure boundary plan version")
        for value, label in (
            (self.target_population_digest, "target population"),
            (self.simulation_evidence_digest, "simulation evidence"),
            (self.system_artifact_digest, "system artifact"),
            (self.failure_outcome_definition_digest, "failure outcome definition"),
            (self.coordinate_extraction_digest, "coordinate extraction"),
        ):
            NamedDigest(label, value)
        if not self.dimensions:
            raise ValueError("failure boundary plan requires dimensions")
        dimensions = tuple(self.dimensions)
        _require_unique(
            (item.dimension_id for item in dimensions),
            "boundary dimension IDs",
        )
        _require_unique(
            (item.definition_digest for item in dimensions),
            "boundary dimension definitions",
        )
        if not self.anchors:
            raise ValueError("failure boundary plan requires anchors")
        anchors = tuple(sorted(self.anchors, key=lambda item: item.anchor_id))
        _require_unique(
            (item.anchor_id for item in anchors),
            "boundary anchor IDs",
        )
        _require_unique(
            (item.coordinates for item in anchors),
            "boundary anchor coordinates",
        )
        dimension_count = len(dimensions)
        for anchor in anchors:
            if len(anchor.coordinates) != dimension_count:
                raise ValueError("boundary anchor coordinate dimension mismatch")
            for coordinate, dimension in zip(
                anchor.coordinates,
                dimensions,
                strict=True,
            ):
                if (
                    coordinate < dimension.lower_bound
                    or coordinate > dimension.upper_bound
                ):
                    raise ValueError(
                        f"anchor '{anchor.anchor_id}' is outside dimension bounds"
                    )
        threshold = _open_unit_interval(
            self.maximum_acceptable_failure_probability,
            "maximum acceptable failure probability",
        )
        lipschitz_constant = _nonnegative_finite(
            self.failure_probability_lipschitz_constant,
            "failure-probability Lipschitz constant",
        )
        familywise_error = _open_unit_interval(
            self.familywise_error_rate,
            "failure boundary familywise error rate",
        )
        maximum_cells = _positive_integer(
            self.maximum_grid_cell_count,
            "maximum boundary grid cell count",
        )
        grid_cell_count = math.prod(len(item.grid_points) - 1 for item in dimensions)
        if grid_cell_count > maximum_cells:
            raise ValueError("boundary grid exceeds maximum cell count")
        limitations = _unique_text(
            self.limitations,
            "failure boundary plan limitations",
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.CONFIRMATORY:
            raise ValueError("failure boundary plan must be confirmatory")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(
            self,
            "maximum_acceptable_failure_probability",
            threshold,
        )
        object.__setattr__(
            self,
            "failure_probability_lipschitz_constant",
            lipschitz_constant,
        )
        object.__setattr__(self, "familywise_error_rate", familywise_error)
        object.__setattr__(self, "maximum_grid_cell_count", maximum_cells)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_use", evidence_use)

    def to_json(self) -> str:
        """Serialize the boundary plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact boundary-plan content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class BoundaryObservation:
    """One fixed-design binary outcome sampled at a declared anchor."""

    plan_content_digest: str
    anchor_id: str
    sample_index: int
    trial_id: str
    evidence_digest: str
    failed: bool

    def __post_init__(self) -> None:
        """Validate observation identity, ordering, provenance, and outcome."""
        NamedDigest("failure boundary plan", self.plan_content_digest)
        _require_text(self.anchor_id, "boundary observation anchor ID")
        _nonnegative_integer(
            self.sample_index,
            "boundary observation sample index",
        )
        _require_text(self.trial_id, "boundary observation trial ID")
        NamedDigest("boundary observation evidence", self.evidence_digest)
        if not isinstance(self.failed, bool):
            raise ValueError("boundary observation failed must be a boolean")


@dataclass(frozen=True, slots=True)
class AnchorSampleStatus:
    """Observed and planned sample counts for one anchor."""

    anchor_id: str
    observed_sample_count: int
    planned_sample_count: int

    def __post_init__(self) -> None:
        """Validate anchor identity and sample counts."""
        _require_text(self.anchor_id, "anchor sample status ID")
        object.__setattr__(
            self,
            "observed_sample_count",
            _nonnegative_integer(
                self.observed_sample_count,
                "observed anchor sample count",
            ),
        )
        object.__setattr__(
            self,
            "planned_sample_count",
            _positive_integer(
                self.planned_sample_count,
                "planned anchor sample count",
            ),
        )


@dataclass(frozen=True, slots=True)
class AnchorFailureEstimate:
    """Simultaneous failure-probability interval at one anchor."""

    anchor_id: str
    coordinates: tuple[float, ...]
    sample_count: int
    failure_count: int
    failure_probability_estimate: float
    confidence_radius: float
    lower_bound: float
    upper_bound: float
    allocated_error_rate: float

    def __post_init__(self) -> None:
        """Validate counts, estimate, Hoeffding radius, and clipped bounds."""
        _require_text(self.anchor_id, "anchor failure estimate ID")
        coordinates = _finite_vector(
            self.coordinates,
            "anchor estimate coordinates",
        )
        sample_count = _positive_integer(
            self.sample_count,
            "anchor estimate sample count",
        )
        failure_count = _nonnegative_integer(
            self.failure_count,
            "anchor estimate failure count",
        )
        if failure_count > sample_count:
            raise ValueError("anchor failures cannot exceed samples")
        estimate = _closed_unit_interval(
            self.failure_probability_estimate,
            "anchor failure-probability estimate",
        )
        if not math.isclose(
            estimate,
            failure_count / sample_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("anchor estimate is inconsistent with counts")
        allocated_error = _open_unit_interval(
            self.allocated_error_rate,
            "anchor allocated error rate",
        )
        radius = _nonnegative_finite(
            self.confidence_radius,
            "anchor confidence radius",
        )
        expected_radius = math.sqrt(
            math.log(2.0 / allocated_error) / (2.0 * sample_count)
        )
        if not math.isclose(
            radius,
            expected_radius,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("anchor confidence radius is inconsistent")
        lower = _closed_unit_interval(self.lower_bound, "anchor lower bound")
        upper = _closed_unit_interval(self.upper_bound, "anchor upper bound")
        if not (
            math.isclose(
                lower,
                max(0.0, estimate - radius),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                upper,
                min(1.0, estimate + radius),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("anchor confidence bounds are inconsistent")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "failure_count", failure_count)
        object.__setattr__(self, "failure_probability_estimate", estimate)
        object.__setattr__(self, "confidence_radius", radius)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "allocated_error_rate", allocated_error)


@dataclass(frozen=True, slots=True)
class FailureBoundaryModel:
    """Fitted anchor intervals and smoothness-consistency diagnostics."""

    model_schema_version: str
    plan_content_digest: str
    training_data_digest: str
    disposition: BoundaryModelDisposition
    allocated_error_rate_per_anchor: float
    sample_statuses: tuple[AnchorSampleStatus, ...]
    anchor_estimates: tuple[AnchorFailureEstimate, ...]
    maximum_lipschitz_violation: float
    violating_anchor_pair: tuple[str, str] | None
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate fitted identity, readiness, diagnostics, and evidence use."""
        _require_text(self.model_schema_version, "boundary model schema version")
        NamedDigest("failure boundary plan", self.plan_content_digest)
        NamedDigest("failure boundary training data", self.training_data_digest)
        disposition = BoundaryModelDisposition(self.disposition)
        allocated_error = _open_unit_interval(
            self.allocated_error_rate_per_anchor,
            "boundary model allocated anchor error rate",
        )
        statuses = tuple(sorted(self.sample_statuses, key=lambda item: item.anchor_id))
        if not statuses:
            raise ValueError("boundary model requires anchor sample statuses")
        _require_unique(
            (item.anchor_id for item in statuses),
            "boundary model sample status IDs",
        )
        estimates = tuple(
            sorted(self.anchor_estimates, key=lambda item: item.anchor_id)
        )
        _require_unique(
            (item.anchor_id for item in estimates),
            "boundary model estimate IDs",
        )
        violation = _nonnegative_finite(
            self.maximum_lipschitz_violation,
            "maximum Lipschitz violation",
        )
        if disposition is BoundaryModelDisposition.READY:
            if len(estimates) != len(statuses) or violation != 0.0:
                raise ValueError("ready boundary model requires complete valid anchors")
            if self.violating_anchor_pair is not None:
                raise ValueError("ready boundary model cannot have violating anchors")
        elif disposition is BoundaryModelDisposition.ASSUMPTIONS_VIOLATED:
            if len(estimates) != len(statuses) or violation <= 0.0:
                raise ValueError(
                    "assumption violation requires complete contradictory anchors"
                )
            if self.violating_anchor_pair is None:
                raise ValueError("assumption violation requires an anchor pair")
        elif estimates or violation != 0.0 or self.violating_anchor_pair is not None:
            raise ValueError("terminal sampling state cannot contain fitted estimates")
        if self.violating_anchor_pair is not None:
            pair = self.violating_anchor_pair
            if len(pair) != 2 or pair[0] >= pair[1]:
                raise ValueError("violating anchor pair must be ordered and distinct")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.CONFIRMATORY:
            raise ValueError("failure boundary model must be confirmatory")
        limitations = _unique_text(
            self.limitations,
            "failure boundary model limitations",
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "allocated_error_rate_per_anchor",
            allocated_error,
        )
        object.__setattr__(self, "sample_statuses", statuses)
        object.__setattr__(self, "anchor_estimates", estimates)
        object.__setattr__(self, "maximum_lipschitz_violation", violation)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the fitted boundary model to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact boundary-model content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class CellDimensionRange:
    """One dimension's closed coordinate range within a grid cell."""

    dimension_id: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        """Validate cell dimension identity and ordered finite bounds."""
        _require_text(self.dimension_id, "cell dimension ID")
        lower = _finite(self.lower_bound, "cell dimension lower bound")
        upper = _finite(self.upper_bound, "cell dimension upper bound")
        if lower >= upper:
            raise ValueError("cell dimension lower bound must be below upper")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)


@dataclass(frozen=True, slots=True)
class BoundaryCellAssessment:
    """Uniform propagated failure bounds over one continuous cell."""

    cell_id: str
    ranges: tuple[CellDimensionRange, ...]
    normalized_geometric_volume: float
    failure_probability_lower_bound: float
    failure_probability_upper_bound: float
    disposition: BoundaryCellDisposition

    def __post_init__(self) -> None:
        """Validate cell geometry, bounds, and disposition."""
        _require_text(self.cell_id, "boundary cell ID")
        if not self.ranges:
            raise ValueError("boundary cell requires dimension ranges")
        _require_unique(
            (item.dimension_id for item in self.ranges),
            "boundary cell dimension IDs",
        )
        volume = _positive_unit_interval(
            self.normalized_geometric_volume,
            "normalized boundary cell volume",
        )
        lower = _closed_unit_interval(
            self.failure_probability_lower_bound,
            "boundary cell failure lower bound",
        )
        upper = _closed_unit_interval(
            self.failure_probability_upper_bound,
            "boundary cell failure upper bound",
        )
        if lower > upper:
            raise ValueError("boundary cell lower bound must not exceed upper")
        disposition = BoundaryCellDisposition(self.disposition)
        object.__setattr__(self, "normalized_geometric_volume", volume)
        object.__setattr__(self, "failure_probability_lower_bound", lower)
        object.__setattr__(self, "failure_probability_upper_bound", upper)
        object.__setattr__(self, "disposition", disposition)


@dataclass(frozen=True, slots=True)
class FailureBoundaryMap:
    """Content-addressed continuous grid-cell boundary map."""

    report_schema_version: str
    model_content_digest: str
    disposition: BoundaryMapDisposition
    cells: tuple[BoundaryCellAssessment, ...]
    reliable_geometric_volume: float
    unreliable_geometric_volume: float
    unresolved_geometric_volume: float
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate map identity, cells, geometric volumes, and evidence use."""
        _require_text(self.report_schema_version, "boundary map schema version")
        NamedDigest("failure boundary model", self.model_content_digest)
        disposition = BoundaryMapDisposition(self.disposition)
        cells = tuple(sorted(self.cells, key=lambda item: item.cell_id))
        _require_unique(
            (item.cell_id for item in cells),
            "boundary map cell IDs",
        )
        volumes = tuple(
            _closed_unit_interval(value, label)
            for value, label in (
                (self.reliable_geometric_volume, "reliable geometric volume"),
                (self.unreliable_geometric_volume, "unreliable geometric volume"),
                (self.unresolved_geometric_volume, "unresolved geometric volume"),
            )
        )
        if cells:
            if not math.isclose(
                math.fsum(item.normalized_geometric_volume for item in cells),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("boundary cell geometric volumes must sum to one")
            expected_volumes = tuple(
                math.fsum(
                    item.normalized_geometric_volume
                    for item in cells
                    if item.disposition is cell_disposition
                )
                for cell_disposition in BoundaryCellDisposition
            )
            if any(
                not math.isclose(
                    observed,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for observed, expected in zip(volumes, expected_volumes, strict=True)
            ):
                raise ValueError("boundary map geometric volumes are inconsistent")
            expected_disposition = (
                BoundaryMapDisposition.PARTIAL
                if volumes[2] > 0.0
                else BoundaryMapDisposition.COMPLETE
            )
            if disposition is not expected_disposition:
                raise ValueError("boundary map disposition is inconsistent with cells")
        elif any(value != 0.0 for value in volumes):
            raise ValueError("terminal boundary map cannot contain geometric volume")
        elif disposition in (
            BoundaryMapDisposition.COMPLETE,
            BoundaryMapDisposition.PARTIAL,
        ):
            raise ValueError("mapped boundary disposition requires cells")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.CONFIRMATORY:
            raise ValueError("failure boundary map must be confirmatory")
        limitations = _unique_text(
            self.limitations,
            "failure boundary map limitations",
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "reliable_geometric_volume", volumes[0])
        object.__setattr__(self, "unreliable_geometric_volume", volumes[1])
        object.__setattr__(self, "unresolved_geometric_volume", volumes[2])
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the boundary map to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact boundary-map content digest."""
        return sha256_digest(self.to_json())


def fit_failure_boundary_model(
    plan: FailureBoundaryPlan,
    observations: Sequence[BoundaryObservation],
) -> FailureBoundaryModel:
    """Fit simultaneous anchor intervals and test smoothness compatibility.

    Args:
        plan: Frozen dimensions, anchors, sampling, threshold, and smoothness.
        observations: Binary fixed-design outcomes at declared anchors.

    Returns:
        A ready model or an explicit terminal sampling/assumption state.

    Raises:
        ValueError: If identities, sample indices, trials, or evidence are
            duplicated or do not belong to the declared plan.
    """
    plan_digest = plan.content_digest()
    anchors = {item.anchor_id: item for item in plan.anchors}
    grouped: dict[str, list[BoundaryObservation]] = {
        anchor_id: [] for anchor_id in anchors
    }
    trial_ids: set[str] = set()
    evidence_digests: set[str] = set()
    for observation in observations:
        if observation.plan_content_digest != plan_digest:
            raise ValueError(f"boundary trial '{observation.trial_id}' plan mismatch")
        if observation.anchor_id not in anchors:
            raise ValueError(f"unknown boundary anchor: {observation.anchor_id}")
        if observation.trial_id in trial_ids:
            raise ValueError("boundary trial IDs must be unique")
        if observation.evidence_digest in evidence_digests:
            raise ValueError("boundary evidence digests must be unique")
        trial_ids.add(observation.trial_id)
        evidence_digests.add(observation.evidence_digest)
        grouped[observation.anchor_id].append(observation)

    canonical_observations: list[BoundaryObservation] = []
    statuses: list[AnchorSampleStatus] = []
    for anchor in plan.anchors:
        ordered = tuple(
            sorted(
                grouped[anchor.anchor_id],
                key=lambda item: item.sample_index,
            )
        )
        if tuple(item.sample_index for item in ordered) != tuple(range(len(ordered))):
            raise ValueError(
                f"anchor '{anchor.anchor_id}' sample indices must be contiguous "
                "from zero"
            )
        canonical_observations.extend(ordered)
        statuses.append(
            AnchorSampleStatus(
                anchor_id=anchor.anchor_id,
                observed_sample_count=len(ordered),
                planned_sample_count=anchor.planned_sample_count,
            )
        )

    training_digest = sha256_digest(_canonical_json(tuple(canonical_observations)))
    allocated_error = plan.familywise_error_rate / len(plan.anchors)
    if any(item.observed_sample_count > item.planned_sample_count for item in statuses):
        return _terminal_model(
            plan,
            training_digest,
            allocated_error,
            statuses,
            BoundaryModelDisposition.DESIGN_VIOLATED,
            "At least one anchor exceeded its fixed planned sample count.",
        )
    if any(item.observed_sample_count < item.planned_sample_count for item in statuses):
        return _terminal_model(
            plan,
            training_digest,
            allocated_error,
            statuses,
            BoundaryModelDisposition.INSUFFICIENT_EVIDENCE,
            "At least one anchor has not reached its fixed planned sample count.",
        )

    estimates = tuple(
        _anchor_estimate(
            anchor,
            grouped[anchor.anchor_id],
            allocated_error,
        )
        for anchor in plan.anchors
    )
    violation, violating_pair = _maximum_lipschitz_violation(
        plan,
        estimates,
    )
    disposition = (
        BoundaryModelDisposition.ASSUMPTIONS_VIOLATED
        if violation > 0.0
        else BoundaryModelDisposition.READY
    )
    limitations = [
        "Anchor intervals assume independent Bernoulli trials under a fixed design.",
        "Anchor consistency diagnostics cannot prove global Lipschitz smoothness.",
        "Content addressing does not prove that the plan preceded outcomes.",
    ]
    if violation > 0.0:
        limitations.append(
            "Observed anchor intervals contradict the declared Lipschitz constant."
        )
    return FailureBoundaryModel(
        model_schema_version="iso-obs.failure-boundary-model.v1",
        plan_content_digest=plan_digest,
        training_data_digest=training_digest,
        disposition=disposition,
        allocated_error_rate_per_anchor=allocated_error,
        sample_statuses=tuple(statuses),
        anchor_estimates=estimates,
        maximum_lipschitz_violation=violation,
        violating_anchor_pair=violating_pair,
        evidence_use=EvidenceUse.CONFIRMATORY,
        limitations=tuple(limitations),
    )


def map_failure_boundary(
    plan: FailureBoundaryPlan,
    model: FailureBoundaryModel,
) -> FailureBoundaryMap:
    """Propagate fitted anchor intervals over every continuous grid cell.

    Args:
        plan: Exact boundary plan used to fit ``model``.
        model: Fitted anchor model or terminal model state.

    Returns:
        A continuous cell map or matching terminal report.

    Raises:
        ValueError: If the fitted model does not belong to ``plan``.
    """
    if model.plan_content_digest != plan.content_digest():
        raise ValueError("failure boundary model does not belong to plan")
    if model.disposition is not BoundaryModelDisposition.READY:
        return _terminal_map(model)

    cells = tuple(_map_cells(plan, model.anchor_estimates))
    reliable_volume = _cell_volume(
        cells,
        BoundaryCellDisposition.RELIABLE_CERTIFIED,
    )
    unreliable_volume = _cell_volume(
        cells,
        BoundaryCellDisposition.UNRELIABLE_CERTIFIED,
    )
    unresolved_volume = _cell_volume(
        cells,
        BoundaryCellDisposition.UNRESOLVED_BOUNDARY,
    )
    return FailureBoundaryMap(
        report_schema_version="iso-obs.failure-boundary-map.v1",
        model_content_digest=model.content_digest(),
        disposition=(
            BoundaryMapDisposition.PARTIAL
            if unresolved_volume > 0.0
            else BoundaryMapDisposition.COMPLETE
        ),
        cells=cells,
        reliable_geometric_volume=reliable_volume,
        unreliable_geometric_volume=unreliable_volume,
        unresolved_geometric_volume=unresolved_volume,
        evidence_use=EvidenceUse.CONFIRMATORY,
        limitations=(
            "Cell certification is uniform only under the declared Lipschitz bound.",
            "Geometric volume is not target-population probability mass.",
            "Unresolved cells require additional anchors, samples, or stronger "
            "validated structure.",
            "The map does not establish failure mechanism or authorize release.",
        ),
    )


def _terminal_model(
    plan: FailureBoundaryPlan,
    training_digest: str,
    allocated_error: float,
    statuses: Sequence[AnchorSampleStatus],
    disposition: BoundaryModelDisposition,
    limitation: str,
) -> FailureBoundaryModel:
    """Build a terminal model without inferential anchor estimates."""
    return FailureBoundaryModel(
        model_schema_version="iso-obs.failure-boundary-model.v1",
        plan_content_digest=plan.content_digest(),
        training_data_digest=training_digest,
        disposition=disposition,
        allocated_error_rate_per_anchor=allocated_error,
        sample_statuses=tuple(statuses),
        anchor_estimates=(),
        maximum_lipschitz_violation=0.0,
        violating_anchor_pair=None,
        evidence_use=EvidenceUse.CONFIRMATORY,
        limitations=(limitation,),
    )


def _terminal_map(model: FailureBoundaryModel) -> FailureBoundaryMap:
    """Map a terminal fitted-model disposition to an empty report."""
    disposition = {
        BoundaryModelDisposition.INSUFFICIENT_EVIDENCE: (
            BoundaryMapDisposition.INSUFFICIENT_EVIDENCE
        ),
        BoundaryModelDisposition.DESIGN_VIOLATED: (
            BoundaryMapDisposition.DESIGN_VIOLATED
        ),
        BoundaryModelDisposition.ASSUMPTIONS_VIOLATED: (
            BoundaryMapDisposition.ASSUMPTIONS_VIOLATED
        ),
    }[model.disposition]
    return FailureBoundaryMap(
        report_schema_version="iso-obs.failure-boundary-map.v1",
        model_content_digest=model.content_digest(),
        disposition=disposition,
        cells=(),
        reliable_geometric_volume=0.0,
        unreliable_geometric_volume=0.0,
        unresolved_geometric_volume=0.0,
        evidence_use=EvidenceUse.CONFIRMATORY,
        limitations=("No continuous boundary map is available from this model state.",),
    )


def _anchor_estimate(
    anchor: BoundaryAnchor,
    observations: Sequence[BoundaryObservation],
    allocated_error: float,
) -> AnchorFailureEstimate:
    """Calculate one fixed-sample simultaneous Hoeffding interval."""
    sample_count = len(observations)
    failure_count = sum(item.failed for item in observations)
    estimate = failure_count / sample_count
    radius = math.sqrt(math.log(2.0 / allocated_error) / (2.0 * sample_count))
    return AnchorFailureEstimate(
        anchor_id=anchor.anchor_id,
        coordinates=anchor.coordinates,
        sample_count=sample_count,
        failure_count=failure_count,
        failure_probability_estimate=estimate,
        confidence_radius=radius,
        lower_bound=max(0.0, estimate - radius),
        upper_bound=min(1.0, estimate + radius),
        allocated_error_rate=allocated_error,
    )


def _maximum_lipschitz_violation(
    plan: FailureBoundaryPlan,
    estimates: Sequence[AnchorFailureEstimate],
) -> tuple[float, tuple[str, str] | None]:
    """Find the largest anchor-interval contradiction to smoothness."""
    maximum_violation = 0.0
    violating_pair: tuple[str, str] | None = None
    for left, right in itertools.combinations(estimates, 2):
        interval_separation = max(
            0.0,
            left.lower_bound - right.upper_bound,
            right.lower_bound - left.upper_bound,
        )
        allowed_change = (
            plan.failure_probability_lipschitz_constant
            * _normalized_distance(plan.dimensions, left.coordinates, right.coordinates)
        )
        violation = max(0.0, interval_separation - allowed_change)
        first_anchor_id, second_anchor_id = sorted((left.anchor_id, right.anchor_id))
        pair = (first_anchor_id, second_anchor_id)
        if violation > maximum_violation or (
            violation == maximum_violation
            and violation > 0.0
            and violating_pair is not None
            and pair < violating_pair
        ):
            maximum_violation = violation
            violating_pair = pair
    return maximum_violation, violating_pair


def _map_cells(
    plan: FailureBoundaryPlan,
    estimates: Sequence[AnchorFailureEstimate],
) -> Iterable[BoundaryCellAssessment]:
    """Yield every grid cell with uniform Lipschitz-propagated bounds."""
    index_ranges = tuple(
        range(len(dimension.grid_points) - 1) for dimension in plan.dimensions
    )
    for indices in itertools.product(*index_ranges):
        ranges = tuple(
            CellDimensionRange(
                dimension_id=dimension.dimension_id,
                lower_bound=dimension.grid_points[index],
                upper_bound=dimension.grid_points[index + 1],
            )
            for dimension, index in zip(plan.dimensions, indices, strict=True)
        )
        volume = math.prod(
            (item.upper_bound - item.lower_bound)
            / (dimension.upper_bound - dimension.lower_bound)
            for item, dimension in zip(ranges, plan.dimensions, strict=True)
        )
        propagated_lower = max(
            0.0,
            max(
                estimate.lower_bound
                - plan.failure_probability_lipschitz_constant
                * _maximum_distance_to_cell(
                    plan.dimensions,
                    estimate.coordinates,
                    ranges,
                )
                for estimate in estimates
            ),
        )
        propagated_upper = min(
            1.0,
            min(
                estimate.upper_bound
                + plan.failure_probability_lipschitz_constant
                * _maximum_distance_to_cell(
                    plan.dimensions,
                    estimate.coordinates,
                    ranges,
                )
                for estimate in estimates
            ),
        )
        if propagated_lower > propagated_upper:
            raise ValueError(
                "propagated cell bounds contradict the Lipschitz assumption"
            )
        threshold = plan.maximum_acceptable_failure_probability
        if propagated_upper <= threshold:
            disposition = BoundaryCellDisposition.RELIABLE_CERTIFIED
        elif propagated_lower > threshold:
            disposition = BoundaryCellDisposition.UNRELIABLE_CERTIFIED
        else:
            disposition = BoundaryCellDisposition.UNRESOLVED_BOUNDARY
        yield BoundaryCellAssessment(
            cell_id="cell:" + ":".join(str(index) for index in indices),
            ranges=ranges,
            normalized_geometric_volume=volume,
            failure_probability_lower_bound=propagated_lower,
            failure_probability_upper_bound=propagated_upper,
            disposition=disposition,
        )


def _normalized_distance(
    dimensions: Sequence[BoundaryDimension],
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Calculate weighted normalized root-mean-square coordinate distance."""
    weighted_square_sum = math.fsum(
        dimension.distance_weight
        * ((left_value - right_value) / (dimension.upper_bound - dimension.lower_bound))
        ** 2
        for dimension, left_value, right_value in zip(
            dimensions,
            left,
            right,
            strict=True,
        )
    )
    return math.sqrt(
        weighted_square_sum / math.fsum(item.distance_weight for item in dimensions)
    )


def _maximum_distance_to_cell(
    dimensions: Sequence[BoundaryDimension],
    coordinates: Sequence[float],
    ranges: Sequence[CellDimensionRange],
) -> float:
    """Calculate the farthest weighted normalized distance to a cell."""
    farthest = tuple(
        (
            item.upper_bound
            if abs(coordinate - item.upper_bound) >= abs(coordinate - item.lower_bound)
            else item.lower_bound
        )
        for coordinate, item in zip(coordinates, ranges, strict=True)
    )
    return _normalized_distance(dimensions, coordinates, farthest)


def _cell_volume(
    cells: Sequence[BoundaryCellAssessment],
    disposition: BoundaryCellDisposition,
) -> float:
    """Sum normalized geometric volume for one cell disposition."""
    return math.fsum(
        item.normalized_geometric_volume
        for item in cells
        if item.disposition is disposition
    )


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


def _positive_finite(value: float, label: str) -> float:
    """Require a finite number greater than zero."""
    resolved = _finite(value, label)
    if resolved <= 0.0:
        raise ValueError(f"{label} must be positive")
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
