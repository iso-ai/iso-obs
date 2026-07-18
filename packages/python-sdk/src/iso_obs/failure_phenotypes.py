"""Conformal failure-phenotype classification with explicit abstention.

The model uses robustly scaled, phenotype-specific prototypes and held-out
class-conditional calibration scores. Predictions are sets rather than forced
labels: a singleton is assigned, multiple supported phenotypes are ambiguous,
and an empty set is a novel-failure candidate requiring review.

Class-conditional coverage relies on exchangeability between calibration and
future examples within each phenotype. Distribution shift, label error, and
dependent sampling can invalidate that guarantee.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse

_MAD_CONSISTENCY_FACTOR = 1.4826


class PhenotypeTrainingSplit(StrEnum):
    """Role of one labeled example in model development."""

    FIT = "fit"
    CALIBRATION = "calibration"


class PhenotypeDisposition(StrEnum):
    """Interpretation of a conformal phenotype prediction set."""

    ASSIGNED = "assigned"
    AMBIGUOUS = "ambiguous"
    NOVEL_CANDIDATE = "novel_candidate"


@dataclass(frozen=True, slots=True)
class PhenotypeFeature:
    """One trace-derived feature used by the phenotype model."""

    feature_id: str
    definition_digest: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate feature identity, provenance, and distance weight."""
        _require_text(self.feature_id, "phenotype feature ID")
        NamedDigest("phenotype feature definition", self.definition_digest)
        object.__setattr__(
            self,
            "weight",
            _positive_finite(self.weight, "phenotype feature weight"),
        )


@dataclass(frozen=True, slots=True)
class FailurePhenotypePlan:
    """Content-addressed design for fitting and calibrating the model."""

    plan_id: str
    plan_version: str
    feature_extraction_digest: str
    target_population_digest: str
    phenotype_taxonomy_digest: str
    features: tuple[PhenotypeFeature, ...]
    phenotype_ids: tuple[str, ...]
    miscoverage_rate: float
    minimum_fit_examples_per_phenotype: int
    minimum_calibration_examples_per_phenotype: int
    minimum_feature_scale: float = 1e-6
    limitations: tuple[str, ...] = ()
    evidence_use: EvidenceUse = EvidenceUse.DISCOVERY_ONLY

    def __post_init__(self) -> None:
        """Validate identities, feature space, classes, and calibration target."""
        _require_text(self.plan_id, "phenotype plan ID")
        _require_text(self.plan_version, "phenotype plan version")
        for value, label in (
            (self.feature_extraction_digest, "feature extraction"),
            (self.target_population_digest, "target population"),
            (self.phenotype_taxonomy_digest, "phenotype taxonomy"),
        ):
            NamedDigest(label, value)
        if not self.features:
            raise ValueError("phenotype plan requires at least one feature")
        features = tuple(self.features)
        _require_unique(
            (item.feature_id for item in features),
            "phenotype feature IDs",
        )
        _require_unique(
            (item.definition_digest for item in features),
            "phenotype feature definitions",
        )
        if len(self.phenotype_ids) < 2:
            raise ValueError("phenotype plan requires at least two phenotypes")
        for phenotype_id in self.phenotype_ids:
            _require_text(phenotype_id, "phenotype ID")
        _require_unique(self.phenotype_ids, "phenotype IDs")
        miscoverage_rate = _open_unit_interval(
            self.miscoverage_rate,
            "phenotype miscoverage rate",
        )
        minimum_fit = _positive_integer(
            self.minimum_fit_examples_per_phenotype,
            "minimum fit examples per phenotype",
        )
        minimum_calibration = _positive_integer(
            self.minimum_calibration_examples_per_phenotype,
            "minimum calibration examples per phenotype",
        )
        minimum_scale = _positive_finite(
            self.minimum_feature_scale,
            "minimum phenotype feature scale",
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("failure phenotype classification must be discovery-only")
        limitations = _unique_text(self.limitations, "phenotype plan limitations")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "phenotype_ids", tuple(sorted(self.phenotype_ids)))
        object.__setattr__(self, "miscoverage_rate", miscoverage_rate)
        object.__setattr__(
            self,
            "minimum_fit_examples_per_phenotype",
            minimum_fit,
        )
        object.__setattr__(
            self,
            "minimum_calibration_examples_per_phenotype",
            minimum_calibration,
        )
        object.__setattr__(self, "minimum_feature_scale", minimum_scale)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_use", evidence_use)

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact plan content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class FailurePhenotypeExample:
    """One independently sampled, labeled feature vector."""

    plan_content_digest: str
    evidence_digest: str
    independence_unit_id: str
    phenotype_id: str
    split: PhenotypeTrainingSplit
    feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate example provenance, role, label, and numeric values."""
        NamedDigest("phenotype plan", self.plan_content_digest)
        NamedDigest("phenotype example evidence", self.evidence_digest)
        _require_text(self.independence_unit_id, "phenotype independence unit ID")
        _require_text(self.phenotype_id, "example phenotype ID")
        split = PhenotypeTrainingSplit(self.split)
        values = _finite_vector(self.feature_values, "phenotype feature values")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "feature_values", values)


@dataclass(frozen=True, slots=True)
class PhenotypePrototype:
    """One robust prototype and its held-out nonconformity distribution."""

    phenotype_id: str
    fit_example_count: int
    calibration_example_count: int
    standardized_center: tuple[float, ...]
    calibration_nonconformity_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate class counts, center, and ordered calibration scores."""
        _require_text(self.phenotype_id, "prototype phenotype ID")
        fit_count = _positive_integer(
            self.fit_example_count,
            "prototype fit example count",
        )
        calibration_count = _positive_integer(
            self.calibration_example_count,
            "prototype calibration example count",
        )
        center = _finite_vector(
            self.standardized_center,
            "prototype standardized center",
        )
        scores = tuple(
            _nonnegative_finite(score, "calibration nonconformity score")
            for score in self.calibration_nonconformity_scores
        )
        if len(scores) != calibration_count:
            raise ValueError("calibration score count must match calibration examples")
        if scores != tuple(sorted(scores)):
            raise ValueError("calibration nonconformity scores must be sorted")
        object.__setattr__(self, "fit_example_count", fit_count)
        object.__setattr__(self, "calibration_example_count", calibration_count)
        object.__setattr__(self, "standardized_center", center)
        object.__setattr__(
            self,
            "calibration_nonconformity_scores",
            scores,
        )


@dataclass(frozen=True, slots=True)
class FailurePhenotypeModel:
    """Content-addressed fitted prototype and conformal calibration state."""

    model_schema_version: str
    plan_content_digest: str
    training_data_digest: str
    feature_ids: tuple[str, ...]
    feature_weights: tuple[float, ...]
    feature_medians: tuple[float, ...]
    feature_scales: tuple[float, ...]
    prototypes: tuple[PhenotypePrototype, ...]
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate model identity, aligned feature state, and prototypes."""
        _require_text(self.model_schema_version, "phenotype model schema version")
        NamedDigest("phenotype plan", self.plan_content_digest)
        NamedDigest("phenotype training data", self.training_data_digest)
        if not self.feature_ids:
            raise ValueError("phenotype model requires features")
        for feature_id in self.feature_ids:
            _require_text(feature_id, "phenotype model feature ID")
        _require_unique(self.feature_ids, "phenotype model feature IDs")
        dimension = len(self.feature_ids)
        weights = _positive_vector(
            self.feature_weights,
            "phenotype model feature weights",
        )
        medians = _finite_vector(
            self.feature_medians,
            "phenotype model feature medians",
        )
        scales = _positive_vector(
            self.feature_scales,
            "phenotype model feature scales",
        )
        if not (len(weights) == len(medians) == len(scales) == dimension):
            raise ValueError("phenotype model feature state dimensions must align")
        if len(self.prototypes) < 2:
            raise ValueError("phenotype model requires at least two prototypes")
        prototypes = tuple(sorted(self.prototypes, key=lambda item: item.phenotype_id))
        _require_unique(
            (item.phenotype_id for item in prototypes),
            "phenotype model prototype IDs",
        )
        if any(len(item.standardized_center) != dimension for item in prototypes):
            raise ValueError("prototype dimensions must match model features")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("failure phenotype model must remain discovery-only")
        limitations = _unique_text(self.limitations, "phenotype model limitations")
        object.__setattr__(self, "feature_weights", weights)
        object.__setattr__(self, "feature_medians", medians)
        object.__setattr__(self, "feature_scales", scales)
        object.__setattr__(self, "prototypes", prototypes)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the fitted model to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact fitted-model content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class FailurePhenotypeQuery:
    """One trace-derived vector to classify against known phenotypes."""

    plan_content_digest: str
    evidence_digest: str
    query_id: str
    feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate query identity, provenance, and feature values."""
        NamedDigest("phenotype plan", self.plan_content_digest)
        NamedDigest("phenotype query evidence", self.evidence_digest)
        _require_text(self.query_id, "phenotype query ID")
        object.__setattr__(
            self,
            "feature_values",
            _finite_vector(self.feature_values, "phenotype query feature values"),
        )

    def to_json(self) -> str:
        """Serialize the query to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact query content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class FeatureDistanceContribution:
    """Contribution of one feature to a prototype distance."""

    feature_id: str
    weighted_squared_distance: float
    fraction_of_squared_distance: float

    def __post_init__(self) -> None:
        """Validate the non-negative feature contribution."""
        _require_text(self.feature_id, "distance contribution feature ID")
        object.__setattr__(
            self,
            "weighted_squared_distance",
            _nonnegative_finite(
                self.weighted_squared_distance,
                "weighted squared distance",
            ),
        )
        object.__setattr__(
            self,
            "fraction_of_squared_distance",
            _closed_unit_interval(
                self.fraction_of_squared_distance,
                "fraction of squared distance",
            ),
        )


@dataclass(frozen=True, slots=True)
class PhenotypeCandidate:
    """One phenotype's conformal support and distance explanation."""

    phenotype_id: str
    nonconformity_score: float
    conformal_p_value: float
    included_in_prediction_set: bool
    feature_contributions: tuple[FeatureDistanceContribution, ...]

    def __post_init__(self) -> None:
        """Validate candidate score, support, and contribution decomposition."""
        _require_text(self.phenotype_id, "candidate phenotype ID")
        score = _nonnegative_finite(
            self.nonconformity_score,
            "phenotype nonconformity score",
        )
        p_value = _closed_unit_interval(
            self.conformal_p_value,
            "phenotype conformal p-value",
        )
        if not isinstance(self.included_in_prediction_set, bool):
            raise ValueError("prediction-set inclusion must be a boolean")
        contributions = tuple(
            sorted(
                self.feature_contributions,
                key=lambda item: (
                    -item.weighted_squared_distance,
                    item.feature_id,
                ),
            )
        )
        _require_unique(
            (item.feature_id for item in contributions),
            "candidate contribution feature IDs",
        )
        fraction_sum = math.fsum(
            item.fraction_of_squared_distance for item in contributions
        )
        expected_sum = 0.0 if score == 0.0 else 1.0
        if not math.isclose(
            fraction_sum,
            expected_sum,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("feature contribution fractions are inconsistent")
        object.__setattr__(self, "nonconformity_score", score)
        object.__setattr__(self, "conformal_p_value", p_value)
        object.__setattr__(self, "feature_contributions", contributions)


@dataclass(frozen=True, slots=True)
class FailurePhenotypeReport:
    """Auditable conformal classification or abstention report."""

    report_schema_version: str
    model_content_digest: str
    query_content_digest: str
    disposition: PhenotypeDisposition
    assigned_phenotype_id: str | None
    nearest_phenotype_id: str
    prediction_set: tuple[str, ...]
    candidates: tuple[PhenotypeCandidate, ...]
    miscoverage_rate: float
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate report identity, prediction set, and disposition."""
        _require_text(self.report_schema_version, "phenotype report schema version")
        NamedDigest("phenotype model", self.model_content_digest)
        NamedDigest("phenotype query", self.query_content_digest)
        disposition = PhenotypeDisposition(self.disposition)
        _require_text(self.nearest_phenotype_id, "nearest phenotype ID")
        candidates = tuple(sorted(self.candidates, key=lambda item: item.phenotype_id))
        if not candidates:
            raise ValueError("phenotype report requires candidates")
        _require_unique(
            (item.phenotype_id for item in candidates),
            "phenotype report candidate IDs",
        )
        candidate_ids = {item.phenotype_id for item in candidates}
        if self.nearest_phenotype_id not in candidate_ids:
            raise ValueError("nearest phenotype must be a report candidate")
        expected_nearest = min(
            candidates,
            key=lambda item: (item.nonconformity_score, item.phenotype_id),
        )
        if self.nearest_phenotype_id != expected_nearest.phenotype_id:
            raise ValueError("nearest phenotype must have the smallest distance")
        miscoverage_rate = _open_unit_interval(
            self.miscoverage_rate,
            "phenotype report miscoverage rate",
        )
        if any(
            item.included_in_prediction_set
            != (item.conformal_p_value > miscoverage_rate)
            for item in candidates
        ):
            raise ValueError(
                "prediction-set inclusion must follow the conformal threshold"
            )
        prediction_set = tuple(sorted(self.prediction_set))
        expected_set = tuple(
            item.phenotype_id for item in candidates if item.included_in_prediction_set
        )
        if prediction_set != expected_set:
            raise ValueError("prediction set must match included candidates")
        if disposition is PhenotypeDisposition.ASSIGNED:
            if len(prediction_set) != 1:
                raise ValueError("assigned disposition requires a singleton set")
            if self.assigned_phenotype_id != prediction_set[0]:
                raise ValueError("assigned phenotype must equal the prediction set")
        elif disposition is PhenotypeDisposition.AMBIGUOUS:
            if len(prediction_set) < 2 or self.assigned_phenotype_id is not None:
                raise ValueError("ambiguous disposition requires multiple candidates")
        elif prediction_set or self.assigned_phenotype_id is not None:
            raise ValueError("novel candidate requires an empty prediction set")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("phenotype report must remain discovery-only")
        limitations = _unique_text(self.limitations, "phenotype report limitations")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "prediction_set", prediction_set)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "miscoverage_rate", miscoverage_rate)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact report content digest."""
        return sha256_digest(self.to_json())


def fit_failure_phenotype_model(
    plan: FailurePhenotypePlan,
    examples: Sequence[FailurePhenotypeExample],
) -> FailurePhenotypeModel:
    """Fit robust prototypes and class-conditional conformal calibration.

    Args:
        plan: Exact feature, taxonomy, sampling, and error-rate design.
        examples: Independently sampled labeled fit and calibration examples.

    Returns:
        A deterministic, content-addressed fitted model.

    Raises:
        ValueError: If evidence identity, dimensions, labels, split sizes, or
            independence units violate the declared plan.
    """
    plan_digest = plan.content_digest()
    dimension = len(plan.features)
    units: set[str] = set()
    evidence_digests: set[str] = set()
    by_split_and_class: dict[
        tuple[PhenotypeTrainingSplit, str],
        list[FailurePhenotypeExample],
    ] = {
        (split, phenotype_id): []
        for split in PhenotypeTrainingSplit
        for phenotype_id in plan.phenotype_ids
    }
    for example in examples:
        if example.plan_content_digest != plan_digest:
            raise ValueError(
                f"phenotype example '{example.independence_unit_id}' plan mismatch"
            )
        if example.phenotype_id not in plan.phenotype_ids:
            raise ValueError(f"unknown phenotype ID: {example.phenotype_id}")
        if len(example.feature_values) != dimension:
            raise ValueError(
                f"phenotype example '{example.independence_unit_id}' "
                "feature dimension mismatch"
            )
        if example.independence_unit_id in units:
            raise ValueError(
                "phenotype independence unit IDs must be unique across all splits"
            )
        if example.evidence_digest in evidence_digests:
            raise ValueError("phenotype example evidence digests must be unique")
        units.add(example.independence_unit_id)
        evidence_digests.add(example.evidence_digest)
        by_split_and_class[(example.split, example.phenotype_id)].append(example)

    for phenotype_id in plan.phenotype_ids:
        fit_count = len(by_split_and_class[(PhenotypeTrainingSplit.FIT, phenotype_id)])
        calibration_count = len(
            by_split_and_class[(PhenotypeTrainingSplit.CALIBRATION, phenotype_id)]
        )
        if fit_count < plan.minimum_fit_examples_per_phenotype:
            raise ValueError(
                f"phenotype '{phenotype_id}' has insufficient fit examples"
            )
        if calibration_count < plan.minimum_calibration_examples_per_phenotype:
            raise ValueError(
                f"phenotype '{phenotype_id}' has insufficient calibration examples"
            )

    fit_examples = tuple(
        example for example in examples if example.split is PhenotypeTrainingSplit.FIT
    )
    feature_columns = tuple(
        tuple(example.feature_values[index] for example in fit_examples)
        for index in range(dimension)
    )
    medians = tuple(float(statistics.median(column)) for column in feature_columns)
    scales = tuple(
        max(
            plan.minimum_feature_scale,
            _MAD_CONSISTENCY_FACTOR
            * float(statistics.median(abs(value - median) for value in column)),
        )
        for column, median in zip(feature_columns, medians, strict=True)
    )
    weights = tuple(item.weight for item in plan.features)

    prototypes: list[PhenotypePrototype] = []
    for phenotype_id in plan.phenotype_ids:
        class_fit = by_split_and_class[(PhenotypeTrainingSplit.FIT, phenotype_id)]
        standardized_fit = tuple(
            _standardize(example.feature_values, medians, scales)
            for example in class_fit
        )
        center = tuple(
            float(statistics.median(vector[index] for vector in standardized_fit))
            for index in range(dimension)
        )
        class_calibration = by_split_and_class[
            (PhenotypeTrainingSplit.CALIBRATION, phenotype_id)
        ]
        calibration_scores = tuple(
            sorted(
                _weighted_rms_distance(
                    _standardize(example.feature_values, medians, scales),
                    center,
                    weights,
                )
                for example in class_calibration
            )
        )
        prototypes.append(
            PhenotypePrototype(
                phenotype_id=phenotype_id,
                fit_example_count=len(class_fit),
                calibration_example_count=len(class_calibration),
                standardized_center=center,
                calibration_nonconformity_scores=calibration_scores,
            )
        )

    canonical_examples = tuple(
        sorted(
            examples,
            key=lambda item: (
                item.split.value,
                item.phenotype_id,
                item.independence_unit_id,
            ),
        )
    )
    return FailurePhenotypeModel(
        model_schema_version="iso-obs.failure-phenotype-model.v1",
        plan_content_digest=plan_digest,
        training_data_digest=sha256_digest(_canonical_json(canonical_examples)),
        feature_ids=tuple(item.feature_id for item in plan.features),
        feature_weights=weights,
        feature_medians=medians,
        feature_scales=scales,
        prototypes=tuple(prototypes),
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=(
            "Class-conditional coverage requires exchangeability within each "
            "phenotype between calibration and future examples.",
            "An empty prediction set is a novel-failure candidate, not proof of "
            "a new failure mechanism.",
            "The model does not establish causality or authorize a release decision.",
            "Robust prototypes do not represent multimodal variation within a "
            "declared phenotype.",
        ),
    )


def classify_failure_phenotype(
    plan: FailurePhenotypePlan,
    model: FailurePhenotypeModel,
    query: FailurePhenotypeQuery,
) -> FailurePhenotypeReport:
    """Classify a query with a conformal set or explicitly abstain.

    Args:
        plan: Exact plan used to fit ``model``.
        model: Fitted robust phenotype model.
        query: Trace-derived feature vector with evidence provenance.

    Returns:
        An assigned, ambiguous, or novel-candidate report.

    Raises:
        ValueError: If model or query identity and dimensions do not match.
    """
    plan_digest = plan.content_digest()
    if model.plan_content_digest != plan_digest:
        raise ValueError("phenotype model does not belong to plan")
    if query.plan_content_digest != plan_digest:
        raise ValueError("phenotype query does not belong to plan")
    if len(query.feature_values) != len(model.feature_ids):
        raise ValueError("phenotype query feature dimension mismatch")
    standardized = _standardize(
        query.feature_values,
        model.feature_medians,
        model.feature_scales,
    )
    candidates: list[PhenotypeCandidate] = []
    for prototype in model.prototypes:
        contributions = _feature_contributions(
            model.feature_ids,
            standardized,
            prototype.standardized_center,
            model.feature_weights,
        )
        score = _weighted_rms_distance(
            standardized,
            prototype.standardized_center,
            model.feature_weights,
        )
        calibration_scores = prototype.calibration_nonconformity_scores
        tail_count = sum(item >= score for item in calibration_scores)
        p_value = (tail_count + 1.0) / (len(calibration_scores) + 1.0)
        candidates.append(
            PhenotypeCandidate(
                phenotype_id=prototype.phenotype_id,
                nonconformity_score=score,
                conformal_p_value=p_value,
                included_in_prediction_set=p_value > plan.miscoverage_rate,
                feature_contributions=contributions,
            )
        )
    prediction_set = tuple(
        sorted(
            item.phenotype_id for item in candidates if item.included_in_prediction_set
        )
    )
    if len(prediction_set) == 1:
        disposition = PhenotypeDisposition.ASSIGNED
        assigned_phenotype_id = prediction_set[0]
    elif prediction_set:
        disposition = PhenotypeDisposition.AMBIGUOUS
        assigned_phenotype_id = None
    else:
        disposition = PhenotypeDisposition.NOVEL_CANDIDATE
        assigned_phenotype_id = None
    nearest = min(
        candidates,
        key=lambda item: (item.nonconformity_score, item.phenotype_id),
    )
    return FailurePhenotypeReport(
        report_schema_version="iso-obs.failure-phenotype-report.v1",
        model_content_digest=model.content_digest(),
        query_content_digest=query.content_digest(),
        disposition=disposition,
        assigned_phenotype_id=assigned_phenotype_id,
        nearest_phenotype_id=nearest.phenotype_id,
        prediction_set=prediction_set,
        candidates=tuple(candidates),
        miscoverage_rate=plan.miscoverage_rate,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=(
            "Conformal support is not a posterior probability or causal claim.",
            "The coverage target depends on within-phenotype exchangeability.",
            "Novel candidates require human review and independent reproduction.",
        ),
    )


def _standardize(
    values: Sequence[float],
    medians: Sequence[float],
    scales: Sequence[float],
) -> tuple[float, ...]:
    """Robustly standardize one aligned feature vector."""
    return tuple(
        (value - median) / scale
        for value, median, scale in zip(values, medians, scales, strict=True)
    )


def _weighted_rms_distance(
    values: Sequence[float],
    center: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Calculate weighted root-mean-square distance to a prototype."""
    weighted_square_sum = math.fsum(
        weight * (value - reference) ** 2
        for value, reference, weight in zip(values, center, weights, strict=True)
    )
    return math.sqrt(weighted_square_sum / math.fsum(weights))


def _feature_contributions(
    feature_ids: Sequence[str],
    values: Sequence[float],
    center: Sequence[float],
    weights: Sequence[float],
) -> tuple[FeatureDistanceContribution, ...]:
    """Decompose squared prototype distance across aligned features."""
    squared_distances = tuple(
        weight * (value - reference) ** 2
        for value, reference, weight in zip(values, center, weights, strict=True)
    )
    total = math.fsum(squared_distances)
    return tuple(
        FeatureDistanceContribution(
            feature_id=feature_id,
            weighted_squared_distance=squared_distance,
            fraction_of_squared_distance=(
                squared_distance / total if total > 0.0 else 0.0
            ),
        )
        for feature_id, squared_distance in zip(
            feature_ids,
            squared_distances,
            strict=True,
        )
    )


def _finite_vector(values: Iterable[float], label: str) -> tuple[float, ...]:
    """Require a non-empty vector containing only finite numbers."""
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError(f"{label} must not be empty")
    if any(not math.isfinite(value) for value in resolved):
        raise ValueError(f"{label} must contain only finite numbers")
    return resolved


def _positive_vector(values: Iterable[float], label: str) -> tuple[float, ...]:
    """Require a non-empty vector containing only positive finite numbers."""
    resolved = tuple(_positive_finite(value, label) for value in values)
    if not resolved:
        raise ValueError(f"{label} must not be empty")
    return resolved


def _positive_finite(value: float, label: str) -> float:
    """Require a finite number greater than zero."""
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return resolved


def _nonnegative_finite(value: float, label: str) -> float:
    """Require a finite number greater than or equal to zero."""
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return resolved


def _open_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``(0, 1)``."""
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0 or resolved >= 1.0:
        raise ValueError(f"{label} must be in (0, 1)")
    return resolved


def _closed_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``[0, 1]``."""
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return resolved


def _positive_integer(value: int, label: str) -> int:
    """Require an integer greater than zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
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
