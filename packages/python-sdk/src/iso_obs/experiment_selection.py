"""Bayesian experiment selection for exploratory simulation campaigns.

The selector learns a separate Beta-Bernoulli failure-rate posterior for each
declared operating region. It recommends the next simulation by combining
severity-weighted failure probability, posterior uncertainty, exact expected
variance reduction, target-population importance, and expected execution cost.

This model is deliberately small, interpretable, and discovery-only. Its
adaptive observations must not be reused as if they came from a fixed
confirmatory design, and its recommendations cannot certify safety.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse


class SelectionReason(StrEnum):
    """Reason a region is eligible for the next experiment."""

    REQUIRED_COVERAGE = "required_coverage"
    ACQUISITION_SCORE = "acquisition_score"


class SelectionDisposition(StrEnum):
    """Outcome of attempting to select another experiment."""

    RECOMMENDED = "recommended"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DESIGN_COMPLETE = "design_complete"


@dataclass(frozen=True, slots=True)
class ExperimentRegion:
    """One disjoint operating region available to the exploratory selector."""

    region_id: str
    condition_definition_digest: str
    target_population_mass: float
    severity_weight: float
    expected_execution_cost: float
    minimum_exploration_count: int
    maximum_experiment_count: int

    def __post_init__(self) -> None:
        """Validate region identity, importance, cost, and sampling limits."""
        _require_text(self.region_id, "experiment region ID")
        NamedDigest("region condition definition", self.condition_definition_digest)
        mass = _positive_unit_interval(
            self.target_population_mass,
            "region target population mass",
        )
        severity = _positive_finite(
            self.severity_weight,
            "region severity weight",
        )
        cost = _positive_finite(
            self.expected_execution_cost,
            "region expected execution cost",
        )
        minimum = _nonnegative_integer(
            self.minimum_exploration_count,
            "region minimum exploration count",
        )
        maximum = _positive_integer(
            self.maximum_experiment_count,
            "region maximum experiment count",
        )
        if minimum > maximum:
            raise ValueError("region minimum exploration count must not exceed maximum")
        object.__setattr__(self, "target_population_mass", mass)
        object.__setattr__(self, "severity_weight", severity)
        object.__setattr__(self, "expected_execution_cost", cost)
        object.__setattr__(self, "minimum_exploration_count", minimum)
        object.__setattr__(self, "maximum_experiment_count", maximum)


@dataclass(frozen=True, slots=True)
class BayesianSelectorConfig:
    """Prior and acquisition weights for a Bayesian failure-search model."""

    prior_failure_count: float = 0.5
    prior_success_count: float = 0.5
    failure_probability_weight: float = 0.4
    uncertainty_weight: float = 0.3
    information_gain_weight: float = 0.3

    def __post_init__(self) -> None:
        """Validate proper priors and a normalized acquisition objective."""
        prior_failure = _positive_finite(
            self.prior_failure_count,
            "prior failure count",
        )
        prior_success = _positive_finite(
            self.prior_success_count,
            "prior success count",
        )
        weights = (
            _closed_unit_interval(
                self.failure_probability_weight,
                "failure probability weight",
            ),
            _closed_unit_interval(
                self.uncertainty_weight,
                "uncertainty weight",
            ),
            _closed_unit_interval(
                self.information_gain_weight,
                "information gain weight",
            ),
        )
        if not math.isclose(
            math.fsum(weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selector acquisition weights must sum to one")
        object.__setattr__(self, "prior_failure_count", prior_failure)
        object.__setattr__(self, "prior_success_count", prior_success)
        object.__setattr__(self, "failure_probability_weight", weights[0])
        object.__setattr__(self, "uncertainty_weight", weights[1])
        object.__setattr__(self, "information_gain_weight", weights[2])


@dataclass(frozen=True, slots=True)
class ExperimentSelectionPlan:
    """Content-addressed exploratory design for model-guided simulation."""

    plan_id: str
    plan_version: str
    target_population_digest: str
    partition_definition_digest: str
    simulation_manifest_digest: str
    system_artifact_digest: str
    outcome_definition_digest: str
    total_execution_budget: float
    regions: tuple[ExperimentRegion, ...]
    config: BayesianSelectorConfig
    limitations: tuple[str, ...]
    evidence_use: EvidenceUse = EvidenceUse.DISCOVERY_ONLY

    def __post_init__(self) -> None:
        """Validate plan identity, partition, budget, and evidence boundary."""
        _require_text(self.plan_id, "experiment selection plan ID")
        _require_text(self.plan_version, "experiment selection plan version")
        for value, label in (
            (self.target_population_digest, "target population"),
            (self.partition_definition_digest, "partition definition"),
            (self.simulation_manifest_digest, "simulation manifest"),
            (self.system_artifact_digest, "system artifact"),
            (self.outcome_definition_digest, "failure outcome definition"),
        ):
            NamedDigest(label, value)
        budget = _positive_finite(
            self.total_execution_budget,
            "total execution budget",
        )
        if not self.regions:
            raise ValueError("experiment selection plan requires regions")
        _require_unique(
            (item.region_id for item in self.regions),
            "experiment region IDs",
        )
        _require_unique(
            (item.condition_definition_digest for item in self.regions),
            "experiment region condition definitions",
        )
        target_mass = math.fsum(item.target_population_mass for item in self.regions)
        if not math.isclose(target_mass, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("experiment region target masses must sum to one")
        if not isinstance(self.config, BayesianSelectorConfig):
            raise ValueError("selector config must be a BayesianSelectorConfig")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("adaptive experiment selection must remain discovery-only")
        limitations = _unique_text(
            self.limitations,
            "experiment selection plan limitations",
        )
        object.__setattr__(self, "total_execution_budget", budget)
        object.__setattr__(
            self,
            "regions",
            tuple(sorted(self.regions, key=lambda item: item.region_id)),
        )
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_use", evidence_use)

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact experiment-selection plan digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class ExperimentSelectionObservation:
    """One exploratory binary outcome used to fit the selector."""

    plan_content_digest: str
    region_id: str
    sample_index: int
    trial_id: str
    failed: bool
    evidence_digest: str

    def __post_init__(self) -> None:
        """Validate training provenance, ordering coordinate, and outcome."""
        NamedDigest("experiment selection plan", self.plan_content_digest)
        _require_text(self.region_id, "experiment observation region ID")
        _nonnegative_integer(
            self.sample_index,
            "experiment observation sample index",
        )
        _require_text(self.trial_id, "experiment selection trial ID")
        if not isinstance(self.failed, bool):
            raise ValueError("experiment observation failed must be a boolean")
        NamedDigest("experiment observation evidence", self.evidence_digest)


@dataclass(frozen=True, slots=True)
class RegionPosterior:
    """Fitted Beta-Bernoulli state and uncertainty for one region."""

    region_id: str
    observation_count: int
    failure_count: int
    posterior_failure_count: float
    posterior_success_count: float
    mean_failure_probability: float
    posterior_standard_deviation: float
    expected_variance_reduction: float

    def __post_init__(self) -> None:
        """Validate posterior parameters and derived statistics."""
        _require_text(self.region_id, "posterior region ID")
        observation_count = _nonnegative_integer(
            self.observation_count,
            "posterior observation count",
        )
        failure_count = _nonnegative_integer(
            self.failure_count,
            "posterior failure count",
        )
        if failure_count > observation_count:
            raise ValueError("posterior failures cannot exceed observations")
        alpha = _positive_finite(
            self.posterior_failure_count,
            "posterior failure count parameter",
        )
        beta = _positive_finite(
            self.posterior_success_count,
            "posterior success count parameter",
        )
        mean = _closed_unit_interval(
            self.mean_failure_probability,
            "posterior mean failure probability",
        )
        expected_mean = alpha / (alpha + beta)
        if not math.isclose(mean, expected_mean, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("posterior mean is inconsistent with its parameters")
        standard_deviation = _nonnegative_finite(
            self.posterior_standard_deviation,
            "posterior standard deviation",
        )
        expected_standard_deviation = math.sqrt(_beta_variance(alpha, beta))
        if not math.isclose(
            standard_deviation,
            expected_standard_deviation,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "posterior standard deviation is inconsistent with its parameters"
            )
        variance_reduction = _nonnegative_finite(
            self.expected_variance_reduction,
            "posterior expected variance reduction",
        )
        expected_reduction = _expected_variance_reduction(alpha, beta)
        if not math.isclose(
            variance_reduction,
            expected_reduction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "posterior variance reduction is inconsistent with its parameters"
            )
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "failure_count", failure_count)
        object.__setattr__(self, "posterior_failure_count", alpha)
        object.__setattr__(self, "posterior_success_count", beta)
        object.__setattr__(self, "mean_failure_probability", mean)
        object.__setattr__(
            self,
            "posterior_standard_deviation",
            standard_deviation,
        )
        object.__setattr__(
            self,
            "expected_variance_reduction",
            variance_reduction,
        )


@dataclass(frozen=True, slots=True)
class BayesianExperimentSelectorState:
    """Content-addressed fitted state of the compact reliability model."""

    model_schema_version: str
    plan_content_digest: str
    training_data_digest: str
    observation_count: int
    expected_budget_spent: float
    expected_budget_remaining: float
    region_posteriors: tuple[RegionPosterior, ...]
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate fitted-state identity, budget, ordering, and use."""
        _require_text(self.model_schema_version, "selector model schema version")
        NamedDigest("experiment selection plan", self.plan_content_digest)
        NamedDigest("selector training data", self.training_data_digest)
        _nonnegative_integer(self.observation_count, "selector observation count")
        spent = _nonnegative_finite(
            self.expected_budget_spent,
            "expected budget spent",
        )
        remaining = _nonnegative_finite(
            self.expected_budget_remaining,
            "expected budget remaining",
        )
        posteriors = tuple(
            sorted(self.region_posteriors, key=lambda item: item.region_id)
        )
        _require_unique(
            (item.region_id for item in posteriors),
            "selector posterior region IDs",
        )
        if sum(item.observation_count for item in posteriors) != self.observation_count:
            raise ValueError("selector posterior counts must equal total observations")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("fitted selector state must remain discovery-only")
        limitations = _unique_text(
            self.limitations,
            "selector state limitations",
        )
        object.__setattr__(self, "expected_budget_spent", spent)
        object.__setattr__(self, "expected_budget_remaining", remaining)
        object.__setattr__(self, "region_posteriors", posteriors)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize fitted model state to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact fitted-state content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class RegionAcquisitionScore:
    """Explainable score decomposition for one eligible region."""

    rank: int
    region_id: str
    reason: SelectionReason
    coverage_deficit: int
    observation_count: int
    posterior_failure_probability: float
    posterior_standard_deviation: float
    normalized_expected_variance_reduction: float
    target_population_mass: float
    severity_weight: float
    expected_execution_cost: float
    unadjusted_acquisition_score: float
    cost_adjusted_acquisition_score: float

    def __post_init__(self) -> None:
        """Validate the public acquisition-score decomposition."""
        rank = _positive_integer(self.rank, "acquisition rank")
        _require_text(self.region_id, "acquisition region ID")
        reason = SelectionReason(self.reason)
        coverage_deficit = _nonnegative_integer(
            self.coverage_deficit,
            "acquisition coverage deficit",
        )
        observation_count = _nonnegative_integer(
            self.observation_count,
            "acquisition observation count",
        )
        failure_probability = _closed_unit_interval(
            self.posterior_failure_probability,
            "acquisition posterior failure probability",
        )
        standard_deviation = _nonnegative_finite(
            self.posterior_standard_deviation,
            "acquisition posterior standard deviation",
        )
        variance_reduction = _closed_unit_interval(
            self.normalized_expected_variance_reduction,
            "normalized expected variance reduction",
        )
        target_mass = _closed_unit_interval(
            self.target_population_mass,
            "acquisition target population mass",
        )
        severity = _positive_finite(
            self.severity_weight,
            "acquisition severity weight",
        )
        cost = _positive_finite(
            self.expected_execution_cost,
            "acquisition expected execution cost",
        )
        unadjusted_score = _nonnegative_finite(
            self.unadjusted_acquisition_score,
            "unadjusted acquisition score",
        )
        cost_adjusted_score = _nonnegative_finite(
            self.cost_adjusted_acquisition_score,
            "cost-adjusted acquisition score",
        )
        if not math.isclose(
            cost_adjusted_score,
            unadjusted_score / cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "cost-adjusted acquisition score is inconsistent with cost"
            )
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "coverage_deficit", coverage_deficit)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(
            self,
            "posterior_failure_probability",
            failure_probability,
        )
        object.__setattr__(
            self,
            "posterior_standard_deviation",
            standard_deviation,
        )
        object.__setattr__(
            self,
            "normalized_expected_variance_reduction",
            variance_reduction,
        )
        object.__setattr__(self, "target_population_mass", target_mass)
        object.__setattr__(self, "severity_weight", severity)
        object.__setattr__(self, "expected_execution_cost", cost)
        object.__setattr__(
            self,
            "unadjusted_acquisition_score",
            unadjusted_score,
        )
        object.__setattr__(
            self,
            "cost_adjusted_acquisition_score",
            cost_adjusted_score,
        )


@dataclass(frozen=True, slots=True)
class ExperimentSelectionReport:
    """Ranked next-experiment recommendation from an exact fitted state."""

    report_schema_version: str
    selector_state_digest: str
    disposition: SelectionDisposition
    selected_region_id: str | None
    ranked_candidates: tuple[RegionAcquisitionScore, ...]
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate recommendation consistency and evidence boundary."""
        _require_text(self.report_schema_version, "selection report schema version")
        NamedDigest("experiment selector state", self.selector_state_digest)
        disposition = SelectionDisposition(self.disposition)
        candidates = tuple(sorted(self.ranked_candidates, key=lambda item: item.rank))
        if tuple(item.rank for item in candidates) != tuple(
            range(1, len(candidates) + 1)
        ):
            raise ValueError("selection candidate ranks must be contiguous from one")
        _require_unique(
            (item.region_id for item in candidates),
            "selection candidate region IDs",
        )
        if disposition is SelectionDisposition.RECOMMENDED:
            if not candidates:
                raise ValueError("recommended selection requires candidates")
            if self.selected_region_id != candidates[0].region_id:
                raise ValueError("selected region must be the top-ranked candidate")
        elif self.selected_region_id is not None or candidates:
            raise ValueError("terminal selection disposition cannot include candidates")
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("selection report must remain discovery-only")
        limitations = _unique_text(
            self.limitations,
            "selection report limitations",
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "ranked_candidates", candidates)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the recommendation report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact recommendation-report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def fit_bayesian_experiment_selector(
    plan: ExperimentSelectionPlan,
    observations: Sequence[ExperimentSelectionObservation],
) -> BayesianExperimentSelectorState:
    """Fit region-level Beta posteriors from exploratory outcomes.

    Args:
        plan: Exact exploratory partition, budget, prior, and objective.
        observations: Binary outcomes in any input order.

    Returns:
        Deterministic fitted state with one posterior per region.

    Raises:
        ValueError: If observations have wrong identity, duplicate trials,
            unknown regions, gaps, excess samples, or exceed the budget.
    """
    plan_digest = plan.content_digest()
    regions = {item.region_id: item for item in plan.regions}
    by_region: dict[str, list[ExperimentSelectionObservation]] = {
        region_id: [] for region_id in regions
    }
    trial_ids: set[str] = set()
    for observation in observations:
        if observation.plan_content_digest != plan_digest:
            raise ValueError(
                f"trial '{observation.trial_id}' plan content digest mismatch"
            )
        if observation.region_id not in regions:
            raise ValueError(
                f"unknown experiment selection region: {observation.region_id}"
            )
        if observation.trial_id in trial_ids:
            raise ValueError(
                f"duplicate experiment selection trial ID: {observation.trial_id}"
            )
        trial_ids.add(observation.trial_id)
        by_region[observation.region_id].append(observation)

    posteriors: list[RegionPosterior] = []
    expected_costs: list[float] = []
    canonical_observations: list[ExperimentSelectionObservation] = []
    for region in plan.regions:
        ordered = tuple(
            sorted(
                by_region[region.region_id],
                key=lambda item: item.sample_index,
            )
        )
        if len(ordered) > region.maximum_experiment_count:
            raise ValueError(f"region '{region.region_id}' exceeds its experiment cap")
        if tuple(item.sample_index for item in ordered) != tuple(range(len(ordered))):
            raise ValueError(
                f"region '{region.region_id}' sample indices must be "
                "contiguous from zero"
            )
        canonical_observations.extend(ordered)
        expected_costs.append(
            math.fsum(region.expected_execution_cost for _ in ordered)
        )
        failures = sum(item.failed for item in ordered)
        alpha = plan.config.prior_failure_count + failures
        beta = plan.config.prior_success_count + len(ordered) - failures
        variance = _beta_variance(alpha, beta)
        posteriors.append(
            RegionPosterior(
                region_id=region.region_id,
                observation_count=len(ordered),
                failure_count=failures,
                posterior_failure_count=alpha,
                posterior_success_count=beta,
                mean_failure_probability=alpha / (alpha + beta),
                posterior_standard_deviation=math.sqrt(variance),
                expected_variance_reduction=_expected_variance_reduction(
                    alpha,
                    beta,
                ),
            )
        )
    expected_spent = math.fsum(expected_costs)
    if expected_spent > plan.total_execution_budget and not math.isclose(
        expected_spent,
        plan.total_execution_budget,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("experiment observations exceed total execution budget")
    expected_remaining = max(0.0, plan.total_execution_budget - expected_spent)
    training_data_digest = sha256_digest(_canonical_json(tuple(canonical_observations)))
    return BayesianExperimentSelectorState(
        model_schema_version="iso-obs.bayesian-experiment-selector.v1",
        plan_content_digest=plan_digest,
        training_data_digest=training_data_digest,
        observation_count=len(observations),
        expected_budget_spent=expected_spent,
        expected_budget_remaining=expected_remaining,
        region_posteriors=tuple(posteriors),
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=(
            "The model assumes conditionally independent Bernoulli outcomes "
            "within each declared region.",
            "Posteriors do not share statistical strength across regions.",
            "Adaptive observations are discovery evidence and cannot be treated "
            "as a fixed confirmatory sample.",
            "Expected execution costs are planning estimates, not measured costs.",
        ),
    )


def recommend_next_experiment(
    plan: ExperimentSelectionPlan,
    state: BayesianExperimentSelectorState,
) -> ExperimentSelectionReport:
    """Rank eligible regions and recommend the next exploratory simulation.

    Required coverage is completed before acquisition-score optimization.
    Within the active candidate set, the score combines posterior failure
    probability, posterior standard deviation, and normalized expected
    posterior-variance reduction. Target-population mass and severity weight
    scale scientific importance, and expected cost scales resource efficiency.

    Args:
        plan: Exact exploratory plan used to fit ``state``.
        state: Fitted Bayesian selector state.

    Returns:
        Ranked explainable candidates or a terminal disposition.

    Raises:
        ValueError: If ``state`` was not fitted from ``plan``.
    """
    if state.plan_content_digest != plan.content_digest():
        raise ValueError("selector state does not belong to experiment plan")
    posteriors = {item.region_id: item for item in state.region_posteriors}
    incomplete_regions = tuple(
        region
        for region in plan.regions
        if posteriors[region.region_id].observation_count
        < region.maximum_experiment_count
    )
    if not incomplete_regions:
        return _terminal_report(state, SelectionDisposition.DESIGN_COMPLETE)

    affordable_regions = tuple(
        region
        for region in incomplete_regions
        if region.expected_execution_cost <= state.expected_budget_remaining + 1e-12
    )
    if not affordable_regions:
        return _terminal_report(state, SelectionDisposition.BUDGET_EXHAUSTED)

    uncovered_regions = tuple(
        region
        for region in incomplete_regions
        if posteriors[region.region_id].observation_count
        < region.minimum_exploration_count
    )
    required_regions = tuple(
        region for region in uncovered_regions if region in affordable_regions
    )
    if uncovered_regions and not required_regions:
        return _terminal_report(state, SelectionDisposition.BUDGET_EXHAUSTED)
    active_regions = required_regions or affordable_regions
    reason = (
        SelectionReason.REQUIRED_COVERAGE
        if required_regions
        else SelectionReason.ACQUISITION_SCORE
    )
    prior_variance = _beta_variance(
        plan.config.prior_failure_count,
        plan.config.prior_success_count,
    )
    scores = [
        _score_region(
            plan,
            region,
            posteriors[region.region_id],
            reason=reason,
            prior_variance=prior_variance,
        )
        for region in active_regions
    ]
    ordered = sorted(
        scores,
        key=lambda item: (
            -item.coverage_deficit,
            -item.cost_adjusted_acquisition_score,
            item.region_id,
        ),
    )
    ranked = tuple(
        RegionAcquisitionScore(
            rank=index,
            region_id=item.region_id,
            reason=item.reason,
            coverage_deficit=item.coverage_deficit,
            observation_count=item.observation_count,
            posterior_failure_probability=item.posterior_failure_probability,
            posterior_standard_deviation=item.posterior_standard_deviation,
            normalized_expected_variance_reduction=(
                item.normalized_expected_variance_reduction
            ),
            target_population_mass=item.target_population_mass,
            severity_weight=item.severity_weight,
            expected_execution_cost=item.expected_execution_cost,
            unadjusted_acquisition_score=item.unadjusted_acquisition_score,
            cost_adjusted_acquisition_score=item.cost_adjusted_acquisition_score,
        )
        for index, item in enumerate(ordered, start=1)
    )
    limitations = [
        "The recommendation is discovery-only and cannot support a release gate.",
        "Required regional coverage takes precedence over acquisition score.",
        "The score is conditional on declared regions, priors, severity weights, "
        "population masses, and expected costs.",
    ]
    if required_regions:
        limitations.append(
            "Regions that already satisfy minimum coverage are omitted until "
            "required coverage is complete."
        )
    return ExperimentSelectionReport(
        report_schema_version="iso-obs.experiment-selection-report.v1",
        selector_state_digest=state.content_digest(),
        disposition=SelectionDisposition.RECOMMENDED,
        selected_region_id=ranked[0].region_id,
        ranked_candidates=ranked,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True, slots=True)
class _UnrankedRegionScore:
    """Internal acquisition score before deterministic ranking."""

    region_id: str
    reason: SelectionReason
    coverage_deficit: int
    observation_count: int
    posterior_failure_probability: float
    posterior_standard_deviation: float
    normalized_expected_variance_reduction: float
    target_population_mass: float
    severity_weight: float
    expected_execution_cost: float
    unadjusted_acquisition_score: float
    cost_adjusted_acquisition_score: float


def _score_region(
    plan: ExperimentSelectionPlan,
    region: ExperimentRegion,
    posterior: RegionPosterior,
    *,
    reason: SelectionReason,
    prior_variance: float,
) -> _UnrankedRegionScore:
    """Calculate an explainable cost-adjusted acquisition score."""
    normalized_gain = (
        posterior.expected_variance_reduction / prior_variance
        if prior_variance > 0.0
        else 0.0
    )
    objective = (
        plan.config.failure_probability_weight * posterior.mean_failure_probability
        + plan.config.uncertainty_weight * posterior.posterior_standard_deviation
        + plan.config.information_gain_weight * normalized_gain
    )
    unadjusted = region.target_population_mass * region.severity_weight * objective
    coverage_deficit = max(
        0,
        region.minimum_exploration_count - posterior.observation_count,
    )
    return _UnrankedRegionScore(
        region_id=region.region_id,
        reason=reason,
        coverage_deficit=coverage_deficit,
        observation_count=posterior.observation_count,
        posterior_failure_probability=posterior.mean_failure_probability,
        posterior_standard_deviation=posterior.posterior_standard_deviation,
        normalized_expected_variance_reduction=normalized_gain,
        target_population_mass=region.target_population_mass,
        severity_weight=region.severity_weight,
        expected_execution_cost=region.expected_execution_cost,
        unadjusted_acquisition_score=unadjusted,
        cost_adjusted_acquisition_score=(unadjusted / region.expected_execution_cost),
    )


def _terminal_report(
    state: BayesianExperimentSelectorState,
    disposition: SelectionDisposition,
) -> ExperimentSelectionReport:
    """Build a terminal report with no recommendation."""
    return ExperimentSelectionReport(
        report_schema_version="iso-obs.experiment-selection-report.v1",
        selector_state_digest=state.content_digest(),
        disposition=disposition,
        selected_region_id=None,
        ranked_candidates=(),
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=(
            "No experiment was recommended.",
            "The selector remains discovery-only and makes no safety claim.",
        ),
    )


def _beta_variance(alpha: float, beta: float) -> float:
    """Calculate Beta-distribution variance."""
    total = alpha + beta
    return alpha * beta / (total * total * (total + 1.0))


def _expected_variance_reduction(alpha: float, beta: float) -> float:
    """Calculate expected posterior-variance reduction from one more outcome."""
    total = alpha + beta
    failure_probability = alpha / total
    current = _beta_variance(alpha, beta)
    expected_next = failure_probability * _beta_variance(alpha + 1.0, beta) + (
        1.0 - failure_probability
    ) * _beta_variance(alpha, beta + 1.0)
    reduction = current - expected_next
    return max(0.0, reduction)


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


def _positive_unit_interval(value: float, label: str) -> float:
    """Require a finite number in ``(0, 1]``."""
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0 or resolved > 1.0:
        raise ValueError(f"{label} must be in (0, 1]")
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
