"""Simulator-neutral regression packs derived from failure evidence.

A regression pack tests more than one memorized failure trajectory. It includes
the canonical reproduction, nearby conditions, a boundary probe, and controls
that demonstrate both expected safety and failure-detection sensitivity.
Release criteria may use exact conformance or a one-sided Wilson lower bound,
so small samples cannot silently support strong reliability claims.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist

from .evidence import NamedDigest, _canonical_json, sha256_digest


class CaseRole(StrEnum):
    """Scientific role of one case within a regression pack."""

    CANONICAL_REPRODUCTION = "canonical_reproduction"
    NEIGHBORHOOD = "neighborhood"
    BOUNDARY = "boundary"
    NEGATIVE_CONTROL = "negative_control"
    POSITIVE_CONTROL = "positive_control"


class ExpectedOutcome(StrEnum):
    """Expected relationship between a run and its invariant oracles."""

    SATISFY_ALL = "satisfy_all"
    VIOLATE_ANY = "violate_any"


class ComparisonOperator(StrEnum):
    """Supported numeric invariant comparison."""

    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    BETWEEN_INCLUSIVE = "between_inclusive"


class SeedStrategy(StrEnum):
    """Experimental strategy for stochastic regression executions."""

    EXACT_REPLAY = "exact_replay"
    COMMON_RANDOM_NUMBERS = "common_random_numbers"
    INDEPENDENT = "independent"


class GateMethod(StrEnum):
    """Method used to compare observed conformance with a release threshold."""

    EXACT = "exact"
    WILSON_LOWER_BOUND = "wilson_lower_bound"


@dataclass(frozen=True, slots=True)
class ScenarioParameter:
    """One typed scenario or perturbation parameter assignment."""

    name: str
    value: str | int | float | bool
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate a portable scalar parameter value."""
        _require_text(self.name, "scenario parameter name")
        if not isinstance(self.value, (str, int, float, bool)):
            raise ValueError("scenario parameter value must be a scalar")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("scenario parameter float value must be finite")
        if isinstance(self.value, str):
            _require_text(self.value, f"scenario parameter '{self.name}' value")
        if self.unit is not None:
            _require_text(self.unit, f"scenario parameter '{self.name}' unit")


@dataclass(frozen=True, slots=True)
class InvariantOracle:
    """Versioned numeric oracle for deciding whether a run is acceptable."""

    invariant_id: str
    invariant_version: str
    signal: str
    operator: ComparisonOperator
    threshold: float | None = None
    lower: float | None = None
    upper: float | None = None
    persistence_steps: int = 1

    def __post_init__(self) -> None:
        """Validate thresholds for the selected comparison operator."""
        _require_text(self.invariant_id, "invariant ID")
        _require_text(self.invariant_version, "invariant version")
        _require_text(self.signal, "invariant signal")
        object.__setattr__(self, "operator", ComparisonOperator(self.operator))
        if (
            isinstance(self.persistence_steps, bool)
            or not isinstance(self.persistence_steps, int)
            or self.persistence_steps < 1
        ):
            raise ValueError("persistence_steps must be a positive integer")

        if self.operator is ComparisonOperator.BETWEEN_INCLUSIVE:
            if self.threshold is not None or self.lower is None or self.upper is None:
                raise ValueError("between_inclusive requires lower and upper only")
            lower = _finite_float(self.lower, "oracle lower bound")
            upper = _finite_float(self.upper, "oracle upper bound")
            if lower > upper:
                raise ValueError("oracle lower bound must not exceed upper")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
            return

        if self.threshold is None or self.lower is not None or self.upper is not None:
            raise ValueError(f"{self.operator.value} requires threshold only")
        object.__setattr__(
            self,
            "threshold",
            _finite_float(self.threshold, "oracle threshold"),
        )


@dataclass(frozen=True, slots=True)
class RegressionCase:
    """One simulator-neutral scenario variant and its expected result."""

    case_id: str
    role: CaseRole
    scenario_id: str
    scenario_version: str
    parameters: tuple[ScenarioParameter, ...]
    oracles: tuple[InvariantOracle, ...]
    expected_outcome: ExpectedOutcome
    artifact_digests: tuple[NamedDigest, ...] = ()
    source_evidence_event_ids: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        """Validate and canonicalize a regression case."""
        _require_text(self.case_id, "regression case ID")
        _require_text(self.scenario_id, "scenario ID")
        _require_text(self.scenario_version, "scenario version")
        _require_text(self.rationale, "regression case rationale")
        object.__setattr__(self, "role", CaseRole(self.role))
        object.__setattr__(
            self,
            "expected_outcome",
            ExpectedOutcome(self.expected_outcome),
        )
        if (
            self.role is CaseRole.POSITIVE_CONTROL
            and self.expected_outcome is not ExpectedOutcome.VIOLATE_ANY
        ):
            raise ValueError(
                "positive control must expect at least one invariant violation"
            )
        if (
            self.role is CaseRole.NEGATIVE_CONTROL
            and self.expected_outcome is not ExpectedOutcome.SATISFY_ALL
        ):
            raise ValueError(
                "negative control must expect all invariants to be satisfied"
            )
        if not self.oracles:
            raise ValueError("regression case requires at least one oracle")
        _require_unique(
            (item.name for item in self.parameters),
            "scenario parameter names",
        )
        _require_unique(
            (
                f"{item.invariant_id}:{item.invariant_version}:{item.signal}"
                for item in self.oracles
            ),
            "regression case oracles",
        )
        _require_unique(
            (item.name for item in self.artifact_digests),
            "regression case artifacts",
        )
        _require_unique_text(
            self.source_evidence_event_ids,
            "source evidence event IDs",
            required=False,
        )
        object.__setattr__(
            self,
            "parameters",
            tuple(sorted(self.parameters, key=lambda item: item.name)),
        )
        object.__setattr__(
            self,
            "oracles",
            tuple(
                sorted(
                    self.oracles,
                    key=lambda item: (
                        item.invariant_id,
                        item.invariant_version,
                        item.signal,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "artifact_digests",
            tuple(sorted(self.artifact_digests, key=lambda item: item.name)),
        )
        object.__setattr__(
            self,
            "source_evidence_event_ids",
            tuple(sorted(self.source_evidence_event_ids)),
        )


@dataclass(frozen=True, slots=True)
class SeedPanel:
    """Seeds and stochastic pairing strategy for every regression case."""

    seeds: tuple[int, ...]
    strategy: SeedStrategy
    random_stream_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate seeds and pairing metadata."""
        object.__setattr__(self, "strategy", SeedStrategy(self.strategy))
        if not self.seeds:
            raise ValueError("seed panel must not be empty")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds
        ):
            raise ValueError("seed panel values must be integers")
        _require_unique(self.seeds, "seed panel values")
        object.__setattr__(self, "seeds", tuple(sorted(self.seeds)))
        if self.strategy is SeedStrategy.EXACT_REPLAY and len(self.seeds) != 1:
            raise ValueError("exact replay requires exactly one seed")
        if self.strategy is SeedStrategy.COMMON_RANDOM_NUMBERS and len(self.seeds) < 2:
            raise ValueError("common random numbers require at least two seeds")
        if self.random_stream_digest is not None:
            NamedDigest("random stream", self.random_stream_digest)
        if (
            self.strategy is SeedStrategy.COMMON_RANDOM_NUMBERS
            and self.random_stream_digest is None
        ):
            raise ValueError("common random numbers require a random stream digest")


@dataclass(frozen=True, slots=True)
class GateCriterion:
    """Release criterion covering a declared subset of regression cases."""

    name: str
    case_ids: tuple[str, ...]
    method: GateMethod
    minimum_expected_outcome_rate: float
    minimum_trials: int
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate statistical release-gate configuration."""
        _require_text(self.name, "gate criterion name")
        _require_unique_text(self.case_ids, "gate criterion case IDs")
        object.__setattr__(self, "case_ids", tuple(sorted(self.case_ids)))
        object.__setattr__(self, "method", GateMethod(self.method))
        rate = _unit_interval(
            self.minimum_expected_outcome_rate,
            "minimum_expected_outcome_rate",
            include_zero=False,
        )
        confidence = _unit_interval(
            self.confidence_level,
            "confidence_level",
            include_zero=False,
            include_one=False,
        )
        object.__setattr__(self, "minimum_expected_outcome_rate", rate)
        object.__setattr__(self, "confidence_level", confidence)
        if (
            isinstance(self.minimum_trials, bool)
            or not isinstance(self.minimum_trials, int)
            or self.minimum_trials < 1
        ):
            raise ValueError("minimum_trials must be a positive integer")
        if self.method is GateMethod.EXACT and rate != 1.0:
            raise ValueError("exact gate requires an expected outcome rate of 1.0")
        if self.method is GateMethod.WILSON_LOWER_BOUND and self.minimum_trials < 5:
            raise ValueError("Wilson gate requires at least five trials")


@dataclass(frozen=True, slots=True)
class RegressionPack:
    """Portable regression definition derived from one failure bundle."""

    pack_id: str
    pack_version: str
    source_failure_content_digest: str
    source_failure_fingerprint: str
    simulation_manifest_digest: str
    cases: tuple[RegressionCase, ...]
    seed_panel: SeedPanel
    gate_criteria: tuple[GateCriterion, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate scientific coverage and release-gate completeness."""
        _require_text(self.pack_id, "regression pack ID")
        _require_text(self.pack_version, "regression pack version")
        NamedDigest("source failure content", self.source_failure_content_digest)
        NamedDigest("source failure fingerprint", self.source_failure_fingerprint)
        NamedDigest("simulation manifest", self.simulation_manifest_digest)
        _require_unique((item.case_id for item in self.cases), "regression case IDs")
        _require_unique((item.name for item in self.gate_criteria), "gate names")
        _require_unique_text(self.limitations, "regression pack limitations")

        role_counts = {
            role: sum(item.role is role for item in self.cases) for role in CaseRole
        }
        if role_counts[CaseRole.CANONICAL_REPRODUCTION] != 1:
            raise ValueError(
                "regression pack requires exactly one canonical reproduction"
            )
        missing_roles = [
            role.value for role, count in role_counts.items() if count == 0
        ]
        if missing_roles:
            raise ValueError(
                "regression pack is missing required roles: " + ", ".join(missing_roles)
            )

        case_ids = {item.case_id for item in self.cases}
        covered: set[str] = set()
        for criterion in self.gate_criteria:
            unknown = sorted(set(criterion.case_ids) - case_ids)
            if unknown:
                raise ValueError(
                    f"gate '{criterion.name}' references unknown cases: "
                    + ", ".join(unknown)
                )
            available_trials = len(criterion.case_ids) * len(self.seed_panel.seeds)
            if criterion.minimum_trials > available_trials:
                raise ValueError(
                    f"gate '{criterion.name}' requires more trials than the "
                    "pack defines"
                )
            covered.update(criterion.case_ids)
        uncovered = sorted(case_ids - covered)
        if uncovered:
            raise ValueError(
                "regression cases are not covered by a release gate: "
                + ", ".join(uncovered)
            )
        object.__setattr__(
            self,
            "cases",
            tuple(sorted(self.cases, key=lambda item: item.case_id)),
        )
        object.__setattr__(
            self,
            "gate_criteria",
            tuple(sorted(self.gate_criteria, key=lambda item: item.name)),
        )

    def to_json(self) -> str:
        """Serialize the regression pack to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact regression-pack content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class RegressionObservation:
    """Observed invariant outcome for one case and seed."""

    case_id: str
    seed: int
    outcome: ExpectedOutcome
    run_id: str

    def __post_init__(self) -> None:
        """Validate an execution result and canonicalize its outcome."""
        _require_text(self.case_id, "observation case ID")
        _require_text(self.run_id, "observation run ID")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("observation seed must be an integer")
        object.__setattr__(self, "outcome", ExpectedOutcome(self.outcome))


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """Observed conformance and uncertainty for one gate criterion."""

    name: str
    trials: int
    matches: int
    observed_match_rate: float
    lower_confidence_bound: float
    passed: bool


@dataclass(frozen=True, slots=True)
class RegressionGateReport:
    """Deterministic release decision across all gate criteria."""

    pack_id: str
    pack_version: str
    observation_count: int
    criterion_results: tuple[CriterionResult, ...]
    passed: bool
    limitations: tuple[str, ...]
    report_schema_version: str = "iso-obs.regression-gate-report.v1"

    def to_json(self) -> str:
        """Serialize the gate report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact regression-gate report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def evaluate_regression_pack(
    pack: RegressionPack,
    observations: Sequence[RegressionObservation],
) -> RegressionGateReport:
    """Evaluate complete case-by-seed observations against release gates.

    Args:
        pack: Versioned regression definition.
        observations: Exactly one outcome for every case and declared seed.

    Returns:
        Per-criterion diagnostics and an all-criteria release decision.

    Raises:
        ValueError: If observations are duplicate, missing, or outside the pack.
    """
    case_by_id = {item.case_id: item for item in pack.cases}
    expected_keys = {
        (case.case_id, seed) for case in pack.cases for seed in pack.seed_panel.seeds
    }
    observed_by_key: dict[tuple[str, int], RegressionObservation] = {}
    for observation in observations:
        key = (observation.case_id, observation.seed)
        if key in observed_by_key:
            raise ValueError(
                f"duplicate regression observation: {observation.case_id}, "
                f"seed {observation.seed}"
            )
        observed_by_key[key] = observation
    unknown = sorted(set(observed_by_key) - expected_keys)
    if unknown:
        raise ValueError(f"observations are outside the regression pack: {unknown}")
    missing = sorted(expected_keys - set(observed_by_key))
    if missing:
        raise ValueError(f"regression observations are incomplete: {missing}")

    results = []
    for criterion in pack.gate_criteria:
        selected = [
            observation
            for key, observation in observed_by_key.items()
            if key[0] in criterion.case_ids
        ]
        matches = sum(
            item.outcome is case_by_id[item.case_id].expected_outcome
            for item in selected
        )
        trials = len(selected)
        rate = matches / trials
        lower_bound = (
            rate
            if criterion.method is GateMethod.EXACT
            else _wilson_lower_bound(
                matches,
                trials,
                criterion.confidence_level,
            )
        )
        results.append(
            CriterionResult(
                name=criterion.name,
                trials=trials,
                matches=matches,
                observed_match_rate=rate,
                lower_confidence_bound=lower_bound,
                passed=(
                    trials >= criterion.minimum_trials
                    and lower_bound >= criterion.minimum_expected_outcome_rate
                ),
            )
        )
    resolved_results = tuple(results)
    return RegressionGateReport(
        pack_id=pack.pack_id,
        pack_version=pack.pack_version,
        observation_count=len(observations),
        criterion_results=resolved_results,
        passed=all(item.passed for item in resolved_results),
        limitations=(
            "A passing gate supports only the scenarios, seeds, simulator "
            "validity envelope, and invariant versions declared by this pack.",
            "Gate results do not remove simulator, measurement, or model-form "
            "bias recorded in the linked simulation evidence manifest.",
            "Repeated tuning against this regression pack can overfit it; "
            "independent held-out confirmation remains necessary.",
        ),
    )


def _wilson_lower_bound(
    successes: int,
    trials: int,
    confidence_level: float,
) -> float:
    """Calculate a one-sided Wilson score lower confidence bound.

    Args:
        successes: Number of expected outcomes observed.
        trials: Total observations.
        confidence_level: One-sided confidence level.

    Returns:
        Lower confidence bound for the Bernoulli success probability.
    """
    proportion = successes / trials
    z_score = NormalDist().inv_cdf(confidence_level)
    z_squared = z_score**2
    denominator = 1.0 + z_squared / trials
    center = proportion + z_squared / (2.0 * trials)
    radius = z_score * math.sqrt(
        proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials**2)
    )
    return max(0.0, (center - radius) / denominator)


def _finite_float(value: float, label: str) -> float:
    """Convert and require a finite floating-point value.

    Args:
        value: Candidate numeric value.
        label: Human-readable field label.

    Returns:
        Validated floating-point value.
    """
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _unit_interval(
    value: float,
    label: str,
    *,
    include_zero: bool = True,
    include_one: bool = True,
) -> float:
    """Validate a finite probability-like value.

    Args:
        value: Candidate probability.
        label: Human-readable field label.
        include_zero: Whether zero is valid.
        include_one: Whether one is valid.

    Returns:
        Validated floating-point probability.
    """
    resolved = _finite_float(value, label)
    lower_valid = resolved >= 0.0 if include_zero else resolved > 0.0
    upper_valid = resolved <= 1.0 if include_one else resolved < 1.0
    if not lower_valid or not upper_valid:
        raise ValueError(f"{label} must be within the required unit interval")
    return resolved


def _require_text(value: str, label: str) -> None:
    """Require a non-empty string.

    Args:
        value: Candidate text.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_unique(values: Iterable[str | int], label: str) -> None:
    """Require unique values from a sequence or generator.

    Args:
        values: Candidate values.
        label: Human-readable collection label.
    """
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} must be unique")


def _require_unique_text(
    values: Sequence[str],
    label: str,
    *,
    required: bool = True,
) -> None:
    """Require non-empty, unique strings in a sequence.

    Args:
        values: Candidate string sequence.
        label: Human-readable collection label.
        required: Whether at least one value is required.
    """
    if required and not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_text(value, label)
    _require_unique(values, label)
