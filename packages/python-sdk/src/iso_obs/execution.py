"""Evaluate simulator results without conflating system and experiment failure.

Completed traces are evaluated against versioned invariant oracles using
contiguous-step persistence. Simulator failures, missing evidence, identity
mismatches, and absent world-model diagnostics make an experiment incomplete;
they are never counted as ordinary system passes or failures. Out-of-domain
rollouts remain available for discovery but cannot enter release evidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .divergence import TraceStep
from .evidence import NamedDigest, _canonical_json, sha256_digest
from .regression import (
    ComparisonOperator,
    ExpectedOutcome,
    InvariantOracle,
    RegressionGateReport,
    RegressionObservation,
    RegressionPack,
    evaluate_regression_pack,
)
from .simulation import (
    SimulationCapability,
    SimulationExecutionPlan,
    SimulationWorkItem,
)


class SimulationRunStatus(StrEnum):
    """Execution state reported by a simulator adapter."""

    COMPLETED = "completed"
    EXECUTION_FAILED = "execution_failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class CampaignIssueCode(StrEnum):
    """Machine-readable reason campaign evidence is incomplete or inadmissible."""

    PACK_IDENTITY_MISMATCH = "pack_identity_mismatch"
    UNKNOWN_WORK_ITEM = "unknown_work_item"
    DUPLICATE_RESULT = "duplicate_result"
    MISSING_RESULT = "missing_result"
    WORK_ITEM_IDENTITY_MISMATCH = "work_item_identity_mismatch"
    EXECUTION_INCOMPLETE = "execution_incomplete"
    MISSING_SIGNAL = "missing_signal"
    MODEL_DIAGNOSTICS_MISSING = "model_diagnostics_missing"
    MODEL_ROLLOUT_OUT_OF_DOMAIN = "model_rollout_out_of_domain"


@dataclass(frozen=True, slots=True)
class ModelRolloutDiagnostics:
    """Calibrated uncertainty and OOD evidence for a learned rollout."""

    predictive_uncertainty: float
    maximum_supported_uncertainty: float
    out_of_distribution_score: float
    maximum_supported_ood_score: float
    calibration_digest: str

    def __post_init__(self) -> None:
        """Validate finite non-negative scores and calibration identity."""
        for field_name in (
            "predictive_uncertainty",
            "maximum_supported_uncertainty",
            "out_of_distribution_score",
            "maximum_supported_ood_score",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)
        NamedDigest("model diagnostic calibration", self.calibration_digest)

    def is_within_supported_domain(self) -> bool:
        """Determine whether both calibrated diagnostic limits are satisfied.

        Returns:
            True when uncertainty and OOD scores are within their limits.
        """
        return (
            self.predictive_uncertainty <= self.maximum_supported_uncertainty
            and self.out_of_distribution_score <= self.maximum_supported_ood_score
        )


@dataclass(frozen=True, slots=True)
class SimulationRunResult:
    """Raw adapter result for one exact simulation work item."""

    work_item_id: str
    work_item_digest: str
    run_id: str
    status: SimulationRunStatus
    trace: tuple[TraceStep, ...] = ()
    model_diagnostics: ModelRolloutDiagnostics | None = None
    artifact_digests: tuple[NamedDigest, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate execution state, trace coordinates, and artifact identity."""
        _require_text(self.work_item_id, "work item ID")
        _require_text(self.run_id, "run ID")
        NamedDigest("work item", self.work_item_digest)
        object.__setattr__(self, "status", SimulationRunStatus(self.status))
        steps = tuple(item.step for item in self.trace)
        if len(set(steps)) != len(steps):
            raise ValueError("simulation result trace steps must be unique")
        object.__setattr__(
            self,
            "trace",
            tuple(sorted(self.trace, key=lambda item: item.step)),
        )
        names = tuple(item.name for item in self.artifact_digests)
        if len(set(names)) != len(names):
            raise ValueError("simulation result artifact names must be unique")
        object.__setattr__(
            self,
            "artifact_digests",
            tuple(sorted(self.artifact_digests, key=lambda item: item.name)),
        )
        if self.status is SimulationRunStatus.COMPLETED:
            if not self.trace:
                raise ValueError("completed simulation result requires a trace")
            if self.failure_reason is not None:
                raise ValueError(
                    "completed simulation result must not have a failure reason"
                )
        else:
            if self.failure_reason is None:
                raise ValueError(
                    "incomplete simulation result requires a failure reason"
                )
            _require_text(self.failure_reason, "simulation failure reason")


@dataclass(frozen=True, slots=True)
class OracleEvaluation:
    """Sustained-violation result for one invariant oracle."""

    invariant_id: str
    invariant_version: str
    signal: str
    satisfied: bool
    first_violation_step: int | None
    confirmed_violation_step: int | None
    worst_value: float
    evidence_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkItemEvaluation:
    """Evidence-linked behavioral result for one completed work item."""

    work_item_id: str
    run_id: str
    trace_digest: str
    oracle_evaluations: tuple[OracleEvaluation, ...]
    observed_outcome: ExpectedOutcome
    expected_outcome: ExpectedOutcome
    matches_expected_outcome: bool
    within_supported_model_domain: bool | None


@dataclass(frozen=True, slots=True)
class CampaignIssue:
    """Structured problem affecting campaign completeness or admissibility."""

    code: CampaignIssueCode
    work_item_ids: tuple[str, ...]
    detail: str
    blocks_completion: bool
    blocks_release: bool

    def __post_init__(self) -> None:
        """Validate and canonicalize a campaign issue."""
        object.__setattr__(self, "code", CampaignIssueCode(self.code))
        _require_text(self.detail, "campaign issue detail")
        _require_unique_text(
            self.work_item_ids,
            "campaign issue work item IDs",
            required=False,
        )
        if not isinstance(self.blocks_completion, bool):
            raise ValueError("blocks_completion must be a boolean")
        if not isinstance(self.blocks_release, bool):
            raise ValueError("blocks_release must be a boolean")
        object.__setattr__(
            self,
            "work_item_ids",
            tuple(sorted(self.work_item_ids)),
        )


@dataclass(frozen=True, slots=True)
class SimulationCampaignReport:
    """Aggregate integrity, behavioral, and release-gate evaluation."""

    plan_content_digest: str
    pack_content_digest: str
    result_count: int
    work_item_evaluations: tuple[WorkItemEvaluation, ...]
    issues: tuple[CampaignIssue, ...]
    execution_complete: bool
    release_evidence_eligible: bool
    regression_gate_report: RegressionGateReport | None
    limitations: tuple[str, ...]
    report_schema_version: str = "iso-obs.simulation-campaign-report.v1"

    def to_json(self) -> str:
        """Serialize the campaign report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact campaign-report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def simulation_work_item_digest(item: SimulationWorkItem) -> str:
    """Calculate the exact digest an adapter must echo in its result.

    Args:
        item: Planned simulation work item.

    Returns:
        A prefixed lowercase SHA-256 digest.
    """
    return sha256_digest(_canonical_json(item))


def evaluate_simulation_campaign(
    plan: SimulationExecutionPlan,
    pack: RegressionPack,
    results: Sequence[SimulationRunResult],
) -> SimulationCampaignReport:
    """Evaluate simulator results and conditionally calculate a release gate.

    Args:
        plan: Exact simulator execution plan.
        pack: Regression pack linked by the plan.
        results: Adapter results to validate and evaluate.

    Returns:
        Campaign integrity issues, behavioral outcomes, and an optional gate.
    """
    issues: list[CampaignIssue] = []
    pack_digest = pack.content_digest()
    if pack_digest != plan.pack_content_digest:
        issues.append(
            CampaignIssue(
                code=CampaignIssueCode.PACK_IDENTITY_MISMATCH,
                work_item_ids=(),
                detail="Regression pack content does not match the execution plan.",
                blocks_completion=True,
                blocks_release=True,
            )
        )

    item_by_id = {item.work_item_id: item for item in plan.work_items}
    result_by_id: dict[str, SimulationRunResult] = {}
    for result in results:
        if result.work_item_id not in item_by_id:
            issues.append(
                CampaignIssue(
                    code=CampaignIssueCode.UNKNOWN_WORK_ITEM,
                    work_item_ids=(result.work_item_id,),
                    detail="Result does not belong to the execution plan.",
                    blocks_completion=True,
                    blocks_release=True,
                )
            )
            continue
        if result.work_item_id in result_by_id:
            issues.append(
                CampaignIssue(
                    code=CampaignIssueCode.DUPLICATE_RESULT,
                    work_item_ids=(result.work_item_id,),
                    detail="More than one result was supplied for a work item.",
                    blocks_completion=True,
                    blocks_release=True,
                )
            )
            continue
        result_by_id[result.work_item_id] = result

    missing_ids = tuple(sorted(set(item_by_id) - set(result_by_id)))
    if missing_ids:
        issues.append(
            CampaignIssue(
                code=CampaignIssueCode.MISSING_RESULT,
                work_item_ids=missing_ids,
                detail="The campaign does not contain every planned result.",
                blocks_completion=True,
                blocks_release=True,
            )
        )

    requires_model_diagnostics = {
        SimulationCapability.UNCERTAINTY_QUANTIFICATION,
        SimulationCapability.OUT_OF_DISTRIBUTION_DETECTION,
    }.issubset(set(plan.required_capabilities))
    evaluations: list[WorkItemEvaluation] = []
    if pack_digest == plan.pack_content_digest:
        for item in plan.work_items:
            run_result = result_by_id.get(item.work_item_id)
            if run_result is None:
                continue
            expected_digest = simulation_work_item_digest(item)
            if run_result.work_item_digest != expected_digest:
                issues.append(
                    CampaignIssue(
                        code=CampaignIssueCode.WORK_ITEM_IDENTITY_MISMATCH,
                        work_item_ids=(item.work_item_id,),
                        detail=(
                            "Result work-item digest does not match the planned "
                            "scenario, seed, adapter, and evidence identity."
                        ),
                        blocks_completion=True,
                        blocks_release=True,
                    )
                )
                continue
            if run_result.status is not SimulationRunStatus.COMPLETED:
                issues.append(
                    CampaignIssue(
                        code=CampaignIssueCode.EXECUTION_INCOMPLETE,
                        work_item_ids=(item.work_item_id,),
                        detail=(
                            "Simulator execution ended as "
                            f"{run_result.status.value}: "
                            f"{run_result.failure_reason}"
                        ),
                        blocks_completion=True,
                        blocks_release=True,
                    )
                )
                continue

            within_domain: bool | None = None
            if requires_model_diagnostics:
                if run_result.model_diagnostics is None:
                    issues.append(
                        CampaignIssue(
                            code=CampaignIssueCode.MODEL_DIAGNOSTICS_MISSING,
                            work_item_ids=(item.work_item_id,),
                            detail=(
                                "Learned-backend result is missing calibrated "
                                "uncertainty and OOD diagnostics."
                            ),
                            blocks_completion=True,
                            blocks_release=True,
                        )
                    )
                    continue
                within_domain = (
                    run_result.model_diagnostics.is_within_supported_domain()
                )
                if not within_domain:
                    issues.append(
                        CampaignIssue(
                            code=CampaignIssueCode.MODEL_ROLLOUT_OUT_OF_DOMAIN,
                            work_item_ids=(item.work_item_id,),
                            detail=(
                                "World-model uncertainty or OOD score exceeds "
                                "its calibrated support limit."
                            ),
                            blocks_completion=False,
                            blocks_release=True,
                        )
                    )

            try:
                evaluations.append(
                    _evaluate_work_item(
                        item,
                        run_result,
                        within_supported_model_domain=within_domain,
                    )
                )
            except _MissingSignalError as error:
                issues.append(
                    CampaignIssue(
                        code=CampaignIssueCode.MISSING_SIGNAL,
                        work_item_ids=(item.work_item_id,),
                        detail=str(error),
                        blocks_completion=True,
                        blocks_release=True,
                    )
                )

    resolved_issues = tuple(
        sorted(
            issues,
            key=lambda item: (
                item.code.value,
                item.work_item_ids,
                item.detail,
            ),
        )
    )
    execution_complete = len(evaluations) == len(plan.work_items) and not any(
        item.blocks_completion for item in resolved_issues
    )
    release_eligible = (
        execution_complete
        and plan.eligible_for_release_gate
        and not any(item.blocks_release for item in resolved_issues)
    )
    gate_report = None
    if release_eligible:
        observations = tuple(
            RegressionObservation(
                case_id=item_by_id[evaluation.work_item_id].case_id,
                seed=item_by_id[evaluation.work_item_id].seed,
                outcome=evaluation.observed_outcome,
                run_id=evaluation.run_id,
            )
            for evaluation in evaluations
        )
        gate_report = evaluate_regression_pack(pack, observations)

    return SimulationCampaignReport(
        plan_content_digest=plan.content_digest(),
        pack_content_digest=pack_digest,
        result_count=len(results),
        work_item_evaluations=tuple(evaluations),
        issues=resolved_issues,
        execution_complete=execution_complete,
        release_evidence_eligible=release_eligible,
        regression_gate_report=gate_report,
        limitations=(
            "A complete campaign establishes only behavior within the exact "
            "plan, invariant versions, and linked simulator validity envelope.",
            "Simulator completion does not establish physical-world validity; "
            "model-form, measurement, and transfer uncertainty remain.",
            "Out-of-domain model rollouts may guide discovery but require "
            "independent in-domain simulation or physical confirmation.",
        ),
    )


def _evaluate_work_item(
    item: SimulationWorkItem,
    result: SimulationRunResult,
    *,
    within_supported_model_domain: bool | None,
) -> WorkItemEvaluation:
    """Evaluate all invariant oracles for a completed trace.

    Args:
        item: Planned work item containing expected oracles.
        result: Identity-verified completed simulator result.
        within_supported_model_domain: Learned-model diagnostic outcome.

    Returns:
        Evidence-linked behavioral outcome for the work item.
    """
    oracle_evaluations = tuple(
        _evaluate_oracle(oracle, result.trace) for oracle in item.oracles
    )
    observed_outcome = (
        ExpectedOutcome.VIOLATE_ANY
        if any(not evaluation.satisfied for evaluation in oracle_evaluations)
        else ExpectedOutcome.SATISFY_ALL
    )
    return WorkItemEvaluation(
        work_item_id=item.work_item_id,
        run_id=result.run_id,
        trace_digest=sha256_digest(_canonical_json(result.trace)),
        oracle_evaluations=oracle_evaluations,
        observed_outcome=observed_outcome,
        expected_outcome=item.expected_outcome,
        matches_expected_outcome=observed_outcome is item.expected_outcome,
        within_supported_model_domain=within_supported_model_domain,
    )


def _evaluate_oracle(
    oracle: InvariantOracle,
    trace: tuple[TraceStep, ...],
) -> OracleEvaluation:
    """Evaluate one oracle with contiguous-step persistence.

    Args:
        oracle: Versioned numeric invariant.
        trace: Ordered simulator trace.

    Returns:
        Sustained-violation timing, worst value, and supporting event IDs.

    Raises:
        _MissingSignalError: If any trace step lacks the oracle signal.
    """
    missing_steps = tuple(
        step.step for step in trace if oracle.signal not in step.values
    )
    if missing_steps:
        raise _MissingSignalError(oracle.signal, missing_steps)

    values = tuple((step.step, step.values[oracle.signal], step) for step in trace)
    worst_value = _worst_value(oracle, tuple(value for _, value, _ in values))
    window: list[tuple[int, TraceStep]] = []
    first_step: int | None = None
    confirmed_step: int | None = None
    evidence_ids: tuple[str, ...] = ()
    previous_step: int | None = None
    for step_number, value, step in values:
        violates = _violates(oracle, value)
        contiguous = previous_step is not None and step_number == previous_step + 1
        if violates:
            if not contiguous:
                window = []
            window.append((step_number, step))
            if len(window) > oracle.persistence_steps:
                window.pop(0)
            if len(window) == oracle.persistence_steps:
                first_step = window[0][0]
                confirmed_step = window[-1][0]
                evidence_ids = tuple(
                    event_id
                    for _, evidence_step in window
                    if (event_id := evidence_step.evidence_event_ids.get(oracle.signal))
                    is not None
                )
                break
        else:
            window = []
        previous_step = step_number

    return OracleEvaluation(
        invariant_id=oracle.invariant_id,
        invariant_version=oracle.invariant_version,
        signal=oracle.signal,
        satisfied=confirmed_step is None,
        first_violation_step=first_step,
        confirmed_violation_step=confirmed_step,
        worst_value=worst_value,
        evidence_event_ids=evidence_ids,
    )


def _violates(oracle: InvariantOracle, value: float) -> bool:
    """Determine whether one numeric value violates an oracle.

    Args:
        oracle: Numeric invariant definition.
        value: Observed signal value.

    Returns:
        True when the value is outside the accepted region.
    """
    if oracle.operator is ComparisonOperator.LESS_THAN_OR_EQUAL:
        return value > _required_threshold(oracle)
    if oracle.operator is ComparisonOperator.GREATER_THAN_OR_EQUAL:
        return value < _required_threshold(oracle)
    return not (_required_lower(oracle) <= value <= _required_upper(oracle))


def _worst_value(oracle: InvariantOracle, values: tuple[float, ...]) -> float:
    """Select the observed value with the greatest invariant risk.

    Args:
        oracle: Numeric invariant definition.
        values: Complete signal values.

    Returns:
        Maximum, minimum, or farthest-outside value as appropriate.
    """
    if oracle.operator is ComparisonOperator.LESS_THAN_OR_EQUAL:
        return max(values)
    if oracle.operator is ComparisonOperator.GREATER_THAN_OR_EQUAL:
        return min(values)
    lower = _required_lower(oracle)
    upper = _required_upper(oracle)
    return max(
        values,
        key=lambda value: max(lower - value, value - upper, 0.0),
    )


def _required_threshold(oracle: InvariantOracle) -> float:
    """Return a threshold already guaranteed by oracle validation."""
    if oracle.threshold is None:
        raise RuntimeError("threshold oracle is internally inconsistent")
    return oracle.threshold


def _required_lower(oracle: InvariantOracle) -> float:
    """Return a lower bound already guaranteed by oracle validation."""
    if oracle.lower is None:
        raise RuntimeError("bounded oracle is internally inconsistent")
    return oracle.lower


def _required_upper(oracle: InvariantOracle) -> float:
    """Return an upper bound already guaranteed by oracle validation."""
    if oracle.upper is None:
        raise RuntimeError("bounded oracle is internally inconsistent")
    return oracle.upper


class _MissingSignalError(ValueError):
    """Internal structured missing-signal failure."""

    def __init__(self, signal: str, steps: tuple[int, ...]) -> None:
        """Describe missing signal evidence.

        Args:
            signal: Required oracle signal.
            steps: Trace steps missing that signal.
        """
        self.signal = signal
        self.steps = steps
        super().__init__(
            f"Signal '{signal}' is missing at trace steps: "
            + ", ".join(str(step) for step in steps)
        )


def _require_text(value: str, label: str) -> None:
    """Require a non-empty string.

    Args:
        value: Candidate text.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")


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
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
