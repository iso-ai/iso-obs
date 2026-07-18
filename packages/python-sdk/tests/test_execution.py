"""Tests for evidence-safe simulator result evaluation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.divergence import TraceStep
from iso_obs.evidence import sha256_digest
from iso_obs.execution import (
    CampaignIssueCode,
    ModelRolloutDiagnostics,
    SimulationRunResult,
    SimulationRunStatus,
    evaluate_simulation_campaign,
    simulation_work_item_digest,
)
from iso_obs.regression import (
    CaseRole,
    ComparisonOperator,
    ExpectedOutcome,
    GateCriterion,
    GateMethod,
    InvariantOracle,
    RegressionCase,
    RegressionPack,
    ScenarioParameter,
    SeedPanel,
    SeedStrategy,
)
from iso_obs.simulation import (
    BackendType,
    EvidenceUse,
    ExecutionIntent,
    SimulationAdapterManifest,
    SimulationCapability,
    SimulationExecutionPlan,
    SimulationWorkItem,
    compile_simulation_plan,
)


def regression_pack() -> RegressionPack:
    """Build a five-role exact-replay regression pack."""
    cases = tuple(
        RegressionCase(
            case_id=f"{index:02d}-{role.value}",
            role=role,
            scenario_id="blind-intersection",
            scenario_version="3",
            parameters=(ScenarioParameter("latency_ms", 100 + index, "ms"),),
            oracles=(
                InvariantOracle(
                    invariant_id="minimum-stopping-distance",
                    invariant_version="2",
                    signal="stopping_margin_m",
                    operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                    threshold=0.25,
                    persistence_steps=2,
                ),
            ),
            expected_outcome=(
                ExpectedOutcome.VIOLATE_ANY
                if role is CaseRole.POSITIVE_CONTROL
                else ExpectedOutcome.SATISFY_ALL
            ),
            rationale=f"Exercise {role.value}.",
        )
        for index, role in enumerate(CaseRole)
    )
    return RegressionPack(
        pack_id="stopping-margin",
        pack_version="1",
        source_failure_content_digest=sha256_digest("failure"),
        source_failure_fingerprint=sha256_digest("fingerprint"),
        simulation_manifest_digest=sha256_digest("simulation"),
        cases=cases,
        seed_panel=SeedPanel(seeds=(42,), strategy=SeedStrategy.EXACT_REPLAY),
        gate_criteria=(
            GateCriterion(
                name="all-cases",
                case_ids=tuple(item.case_id for item in cases),
                method=GateMethod.WILSON_LOWER_BOUND,
                minimum_expected_outcome_rate=0.50,
                minimum_trials=5,
            ),
        ),
        limitations=("Indoor low-speed operation only.",),
    )


def adapter(
    pack: RegressionPack,
    *,
    backend_type: BackendType = BackendType.PHYSICS_SIMULATOR,
    model_capabilities: bool = False,
) -> SimulationAdapterManifest:
    """Build an adapter compatible with the test pack."""
    capabilities = [
        SimulationCapability.SEEDED_RESET,
        SimulationCapability.PARAMETER_OVERRIDE,
        SimulationCapability.INVARIANT_SIGNAL_EXPORT,
    ]
    if model_capabilities:
        capabilities.extend(
            [
                SimulationCapability.UNCERTAINTY_QUANTIFICATION,
                SimulationCapability.OUT_OF_DISTRIBUTION_DETECTION,
            ]
        )
    return SimulationAdapterManifest(
        adapter_name="warehouse-adapter",
        adapter_version="1",
        backend_type=backend_type,
        simulator_name="warehouse-twin",
        simulator_version="4",
        simulation_evidence_manifest_digest=pack.simulation_manifest_digest,
        capabilities=tuple(capabilities),
        supported_parameters=("latency_ms",),
        supported_artifacts=(),
        max_parallelism=2,
    )


def execution_plan(
    pack: RegressionPack | None = None,
    *,
    backend_type: BackendType = BackendType.PHYSICS_SIMULATOR,
    intent: ExecutionIntent = ExecutionIntent.REPRODUCTION,
    evidence_use: EvidenceUse = EvidenceUse.CONFIRMATORY,
    model_capabilities: bool = False,
) -> tuple[RegressionPack, SimulationExecutionPlan]:
    """Compile a plan with requested backend and evidence semantics."""
    resolved_pack = pack or regression_pack()
    resolved_adapter = adapter(
        resolved_pack,
        backend_type=backend_type,
        model_capabilities=model_capabilities,
    )
    if intent is ExecutionIntent.CROSS_BACKEND_VALIDATION:
        resolved_adapter = replace(
            resolved_adapter,
            simulation_evidence_manifest_digest=sha256_digest("other-backend"),
        )
    plan = compile_simulation_plan(
        resolved_pack,
        resolved_adapter,
        intent=intent,
        evidence_use=evidence_use,
    )
    return resolved_pack, plan


def trace(
    *values: float, steps: tuple[int, ...] | None = None
) -> tuple[TraceStep, ...]:
    """Build signal evidence at explicit or sequential steps."""
    resolved_steps = steps or tuple(range(len(values)))
    return tuple(
        TraceStep(
            step=step,
            values={"stopping_margin_m": value},
            evidence_event_ids={"stopping_margin_m": f"evt-{step}"},
        )
        for step, value in zip(resolved_steps, values, strict=True)
    )


def model_diagnostics(*, in_domain: bool = True) -> ModelRolloutDiagnostics:
    """Build calibrated learned-rollout diagnostics."""
    return ModelRolloutDiagnostics(
        predictive_uncertainty=0.1 if in_domain else 0.8,
        maximum_supported_uncertainty=0.5,
        out_of_distribution_score=0.2,
        maximum_supported_ood_score=0.5,
        calibration_digest=sha256_digest("calibration"),
    )


def result_for(
    item: SimulationWorkItem,
    *,
    violating: bool | None = None,
    status: SimulationRunStatus = SimulationRunStatus.COMPLETED,
    diagnostics: ModelRolloutDiagnostics | None = None,
) -> SimulationRunResult:
    """Build a result matching one planned work item."""
    work_item = item
    expected_violation = work_item.expected_outcome is ExpectedOutcome.VIOLATE_ANY
    resolved_violation = expected_violation if violating is None else violating
    result_trace = trace(0.5, 0.1, 0.1) if resolved_violation else trace(0.5, 0.5, 0.5)
    return SimulationRunResult(
        work_item_id=work_item.work_item_id,
        work_item_digest=simulation_work_item_digest(work_item),
        run_id=f"run-{work_item.work_item_id}",
        status=status,
        trace=result_trace,
        model_diagnostics=diagnostics,
        failure_reason=None if status is SimulationRunStatus.COMPLETED else "timeout",
    )


def complete_results(
    plan: SimulationExecutionPlan,
    *,
    diagnostics: ModelRolloutDiagnostics | None = None,
) -> tuple[SimulationRunResult, ...]:
    """Build one expected result for every planned work item."""
    return tuple(result_for(item, diagnostics=diagnostics) for item in plan.work_items)


def test_complete_campaign_produces_passing_release_gate() -> None:
    pack, plan = execution_plan()

    report = evaluate_simulation_campaign(
        plan,
        pack,
        complete_results(plan),
    )

    assert report.execution_complete
    assert report.release_evidence_eligible
    assert report.regression_gate_report is not None
    assert report.regression_gate_report.passed
    assert len(report.work_item_evaluations) == 5
    assert all(item.matches_expected_outcome for item in report.work_item_evaluations)
    assert not report.issues


def test_campaign_report_is_input_order_invariant() -> None:
    pack, plan = execution_plan()
    results = complete_results(plan)

    forward = evaluate_simulation_campaign(plan, pack, results)
    reverse = evaluate_simulation_campaign(plan, pack, tuple(reversed(results)))

    assert forward == reverse
    assert forward.content_digest() == reverse.content_digest()


def test_valid_system_regression_fails_gate_without_invalidating_evidence() -> None:
    pack, plan = execution_plan()
    results = list(complete_results(plan))
    negative = next(
        item for item in plan.work_items if item.role is CaseRole.NEGATIVE_CONTROL
    )
    index = plan.work_items.index(negative)
    results[index] = result_for(negative, violating=True)

    report = evaluate_simulation_campaign(plan, pack, results)

    assert report.execution_complete
    assert report.release_evidence_eligible
    assert report.regression_gate_report is not None
    assert not report.regression_gate_report.passed
    evaluation = next(
        item
        for item in report.work_item_evaluations
        if item.work_item_id == negative.work_item_id
    )
    assert not evaluation.matches_expected_outcome


def test_sustained_violation_reports_first_and_confirmed_steps() -> None:
    pack, plan = execution_plan()
    positive = next(
        item for item in plan.work_items if item.role is CaseRole.POSITIVE_CONTROL
    )

    report = evaluate_simulation_campaign(
        plan,
        pack,
        complete_results(plan),
    )
    evaluation = next(
        item
        for item in report.work_item_evaluations
        if item.work_item_id == positive.work_item_id
    )
    oracle = evaluation.oracle_evaluations[0]

    assert not oracle.satisfied
    assert oracle.first_violation_step == 1
    assert oracle.confirmed_violation_step == 2
    assert oracle.worst_value == 0.1
    assert oracle.evidence_event_ids == ("evt-1", "evt-2")


def test_isolated_violation_does_not_satisfy_persistence() -> None:
    pack, plan = execution_plan()
    item = next(work for work in plan.work_items if work.role is CaseRole.NEIGHBORHOOD)
    results = list(complete_results(plan))
    index = plan.work_items.index(item)
    results[index] = replace(results[index], trace=trace(0.5, 0.1, 0.5))

    report = evaluate_simulation_campaign(plan, pack, tuple(results))
    evaluation = next(
        value
        for value in report.work_item_evaluations
        if value.work_item_id == item.work_item_id
    )

    assert evaluation.oracle_evaluations[0].satisfied


def test_step_gap_breaks_violation_persistence() -> None:
    pack, plan = execution_plan()
    item = next(work for work in plan.work_items if work.role is CaseRole.NEIGHBORHOOD)
    results = list(complete_results(plan))
    index = plan.work_items.index(item)
    results[index] = replace(
        results[index],
        trace=trace(0.1, 0.1, steps=(0, 2)),
    )

    report = evaluate_simulation_campaign(plan, pack, tuple(results))
    evaluation = next(
        value
        for value in report.work_item_evaluations
        if value.work_item_id == item.work_item_id
    )

    assert evaluation.oracle_evaluations[0].satisfied


def test_simulator_failure_is_not_counted_as_system_failure() -> None:
    pack, plan = execution_plan()
    results = list(complete_results(plan))
    results[0] = result_for(
        plan.work_items[0],
        status=SimulationRunStatus.EXECUTION_FAILED,
    )

    report = evaluate_simulation_campaign(plan, pack, tuple(results))

    assert not report.execution_complete
    assert not report.release_evidence_eligible
    assert report.regression_gate_report is None
    assert CampaignIssueCode.EXECUTION_INCOMPLETE in {
        item.code for item in report.issues
    }
    assert len(report.work_item_evaluations) == 4


def test_missing_signal_makes_experiment_incomplete() -> None:
    pack, plan = execution_plan()
    results = list(complete_results(plan))
    results[0] = replace(
        results[0],
        trace=(
            TraceStep(step=0, values={}, evidence_event_ids={}),
            TraceStep(step=1, values={}, evidence_event_ids={}),
        ),
    )

    report = evaluate_simulation_campaign(plan, pack, tuple(results))

    assert not report.execution_complete
    assert CampaignIssueCode.MISSING_SIGNAL in {item.code for item in report.issues}
    assert "trace steps: 0, 1" in next(
        item.detail
        for item in report.issues
        if item.code is CampaignIssueCode.MISSING_SIGNAL
    )


def test_work_item_digest_mismatch_blocks_evidence() -> None:
    pack, plan = execution_plan()
    results = list(complete_results(plan))
    results[0] = replace(
        results[0],
        work_item_digest=sha256_digest("tampered"),
    )

    report = evaluate_simulation_campaign(plan, pack, tuple(results))

    assert CampaignIssueCode.WORK_ITEM_IDENTITY_MISMATCH in {
        item.code for item in report.issues
    }
    assert not report.execution_complete


def test_unknown_duplicate_and_missing_results_are_distinct() -> None:
    pack, plan = execution_plan()
    results = list(complete_results(plan))
    removed = results.pop()
    results.append(results[0])
    results.append(replace(removed, work_item_id="work_unknown"))

    report = evaluate_simulation_campaign(plan, pack, tuple(results))

    assert {
        CampaignIssueCode.UNKNOWN_WORK_ITEM,
        CampaignIssueCode.DUPLICATE_RESULT,
        CampaignIssueCode.MISSING_RESULT,
    }.issubset({item.code for item in report.issues})


def test_pack_identity_mismatch_blocks_all_evaluation() -> None:
    pack, plan = execution_plan()
    changed_pack = replace(pack, pack_version="2")

    report = evaluate_simulation_campaign(
        plan,
        changed_pack,
        complete_results(plan),
    )

    assert CampaignIssueCode.PACK_IDENTITY_MISMATCH in {
        item.code for item in report.issues
    }
    assert not report.work_item_evaluations
    assert not report.execution_complete


def test_world_model_result_requires_diagnostics() -> None:
    pack, plan = execution_plan(
        backend_type=BackendType.WORLD_MODEL,
        intent=ExecutionIntent.CROSS_BACKEND_VALIDATION,
        evidence_use=EvidenceUse.CONFIRMATORY,
        model_capabilities=True,
    )

    report = evaluate_simulation_campaign(
        plan,
        pack,
        complete_results(plan),
    )

    assert CampaignIssueCode.MODEL_DIAGNOSTICS_MISSING in {
        item.code for item in report.issues
    }
    assert not report.execution_complete


def test_in_domain_world_model_can_produce_confirmatory_gate() -> None:
    pack, plan = execution_plan(
        backend_type=BackendType.WORLD_MODEL,
        intent=ExecutionIntent.CROSS_BACKEND_VALIDATION,
        evidence_use=EvidenceUse.CONFIRMATORY,
        model_capabilities=True,
    )

    report = evaluate_simulation_campaign(
        plan,
        pack,
        complete_results(plan, diagnostics=model_diagnostics()),
    )

    assert report.execution_complete
    assert report.release_evidence_eligible
    assert report.regression_gate_report is not None
    assert all(
        item.within_supported_model_domain for item in report.work_item_evaluations
    )


def test_out_of_domain_model_rollout_remains_discovery_only() -> None:
    pack, plan = execution_plan(
        backend_type=BackendType.WORLD_MODEL,
        intent=ExecutionIntent.SURROGATE_SCREENING,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        model_capabilities=True,
    )

    report = evaluate_simulation_campaign(
        plan,
        pack,
        complete_results(plan, diagnostics=model_diagnostics(in_domain=False)),
    )

    assert report.execution_complete
    assert not report.release_evidence_eligible
    assert report.regression_gate_report is None
    assert CampaignIssueCode.MODEL_ROLLOUT_OUT_OF_DOMAIN in {
        item.code for item in report.issues
    }
    assert all(
        item.within_supported_model_domain is False
        for item in report.work_item_evaluations
    )


def test_discovery_plan_never_generates_release_gate() -> None:
    pack, plan = execution_plan(
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
    )

    report = evaluate_simulation_campaign(
        plan,
        pack,
        complete_results(plan),
    )

    assert report.execution_complete
    assert not report.release_evidence_eligible
    assert report.regression_gate_report is None


def test_completed_result_requires_trace_and_no_failure_reason() -> None:
    _, plan = execution_plan()
    item = plan.work_items[0]

    with pytest.raises(ValueError, match="requires a trace"):
        SimulationRunResult(
            work_item_id=item.work_item_id,
            work_item_digest=simulation_work_item_digest(item),
            run_id="run",
            status=SimulationRunStatus.COMPLETED,
        )
    with pytest.raises(ValueError, match="must not have a failure reason"):
        replace(result_for(item), failure_reason="should not exist")


def test_incomplete_result_requires_failure_reason() -> None:
    _, plan = execution_plan()
    item = plan.work_items[0]

    with pytest.raises(ValueError, match="requires a failure reason"):
        replace(
            result_for(item),
            status=SimulationRunStatus.TIMED_OUT,
            failure_reason=None,
        )


def test_duplicate_trace_steps_are_rejected() -> None:
    _, plan = execution_plan()
    item = plan.work_items[0]
    duplicate_steps = (
        TraceStep(
            step=0,
            values={"stopping_margin_m": 0.5},
            evidence_event_ids={},
        ),
        TraceStep(
            step=0,
            values={"stopping_margin_m": 0.4},
            evidence_event_ids={},
        ),
    )

    with pytest.raises(ValueError, match="steps must be unique"):
        replace(result_for(item), trace=duplicate_steps)


def test_invalid_model_diagnostics_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(model_diagnostics(), predictive_uncertainty=-0.1)
    with pytest.raises(ValueError, match="sha256"):
        replace(model_diagnostics(), calibration_digest="unversioned")


def test_trace_digest_changes_with_behavior() -> None:
    pack, plan = execution_plan()
    original_results = complete_results(plan)
    changed_results = list(original_results)
    changed_results[0] = replace(
        changed_results[0],
        trace=trace(0.6, 0.6, 0.6),
    )

    original = evaluate_simulation_campaign(plan, pack, original_results)
    changed = evaluate_simulation_campaign(plan, pack, tuple(changed_results))

    assert (
        original.work_item_evaluations[0].trace_digest
        != changed.work_item_evaluations[0].trace_digest
    )


def test_report_states_scope_physical_validity_and_ood_limitations() -> None:
    pack, plan = execution_plan()

    report = evaluate_simulation_campaign(
        plan,
        pack,
        complete_results(plan),
    )

    assert "exact plan" in report.limitations[0]
    assert "physical-world validity" in report.limitations[1]
    assert "Out-of-domain" in report.limitations[2]
