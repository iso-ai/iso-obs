"""Tests for deterministic simulator execution-plan compilation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.evidence import NamedDigest, sha256_digest
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
    CompatibilityIssueCode,
    EvidenceUse,
    ExecutionIntent,
    IncompatibleSimulationAdapterError,
    SimulationAdapterManifest,
    SimulationCapability,
    assess_adapter_compatibility,
    compile_simulation_plan,
)


def regression_pack() -> RegressionPack:
    """Build a compact pack spanning every required scientific role."""
    pack_cases = tuple(
        RegressionCase(
            case_id=f"{index:02d}-{role.value}",
            role=role,
            scenario_id="blind-intersection",
            scenario_version="3",
            parameters=(ScenarioParameter("latency_ms", 100 + index * 5, "ms"),),
            oracles=(
                InvariantOracle(
                    invariant_id="minimum-stopping-distance",
                    invariant_version="2",
                    signal="stopping_margin_m",
                    operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                    threshold=0.25,
                ),
            ),
            expected_outcome=(
                ExpectedOutcome.VIOLATE_ANY
                if role is CaseRole.POSITIVE_CONTROL
                else ExpectedOutcome.SATISFY_ALL
            ),
            artifact_digests=(
                NamedDigest("scenario", sha256_digest(f"scenario-{index}")),
            ),
            source_evidence_event_ids=(f"evt-{index}",),
            rationale=f"Exercise {role.value}.",
        )
        for index, role in enumerate(CaseRole)
    )
    return RegressionPack(
        pack_id="blind-intersection-stop-margin",
        pack_version="1",
        source_failure_content_digest=sha256_digest("failure"),
        source_failure_fingerprint=sha256_digest("fingerprint"),
        simulation_manifest_digest=sha256_digest("simulation"),
        cases=pack_cases,
        seed_panel=SeedPanel(
            seeds=(10, 20),
            strategy=SeedStrategy.COMMON_RANDOM_NUMBERS,
            random_stream_digest=sha256_digest("streams"),
        ),
        gate_criteria=(
            GateCriterion(
                name="all-cases",
                case_ids=tuple(item.case_id for item in pack_cases),
                method=GateMethod.WILSON_LOWER_BOUND,
                minimum_expected_outcome_rate=0.5,
                minimum_trials=10,
            ),
        ),
        limitations=("Low-speed indoor operation only.",),
    )


def adapter(
    *,
    backend_type: BackendType = BackendType.PHYSICS_SIMULATOR,
    capabilities: tuple[SimulationCapability, ...] | None = None,
    evidence_digest: str | None = None,
) -> SimulationAdapterManifest:
    """Build a fully capable test adapter."""
    resolved_capabilities = (
        capabilities
        if capabilities is not None
        else (
            SimulationCapability.SEEDED_RESET,
            SimulationCapability.PARAMETER_OVERRIDE,
            SimulationCapability.ARTIFACT_LOADING,
            SimulationCapability.RANDOM_STREAM_CONTROL,
            SimulationCapability.INVARIANT_SIGNAL_EXPORT,
        )
    )
    return SimulationAdapterManifest(
        adapter_name="warehouse-adapter",
        adapter_version="1.2.0",
        backend_type=backend_type,
        simulator_name="warehouse-twin",
        simulator_version="4.2.0",
        simulation_evidence_manifest_digest=(
            evidence_digest or regression_pack().simulation_manifest_digest
        ),
        capabilities=resolved_capabilities,
        supported_parameters=("latency_ms",),
        supported_artifacts=("scenario",),
        max_parallelism=3,
        limitations=("Contact severity is not validated.",),
    )


def test_compatible_pack_compiles_to_complete_deterministic_plan() -> None:
    pack = regression_pack()
    manifest = adapter(evidence_digest=pack.simulation_manifest_digest)

    first = compile_simulation_plan(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )
    second = compile_simulation_plan(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    assert first == second
    assert first.content_digest() == second.content_digest()
    assert first.eligible_for_release_gate
    assert len(first.work_items) == 10
    assert len(first.batches) == 4
    assert [len(batch.work_items) for batch in first.batches] == [3, 3, 3, 1]
    assert len({item.work_item_id for item in first.work_items}) == 10
    assert all(
        item.pack_content_digest == pack.content_digest() for item in first.work_items
    )
    assert all(item.random_stream_digest is not None for item in first.work_items)


def test_adapter_manifest_is_order_invariant() -> None:
    original = adapter()
    reordered = replace(
        original,
        capabilities=tuple(reversed(original.capabilities)),
        supported_parameters=tuple(reversed(original.supported_parameters)),
        supported_artifacts=tuple(reversed(original.supported_artifacts)),
    )

    assert original == reordered
    assert original.content_digest() == reordered.content_digest()


def test_missing_capabilities_are_reported_before_compilation() -> None:
    pack = regression_pack()
    manifest = adapter(
        capabilities=(SimulationCapability.SEEDED_RESET,),
        evidence_digest=pack.simulation_manifest_digest,
    )

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    assert not report.compatible
    assert (
        '"report_schema_version":"iso-obs.simulation-compatibility-report.v1"'
        in report.to_json()
    )
    assert report.content_digest().startswith("sha256:")
    assert {
        issue.subject
        for issue in report.issues
        if issue.code is CompatibilityIssueCode.MISSING_CAPABILITY
    } == {
        "artifact_loading",
        "invariant_signal_export",
        "parameter_override",
        "random_stream_control",
    }


def test_compile_error_preserves_structured_report() -> None:
    pack = regression_pack()
    manifest = adapter(
        capabilities=(SimulationCapability.SEEDED_RESET,),
        evidence_digest=pack.simulation_manifest_digest,
    )

    with pytest.raises(IncompatibleSimulationAdapterError) as captured:
        compile_simulation_plan(
            pack,
            manifest,
            intent=ExecutionIntent.REPRODUCTION,
            evidence_use=EvidenceUse.CONFIRMATORY,
        )

    assert not captured.value.report.compatible
    assert "artifact_loading" in str(captured.value)


def test_unsupported_parameter_identifies_affected_cases() -> None:
    pack = regression_pack()
    manifest = replace(
        adapter(evidence_digest=pack.simulation_manifest_digest),
        supported_parameters=(),
    )

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    issue = next(
        item
        for item in report.issues
        if item.code is CompatibilityIssueCode.UNSUPPORTED_PARAMETER
    )
    assert issue.subject == "latency_ms"
    assert issue.affected_case_ids == tuple(item.case_id for item in pack.cases)


def test_unsupported_artifact_identifies_affected_cases() -> None:
    pack = regression_pack()
    manifest = replace(
        adapter(evidence_digest=pack.simulation_manifest_digest),
        supported_artifacts=(),
    )

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    issue = next(
        item
        for item in report.issues
        if item.code is CompatibilityIssueCode.UNSUPPORTED_ARTIFACT
    )
    assert issue.subject == "scenario"
    assert issue.affected_case_ids == tuple(item.case_id for item in pack.cases)


def test_explicit_arbitrary_support_allows_parameters_and_artifacts() -> None:
    pack = regression_pack()
    manifest = replace(
        adapter(evidence_digest=pack.simulation_manifest_digest),
        supported_parameters=(),
        supported_artifacts=(),
        accepts_arbitrary_parameters=True,
        accepts_arbitrary_artifacts=True,
    )

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    assert report.compatible


def test_reproduction_requires_linked_simulation_evidence() -> None:
    pack = regression_pack()
    manifest = adapter(evidence_digest=sha256_digest("different-simulation"))

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    assert CompatibilityIssueCode.EVIDENCE_MANIFEST_MISMATCH in {
        item.code for item in report.issues
    }


def test_cross_backend_validation_allows_explicit_manifest_difference() -> None:
    pack = regression_pack()
    manifest = adapter(evidence_digest=sha256_digest("second-simulator"))

    plan = compile_simulation_plan(
        pack,
        manifest,
        intent=ExecutionIntent.CROSS_BACKEND_VALIDATION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    assert plan.eligible_for_release_gate
    assert "not exact reproductions" in plan.limitations[2]


def test_world_model_requires_uncertainty_and_ood_capabilities() -> None:
    pack = regression_pack()
    manifest = adapter(
        backend_type=BackendType.WORLD_MODEL,
        evidence_digest=sha256_digest("world-model"),
    )

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.SURROGATE_SCREENING,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
    )

    assert {
        item.subject
        for item in report.issues
        if item.code is CompatibilityIssueCode.MISSING_CAPABILITY
    } == {"out_of_distribution_detection", "uncertainty_quantification"}


def test_world_model_screening_is_discovery_only() -> None:
    pack = regression_pack()
    manifest = adapter(
        backend_type=BackendType.WORLD_MODEL,
        capabilities=adapter().capabilities
        + (
            SimulationCapability.UNCERTAINTY_QUANTIFICATION,
            SimulationCapability.OUT_OF_DISTRIBUTION_DETECTION,
        ),
        evidence_digest=sha256_digest("world-model"),
    )

    plan = compile_simulation_plan(
        pack,
        manifest,
        intent=ExecutionIntent.SURROGATE_SCREENING,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
    )

    assert not plan.eligible_for_release_gate
    assert "not eligible for release-gate evidence" in plan.limitations[-1]


def test_surrogate_screening_rejects_confirmatory_evidence_use() -> None:
    pack = regression_pack()
    manifest = adapter(
        backend_type=BackendType.WORLD_MODEL,
        capabilities=adapter().capabilities
        + (
            SimulationCapability.UNCERTAINTY_QUANTIFICATION,
            SimulationCapability.OUT_OF_DISTRIBUTION_DETECTION,
        ),
        evidence_digest=sha256_digest("world-model"),
    )

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.SURROGATE_SCREENING,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    assert CompatibilityIssueCode.INVALID_INTENT in {
        item.code for item in report.issues
    }


def test_surrogate_screening_rejects_non_surrogate_backend() -> None:
    pack = regression_pack()
    manifest = adapter(
        capabilities=adapter().capabilities
        + (
            SimulationCapability.UNCERTAINTY_QUANTIFICATION,
            SimulationCapability.OUT_OF_DISTRIBUTION_DETECTION,
        ),
        evidence_digest=pack.simulation_manifest_digest,
    )

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.SURROGATE_SCREENING,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
    )

    assert any(item.subject == "surrogate_screening_backend" for item in report.issues)


def test_discovery_only_reproduction_is_not_release_eligible() -> None:
    pack = regression_pack()
    manifest = adapter(evidence_digest=pack.simulation_manifest_digest)

    plan = compile_simulation_plan(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
    )

    assert not plan.eligible_for_release_gate


def test_adapter_identity_changes_work_item_identity() -> None:
    pack = regression_pack()
    original = adapter(evidence_digest=pack.simulation_manifest_digest)
    updated = replace(original, adapter_version="1.3.0")

    first = compile_simulation_plan(
        pack,
        original,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )
    second = compile_simulation_plan(
        pack,
        updated,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    assert first.work_items[0].work_item_id != second.work_items[0].work_item_id
    assert first.content_digest() != second.content_digest()


def test_exact_replay_does_not_require_random_stream_control() -> None:
    original = regression_pack()
    pack = replace(
        original,
        seed_panel=SeedPanel(seeds=(10,), strategy=SeedStrategy.EXACT_REPLAY),
        gate_criteria=(replace(original.gate_criteria[0], minimum_trials=5),),
    )
    manifest = adapter(
        capabilities=tuple(
            item
            for item in adapter().capabilities
            if item is not SimulationCapability.RANDOM_STREAM_CONTROL
        ),
        evidence_digest=pack.simulation_manifest_digest,
    )

    report = assess_adapter_compatibility(
        pack,
        manifest,
        intent=ExecutionIntent.REPRODUCTION,
        evidence_use=EvidenceUse.CONFIRMATORY,
    )

    assert report.compatible


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_parallelism_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        replace(adapter(), max_parallelism=value)  # type: ignore[arg-type]


def test_duplicate_capability_is_rejected() -> None:
    original = adapter()

    with pytest.raises(ValueError, match="capabilities must be unique"):
        replace(
            original,
            capabilities=original.capabilities + (original.capabilities[0],),
        )
