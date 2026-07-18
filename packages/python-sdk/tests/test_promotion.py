"""Tests for evidence-qualified regression promotion."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.counterfactual import (
    ConditionSetTrial,
    CounterfactualDesign,
    CounterfactualEstimand,
    CounterfactualEvidenceLevel,
    CounterfactualObservation,
    CounterfactualPair,
    EffectDirection,
    FailureCondition,
    InterventionOperation,
    InterventionScope,
    InterventionSpec,
    ReplayControl,
    analyze_counterfactual_design,
    assess_minimal_counterexample,
)
from iso_obs.evidence import (
    AnalysisKind,
    AnalysisProvenance,
    EvidenceLevel,
    FailureEvidenceBundle,
    NamedDigest,
    ReproductionManifest,
    SimulationEvidenceManifest,
    SimulationModelType,
    UncertaintyKind,
    UncertaintySource,
    ValidityRange,
    sha256_digest,
)
from iso_obs.promotion import (
    ConditionParameterBinding,
    PromotionIntent,
    PromotionRequirement,
    PromotionStatus,
    RegressionPromotionRequest,
    review_regression_promotion,
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
from iso_obs.replication import PairedAnalysisConfig


def digest(value: str) -> str:
    """Build a valid digest for test content."""
    return sha256_digest(value)


def source_failure(
    *,
    validated: bool = True,
    anchored: bool = True,
) -> FailureEvidenceBundle:
    """Build a failure bundle with configurable simulator credibility."""
    simulation = SimulationEvidenceManifest(
        simulator_name="warehouse-twin",
        simulator_version="4.2.0",
        model_type=SimulationModelType.HYBRID,
        artifact_digest=digest("warehouse-twin"),
        intended_use="Low-speed indoor robot navigation.",
        validity_envelope=(ValidityRange("speed", 0.0, 2.0, "m/s"),),
        uncertainty_sources=(
            UncertaintySource(
                name="contact model",
                kind=UncertaintyKind.MODEL_FORM_EPISTEMIC,
                characterization="Rigid contact approximation.",
                quantified=False,
            ),
        ),
        verification_evidence=(
            NamedDigest("solver convergence", digest("verification")),
        ),
        validation_evidence=(
            (NamedDigest("motion capture", digest("validation")),) if validated else ()
        ),
        real_world_anchors=(
            (NamedDigest("warehouse trials", digest("anchors")),) if anchored else ()
        ),
        known_omissions=("Tire wear is not represented.",),
    )
    return FailureEvidenceBundle(
        failure_category="unsafe_stop_margin",
        scenario_family="blind_intersection",
        scenario_version="3",
        perturbation_family="visibility_and_friction",
        invariant_id="minimum-stopping-distance",
        invariant_version="2",
        signal_group=("stopping_margin_m",),
        execution_phase="closed_loop_control",
        component="navigation_stack",
        evidence_level=EvidenceLevel.SUSTAINED_DIVERGENCE,
        summary="Stopping margin became unsafe in a blind intersection.",
        affected_system_versions=("policy-v17",),
        evidence_event_ids=("evt-101", "evt-102"),
        reproduction=ReproductionManifest(
            scenario_id="blind-intersection",
            scenario_version="3",
            environment_id="warehouse-twin",
            environment_version="4.2.0",
            system_versions=("policy-v17",),
            seed=42,
            perturbation_realization_digest=digest("perturbation"),
            sdk_version="0.1.0",
            schema_version="0.1.0",
            configuration_digests=(NamedDigest("policy", digest("policy-config")),),
            trace_digest=digest("trace"),
            command="python evaluate.py --scenario blind-intersection",
        ),
        simulation=simulation,
        analyses=(
            AnalysisProvenance(
                kind=AnalysisKind.SUSTAINED_DIVERGENCE,
                analyzer="iso_obs.divergence",
                analyzer_version="1",
                input_digests=(NamedDigest("trace", digest("trace")),),
                configuration_digest=digest("divergence-config"),
                method="sustained tolerance",
            ),
        ),
        limitations=("Collision severity is outside the validity envelope.",),
    )


def counterfactual_design(
    failure: FailureEvidenceBundle,
) -> CounterfactualDesign:
    """Build a six-pair design linked to the source failure."""
    pairs = tuple(
        CounterfactualPair(
            pair_id=f"pair-{index}",
            control=ReplayControl(
                scenario_id="blind-intersection",
                scenario_version="3",
                simulation_manifest_digest=(failure.simulation.content_digest()),
                initial_state_digest=digest(f"state-{index}"),
                seed=index,
                random_stream_digest=digest(f"stream-{index}"),
            ),
            source_failure_event_ids=("evt-101",),
        )
        for index in range(6)
    )
    return CounterfactualDesign(
        design_id="restore-braking",
        design_version="1",
        preregistration_digest=digest("preregistration"),
        simulation_evidence_manifest_digest=(failure.simulation.content_digest()),
        intervention=InterventionSpec(
            intervention_id="restore-controller",
            mechanism_hypothesis_id="controller-causes-late-braking",
            scope=InterventionScope.COMPONENT,
            operation=InterventionOperation.REPLACE,
            target="controller.braking",
            baseline_value_digest=digest("candidate"),
            intervention_value_digest=digest("reference"),
            rationale="Isolate the braking controller change.",
        ),
        estimand=CounterfactualEstimand(
            outcome_name="unsafe stopping margin",
            summary_statistic="minimum over episode",
            expected_direction=EffectDirection.DECREASE,
            minimum_meaningful_effect=1.0,
            unit="meters",
        ),
        pairs=pairs,
        analysis_config=PairedAnalysisConfig(
            bootstrap_resamples=1_000,
            randomization_resamples=1_000,
        ),
    )


def counterfactual_report(
    design: CounterfactualDesign,
    *,
    difference: float = -2.0,
):
    """Analyze identity-matched counterfactual observations."""
    intervention_digest = design.intervention.content_digest()
    observations = tuple(
        CounterfactualObservation(
            pair_id=pair.pair_id,
            baseline=10.0,
            intervention=10.0 + difference,
            baseline_run_id=f"baseline-{pair.pair_id}",
            intervention_run_id=f"intervention-{pair.pair_id}",
            replay_control_digest=pair.control.content_digest(),
            intervention_spec_digest=intervention_digest,
        )
        for pair in design.pairs
    )
    return analyze_counterfactual_design(design, observations)


def minimal_report(*, one_minimal: bool = True):
    """Build a minimized two-condition reproduction."""
    conditions = tuple(
        FailureCondition(
            condition_id=name,
            value_digest=digest(name),
            category="environment",
        )
        for name in ("floor-friction", "fog-density", "latency")
    )
    trials = [
        ConditionSetTrial(
            retained_condition_ids=(
                "floor-friction",
                "fog-density",
                "latency",
            ),
            failure_reproduced=True,
            run_id="full",
            evidence_digest=digest("full"),
        ),
        ConditionSetTrial(
            retained_condition_ids=("floor-friction", "fog-density"),
            failure_reproduced=True,
            run_id="minimal",
            evidence_digest=digest("minimal"),
        ),
        ConditionSetTrial(
            retained_condition_ids=("floor-friction",),
            failure_reproduced=False,
            run_id="remove-fog",
            evidence_digest=digest("remove-fog"),
        ),
    ]
    if one_minimal:
        trials.append(
            ConditionSetTrial(
                retained_condition_ids=("fog-density",),
                failure_reproduced=False,
                run_id="remove-friction",
                evidence_digest=digest("remove-friction"),
            )
        )
    return assess_minimal_counterexample(conditions, tuple(trials))


def oracle() -> InvariantOracle:
    """Build the source failure's invariant oracle."""
    return InvariantOracle(
        invariant_id="minimum-stopping-distance",
        invariant_version="2",
        signal="stopping_margin_m",
        operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        threshold=0.25,
    )


def regression_case(
    role: CaseRole,
    index: int,
    *,
    expected_outcome: ExpectedOutcome | None = None,
) -> RegressionCase:
    """Build one scientifically distinct regression case."""
    expected = expected_outcome or (
        ExpectedOutcome.VIOLATE_ANY
        if role is CaseRole.POSITIVE_CONTROL
        else ExpectedOutcome.SATISFY_ALL
    )
    return RegressionCase(
        case_id=f"{index}-{role.value}",
        role=role,
        scenario_id="blind-intersection",
        scenario_version="3",
        parameters=(
            ScenarioParameter("floor_friction", 0.45),
            ScenarioParameter("fog_density", 0.8),
        ),
        oracles=(oracle(),),
        expected_outcome=expected,
        source_evidence_event_ids=("evt-101",),
        rationale=f"Exercise the {role.value} role.",
    )


def regression_pack(
    failure: FailureEvidenceBundle,
    *,
    canonical_outcome: ExpectedOutcome = ExpectedOutcome.SATISFY_ALL,
) -> RegressionPack:
    """Build a complete regression pack linked to the source failure."""
    cases = tuple(
        regression_case(
            role,
            index,
            expected_outcome=(
                canonical_outcome if role is CaseRole.CANONICAL_REPRODUCTION else None
            ),
        )
        for index, role in enumerate(CaseRole)
    )
    return RegressionPack(
        pack_id="blind-intersection-regression",
        pack_version="1",
        source_failure_content_digest=failure.content_digest(),
        source_failure_fingerprint=failure.failure_fingerprint(),
        simulation_manifest_digest=failure.simulation.content_digest(),
        cases=cases,
        seed_panel=SeedPanel(
            seeds=tuple(range(5)),
            strategy=SeedStrategy.COMMON_RANDOM_NUMBERS,
            random_stream_digest=digest("regression-streams"),
        ),
        gate_criteria=(
            GateCriterion(
                name="all-cases",
                case_ids=tuple(item.case_id for item in cases),
                method=GateMethod.WILSON_LOWER_BOUND,
                minimum_expected_outcome_rate=0.8,
                minimum_trials=25,
            ),
        ),
        limitations=("Valid only within the linked simulation envelope.",),
    )


def request(
    failure: FailureEvidenceBundle,
    design: CounterfactualDesign,
    report,
    minimal,
    pack: RegressionPack,
    *,
    intent: PromotionIntent = PromotionIntent.RELEASE_GATE_CANDIDATE,
) -> RegressionPromotionRequest:
    """Build a request containing exact artifact identities."""
    return RegressionPromotionRequest(
        request_id="promote-blind-intersection",
        request_version="1",
        intent=intent,
        source_failure_content_digest=failure.content_digest(),
        counterfactual_design_digest=design.content_digest(),
        counterfactual_report_digest=report.content_digest(),
        minimal_counterexample_report_digest=minimal.content_digest(),
        regression_pack_digest=pack.content_digest(),
        condition_bindings=(
            ConditionParameterBinding("floor-friction", "floor_friction"),
            ConditionParameterBinding("fog-density", "fog_density"),
        ),
        limitations=("Requires execution against the proposed system version.",),
    )


def artifacts(
    *,
    difference: float = -2.0,
    one_minimal: bool = True,
    validated: bool = True,
    anchored: bool = True,
    canonical_outcome: ExpectedOutcome = ExpectedOutcome.SATISFY_ALL,
):
    """Build one coherent promotion evidence chain."""
    failure = source_failure(validated=validated, anchored=anchored)
    design = counterfactual_design(failure)
    report = counterfactual_report(design, difference=difference)
    minimal = minimal_report(one_minimal=one_minimal)
    pack = regression_pack(failure, canonical_outcome=canonical_outcome)
    return failure, design, report, minimal, pack


def review(evidence, *, intent=PromotionIntent.RELEASE_GATE_CANDIDATE):
    """Review one coherent evidence tuple."""
    failure, design, report, minimal, pack = evidence
    promotion_request = request(
        failure,
        design,
        report,
        minimal,
        pack,
        intent=intent,
    )
    return review_regression_promotion(
        promotion_request,
        source_failure=failure,
        counterfactual_design=design,
        counterfactual_report=report,
        minimal_counterexample=minimal,
        regression_pack=pack,
    )


def test_complete_evidence_chain_is_release_gate_candidate_ready() -> None:
    result = review(artifacts())

    assert result.status is PromotionStatus.RELEASE_GATE_CANDIDATE_READY
    assert result.unmet_requirements == ()
    assert all(item.satisfied for item in result.requirement_results)


def test_discovery_asset_does_not_require_mechanism_confirmation() -> None:
    evidence = artifacts(
        difference=0.2,
        one_minimal=False,
        validated=False,
        anchored=False,
        canonical_outcome=ExpectedOutcome.VIOLATE_ANY,
    )

    result = review(evidence, intent=PromotionIntent.DISCOVERY_REGRESSION)

    assert result.status is PromotionStatus.DISCOVERY_READY
    assert len(result.unmet_requirements) == len(PromotionRequirement)


def test_release_candidate_reports_every_missing_requirement() -> None:
    evidence = artifacts(
        difference=0.2,
        one_minimal=False,
        validated=False,
        anchored=False,
        canonical_outcome=ExpectedOutcome.VIOLATE_ANY,
    )

    result = review(evidence)

    assert result.status is PromotionStatus.EVIDENCE_INSUFFICIENT
    assert result.unmet_requirements == tuple(PromotionRequirement)


@pytest.mark.parametrize(
    "request_field",
    [
        "source_failure_content_digest",
        "counterfactual_design_digest",
        "counterfactual_report_digest",
        "minimal_counterexample_report_digest",
        "regression_pack_digest",
    ],
)
def test_request_rejects_each_artifact_digest_mismatch(
    request_field: str,
) -> None:
    failure, design, report, minimal, pack = artifacts()
    promotion_request = request(failure, design, report, minimal, pack)
    promotion_request = replace(
        promotion_request,
        **{request_field: digest(f"wrong-{request_field}")},
    )

    with pytest.raises(ValueError, match="content digest mismatch"):
        review_regression_promotion(
            promotion_request,
            source_failure=failure,
            counterfactual_design=design,
            counterfactual_report=report,
            minimal_counterexample=minimal,
            regression_pack=pack,
        )


def test_pack_must_link_exact_failure_fingerprint_and_simulation() -> None:
    failure, design, report, minimal, pack = artifacts()

    for changed, expected in (
        (
            replace(pack, source_failure_fingerprint=digest("wrong")),
            "failure fingerprint",
        ),
        (
            replace(pack, simulation_manifest_digest=digest("wrong")),
            "simulation manifest",
        ),
    ):
        promotion_request = request(
            failure,
            design,
            report,
            minimal,
            changed,
        )
        with pytest.raises(ValueError, match=expected):
            review_regression_promotion(
                promotion_request,
                source_failure=failure,
                counterfactual_design=design,
                counterfactual_report=report,
                minimal_counterexample=minimal,
                regression_pack=changed,
            )


def test_counterfactual_pairs_must_reference_source_failure_events() -> None:
    failure, design, report, minimal, pack = artifacts()
    changed_pair = replace(
        design.pairs[0],
        source_failure_event_ids=("outside-event",),
    )
    changed_design = replace(
        design,
        pairs=(changed_pair,) + design.pairs[1:],
    )
    changed_report = replace(
        report,
        design_content_digest=changed_design.content_digest(),
    )
    promotion_request = request(
        failure,
        changed_design,
        changed_report,
        minimal,
        pack,
    )

    with pytest.raises(ValueError, match="outside the source failure"):
        review_regression_promotion(
            promotion_request,
            source_failure=failure,
            counterfactual_design=changed_design,
            counterfactual_report=changed_report,
            minimal_counterexample=minimal,
            regression_pack=pack,
        )


def test_canonical_case_must_match_reproduction_and_source_events() -> None:
    failure, design, report, minimal, pack = artifacts()
    canonical = next(
        item for item in pack.cases if item.role is CaseRole.CANONICAL_REPRODUCTION
    )
    changed_case = replace(canonical, scenario_id="different-scenario")
    changed_pack = replace(
        pack,
        cases=tuple(
            changed_case if item.case_id == canonical.case_id else item
            for item in pack.cases
        ),
    )
    promotion_request = request(
        failure,
        design,
        report,
        minimal,
        changed_pack,
    )

    with pytest.raises(ValueError, match="does not match failure reproduction"):
        review_regression_promotion(
            promotion_request,
            source_failure=failure,
            counterfactual_design=design,
            counterfactual_report=report,
            minimal_counterexample=minimal,
            regression_pack=changed_pack,
        )


def test_condition_bindings_must_cover_minimal_set_and_canonical_parameters() -> None:
    failure, design, report, minimal, pack = artifacts()
    promotion_request = request(failure, design, report, minimal, pack)

    with pytest.raises(ValueError, match="missing: fog-density"):
        review_regression_promotion(
            replace(
                promotion_request,
                condition_bindings=promotion_request.condition_bindings[:1],
            ),
            source_failure=failure,
            counterfactual_design=design,
            counterfactual_report=report,
            minimal_counterexample=minimal,
            regression_pack=pack,
        )
    with pytest.raises(ValueError, match="unknown canonical parameters"):
        review_regression_promotion(
            replace(
                promotion_request,
                condition_bindings=(
                    promotion_request.condition_bindings[0],
                    ConditionParameterBinding("fog-density", "not-a-parameter"),
                ),
            ),
            source_failure=failure,
            counterfactual_design=design,
            counterfactual_report=report,
            minimal_counterexample=minimal,
            regression_pack=pack,
        )


def test_request_and_review_are_order_invariant() -> None:
    failure, design, report, minimal, pack = artifacts()
    promotion_request = request(failure, design, report, minimal, pack)
    reversed_request = replace(
        promotion_request,
        condition_bindings=tuple(reversed(promotion_request.condition_bindings)),
        limitations=tuple(reversed(promotion_request.limitations)),
    )

    first = review_regression_promotion(
        promotion_request,
        source_failure=failure,
        counterfactual_design=design,
        counterfactual_report=report,
        minimal_counterexample=minimal,
        regression_pack=pack,
    )
    second = review_regression_promotion(
        reversed_request,
        source_failure=failure,
        counterfactual_design=design,
        counterfactual_report=report,
        minimal_counterexample=minimal,
        regression_pack=pack,
    )

    assert promotion_request == reversed_request
    assert first == second
    assert first.content_digest() == second.content_digest()


def test_review_does_not_claim_release_or_real_world_causality() -> None:
    result = review(artifacts())
    limitations = " ".join(result.limitations)

    assert "does not execute the regression pack or approve" in limitations
    assert "sim-to-real limits" in limitations
    assert "rather than proof" in limitations
    assert "Requires execution against the proposed system version." in limitations
    assert (
        result.requirement_results[0].requirement
        is PromotionRequirement.REPLICATED_INTERVENTION_ASSOCIATION
    )
    assert (
        artifacts()[2].evidence_level
        is CounterfactualEvidenceLevel.REPLICATED_INTERVENTION_ASSOCIATION
    )
