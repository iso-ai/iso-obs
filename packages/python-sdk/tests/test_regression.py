"""Tests for simulator-neutral failure-derived regression packs."""

from __future__ import annotations

import math
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
    RegressionObservation,
    RegressionPack,
    ScenarioParameter,
    SeedPanel,
    SeedStrategy,
    evaluate_regression_pack,
)


def oracle() -> InvariantOracle:
    """Build a stopping-margin safety oracle."""
    return InvariantOracle(
        invariant_id="minimum-stopping-distance",
        invariant_version="2",
        signal="stopping_margin_m",
        operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
        threshold=0.25,
        persistence_steps=2,
    )


def case(role: CaseRole, index: int) -> RegressionCase:
    """Build one case with a scientifically distinct role."""
    return RegressionCase(
        case_id=f"{index:02d}-{role.value}",
        role=role,
        scenario_id="blind-intersection",
        scenario_version="3",
        parameters=(
            ScenarioParameter("observation_latency_ms", 100 + index * 5, "ms"),
            ScenarioParameter("floor_friction", 0.7 - index / 100),
        ),
        oracles=(oracle(),),
        expected_outcome=(
            ExpectedOutcome.VIOLATE_ANY
            if role is CaseRole.POSITIVE_CONTROL
            else ExpectedOutcome.SATISFY_ALL
        ),
        artifact_digests=(NamedDigest("scenario", sha256_digest(f"scenario-{index}")),),
        source_evidence_event_ids=(f"evt-{index}",),
        rationale=f"Exercise the {role.value} condition.",
    )


def cases() -> tuple[RegressionCase, ...]:
    """Build the five required regression case roles."""
    return tuple(case(role, index) for index, role in enumerate(CaseRole))


def seed_panel() -> SeedPanel:
    """Build a common-random-number panel with ten matched streams."""
    return SeedPanel(
        seeds=tuple(range(10)),
        strategy=SeedStrategy.COMMON_RANDOM_NUMBERS,
        random_stream_digest=sha256_digest("common-random-streams"),
    )


def pack(
    *,
    panel: SeedPanel | None = None,
    criteria: tuple[GateCriterion, ...] | None = None,
) -> RegressionPack:
    """Build a complete failure-derived regression pack."""
    selected_panel = panel or seed_panel()
    selected_criteria = criteria or (
        GateCriterion(
            name="failure-neighborhood",
            case_ids=tuple(item.case_id for item in cases()),
            method=GateMethod.WILSON_LOWER_BOUND,
            minimum_expected_outcome_rate=0.90,
            minimum_trials=50,
            confidence_level=0.95,
        ),
    )
    return RegressionPack(
        pack_id="blind-intersection-stop-margin",
        pack_version="1",
        source_failure_content_digest=sha256_digest("failure-bundle"),
        source_failure_fingerprint=sha256_digest("failure-family"),
        simulation_manifest_digest=sha256_digest("simulation-manifest"),
        cases=cases(),
        seed_panel=selected_panel,
        gate_criteria=selected_criteria,
        limitations=(
            "Validated only for low-speed indoor navigation.",
            "Contact severity is outside this regression pack.",
        ),
    )


def observations(
    regression_pack: RegressionPack,
) -> list[RegressionObservation]:
    """Build complete observations matching every expected outcome."""
    return [
        RegressionObservation(
            case_id=item.case_id,
            seed=seed,
            outcome=item.expected_outcome,
            run_id=f"run-{item.case_id}-{seed}",
        )
        for item in regression_pack.cases
        for seed in regression_pack.seed_panel.seeds
    ]


def test_pack_is_deterministic_and_order_invariant() -> None:
    original = pack()
    reordered = replace(
        original,
        cases=tuple(reversed(original.cases)),
        gate_criteria=tuple(reversed(original.gate_criteria)),
    )

    assert original == reordered
    assert original.to_json() == reordered.to_json()
    assert original.content_digest() == reordered.content_digest()


def test_complete_expected_panel_passes_wilson_gate() -> None:
    regression_pack = pack()

    report = evaluate_regression_pack(
        regression_pack,
        observations(regression_pack),
    )

    assert report.passed
    assert report.observation_count == 50
    assert report.criterion_results[0].matches == 50
    assert report.criterion_results[0].observed_match_rate == 1.0
    assert report.criterion_results[0].lower_confidence_bound >= 0.90


def test_positive_control_must_be_observed_violating_an_invariant() -> None:
    regression_pack = pack()
    results = observations(regression_pack)
    positive = next(
        item for item in regression_pack.cases if item.role is CaseRole.POSITIVE_CONTROL
    )

    matching = [item for item in results if item.case_id == positive.case_id]

    assert matching
    assert all(item.outcome is ExpectedOutcome.VIOLATE_ANY for item in matching)


def test_mislabeled_positive_control_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive control"):
        replace(
            case(CaseRole.POSITIVE_CONTROL, 4),
            expected_outcome=ExpectedOutcome.SATISFY_ALL,
        )


def test_mislabeled_negative_control_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative control"):
        replace(
            case(CaseRole.NEGATIVE_CONTROL, 3),
            expected_outcome=ExpectedOutcome.VIOLATE_ANY,
        )


def test_exact_gate_fails_one_mismatch() -> None:
    all_case_ids = tuple(item.case_id for item in cases())
    criterion = GateCriterion(
        name="zero-tolerance",
        case_ids=all_case_ids,
        method=GateMethod.EXACT,
        minimum_expected_outcome_rate=1.0,
        minimum_trials=50,
    )
    regression_pack = pack(criteria=(criterion,))
    results = observations(regression_pack)
    first = results[0]
    results[0] = replace(first, outcome=ExpectedOutcome.VIOLATE_ANY)

    report = evaluate_regression_pack(regression_pack, results)

    assert not report.passed
    assert report.criterion_results[0].matches == 49
    assert report.criterion_results[0].lower_confidence_bound == 0.98


def test_wilson_gate_rejects_perfect_but_tiny_sample() -> None:
    panel = SeedPanel(seeds=(42,), strategy=SeedStrategy.EXACT_REPLAY)
    criterion = GateCriterion(
        name="insufficient-evidence",
        case_ids=tuple(item.case_id for item in cases()),
        method=GateMethod.WILSON_LOWER_BOUND,
        minimum_expected_outcome_rate=0.90,
        minimum_trials=5,
    )
    regression_pack = pack(panel=panel, criteria=(criterion,))

    report = evaluate_regression_pack(
        regression_pack,
        observations(regression_pack),
    )

    assert report.criterion_results[0].observed_match_rate == 1.0
    assert report.criterion_results[0].lower_confidence_bound < 0.90
    assert not report.passed


@pytest.mark.parametrize(
    "missing_role",
    [
        CaseRole.NEIGHBORHOOD,
        CaseRole.BOUNDARY,
        CaseRole.NEGATIVE_CONTROL,
        CaseRole.POSITIVE_CONTROL,
    ],
)
def test_pack_requires_every_scientific_case_role(
    missing_role: CaseRole,
) -> None:
    original = pack()
    reduced = tuple(item for item in original.cases if item.role is not missing_role)

    with pytest.raises(ValueError, match="missing required roles"):
        replace(original, cases=reduced)


def test_pack_requires_exactly_one_canonical_reproduction() -> None:
    original = pack()
    duplicate = case(CaseRole.CANONICAL_REPRODUCTION, 99)

    with pytest.raises(ValueError, match="exactly one canonical"):
        replace(original, cases=original.cases + (duplicate,))


def test_unknown_gate_case_is_rejected() -> None:
    original = pack()
    invalid_gate = replace(
        original.gate_criteria[0],
        case_ids=original.gate_criteria[0].case_ids + ("not-a-case",),
    )

    with pytest.raises(ValueError, match="unknown cases"):
        replace(original, gate_criteria=(invalid_gate,))


def test_every_case_must_be_covered_by_a_gate() -> None:
    original = pack()
    partial_gate = replace(
        original.gate_criteria[0],
        case_ids=original.gate_criteria[0].case_ids[:-1],
        minimum_trials=40,
    )

    with pytest.raises(ValueError, match="not covered"):
        replace(original, gate_criteria=(partial_gate,))


def test_gate_cannot_require_more_trials_than_pack_defines() -> None:
    original = pack()
    impossible = replace(original.gate_criteria[0], minimum_trials=51)

    with pytest.raises(ValueError, match="more trials"):
        replace(original, gate_criteria=(impossible,))


def test_incomplete_observations_are_rejected() -> None:
    regression_pack = pack()

    with pytest.raises(ValueError, match="incomplete"):
        evaluate_regression_pack(
            regression_pack,
            observations(regression_pack)[:-1],
        )


def test_duplicate_observations_are_rejected() -> None:
    regression_pack = pack()
    results = observations(regression_pack)

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_regression_pack(regression_pack, results + [results[0]])


def test_observations_outside_pack_are_rejected() -> None:
    regression_pack = pack()
    results = observations(regression_pack)
    results[0] = replace(results[0], seed=999)

    with pytest.raises(ValueError, match="outside"):
        evaluate_regression_pack(regression_pack, results)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_scenario_parameter_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ScenarioParameter("friction", value)


def test_between_oracle_requires_ordered_bounds() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        InvariantOracle(
            invariant_id="temperature-band",
            invariant_version="1",
            signal="temperature_c",
            operator=ComparisonOperator.BETWEEN_INCLUSIVE,
            lower=80.0,
            upper=20.0,
        )


def test_threshold_oracle_rejects_bounds() -> None:
    with pytest.raises(ValueError, match="requires threshold only"):
        replace(oracle(), lower=0.0)


def test_exact_replay_requires_one_seed() -> None:
    with pytest.raises(ValueError, match="exactly one seed"):
        SeedPanel(seeds=(1, 2), strategy=SeedStrategy.EXACT_REPLAY)


def test_common_random_numbers_require_stream_identity() -> None:
    with pytest.raises(ValueError, match="random stream digest"):
        SeedPanel(
            seeds=(1, 2),
            strategy=SeedStrategy.COMMON_RANDOM_NUMBERS,
        )


def test_exact_gate_requires_perfect_expected_rate() -> None:
    with pytest.raises(ValueError, match="rate of 1.0"):
        GateCriterion(
            name="invalid-exact",
            case_ids=("case",),
            method=GateMethod.EXACT,
            minimum_expected_outcome_rate=0.99,
            minimum_trials=1,
        )


def test_wilson_gate_requires_minimum_sample() -> None:
    with pytest.raises(ValueError, match="at least five"):
        GateCriterion(
            name="too-small",
            case_ids=("case",),
            method=GateMethod.WILSON_LOWER_BOUND,
            minimum_expected_outcome_rate=0.90,
            minimum_trials=4,
        )


def test_gate_report_states_scope_bias_and_overfitting_limitations() -> None:
    regression_pack = pack()

    report = evaluate_regression_pack(
        regression_pack,
        observations(regression_pack),
    )

    assert "declared by this pack" in report.limitations[0]
    assert "model-form bias" in report.limitations[1]
    assert "overfit" in report.limitations[2]
