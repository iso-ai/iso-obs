"""Tests for multiplicity control and adaptive-search provenance."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from iso_obs.evidence import sha256_digest
from iso_obs.multiplicity import (
    AdaptiveSearchLedger,
    AdaptiveSearchTrial,
    AlternativeHypothesis,
    AnalysisMode,
    ControlledErrorRate,
    HypothesisTestResult,
    MultiplicityEvidenceLabel,
    MultiplicityMethod,
    MultiplicityPlan,
    MultiplicityReport,
    adjust_hypothesis_family,
)


def hypothesis_result(
    hypothesis_id: str,
    p_value: float,
    *,
    source_discovery_id: str | None = None,
) -> HypothesisTestResult:
    """Build one traceable hypothesis result."""
    return HypothesisTestResult(
        hypothesis_id=hypothesis_id,
        p_value=p_value,
        effect_estimate={"h1": 1.0, "h2": 2.0, "h3": 3.0}.get(
            hypothesis_id,
            0.5,
        ),
        sample_size=20,
        alternative=AlternativeHypothesis.TWO_SIDED,
        analysis_digest=sha256_digest(f"analysis-{hypothesis_id}"),
        standard_error=0.2,
        source_discovery_id=source_discovery_id,
    )


def plan(
    method: MultiplicityMethod,
    *,
    mode: AnalysisMode = AnalysisMode.EXPLORATORY,
) -> MultiplicityPlan:
    """Build a family plan for three hypotheses."""
    confirmatory = mode is not AnalysisMode.EXPLORATORY
    held_out = mode is AnalysisMode.HELD_OUT_CONFIRMATION
    return MultiplicityPlan(
        family_id="warehouse-failure-signals",
        mode=mode,
        method=method,
        target_error_rate=0.05,
        analysis_dataset_digest=sha256_digest("confirmation" if held_out else "data"),
        planned_hypothesis_ids=("h1", "h2", "h3"),
        preregistration_digest=(
            sha256_digest("preregistration") if confirmatory else None
        ),
        selection_dataset_digest=(sha256_digest("discovery") if held_out else None),
    )


def tests(
    *,
    held_out: bool = False,
) -> tuple[HypothesisTestResult, ...]:
    """Build a known p-value family in intentionally unsorted order."""
    return (
        hypothesis_result(
            "h2",
            0.04,
            source_discovery_id="discovery-h2" if held_out else None,
        ),
        hypothesis_result(
            "h1",
            0.01,
            source_discovery_id="discovery-h1" if held_out else None,
        ),
        hypothesis_result(
            "h3",
            0.03,
            source_discovery_id="discovery-h3" if held_out else None,
        ),
    )


def adjusted_by_id(report: MultiplicityReport) -> dict[str, float]:
    """Index report adjusted values by hypothesis ID."""
    return {item.hypothesis_id: item.adjusted_value for item in report.results}


def test_benjamini_hochberg_adjustment_and_labels() -> None:
    report = adjust_hypothesis_family(
        plan(MultiplicityMethod.BENJAMINI_HOCHBERG),
        tests(),
    )

    assert adjusted_by_id(report) == pytest.approx({"h1": 0.03, "h2": 0.04, "h3": 0.04})
    assert report.controlled_error_rate is ControlledErrorRate.FALSE_DISCOVERY_RATE
    assert report.rejected_count == 3
    assert all(
        item.evidence_label is MultiplicityEvidenceLabel.EXPLORATORY_SIGNAL
        for item in report.results
    )


def test_benjamini_yekutieli_is_more_conservative() -> None:
    report = adjust_hypothesis_family(
        plan(MultiplicityMethod.BENJAMINI_YEKUTIELI),
        tests(),
    )

    assert adjusted_by_id(report) == pytest.approx(
        {
            "h1": 0.055,
            "h2": 0.07333333333333333,
            "h3": 0.07333333333333333,
        }
    )
    assert report.rejected_count == 0
    assert all(
        item.evidence_label is MultiplicityEvidenceLabel.NOT_SELECTED
        for item in report.results
    )
    assert "arbitrary dependence" in report.limitations[3]


def test_holm_controls_family_wise_error() -> None:
    report = adjust_hypothesis_family(
        plan(MultiplicityMethod.HOLM),
        tests(),
    )

    assert adjusted_by_id(report) == pytest.approx({"h1": 0.03, "h2": 0.06, "h3": 0.06})
    assert report.controlled_error_rate is ControlledErrorRate.FAMILY_WISE_ERROR_RATE
    assert report.rejected_count == 1


def test_bonferroni_adjustment() -> None:
    report = adjust_hypothesis_family(
        plan(MultiplicityMethod.BONFERRONI),
        tests(),
    )

    assert adjusted_by_id(report) == pytest.approx({"h1": 0.03, "h2": 0.12, "h3": 0.09})
    assert report.rejected_count == 1


def test_input_order_does_not_change_report() -> None:
    family_plan = plan(MultiplicityMethod.BENJAMINI_HOCHBERG)
    family_tests = tests()

    forward = adjust_hypothesis_family(family_plan, family_tests)
    reverse = adjust_hypothesis_family(
        family_plan,
        tuple(reversed(family_tests)),
    )

    assert forward == reverse
    assert forward.content_digest() == reverse.content_digest()
    assert forward.plan_content_digest == family_plan.content_digest()


def test_effect_estimates_are_preserved_not_replaced_by_significance() -> None:
    report = adjust_hypothesis_family(
        plan(MultiplicityMethod.HOLM),
        tests(),
    )

    by_id = {item.hypothesis_id: item for item in report.results}
    assert by_id["h1"].effect_estimate == 1.0
    assert by_id["h2"].effect_estimate == 2.0
    assert by_id["h2"].standard_error == 0.2
    assert by_id["h2"].alternative is AlternativeHypothesis.TWO_SIDED
    assert not by_id["h2"].rejected


def test_preregistered_rejection_has_bounded_association_label() -> None:
    report = adjust_hypothesis_family(
        plan(
            MultiplicityMethod.HOLM,
            mode=AnalysisMode.PREREGISTERED,
        ),
        tests(),
    )

    selected = next(item for item in report.results if item.rejected)
    assert (
        selected.evidence_label is MultiplicityEvidenceLabel.PREREGISTERED_ASSOCIATION
    )
    assert "do not establish a causal mechanism" in report.limitations[2]


def test_held_out_confirmation_links_source_discoveries() -> None:
    report = adjust_hypothesis_family(
        plan(
            MultiplicityMethod.HOLM,
            mode=AnalysisMode.HELD_OUT_CONFIRMATION,
        ),
        tests(held_out=True),
    )

    selected = next(item for item in report.results if item.rejected)
    assert (
        selected.evidence_label
        is MultiplicityEvidenceLabel.HELD_OUT_CONFIRMED_ASSOCIATION
    )
    assert selected.source_discovery_id == "discovery-h1"
    assert "strictly separated" in report.limitations[-1]


def test_held_out_confirmation_requires_source_discovery_ids() -> None:
    with pytest.raises(ValueError, match="source discovery IDs"):
        adjust_hypothesis_family(
            plan(
                MultiplicityMethod.HOLM,
                mode=AnalysisMode.HELD_OUT_CONFIRMATION,
            ),
            tests(),
        )


def test_confirmatory_plan_requires_preregistration_and_hypotheses() -> None:
    base = plan(MultiplicityMethod.HOLM, mode=AnalysisMode.PREREGISTERED)

    with pytest.raises(ValueError, match="preregistration"):
        replace(base, preregistration_digest=None)
    with pytest.raises(ValueError, match="planned hypothesis IDs"):
        replace(base, planned_hypothesis_ids=())


def test_held_out_plan_requires_distinct_selection_data() -> None:
    base = plan(
        MultiplicityMethod.HOLM,
        mode=AnalysisMode.HELD_OUT_CONFIRMATION,
    )

    with pytest.raises(ValueError, match="selection dataset digest"):
        replace(base, selection_dataset_digest=None)
    with pytest.raises(ValueError, match="must be different"):
        replace(
            base,
            selection_dataset_digest=base.analysis_dataset_digest,
        )


def test_selection_data_requires_held_out_mode() -> None:
    base = plan(MultiplicityMethod.HOLM, mode=AnalysisMode.PREREGISTERED)

    with pytest.raises(ValueError, match="use held_out_confirmation"):
        replace(
            base,
            selection_dataset_digest=sha256_digest("selection"),
        )


def test_confirmatory_analysis_rejects_adaptive_hypotheses() -> None:
    base = plan(MultiplicityMethod.HOLM, mode=AnalysisMode.PREREGISTERED)

    with pytest.raises(ValueError, match="cannot adapt hypotheses"):
        replace(
            base,
            adaptive_search_ledger_digest=sha256_digest("adaptive-ledger"),
        )


def test_family_must_match_planned_hypotheses_exactly() -> None:
    family_plan = plan(MultiplicityMethod.HOLM)

    with pytest.raises(ValueError, match="differs from plan"):
        adjust_hypothesis_family(family_plan, tests()[:-1])
    with pytest.raises(ValueError, match="differs from plan"):
        adjust_hypothesis_family(
            family_plan,
            tests() + (hypothesis_result("extra", 0.5),),
        )


def test_duplicate_or_empty_test_family_is_rejected() -> None:
    family_plan = plan(MultiplicityMethod.HOLM)

    with pytest.raises(ValueError, match="at least one test"):
        adjust_hypothesis_family(family_plan, ())
    with pytest.raises(ValueError, match="must be unique"):
        adjust_hypothesis_family(
            replace(family_plan, planned_hypothesis_ids=()),
            (hypothesis_result("h1", 0.1), hypothesis_result("h1", 0.2)),
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_p_value_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="p_value"):
        hypothesis_result("h1", value)


def test_invalid_effect_sample_and_standard_error_are_rejected() -> None:
    base = hypothesis_result("h1", 0.1)

    with pytest.raises(ValueError, match="effect_estimate"):
        replace(base, effect_estimate=math.nan)
    with pytest.raises(ValueError, match="at least two"):
        replace(base, sample_size=1)
    with pytest.raises(ValueError, match="standard_error"):
        replace(base, standard_error=-0.1)


@pytest.mark.parametrize("value", [0.0, 1.0, math.nan])
def test_invalid_target_error_rate_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="target_error_rate"):
        replace(
            plan(MultiplicityMethod.HOLM),
            target_error_rate=value,
        )


def adaptive_trial(iteration: int) -> AdaptiveSearchTrial:
    """Build one content-addressed adaptive proposal."""
    return AdaptiveSearchTrial(
        iteration=iteration,
        candidate_id=f"candidate-{iteration}",
        scenario_digest=sha256_digest(f"scenario-{iteration}"),
        proposal_policy_digest=sha256_digest("bayesian-optimizer-v2"),
        information_available_digest=sha256_digest(f"history-{iteration}"),
        outcome_digest=sha256_digest(f"outcome-{iteration}"),
        proposal_score=0.9 - iteration / 10,
    )


def adaptive_ledger() -> AdaptiveSearchLedger:
    """Build a three-iteration adaptive-search ledger."""
    return AdaptiveSearchLedger(
        search_id="warehouse-boundary-search",
        algorithm="constrained-bayesian-optimization",
        algorithm_version="2",
        objective="minimize stopping margin",
        trials=(adaptive_trial(2), adaptive_trial(0), adaptive_trial(1)),
        limitations=("Acquisition scores depend on the declared surrogate model.",),
    )


def test_adaptive_ledger_is_order_invariant_and_content_addressed() -> None:
    first = adaptive_ledger()
    second = replace(first, trials=tuple(reversed(first.trials)))

    assert first == second
    assert first.content_digest() == second.content_digest()
    assert tuple(item.iteration for item in first.trials) == (0, 1, 2)


def test_exploratory_plan_can_bind_adaptive_search_ledger() -> None:
    ledger = adaptive_ledger()
    family_plan = replace(
        plan(MultiplicityMethod.BENJAMINI_YEKUTIELI),
        planned_hypothesis_ids=(),
        adaptive_search_ledger_digest=ledger.content_digest(),
    )

    report = adjust_hypothesis_family(family_plan, tests())

    assert report.mode is AnalysisMode.EXPLORATORY
    assert "independent held-out data" in report.limitations[-1]


def test_adaptive_ledger_requires_contiguous_unique_trials() -> None:
    ledger = adaptive_ledger()

    with pytest.raises(ValueError, match="contiguous"):
        replace(
            ledger,
            trials=(adaptive_trial(0), adaptive_trial(2)),
        )
    with pytest.raises(ValueError, match="candidate IDs must be unique"):
        replace(
            ledger,
            trials=(
                adaptive_trial(0),
                replace(adaptive_trial(1), candidate_id="candidate-0"),
            ),
        )


def test_invalid_adaptive_trial_is_rejected() -> None:
    trial = adaptive_trial(0)

    with pytest.raises(ValueError, match="non-negative integer"):
        replace(trial, iteration=-1)
    with pytest.raises(ValueError, match="proposal score"):
        replace(trial, proposal_score=math.inf)
    with pytest.raises(ValueError, match="sha256"):
        replace(trial, outcome_digest="raw-output")
