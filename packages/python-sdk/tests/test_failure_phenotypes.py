"""Tests for conformal failure-phenotype classification."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.evidence import sha256_digest
from iso_obs.failure_phenotypes import (
    FailurePhenotypeExample,
    FailurePhenotypePlan,
    FailurePhenotypeQuery,
    PhenotypeDisposition,
    PhenotypeFeature,
    PhenotypeTrainingSplit,
    classify_failure_phenotype,
    fit_failure_phenotype_model,
)
from iso_obs.simulation import EvidenceUse


def plan(*, feature_count: int = 2) -> FailurePhenotypePlan:
    """Build a compact deterministic phenotype plan."""
    features = tuple(
        PhenotypeFeature(
            feature_id=feature_id,
            definition_digest=sha256_digest(f"definition-{feature_id}"),
        )
        for feature_id in ("peak-deceleration", "lateral-error")[:feature_count]
    )
    return FailurePhenotypePlan(
        plan_id="warehouse-failure-phenotypes",
        plan_version="1",
        feature_extraction_digest=sha256_digest("extractor"),
        target_population_digest=sha256_digest("target-population"),
        phenotype_taxonomy_digest=sha256_digest("taxonomy"),
        features=features,
        phenotype_ids=("late-braking", "steering-oscillation"),
        miscoverage_rate=0.2,
        minimum_fit_examples_per_phenotype=3,
        minimum_calibration_examples_per_phenotype=4,
        limitations=("Low-speed warehouse failures only.",),
    )


def example(
    phenotype_plan: FailurePhenotypePlan,
    phenotype_id: str,
    split: PhenotypeTrainingSplit,
    index: int,
    values: tuple[float, ...],
) -> FailurePhenotypeExample:
    """Build one independently identified training example."""
    unit_id = f"{phenotype_id}-{split.value}-{index}"
    return FailurePhenotypeExample(
        plan_content_digest=phenotype_plan.content_digest(),
        evidence_digest=sha256_digest(f"evidence-{unit_id}"),
        independence_unit_id=unit_id,
        phenotype_id=phenotype_id,
        split=split,
        feature_values=values,
    )


def separated_examples(
    phenotype_plan: FailurePhenotypePlan,
) -> tuple[FailurePhenotypeExample, ...]:
    """Build two well-separated phenotypes with held-out calibration."""
    fit_values = {
        "late-braking": ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
        "steering-oscillation": ((9.0, 10.0), (10.0, 10.0), (11.0, 10.0)),
    }
    calibration_values = {
        "late-braking": (
            (-0.5, 0.0),
            (-0.25, 0.0),
            (0.25, 0.0),
            (0.5, 0.0),
        ),
        "steering-oscillation": (
            (9.5, 10.0),
            (9.75, 10.0),
            (10.25, 10.0),
            (10.5, 10.0),
        ),
    }
    examples: list[FailurePhenotypeExample] = []
    for phenotype_id in phenotype_plan.phenotype_ids:
        examples.extend(
            example(
                phenotype_plan,
                phenotype_id,
                PhenotypeTrainingSplit.FIT,
                index,
                values,
            )
            for index, values in enumerate(fit_values[phenotype_id])
        )
        examples.extend(
            example(
                phenotype_plan,
                phenotype_id,
                PhenotypeTrainingSplit.CALIBRATION,
                index,
                values,
            )
            for index, values in enumerate(calibration_values[phenotype_id])
        )
    return tuple(examples)


def query(
    phenotype_plan: FailurePhenotypePlan,
    values: tuple[float, ...],
    *,
    query_id: str = "query-1",
) -> FailurePhenotypeQuery:
    """Build one identity-linked classification query."""
    return FailurePhenotypeQuery(
        plan_content_digest=phenotype_plan.content_digest(),
        evidence_digest=sha256_digest(f"query-evidence-{query_id}"),
        query_id=query_id,
        feature_values=values,
    )


def test_fit_is_order_invariant_and_content_addressed() -> None:
    """Input order cannot alter robust model state or exact identity."""
    phenotype_plan = plan()
    examples = separated_examples(phenotype_plan)

    first = fit_failure_phenotype_model(phenotype_plan, examples)
    second = fit_failure_phenotype_model(
        phenotype_plan,
        tuple(reversed(examples)),
    )

    assert first == second
    assert first.content_digest() == second.content_digest()
    assert first.evidence_use is EvidenceUse.DISCOVERY_ONLY
    assert first.feature_ids == ("peak-deceleration", "lateral-error")
    assert all(item > 0.0 for item in first.feature_scales)
    assert tuple(item.phenotype_id for item in first.prototypes) == (
        "late-braking",
        "steering-oscillation",
    )


def test_singleton_conformal_set_assigns_known_phenotype() -> None:
    """A well-supported query receives one known phenotype."""
    phenotype_plan = plan()
    model = fit_failure_phenotype_model(
        phenotype_plan,
        separated_examples(phenotype_plan),
    )

    report = classify_failure_phenotype(
        phenotype_plan,
        model,
        query(phenotype_plan, (0.0, 0.0)),
    )

    assert report.disposition is PhenotypeDisposition.ASSIGNED
    assert report.assigned_phenotype_id == "late-braking"
    assert report.prediction_set == ("late-braking",)
    assert report.nearest_phenotype_id == "late-braking"
    assert report.evidence_use is EvidenceUse.DISCOVERY_ONLY


def test_empty_conformal_set_marks_novel_candidate() -> None:
    """A vector far from every calibrated class triggers abstention."""
    phenotype_plan = plan()
    model = fit_failure_phenotype_model(
        phenotype_plan,
        separated_examples(phenotype_plan),
    )

    report = classify_failure_phenotype(
        phenotype_plan,
        model,
        query(phenotype_plan, (100.0, -100.0)),
    )

    assert report.disposition is PhenotypeDisposition.NOVEL_CANDIDATE
    assert report.assigned_phenotype_id is None
    assert report.prediction_set == ()
    assert all(
        item.conformal_p_value == pytest.approx(0.2) for item in report.candidates
    )


def test_multiple_supported_classes_produce_ambiguity() -> None:
    """Overlapping calibration support cannot be forced into one label."""
    phenotype_plan = plan(feature_count=1)
    fit = {
        "late-braking": (-1.0, 0.0, 1.0),
        "steering-oscillation": (9.0, 10.0, 11.0),
    }
    calibration = {
        "late-braking": (4.0, 4.5, 5.0, 5.5),
        "steering-oscillation": (4.5, 5.0, 5.5, 6.0),
    }
    examples = tuple(
        example(
            phenotype_plan,
            phenotype_id,
            split,
            index,
            (value,),
        )
        for phenotype_id in phenotype_plan.phenotype_ids
        for split, values in (
            (PhenotypeTrainingSplit.FIT, fit[phenotype_id]),
            (
                PhenotypeTrainingSplit.CALIBRATION,
                calibration[phenotype_id],
            ),
        )
        for index, value in enumerate(values)
    )
    model = fit_failure_phenotype_model(phenotype_plan, examples)

    report = classify_failure_phenotype(
        phenotype_plan,
        model,
        query(phenotype_plan, (5.0,)),
    )

    assert report.disposition is PhenotypeDisposition.AMBIGUOUS
    assert report.assigned_phenotype_id is None
    assert report.prediction_set == (
        "late-braking",
        "steering-oscillation",
    )


def test_distance_explanation_is_complete_and_ranked() -> None:
    """Feature contributions exactly decompose each squared distance."""
    phenotype_plan = plan()
    model = fit_failure_phenotype_model(
        phenotype_plan,
        separated_examples(phenotype_plan),
    )
    report = classify_failure_phenotype(
        phenotype_plan,
        model,
        query(phenotype_plan, (0.0, 1.0)),
    )

    candidate = next(
        item for item in report.candidates if item.phenotype_id == "late-braking"
    )

    assert sum(
        item.fraction_of_squared_distance for item in candidate.feature_contributions
    ) == pytest.approx(1.0)
    assert (
        candidate.feature_contributions[0].weighted_squared_distance
        >= candidate.feature_contributions[1].weighted_squared_distance
    )


def test_constant_fit_feature_uses_declared_scale_floor() -> None:
    """A constant training feature cannot create a zero denominator."""
    phenotype_plan = plan()
    examples = tuple(
        replace(item, feature_values=(item.feature_values[0], 0.0))
        for item in separated_examples(phenotype_plan)
    )
    model = fit_failure_phenotype_model(phenotype_plan, examples)

    lateral_index = model.feature_ids.index("lateral-error")

    assert model.feature_scales[lateral_index] == phenotype_plan.minimum_feature_scale


def test_insufficient_class_evidence_is_rejected() -> None:
    """Every phenotype needs separate fit and calibration evidence."""
    phenotype_plan = plan()
    examples = separated_examples(phenotype_plan)
    missing_calibration = tuple(
        item
        for item in examples
        if not (
            item.phenotype_id == "late-braking"
            and item.split is PhenotypeTrainingSplit.CALIBRATION
            and item.independence_unit_id.endswith("-3")
        )
    )

    with pytest.raises(ValueError, match="insufficient calibration"):
        fit_failure_phenotype_model(phenotype_plan, missing_calibration)


def test_independence_units_cannot_cross_model_development_rows() -> None:
    """Repeated incident units cannot inflate fit or calibration evidence."""
    phenotype_plan = plan()
    examples = separated_examples(phenotype_plan)
    duplicate_unit = replace(
        examples[-1],
        independence_unit_id=examples[0].independence_unit_id,
    )

    with pytest.raises(ValueError, match="independence unit IDs"):
        fit_failure_phenotype_model(
            phenotype_plan,
            (*examples[:-1], duplicate_unit),
        )


def test_unknown_label_and_wrong_dimension_are_rejected() -> None:
    """Training labels and feature dimensions must match the frozen plan."""
    phenotype_plan = plan()
    examples = separated_examples(phenotype_plan)
    unknown = replace(examples[0], phenotype_id="unreviewed-failure")
    with pytest.raises(ValueError, match="unknown phenotype"):
        fit_failure_phenotype_model(
            phenotype_plan,
            (unknown, *examples[1:]),
        )

    wrong_dimension = replace(examples[0], feature_values=(0.0,))
    with pytest.raises(ValueError, match="dimension mismatch"):
        fit_failure_phenotype_model(
            phenotype_plan,
            (wrong_dimension, *examples[1:]),
        )


def test_model_and_query_are_bound_to_exact_plan() -> None:
    """Changed taxonomy or feature design invalidates model and query reuse."""
    first_plan = plan()
    model = fit_failure_phenotype_model(
        first_plan,
        separated_examples(first_plan),
    )
    changed_plan = replace(
        first_plan,
        phenotype_taxonomy_digest=sha256_digest("changed-taxonomy"),
    )

    with pytest.raises(ValueError, match="model does not belong"):
        classify_failure_phenotype(
            changed_plan,
            model,
            query(changed_plan, (0.0, 0.0)),
        )

    wrong_query = replace(
        query(first_plan, (0.0, 0.0)),
        plan_content_digest=changed_plan.content_digest(),
    )
    with pytest.raises(ValueError, match="query does not belong"):
        classify_failure_phenotype(first_plan, model, wrong_query)


def test_confirmatory_evidence_use_is_rejected() -> None:
    """Phenotype discovery cannot be relabeled as confirmatory evidence."""
    with pytest.raises(ValueError, match="discovery-only"):
        replace(plan(), evidence_use=EvidenceUse.CONFIRMATORY)
