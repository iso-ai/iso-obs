"""Tests for paired causal perturbation-response assessment."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.causal_perturbations import (
    CausalCampaignDisposition,
    CausalEffectDisposition,
    CausalPerturbationPlan,
    PairedPerturbationObservation,
    PerturbationContrast,
    assess_causal_perturbations,
)
from iso_obs.evidence import sha256_digest
from iso_obs.simulation import EvidenceUse


def contrast(
    contrast_id: str,
    *,
    magnitude: float = 0.2,
    minimum_pairs: int = 100,
) -> PerturbationContrast:
    """Build one predeclared perturbation contrast."""
    return PerturbationContrast(
        contrast_id=contrast_id,
        perturbation_spec_digest=sha256_digest(f"spec-{contrast_id}"),
        factor_id=f"factor-{contrast_id}",
        magnitude=magnitude,
        unit="normalized",
        minimum_material_failure_rate_change=0.1,
        minimum_pair_count=minimum_pairs,
    )


def plan(
    *,
    contrasts: tuple[PerturbationContrast, ...] | None = None,
) -> CausalPerturbationPlan:
    """Build a deterministic confirmatory intervention plan."""
    return CausalPerturbationPlan(
        plan_id="warehouse-causal-perturbations",
        plan_version="1",
        target_population_digest=sha256_digest("target-population"),
        simulation_evidence_digest=sha256_digest("simulator"),
        system_artifact_digest=sha256_digest("system"),
        failure_outcome_definition_digest=sha256_digest("collision"),
        intervention_protocol_digest=sha256_digest("intervention-protocol"),
        pairing_protocol_digest=sha256_digest("pairing-protocol"),
        contrasts=contrasts or (contrast("rain-intensity"),),
        familywise_error_rate=0.1,
        limitations=("Low-speed warehouse operation only.",),
    )


def paired_observations(
    causal_plan: CausalPerturbationPlan,
    contrast_id: str,
    pair_count: int,
    *,
    harm_count: int = 0,
    benefit_count: int = 0,
    both_fail_count: int = 0,
) -> tuple[PairedPerturbationObservation, ...]:
    """Build paired outcomes with exact discordant transition counts."""
    if harm_count + benefit_count + both_fail_count > pair_count:
        raise ValueError("test transition counts exceed pair count")
    plan_digest = causal_plan.content_digest()
    observations: list[PairedPerturbationObservation] = []
    for index in range(pair_count):
        if index < harm_count:
            baseline_failed = False
            perturbed_failed = True
        elif index < harm_count + benefit_count:
            baseline_failed = True
            perturbed_failed = False
        elif index < harm_count + benefit_count + both_fail_count:
            baseline_failed = True
            perturbed_failed = True
        else:
            baseline_failed = False
            perturbed_failed = False
        pair_id = f"{contrast_id}-pair-{index}"
        observations.append(
            PairedPerturbationObservation(
                plan_content_digest=plan_digest,
                contrast_id=contrast_id,
                pair_id=pair_id,
                matched_context_digest=sha256_digest(f"context-{pair_id}"),
                baseline_evidence_digest=sha256_digest(f"baseline-{pair_id}"),
                perturbed_evidence_digest=sha256_digest(
                    f"perturbed-{contrast_id}-{pair_id}"
                ),
                baseline_failed=baseline_failed,
                perturbed_failed=perturbed_failed,
            )
        )
    return tuple(observations)


def test_material_failure_increase_is_supported() -> None:
    """A positive lower bound beyond threshold supports harmful causality."""
    causal_plan = plan()

    report = assess_causal_perturbations(
        causal_plan,
        paired_observations(
            causal_plan,
            "rain-intensity",
            100,
            harm_count=50,
        ),
    )

    assert report.disposition is CausalCampaignDisposition.HARMFUL_EFFECT_IDENTIFIED
    effect = report.contrast_assessments[0].effect
    assert effect is not None
    assert effect.disposition is CausalEffectDisposition.CAUSAL_INCREASE_SUPPORTED
    assert effect.failure_rate_change_estimate == 0.5
    assert effect.effect_lower_bound > effect.minimum_material_change
    assert report.harm_priority_ranking == ("rain-intensity",)


def test_material_failure_decrease_is_supported() -> None:
    """A negative upper bound beyond threshold supports a beneficial effect."""
    causal_plan = plan()

    report = assess_causal_perturbations(
        causal_plan,
        paired_observations(
            causal_plan,
            "rain-intensity",
            100,
            benefit_count=50,
        ),
    )

    assert report.disposition is CausalCampaignDisposition.NO_HARMFUL_EFFECT_SUPPORTED
    effect = report.contrast_assessments[0].effect
    assert effect is not None
    assert effect.disposition is CausalEffectDisposition.CAUSAL_DECREASE_SUPPORTED


def test_public_effect_rejects_inconsistent_counts_and_bounds() -> None:
    """Effect artifacts cannot contradict their sufficient statistics."""
    causal_plan = plan()
    report = assess_causal_perturbations(
        causal_plan,
        paired_observations(
            causal_plan,
            "rain-intensity",
            100,
            harm_count=50,
        ),
    )
    effect = report.contrast_assessments[0].effect
    assert effect is not None

    with pytest.raises(ValueError, match="totals are inconsistent"):
        replace(effect, baseline_failure_count=1)
    with pytest.raises(ValueError, match="radius is inconsistent"):
        replace(effect, confidence_radius=effect.confidence_radius + 0.1)


def test_equivalence_requires_full_interval_inside_material_band() -> None:
    """A precise null effect can support bounded practical equivalence."""
    causal_plan = plan()

    report = assess_causal_perturbations(
        causal_plan,
        paired_observations(
            causal_plan,
            "rain-intensity",
            600,
        ),
    )

    effect = report.contrast_assessments[0].effect
    assert effect is not None
    assert effect.disposition is CausalEffectDisposition.EQUIVALENCE_SUPPORTED
    assert effect.effect_lower_bound >= -effect.minimum_material_change
    assert effect.effect_upper_bound <= effect.minimum_material_change


def test_interval_overlap_is_inconclusive() -> None:
    """An interval crossing the material threshold cannot establish harm."""
    causal_plan = plan()

    report = assess_causal_perturbations(
        causal_plan,
        paired_observations(
            causal_plan,
            "rain-intensity",
            100,
            harm_count=20,
        ),
    )

    assert report.disposition is CausalCampaignDisposition.INCONCLUSIVE
    effect = report.contrast_assessments[0].effect
    assert effect is not None
    assert effect.disposition is CausalEffectDisposition.INCONCLUSIVE
    assert effect.effect_lower_bound <= effect.minimum_material_change
    assert effect.effect_upper_bound > effect.minimum_material_change


def test_under_sampled_contrast_abstains_without_effect() -> None:
    """A contrast below its fixed pair count reports insufficient evidence."""
    causal_plan = plan()

    report = assess_causal_perturbations(
        causal_plan,
        paired_observations(
            causal_plan,
            "rain-intensity",
            99,
            harm_count=99,
        ),
    )

    assert report.disposition is CausalCampaignDisposition.INSUFFICIENT_EVIDENCE
    assessment = report.contrast_assessments[0]
    assert assessment.effect is None
    assert assessment.disposition is CausalEffectDisposition.INSUFFICIENT_EVIDENCE


def test_harm_is_noncompensatory_and_priority_is_conservative() -> None:
    """A beneficial contrast cannot offset a harmful perturbation."""
    contrasts = (
        contrast("rain-intensity"),
        contrast("sensor-latency"),
    )
    causal_plan = plan(contrasts=contrasts)
    evidence = (
        *paired_observations(
            causal_plan,
            "rain-intensity",
            150,
            harm_count=90,
        ),
        *paired_observations(
            causal_plan,
            "sensor-latency",
            150,
            benefit_count=90,
        ),
    )

    report = assess_causal_perturbations(causal_plan, evidence)

    assert report.disposition is CausalCampaignDisposition.HARMFUL_EFFECT_IDENTIFIED
    assert report.harm_priority_ranking == (
        "rain-intensity",
        "sensor-latency",
    )
    assert report.allocated_error_rate_per_contrast == pytest.approx(0.05)


def test_report_is_order_invariant_and_content_addressed() -> None:
    """Input ordering cannot alter the exact causal report."""
    causal_plan = plan()
    evidence = paired_observations(
        causal_plan,
        "rain-intensity",
        100,
        harm_count=50,
    )

    first = assess_causal_perturbations(causal_plan, evidence)
    second = assess_causal_perturbations(
        causal_plan,
        tuple(reversed(evidence)),
    )

    assert first == second
    assert first.content_digest() == second.content_digest()
    assert first.evidence_use is EvidenceUse.CONFIRMATORY


def test_duplicate_pair_within_contrast_is_rejected() -> None:
    """Repeated matched pairs cannot inflate one contrast's evidence."""
    causal_plan = plan()
    first = paired_observations(
        causal_plan,
        "rain-intensity",
        1,
    )[0]
    duplicate = replace(
        first,
        perturbed_evidence_digest=sha256_digest("other-perturbed-evidence"),
    )

    with pytest.raises(ValueError, match="unique within each contrast"):
        assess_causal_perturbations(causal_plan, (first, duplicate))


def test_reused_baseline_must_remain_consistent_across_contrasts() -> None:
    """Shared baselines may be reused only with identical context and outcome."""
    causal_plan = plan(
        contrasts=(
            contrast("rain-intensity"),
            contrast("sensor-latency"),
        )
    )
    first = paired_observations(
        causal_plan,
        "rain-intensity",
        1,
    )[0]
    second = replace(
        paired_observations(
            causal_plan,
            "sensor-latency",
            1,
        )[0],
        pair_id=first.pair_id,
        matched_context_digest=first.matched_context_digest,
        baseline_evidence_digest=first.baseline_evidence_digest,
        baseline_failed=not first.baseline_failed,
    )

    with pytest.raises(ValueError, match="preserve matched context and baseline"):
        assess_causal_perturbations(causal_plan, (first, second))


def test_context_and_baseline_cannot_create_fake_independent_pairs() -> None:
    """One replay context or baseline cannot masquerade as multiple pairs."""
    causal_plan = plan()
    first = paired_observations(
        causal_plan,
        "rain-intensity",
        1,
    )[0]
    second = replace(
        first,
        pair_id="different-pair",
        perturbed_evidence_digest=sha256_digest("different-perturbed"),
    )
    with pytest.raises(ValueError, match="matched context digests"):
        assess_causal_perturbations(causal_plan, (first, second))

    second = replace(
        first,
        pair_id="different-pair",
        matched_context_digest=sha256_digest("different-context"),
        perturbed_evidence_digest=sha256_digest("different-perturbed"),
    )
    with pytest.raises(ValueError, match="baseline evidence"):
        assess_causal_perturbations(causal_plan, (first, second))


def test_plan_requires_nonzero_intervention_and_confirmatory_use() -> None:
    """Causal claims require a real intervention and fixed evidence design."""
    with pytest.raises(ValueError, match="non-zero"):
        contrast("rain-intensity", magnitude=0.0)

    with pytest.raises(ValueError, match="must be confirmatory"):
        replace(plan(), evidence_use=EvidenceUse.DISCOVERY_ONLY)
