"""Identify failure-causing perturbations with matched simulator replays.

This example compares controlled perturbations against the exact same replay
contexts. It demonstrates simultaneous confidence intervals, material-effect
thresholds, and a non-compensatory campaign conclusion.
"""

from __future__ import annotations

from iso_obs.causal_perturbations import (
    CausalPerturbationPlan,
    PairedPerturbationObservation,
    PerturbationContrast,
    assess_causal_perturbations,
)
from iso_obs.evidence import sha256_digest


def build_plan() -> CausalPerturbationPlan:
    """Declare the intervention family before observing outcomes."""
    return CausalPerturbationPlan(
        plan_id="blind-intersection-perturbations",
        plan_version="1",
        target_population_digest=sha256_digest("low-speed warehouse intersections"),
        simulation_evidence_digest=sha256_digest("warehouse-twin-v4.2"),
        system_artifact_digest=sha256_digest("navigation-policy-v18"),
        failure_outcome_definition_digest=sha256_digest(
            "minimum stopping margin below zero"
        ),
        intervention_protocol_digest=sha256_digest(
            "one-factor matched replay protocol v1"
        ),
        pairing_protocol_digest=sha256_digest("identical state seed and random stream"),
        contrasts=(
            PerturbationContrast(
                contrast_id="observation-latency",
                perturbation_spec_digest=sha256_digest(
                    "add 120 milliseconds observation latency"
                ),
                factor_id="observation_latency_ms",
                magnitude=120.0,
                unit="ms",
                minimum_material_failure_rate_change=0.08,
                minimum_pair_count=160,
            ),
            PerturbationContrast(
                contrast_id="surface-friction-jitter",
                perturbation_spec_digest=sha256_digest(
                    "sample friction within plus or minus 0.03"
                ),
                factor_id="surface_friction_jitter",
                magnitude=0.03,
                unit="coefficient",
                minimum_material_failure_rate_change=0.15,
                minimum_pair_count=600,
            ),
        ),
        familywise_error_rate=0.05,
        limitations=(
            "The causal claim is conditional on isolated interventions.",
            "The target population excludes outdoor and high-speed operation.",
        ),
    )


def build_observations(
    plan: CausalPerturbationPlan,
) -> tuple[PairedPerturbationObservation, ...]:
    """Create deterministic matched outcomes for the demonstration."""
    outcomes = {
        "observation-latency": (160, 80),
        "surface-friction-jitter": (600, 0),
    }
    plan_digest = plan.content_digest()
    observations: list[PairedPerturbationObservation] = []
    for contrast_id, (pair_count, harm_count) in outcomes.items():
        for index in range(pair_count):
            pair_id = f"{contrast_id}-pair-{index:04d}"
            observations.append(
                PairedPerturbationObservation(
                    plan_content_digest=plan_digest,
                    contrast_id=contrast_id,
                    pair_id=pair_id,
                    matched_context_digest=sha256_digest(
                        f"matched-context-{contrast_id}-{index:04d}"
                    ),
                    baseline_evidence_digest=sha256_digest(
                        f"baseline-{contrast_id}-{index:04d}"
                    ),
                    perturbed_evidence_digest=sha256_digest(
                        f"perturbed-{contrast_id}-{index:04d}"
                    ),
                    baseline_failed=False,
                    perturbed_failed=index < harm_count,
                )
            )
    return tuple(observations)


def main() -> None:
    """Run the campaign and print the decision-relevant evidence."""
    plan = build_plan()
    report = assess_causal_perturbations(plan, build_observations(plan))

    print("Causal perturbation campaign")
    print(f"  disposition: {report.disposition.value}")
    print(f"  familywise error rate: {report.familywise_error_rate:.1%}")
    print("  simultaneous contrasts:")
    for assessment in report.contrast_assessments:
        effect = assessment.effect
        if effect is None:
            print(
                f"    {assessment.contrast_id}: "
                f"{assessment.disposition.value} "
                f"({assessment.pair_count}/{assessment.minimum_pair_count} pairs)"
            )
            continue
        print(
            f"    {assessment.contrast_id}: "
            f"{effect.failure_rate_change_estimate:+.1%} "
            f"[{effect.effect_lower_bound:+.1%}, "
            f"{effect.effect_upper_bound:+.1%}] "
            f"vs ±{effect.minimum_material_change:.1%} material band "
            f"→ {effect.disposition.value}"
        )
    print(f"  harm priority: {', '.join(report.harm_priority_ranking)}")
    print(f"  report digest: {report.content_digest()}")
    print("  claim boundary: causal only within the declared replay design")


if __name__ == "__main__":
    main()
