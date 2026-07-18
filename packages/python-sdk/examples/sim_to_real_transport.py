"""Test simulator transport without allowing rare failures to average away."""

from __future__ import annotations

from iso_obs.evidence import NamedDigest, ValidityRange, sha256_digest
from iso_obs.transportability import (
    DiscrepancyKind,
    DiscrepancySource,
    PairedAnchorObservation,
    TransportStratum,
    TransportValidationPlan,
    validate_transportability,
)


def build_plan() -> TransportValidationPlan:
    """Declare the paired anchor envelope and non-compensatory strata."""
    return TransportValidationPlan(
        plan_id="warehouse-twin-transport",
        plan_version="1",
        simulator_manifest_digest=sha256_digest("warehouse-twin-v4.2"),
        real_anchor_dataset_digest=sha256_digest("physical-braking-study-v3"),
        pairing_protocol_digest=sha256_digest(
            "matched initial state and commanded speed"
        ),
        target_population_digest=sha256_digest(
            "production warehouse exposure distribution"
        ),
        partition_definition_digest=sha256_digest("floor friction below or above 0.55"),
        metric_name="minimum stopping margin",
        metric_unit="m",
        outcome_definition_digest=sha256_digest(
            "minimum signed obstacle distance during braking"
        ),
        maximum_acceptable_overall_mean_expanded_discrepancy=0.10,
        family_error_rate=0.05,
        anchor_envelope=(
            ValidityRange("floor_friction", 0.45, 0.90, "coefficient"),
            ValidityRange("robot_speed", 0.0, 2.0, "m/s"),
        ),
        discrepancy_sources=(
            DiscrepancySource(
                name="contact compliance",
                kind=DiscrepancyKind.PHYSICS_MODEL_FORM,
                characterization=(
                    "Rigid simulated contact differs from compliant tire-floor "
                    "interaction."
                ),
                evidence_digest=sha256_digest("contact-compliance-study"),
                quantified=True,
            ),
            DiscrepancySource(
                name="brake wear",
                kind=DiscrepancyKind.ACTUATOR,
                characterization="Brake response varies across maintenance age.",
                evidence_digest=sha256_digest("brake-wear-bench-study"),
                quantified=True,
            ),
        ),
        physics_evidence=(
            NamedDigest(
                "instrumented braking rig",
                sha256_digest("instrumented-braking-rig-results"),
            ),
        ),
        strata=(
            TransportStratum(
                stratum_id="common-floor",
                condition_definition_digest=sha256_digest("friction at least 0.55"),
                target_population_mass=0.98,
                planned_pair_count=500,
                maximum_acceptable_mean_expanded_discrepancy=0.10,
                maximum_possible_expanded_discrepancy=1.0,
                allocated_error_rate=0.025,
            ),
            TransportStratum(
                stratum_id="rare-low-friction",
                condition_definition_digest=sha256_digest("friction below 0.55"),
                target_population_mass=0.02,
                planned_pair_count=500,
                maximum_acceptable_mean_expanded_discrepancy=0.08,
                maximum_possible_expanded_discrepancy=1.0,
                allocated_error_rate=0.025,
            ),
        ),
        limitations=(
            "The evidence does not cover speeds above two meters per second.",
            "Population weights are treated as fixed and must be monitored.",
        ),
    )


def build_observations(
    plan: TransportValidationPlan,
) -> tuple[PairedAnchorObservation, ...]:
    """Build matched anchor pairs with a localized transport discrepancy."""
    residuals = {"common-floor": 0.01, "rare-low-friction": 0.18}
    plan_digest = plan.content_digest()
    observations: list[PairedAnchorObservation] = []
    for stratum in plan.strata:
        residual = residuals[stratum.stratum_id]
        for index in range(stratum.planned_pair_count):
            anchor_id = f"{stratum.stratum_id}-anchor-{index:04d}"
            observations.append(
                PairedAnchorObservation(
                    plan_content_digest=plan_digest,
                    stratum_id=stratum.stratum_id,
                    pair_index=index,
                    anchor_id=anchor_id,
                    simulation_run_id=f"sim-{anchor_id}",
                    real_run_id=f"real-{anchor_id}",
                    simulation_value=0.50,
                    real_value=0.50 + residual,
                    simulation_uncertainty_bound=0.01,
                    measurement_uncertainty_bound=0.01,
                    evidence_digest=sha256_digest(f"evidence-{anchor_id}"),
                )
            )
    return tuple(observations)


def main() -> None:
    """Validate transport and print global and stratum-level evidence."""
    plan = build_plan()
    report = validate_transportability(plan, build_observations(plan))

    print("Simulator-to-real transport validation")
    print(f"  conclusion: {report.conclusion.value}")
    print("  anchor envelope: friction 0.45–0.90; speed 0.0–2.0 m/s")
    if (
        report.overall_mean_expanded_discrepancy is None
        or report.overall_confidence_lower is None
        or report.overall_confidence_upper is None
    ):
        raise RuntimeError("the complete example design must produce bounds")
    print(
        "  overall expanded discrepancy: "
        f"{report.overall_mean_expanded_discrepancy:.3f} "
        f"[{report.overall_confidence_lower:.3f}, "
        f"{report.overall_confidence_upper:.3f}] "
        f"vs {report.maximum_acceptable_overall_mean_expanded_discrepancy:.3f}"
    )
    print("  strata:")
    for estimate in report.stratum_estimates:
        if (
            estimate.mean_expanded_discrepancy is None
            or estimate.confidence_lower is None
            or estimate.confidence_upper is None
        ):
            raise RuntimeError("complete example strata must produce bounds")
        print(
            f"    {estimate.stratum_id} ({estimate.target_population_mass:.0%}): "
            f"{estimate.mean_expanded_discrepancy:.3f} "
            f"[{estimate.confidence_lower:.3f}, "
            f"{estimate.confidence_upper:.3f}] "
            f"vs {estimate.maximum_acceptable_mean_expanded_discrepancy:.3f} "
            f"→ {estimate.conclusion.value}"
        )
    print(f"  limiting strata: {', '.join(report.limiting_stratum_ids)}")
    print(f"  report digest: {report.content_digest()}")
    print("  claim boundary: support never extrapolates past the anchor envelope")


if __name__ == "__main__":
    main()
