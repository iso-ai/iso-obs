"""Tests for distribution-free sim-to-real validity assessment."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.evidence import sha256_digest
from iso_obs.sim_to_real import (
    GapDisposition,
    SimToRealValidityPlan,
    TransferDisposition,
    TransferDomain,
    TransferFeature,
    TransferObservation,
    TransferRegion,
    ValidityClaimKind,
    assess_sim_to_real_validity,
)
from iso_obs.simulation import EvidenceUse


def region(
    region_id: str,
    *,
    mass: float = 1.0,
    minimum: int = 500,
) -> TransferRegion:
    """Build one test operating region."""
    return TransferRegion(
        region_id=region_id,
        condition_definition_digest=sha256_digest(f"condition-{region_id}"),
        target_population_mass=mass,
        minimum_simulation_samples=minimum,
        minimum_real_world_samples=minimum,
    )


def plan(
    *,
    regions: tuple[TransferRegion, ...] | None = None,
) -> SimToRealValidityPlan:
    """Build a deterministic bounded validity plan."""
    return SimToRealValidityPlan(
        plan_id="warehouse-sim-to-real-validity",
        plan_version="1",
        target_population_digest=sha256_digest("target-population"),
        simulation_evidence_digest=sha256_digest("simulator-evidence"),
        real_world_anchor_digest=sha256_digest("real-world-anchor"),
        system_artifact_digest=sha256_digest("system"),
        failure_outcome_definition_digest=sha256_digest("collision"),
        features=(
            TransferFeature(
                feature_id="normalized-stopping-distance",
                definition_digest=sha256_digest("stopping-distance-feature"),
                lower_bound=0.0,
                upper_bound=1.0,
                maximum_acceptable_mean_gap=0.15,
            ),
        ),
        regions=regions or (region("nominal"),),
        maximum_acceptable_failure_rate_gap=0.15,
        familywise_error_rate=0.1,
        limitations=("Low-speed warehouse operation only.",),
    )


def observations(
    validity_plan: SimToRealValidityPlan,
    region_id: str,
    domain: TransferDomain,
    count: int,
    *,
    feature_value: float,
    failure_count: int = 0,
) -> tuple[TransferObservation, ...]:
    """Build deterministic independent observations for one domain."""
    plan_digest = validity_plan.content_digest()
    return tuple(
        TransferObservation(
            plan_content_digest=plan_digest,
            evidence_digest=sha256_digest(
                f"evidence-{region_id}-{domain.value}-{index}"
            ),
            independence_unit_id=f"{region_id}-{domain.value}-{index}",
            domain=domain,
            region_id=region_id,
            feature_values=(feature_value,),
            failed=index < failure_count,
        )
        for index in range(count)
    )


def balanced_evidence(
    validity_plan: SimToRealValidityPlan,
    *,
    simulation_value: float,
    real_value: float,
    count: int = 500,
    region_id: str = "nominal",
) -> tuple[TransferObservation, ...]:
    """Build simulation and real-world evidence for one region."""
    return (
        *observations(
            validity_plan,
            region_id,
            TransferDomain.SIMULATION,
            count,
            feature_value=simulation_value,
        ),
        *observations(
            validity_plan,
            region_id,
            TransferDomain.REAL_WORLD,
            count,
            feature_value=real_value,
        ),
    )


def test_equivalence_requires_upper_bound_within_tolerance() -> None:
    """Close domains support equivalence only when the full bound fits."""
    validity_plan = plan()

    report = assess_sim_to_real_validity(
        validity_plan,
        balanced_evidence(
            validity_plan,
            simulation_value=0.4,
            real_value=0.4,
        ),
    )

    assert report.disposition is TransferDisposition.SUPPORTED_WITHIN_SCOPE
    assert report.supported_target_population_mass == 1.0
    assert report.evidence_use is EvidenceUse.CONFIRMATORY
    region_report = report.region_assessments[0]
    assert all(
        item.disposition is GapDisposition.SUPPORTED_WITHIN_TOLERANCE
        for item in region_report.claims
    )
    assert all(
        item.absolute_gap_upper_bound <= item.maximum_acceptable_gap
        for item in region_report.claims
    )


def test_shift_requires_lower_bound_beyond_tolerance() -> None:
    """A large feature gap produces positive evidence of invalidity."""
    validity_plan = plan()

    report = assess_sim_to_real_validity(
        validity_plan,
        balanced_evidence(
            validity_plan,
            simulation_value=0.1,
            real_value=0.7,
        ),
    )

    assert report.disposition is TransferDisposition.SHIFT_DETECTED
    feature_claim = next(
        item
        for item in report.region_assessments[0].claims
        if item.kind is ValidityClaimKind.FEATURE_MEAN_GAP
    )
    assert feature_claim.disposition is GapDisposition.SHIFT_EXCEEDS_TOLERANCE
    assert feature_claim.absolute_gap_lower_bound > (
        feature_claim.maximum_acceptable_gap
    )


def test_overlap_with_tolerance_is_inconclusive_not_equivalent() -> None:
    """An interval crossing the tolerance cannot support either conclusion."""
    validity_plan = plan()

    report = assess_sim_to_real_validity(
        validity_plan,
        balanced_evidence(
            validity_plan,
            simulation_value=0.3,
            real_value=0.5,
        ),
    )

    assert report.disposition is TransferDisposition.INCONCLUSIVE
    feature_claim = next(
        item
        for item in report.region_assessments[0].claims
        if item.kind is ValidityClaimKind.FEATURE_MEAN_GAP
    )
    assert feature_claim.disposition is GapDisposition.INCONCLUSIVE
    assert feature_claim.absolute_gap_lower_bound <= 0.15
    assert feature_claim.absolute_gap_upper_bound > 0.15


def test_missing_predeclared_sample_size_is_explicit() -> None:
    """Underpowered regions emit no gap claims and remain unsupported."""
    validity_plan = plan()

    report = assess_sim_to_real_validity(
        validity_plan,
        balanced_evidence(
            validity_plan,
            simulation_value=0.4,
            real_value=0.4,
            count=499,
        ),
    )

    assert report.disposition is TransferDisposition.INSUFFICIENT_EVIDENCE
    assert report.supported_target_population_mass == 0.0
    assert report.region_assessments[0].claims == ()


def test_regional_shift_cannot_be_averaged_away() -> None:
    """One shifted region blocks overall support despite another valid region."""
    regions = (
        region("nominal", mass=0.5),
        region("occluded", mass=0.5),
    )
    validity_plan = plan(regions=regions)
    evidence = (
        *balanced_evidence(
            validity_plan,
            simulation_value=0.4,
            real_value=0.4,
            region_id="nominal",
        ),
        *balanced_evidence(
            validity_plan,
            simulation_value=0.1,
            real_value=0.9,
            region_id="occluded",
        ),
    )

    report = assess_sim_to_real_validity(validity_plan, evidence)

    assert report.disposition is TransferDisposition.SHIFT_DETECTED
    assert report.supported_target_population_mass == 0.5
    assert {item.region_id: item.disposition for item in report.region_assessments} == {
        "nominal": TransferDisposition.SUPPORTED_WITHIN_SCOPE,
        "occluded": TransferDisposition.SHIFT_DETECTED,
    }


def test_familywise_error_is_allocated_across_all_planned_claims() -> None:
    """Multiplicity allocation includes each feature and failure claim."""
    validity_plan = plan(
        regions=(
            region("nominal", mass=0.5),
            region("occluded", mass=0.5),
        )
    )

    report = assess_sim_to_real_validity(validity_plan, ())

    assert report.planned_claim_count == 4
    assert report.allocated_error_rate_per_claim == pytest.approx(0.025)
    assert (
        report.allocated_error_rate_per_claim * report.planned_claim_count
        == pytest.approx(report.familywise_error_rate)
    )


def test_result_is_order_invariant_and_content_addressed() -> None:
    """Observation ordering cannot alter the exact validity report."""
    validity_plan = plan()
    evidence = balanced_evidence(
        validity_plan,
        simulation_value=0.4,
        real_value=0.4,
    )

    first = assess_sim_to_real_validity(validity_plan, evidence)
    second = assess_sim_to_real_validity(
        validity_plan,
        tuple(reversed(evidence)),
    )

    assert first == second
    assert first.content_digest() == second.content_digest()


def test_feature_values_outside_declared_bounds_are_rejected() -> None:
    """Distribution-free guarantees require enforced bounded support."""
    validity_plan = plan()
    invalid = observations(
        validity_plan,
        "nominal",
        TransferDomain.SIMULATION,
        1,
        feature_value=1.1,
    )

    with pytest.raises(ValueError, match="outside declared bounds"):
        assess_sim_to_real_validity(validity_plan, invalid)


def test_independence_and_evidence_reuse_are_rejected() -> None:
    """Repeated rows cannot inflate validity evidence."""
    validity_plan = plan()
    first = observations(
        validity_plan,
        "nominal",
        TransferDomain.SIMULATION,
        1,
        feature_value=0.4,
    )[0]
    duplicate_unit = replace(
        first,
        evidence_digest=sha256_digest("different-evidence"),
    )
    with pytest.raises(ValueError, match="unique within each domain"):
        assess_sim_to_real_validity(
            validity_plan,
            (first, duplicate_unit),
        )

    duplicate_evidence = replace(
        first,
        independence_unit_id="another-unit",
    )
    with pytest.raises(ValueError, match="evidence digests"):
        assess_sim_to_real_validity(
            validity_plan,
            (first, duplicate_evidence),
        )


def test_matched_cross_domain_units_must_share_region() -> None:
    """Matched anchors may cross domains but cannot change region labels."""
    validity_plan = plan(
        regions=(
            region("nominal", mass=0.5),
            region("occluded", mass=0.5),
        )
    )
    simulation = observations(
        validity_plan,
        "nominal",
        TransferDomain.SIMULATION,
        1,
        feature_value=0.4,
    )[0]
    real_world = observations(
        validity_plan,
        "occluded",
        TransferDomain.REAL_WORLD,
        1,
        feature_value=0.4,
    )[0]
    real_world = replace(
        real_world,
        independence_unit_id=simulation.independence_unit_id,
    )

    with pytest.raises(ValueError, match="must share a region"):
        assess_sim_to_real_validity(
            validity_plan,
            (simulation, real_world),
        )


def test_plan_requires_confirmatory_use_and_complete_partition() -> None:
    """Validity support requires a fixed confirmatory plan and full partition."""
    with pytest.raises(ValueError, match="must be confirmatory"):
        replace(plan(), evidence_use=EvidenceUse.DISCOVERY_ONLY)

    with pytest.raises(ValueError, match="target masses must sum to one"):
        plan(
            regions=(
                region("nominal", mass=0.4),
                region("occluded", mass=0.4),
            )
        )
