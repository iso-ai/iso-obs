"""Tests for deterministic failure and simulation evidence artifacts."""

from __future__ import annotations

from dataclasses import replace

import pytest

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


def digest(name: str) -> NamedDigest:
    """Build a named digest for a stable test value."""
    return NamedDigest(name=name, digest=sha256_digest(name))


def simulation() -> SimulationEvidenceManifest:
    """Build a representative hybrid-world simulation manifest."""
    return SimulationEvidenceManifest(
        simulator_name="warehouse-twin",
        simulator_version="4.2.0",
        model_type=SimulationModelType.HYBRID,
        artifact_digest=sha256_digest("warehouse-twin:4.2.0"),
        intended_use="Low-speed indoor mobile-robot navigation",
        validity_envelope=(
            ValidityRange("robot_speed", 0.0, 2.0, "m/s"),
            ValidityRange("floor_friction", 0.45, 0.90, "coefficient"),
        ),
        uncertainty_sources=(
            UncertaintySource(
                name="pedestrian arrival",
                kind=UncertaintyKind.ALEATORIC,
                characterization="Empirical arrival distribution",
                quantified=True,
            ),
            UncertaintySource(
                name="contact dynamics",
                kind=UncertaintyKind.MODEL_FORM_EPISTEMIC,
                characterization="Rigid-body contact approximation",
                quantified=False,
            ),
        ),
        verification_evidence=(digest("solver-convergence"),),
        calibration_evidence=(digest("calibration-2026q2"),),
        validation_evidence=(digest("motion-capture-validation"),),
        real_world_anchors=(digest("warehouse-7-trials"),),
        known_omissions=("Tire wear is not modeled.",),
        solver_name="rigid-body-solver",
        solver_version="3.1",
        time_step_seconds=0.01,
    )


def reproduction() -> ReproductionManifest:
    """Build exact failure-reproduction inputs."""
    return ReproductionManifest(
        scenario_id="blind-intersection",
        scenario_version="3",
        environment_id="warehouse-twin",
        environment_version="4.2.0",
        system_versions=("policy-v17", "planner-v8"),
        seed=42,
        perturbation_realization_digest=sha256_digest("perturbation-stream"),
        sdk_version="0.1.0",
        schema_version="0.1.0",
        configuration_digests=(digest("policy-config"), digest("sim-config")),
        trace_digest=sha256_digest("trace"),
        command="python evaluate.py --scenario blind-intersection --seed 42",
    )


def analysis(
    kind: AnalysisKind = AnalysisKind.SUSTAINED_DIVERGENCE,
) -> AnalysisProvenance:
    """Build versioned analysis provenance of a requested kind."""
    return AnalysisProvenance(
        kind=kind,
        analyzer="iso_obs.divergence",
        analyzer_version="1",
        input_digests=(digest("baseline-trace"), digest("candidate-trace")),
        configuration_digest=sha256_digest("signal-specs"),
        method="exact-step sustained tolerance",
        random_seed=None,
    )


def bundle() -> FailureEvidenceBundle:
    """Build a sustained-divergence evidence bundle."""
    return FailureEvidenceBundle(
        failure_category="unsafe_stop_margin",
        scenario_family="blind_intersection",
        scenario_version="3",
        perturbation_family="observation_latency",
        invariant_id="minimum-stopping-distance",
        invariant_version="2",
        signal_group=("stopping_margin_m", "velocity_mps"),
        execution_phase="closed_loop_control",
        component="navigation_stack",
        evidence_level=EvidenceLevel.SUSTAINED_DIVERGENCE,
        summary="Stopping margin diverged before the safety monitor activated.",
        affected_system_versions=("policy-v17",),
        evidence_event_ids=("evt-101", "evt-102"),
        reproduction=reproduction(),
        simulation=simulation(),
        analyses=(analysis(),),
        limitations=(
            "Observed divergence does not establish a causal mechanism.",
            "Contact dynamics have not been validated for collision severity.",
        ),
    )


def test_bundle_serialization_and_digest_are_deterministic() -> None:
    first = bundle()
    second = bundle()

    assert first.to_json() == second.to_json()
    assert first.content_digest() == second.content_digest()
    assert first.content_digest().startswith("sha256:")


def test_manifest_has_independent_content_identity() -> None:
    manifest = simulation()

    assert manifest.to_json().startswith("{")
    assert manifest.content_digest() == sha256_digest(manifest.to_json())


def test_set_like_manifest_fields_are_order_invariant() -> None:
    original = simulation()
    reordered = replace(
        original,
        validity_envelope=tuple(reversed(original.validity_envelope)),
        uncertainty_sources=tuple(reversed(original.uncertainty_sources)),
    )

    assert original == reordered
    assert original.content_digest() == reordered.content_digest()


def test_fingerprint_excludes_observation_specific_fields() -> None:
    original = bundle()
    another_observation = replace(
        original,
        summary="A different human-readable account.",
        affected_system_versions=("policy-v18",),
        evidence_event_ids=("evt-900",),
        reproduction=replace(original.reproduction, seed=900),
    )

    assert original.failure_fingerprint() == another_observation.failure_fingerprint()
    assert original.content_digest() != another_observation.content_digest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failure_category", "collision"),
        ("scenario_family", "loading_dock"),
        ("scenario_version", "4"),
        ("perturbation_family", "floor_friction"),
        ("invariant_id", "maximum-impact-speed"),
        ("invariant_version", "3"),
        ("execution_phase", "emergency_stop"),
        ("component", "safety_monitor"),
        ("signal_group", ("impact_speed_mps",)),
    ],
)
def test_material_identity_change_changes_fingerprint(
    field: str,
    value: str | tuple[str, ...],
) -> None:
    original = bundle()
    changed = replace(original, **{field: value})

    assert original.failure_fingerprint() != changed.failure_fingerprint()


def test_signal_order_does_not_change_failure_fingerprint() -> None:
    original = bundle()
    reordered = replace(original, signal_group=tuple(reversed(original.signal_group)))

    assert original.failure_fingerprint() == reordered.failure_fingerprint()
    assert original.content_digest() == reordered.content_digest()


@pytest.mark.parametrize(
    ("level", "required"),
    [
        (EvidenceLevel.OBSERVED_DIFFERENCE, AnalysisKind.TRACE_COMPARISON),
        (EvidenceLevel.SUSTAINED_DIVERGENCE, AnalysisKind.SUSTAINED_DIVERGENCE),
        (EvidenceLevel.OUTCOME_ASSOCIATION, AnalysisKind.OUTCOME_ASSOCIATION),
        (EvidenceLevel.REPLICATED_ASSOCIATION, AnalysisKind.REPLICATION),
        (EvidenceLevel.INTERVENTION_SUPPORTED, AnalysisKind.INTERVENTION),
    ],
)
def test_each_claim_level_requires_matching_provenance(
    level: EvidenceLevel,
    required: AnalysisKind,
) -> None:
    candidate = replace(bundle(), evidence_level=level, analyses=(analysis(required),))

    assert candidate.evidence_level is level


def test_overstated_claim_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires replication"):
        replace(bundle(), evidence_level=EvidenceLevel.REPLICATED_ASSOCIATION)


def test_string_enums_are_canonicalized() -> None:
    source = UncertaintySource(
        name="friction",
        kind="parametric_epistemic",  # type: ignore[arg-type]
        characterization="Posterior distribution from pull tests",
        quantified=True,
    )

    assert source.kind is UncertaintyKind.PARAMETRIC_EPISTEMIC


@pytest.mark.parametrize("value", ["abc", "sha256:ABC", "sha256:" + "0" * 63])
def test_malformed_digest_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        NamedDigest(name="bad", digest=value)


def test_invalid_validity_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        ValidityRange("speed", 2.0, 1.0, "m/s")


def test_partial_solver_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        replace(simulation(), solver_version=None)


def test_duplicate_uncertainty_sources_are_rejected() -> None:
    source = simulation().uncertainty_sources[0]

    with pytest.raises(ValueError, match="uncertainty sources must be unique"):
        replace(simulation(), uncertainty_sources=(source, source))


def test_empty_limitations_are_rejected() -> None:
    with pytest.raises(ValueError, match="limitations must not be empty"):
        replace(bundle(), limitations=())


def test_seed_rejects_boolean_despite_integer_subclassing() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        replace(reproduction(), seed=True)  # type: ignore[arg-type]
