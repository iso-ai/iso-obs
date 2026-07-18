"""End-to-end tests for local dataset reliability CLI commands."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from iso_obs.dataset_io import (
    DATASET_BUNDLE_SCHEMA_VERSION,
    DATASET_SPLIT_REQUEST_SCHEMA_VERSION,
    DatasetBundle,
    DatasetSplitRequest,
)
from iso_obs.dataset_reliability import (
    DatasetEvidenceRole,
    DatasetLabelAssertion,
    DatasetManifest,
    DatasetReliabilityAuditPlan,
    DatasetScope,
    EpisodeManifest,
    LabelStatus,
    ModeFamily,
)
from iso_obs.dataset_splitting import (
    DatasetSplitAssignment,
    DatasetSplitPlan,
    DatasetSplitRole,
    DatasetSplitUnit,
    DatasetUnitOrigin,
    SplitRoleMinimum,
)
from iso_obs.dataset_synchronization import (
    ChannelSynchronizationRule,
    ChannelTimingTrace,
    ClockBasis,
    ClockDeclaration,
    DatasetSynchronizationPlan,
    InterpolationPolicy,
)
from iso_obs.evidence import NamedDigest, sha256_digest
from iso_obs_cli.main import app


def dataset_bundle() -> DatasetBundle:
    """Build a bundle supporting all local dataset commands."""
    manifest = DatasetManifest(
        dataset_id="robot-failures",
        dataset_version="1",
        content_digest=sha256_digest("raw"),
        source_uri="file:///datasets/robot-failures",
        license_id="Apache-2.0",
        scope=DatasetScope(
            domain_namespace="robotics",
            target_population_digest=sha256_digest("population"),
            collection_protocol_digest=sha256_digest("protocol"),
            modality_ids=("camera",),
            evidence_roles=(DatasetEvidenceRole.FAILURE_LABEL_LEARNING,),
        ),
    )
    episode = EpisodeManifest(
        dataset_manifest_digest=manifest.manifest_digest(),
        episode_id="episode-1",
        independence_unit_id="run-1",
        content_digest=sha256_digest("episode"),
        start_seconds=0.0,
        end_seconds=10.0,
        modality_digests=(NamedDigest("camera", sha256_digest("camera")),),
    )
    assertions = tuple(
        DatasetLabelAssertion(
            assertion_id=f"{family.value}-assertion",
            episode_id=episode.episode_id,
            label_namespace=f"reliability/{family.value}/v1",
            mode_family=family,
            start_seconds=0.0,
            end_seconds=10.0,
            candidate_values=(value,),
            status=LabelStatus.ADJUDICATED,
            evidence=(),
        )
        for family, value in (
            (ModeFamily.OPERATING, "autonomous"),
            (ModeFamily.FAILURE_STATE, "contact_loss"),
        )
    )
    trace = ChannelTimingTrace(
        episode_id=episode.episode_id,
        channel_id="camera-front",
        modality_id="camera",
        clock_id="controller",
        content_digest=sha256_digest("timing"),
        timestamps_seconds=(0.0, 5.0, 10.0),
        timestamp_uncertainty_seconds=0.001,
    )
    return DatasetBundle(
        schema_version=DATASET_BUNDLE_SCHEMA_VERSION,
        manifest=manifest,
        episodes=(episode,),
        label_assertions=assertions,
        timing_traces=(trace,),
    )


def synchronization_plan(bundle: DatasetBundle) -> DatasetSynchronizationPlan:
    """Build a plan aligned to the bundle's controller clock."""
    return DatasetSynchronizationPlan(
        plan_id="camera-sync",
        plan_version="1",
        dataset_manifest_digest=bundle.manifest.manifest_digest(),
        episode_id="episode-1",
        reference_clock_id="controller",
        clocks=(
            ClockDeclaration(
                clock_id="controller",
                basis=ClockBasis.HOST_MONOTONIC,
                implementation_digest=sha256_digest("controller-clock"),
            ),
        ),
        clock_alignments=(),
        channel_rules=(
            ChannelSynchronizationRule(
                channel_id="camera-front",
                modality_id="camera",
                required=True,
                interpolation_policy=InterpolationPolicy.NEAREST,
                maximum_alignment_error_seconds=0.01,
                maximum_sample_gap_seconds=5.1,
            ),
        ),
    )


def audit_plan(bundle: DatasetBundle) -> DatasetReliabilityAuditPlan:
    """Build a plan requiring both supplied label families."""
    return DatasetReliabilityAuditPlan(
        plan_id="label-audit",
        plan_version="1",
        dataset_manifest_digest=bundle.manifest.manifest_digest(),
        label_taxonomy_digest=sha256_digest("taxonomy"),
        required_mode_families=(
            ModeFamily.OPERATING,
            ModeFamily.FAILURE_STATE,
        ),
    )


def split_request(*, contaminated: bool) -> DatasetSplitRequest:
    """Build a valid or deliberately contaminated split proposal."""
    scope = dataset_bundle().manifest.scope
    plan = DatasetSplitPlan(
        plan_id="split",
        plan_version="1",
        dataset_collection_digest=sha256_digest("collection"),
        scope=scope,
        required_roles=(
            DatasetSplitRole.MODEL_FIT,
            DatasetSplitRole.REAL_WORLD_HOLDOUT,
        ),
        role_minimums=(
            SplitRoleMinimum(DatasetSplitRole.MODEL_FIT, 1),
            SplitRoleMinimum(DatasetSplitRole.REAL_WORLD_HOLDOUT, 1),
        ),
        holdout_roles=(DatasetSplitRole.REAL_WORLD_HOLDOUT,),
        protected_holdout_attributes=(),
    )
    units = tuple(
        DatasetSplitUnit(
            unit_id=unit_id,
            episode_id=f"episode-{unit_id}",
            episode_content_digest=sha256_digest(f"episode-{unit_id}"),
            dataset_manifest_digest=sha256_digest(f"manifest-{unit_id}"),
            independence_unit_id="shared" if contaminated else f"run-{unit_id}",
            origin=DatasetUnitOrigin.REAL_OBSERVED,
        )
        for unit_id in ("fit", "holdout")
    )
    assignments = (
        DatasetSplitAssignment("fit", DatasetSplitRole.MODEL_FIT),
        DatasetSplitAssignment("holdout", DatasetSplitRole.REAL_WORLD_HOLDOUT),
    )
    return DatasetSplitRequest(
        schema_version=DATASET_SPLIT_REQUEST_SCHEMA_VERSION,
        plan=plan,
        units=units,
        assignments=assignments,
    )


def write_artifact(path: Path, payload: str) -> Path:
    """Write one canonical test artifact and return its path."""
    path.write_text(payload, encoding="utf-8")
    return path


def test_inspect_emits_versioned_inventory(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Inspect emits machine-readable structural coverage."""
    path = write_artifact(tmp_path / "bundle.json", dataset_bundle().to_json())

    result = runner.invoke(app, ["dataset", "inspect", str(path)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "iso-obs.dataset-bundle-inspection.v1"
    assert payload["episode_count"] == 1
    assert payload["assertion_count"] == 2


def test_synchronize_distinguishes_alignment_from_insufficient_evidence(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Missing required timing evidence uses the abstention exit code."""
    bundle = dataset_bundle()
    bundle_path = write_artifact(tmp_path / "bundle.json", bundle.to_json())
    empty_path = write_artifact(
        tmp_path / "bundle-empty.json",
        replace(bundle, timing_traces=()).to_json(),
    )
    plan_path = write_artifact(
        tmp_path / "sync-plan.json",
        synchronization_plan(bundle).to_json(),
    )

    aligned = runner.invoke(
        app,
        ["dataset", "synchronize", str(bundle_path), "--plan", str(plan_path)],
    )
    insufficient = runner.invoke(
        app,
        ["dataset", "synchronize", str(empty_path), "--plan", str(plan_path)],
    )

    assert aligned.exit_code == 0
    assert json.loads(aligned.stdout)["disposition"] == "aligned_within_scope"
    assert insufficient.exit_code == 5
    assert json.loads(insufficient.stdout)["disposition"] == "insufficient_evidence"


def test_audit_emits_curated_label_report(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Audit delegates label status decisions to the SDK."""
    bundle = dataset_bundle()
    bundle_path = write_artifact(tmp_path / "bundle.json", bundle.to_json())
    plan_path = write_artifact(
        tmp_path / "audit-plan.json",
        audit_plan(bundle).to_json(),
    )

    result = runner.invoke(
        app,
        ["dataset", "audit", str(bundle_path), "--plan", str(plan_path)],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["disposition"] == "curated_within_scope"


def test_audit_uses_review_exit_code_for_machine_suggestion(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """A detector suggestion remains a review obligation in automation."""
    bundle = dataset_bundle()
    suggested = replace(
        bundle.label_assertions[0],
        status=LabelStatus.SUGGESTED,
        method_digest=sha256_digest("mixed-mode-detector"),
    )
    bundle = replace(
        bundle,
        label_assertions=(suggested, *bundle.label_assertions[1:]),
    )
    bundle_path = write_artifact(tmp_path / "bundle.json", bundle.to_json())
    plan_path = write_artifact(
        tmp_path / "audit-plan.json",
        audit_plan(bundle).to_json(),
    )

    result = runner.invoke(
        app,
        ["dataset", "audit", str(bundle_path), "--plan", str(plan_path)],
    )

    assert result.exit_code == 3
    assert json.loads(result.stdout)["disposition"] == "review_required"


def test_split_uses_distinct_contamination_exit_code(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Known cross-role dependence is not flattened into generic failure."""
    request_path = write_artifact(
        tmp_path / "split.json",
        split_request(contaminated=True).to_json(),
    )

    result = runner.invoke(app, ["dataset", "split", str(request_path)])

    assert result.exit_code == 4
    assert json.loads(result.stdout)["disposition"] == "contaminated"


def test_split_accepts_valid_independent_roles(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Independent fit and holdout units support a scope-bound valid report."""
    request_path = write_artifact(
        tmp_path / "split.json",
        split_request(contaminated=False).to_json(),
    )

    result = runner.invoke(app, ["dataset", "split", str(request_path)])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["disposition"] == "valid_within_scope"


def test_unknown_input_field_fails_before_audit(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Misspelled contract fields cannot be silently ignored."""
    payload = (
        dataset_bundle()
        .to_json()
        .replace(
            '"dataset_id":"robot-failures"',
            '"dataset_id":"robot-failures","dataset_typo":"unsafe"',
        )
    )
    path = write_artifact(tmp_path / "invalid.json", payload)

    result = runner.invoke(app, ["dataset", "inspect", str(path)])

    assert result.exit_code == 2
    assert "unknown field" in result.stderr


def test_output_option_never_overwrites_an_existing_artifact(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Canonical output is created once and preserved on a repeated command."""
    bundle_path = write_artifact(
        tmp_path / "bundle.json",
        dataset_bundle().to_json(),
    )
    output_path = tmp_path / "inspection.json"

    first = runner.invoke(
        app,
        [
            "dataset",
            "inspect",
            str(bundle_path),
            "--output",
            str(output_path),
        ],
    )
    original = output_path.read_text(encoding="utf-8")
    second = runner.invoke(
        app,
        [
            "dataset",
            "inspect",
            str(bundle_path),
            "--output",
            str(output_path),
        ],
    )

    assert first.exit_code == 0
    assert first.stdout == ""
    assert second.exit_code == 2
    assert output_path.read_text(encoding="utf-8") == original
