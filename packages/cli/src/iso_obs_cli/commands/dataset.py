"""Local, evidence-preserving dataset reliability commands."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from iso_obs.dataset_ingestion import (
    DatasetIngestionDisposition,
    DatasetIngestionError,
    DatasetIngestionPlan,
    DatasetIngestionResult,
    ingest_dataset_source,
)
from iso_obs.dataset_io import (
    ArtifactFormatError,
    DatasetBundle,
    DatasetSplitRequest,
    ManifestDatasetAdapter,
    inspect_dataset_bundle,
    load_json_artifact,
)
from iso_obs.dataset_mcap import (
    McapDatasetIngestionPlan,
    ingest_mcap_source,
)
from iso_obs.dataset_reliability import (
    DatasetAuditDisposition,
    DatasetReliabilityAuditPlan,
    audit_dataset_reliability,
)
from iso_obs.dataset_rosbag2 import (
    Rosbag2DatasetIngestionResult,
    Rosbag2McapIngestionPlan,
    Rosbag2RecordingEvidence,
    ingest_rosbag2_mcap_source,
)
from iso_obs.dataset_splitting import (
    DatasetSplitDisposition,
    audit_dataset_split,
)
from iso_obs.dataset_synchronization import (
    DatasetSynchronizationPlan,
    DatasetSynchronizationReport,
    SynchronizationDisposition,
    audit_dataset_synchronization,
)
from iso_obs.replay_capsule import (
    FailureClaimQualification,
    FailureReplayRequest,
    ReplayFidelity,
    compile_failure_replay_capsule,
)
from iso_obs.simulation import SimulationAdapterManifest
from iso_obs_cli.output import (
    EXIT_CONTAMINATED,
    EXIT_INPUT_ERROR,
    EXIT_INSUFFICIENT_EVIDENCE,
    EXIT_REVIEW_REQUIRED,
    fail,
)

app = typer.Typer(
    help="Inspect and audit local dataset reliability artifacts.",
    no_args_is_help=True,
)

_OutputOption = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        help="Write canonical JSON to a new file instead of stdout.",
    ),
]


@app.command()
def ingest(
    source_path: Annotated[
        Path,
        typer.Argument(help="JSONL, Parquet, or MCAP source path."),
    ],
    plan_path: Annotated[
        Path,
        typer.Option("--plan", help="Dataset ingestion plan JSON path."),
    ],
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="New dataset bundle JSON path."),
    ],
    report_path: Annotated[
        Path,
        typer.Option("--report", help="New ingestion report JSON path."),
    ],
    recording_report_path: Annotated[
        Path | None,
        typer.Option(
            "--recording-report",
            help="Required rosbag2 recording-evidence JSON output path.",
        ),
    ] = None,
) -> None:
    """Convert a mapped source file into audited local artifacts."""
    plan = _load_ingestion_plan(plan_path)
    result: DatasetIngestionResult | Rosbag2DatasetIngestionResult
    recording_evidence_payload: str | None = None
    try:
        if isinstance(plan, Rosbag2McapIngestionPlan):
            if recording_report_path is None:
                fail(
                    "--recording-report is required for rosbag2 ingestion",
                    code=EXIT_INPUT_ERROR,
                )
            result = ingest_rosbag2_mcap_source(plan, source_path)
            recording_evidence_payload = result.recording_evidence.to_json()
        elif isinstance(plan, McapDatasetIngestionPlan):
            result = ingest_mcap_source(plan, source_path)
        else:
            result = ingest_dataset_source(plan, source_path)
    except (DatasetIngestionError, OSError, ValueError) as exc:
        fail(f"{source_path}: {exc}", code=EXIT_INPUT_ERROR)
    if isinstance(plan, Rosbag2McapIngestionPlan):
        if recording_report_path is None:
            raise AssertionError("rosbag2 recording report path was validated")
        if recording_evidence_payload is None:
            raise AssertionError("rosbag2 recording evidence was produced")
        _emit_artifacts(
            (
                (output_path, result.bundle.to_json()),
                (report_path, result.report.to_json()),
                (
                    recording_report_path,
                    recording_evidence_payload,
                ),
            )
        )
    else:
        if recording_report_path is not None:
            fail(
                "--recording-report is only valid for rosbag2 ingestion",
                code=EXIT_INPUT_ERROR,
            )
        _emit_pair(
            result.bundle.to_json(),
            output_path,
            result.report.to_json(),
            report_path,
        )
    _exit_for_ingestion(result.report.disposition)


@app.command()
def inspect(
    bundle_path: Annotated[Path, typer.Argument(help="Dataset bundle JSON path.")],
    output: _OutputOption = None,
) -> None:
    """Inventory a bundle without claiming scientific adequacy."""
    bundle = _load(bundle_path, DatasetBundle)
    inspection = inspect_dataset_bundle(bundle)
    _emit(inspection.to_json(), output)


@app.command("compile-replay")
def compile_replay(
    request_path: Annotated[
        Path,
        typer.Argument(help="Failure replay request JSON path."),
    ],
    bundle_path: Annotated[
        Path,
        typer.Option("--bundle", help="Source dataset bundle JSON path."),
    ],
    synchronization_path: Annotated[
        Path,
        typer.Option(
            "--synchronization",
            help="Dataset synchronization report JSON path.",
        ),
    ],
    adapter_path: Annotated[
        Path,
        typer.Option("--adapter", help="Simulation adapter manifest JSON path."),
    ],
    recording_evidence_path: Annotated[
        Path | None,
        typer.Option(
            "--recording-evidence",
            help="Optional rosbag2 recording-evidence JSON path.",
        ),
    ] = None,
    output: _OutputOption = None,
) -> None:
    """Compile recorded failure evidence into a portable replay capsule."""
    request = _load(request_path, FailureReplayRequest)
    bundle = _load(bundle_path, DatasetBundle)
    synchronization = _load(
        synchronization_path,
        DatasetSynchronizationReport,
    )
    adapter = _load(adapter_path, SimulationAdapterManifest)
    recording_evidence = (
        _load(recording_evidence_path, Rosbag2RecordingEvidence)
        if recording_evidence_path is not None
        else None
    )
    try:
        capsule = compile_failure_replay_capsule(
            request,
            bundle=bundle,
            synchronization=synchronization,
            adapter=adapter,
            recording_evidence=recording_evidence,
        )
    except ValueError as exc:
        fail(str(exc), code=EXIT_INPUT_ERROR)
    _emit(capsule.to_json(), output)
    if capsule.fidelity is ReplayFidelity.INSUFFICIENT_EVIDENCE:
        raise typer.Exit(EXIT_INSUFFICIENT_EVIDENCE)
    if (
        capsule.fidelity is ReplayFidelity.APPROXIMATE_REPLAY_ONLY
        or capsule.failure_claim_qualification
        is FailureClaimQualification.REVIEW_REQUIRED
    ):
        raise typer.Exit(EXIT_REVIEW_REQUIRED)


@app.command()
def synchronize(
    bundle_path: Annotated[Path, typer.Argument(help="Dataset bundle JSON path.")],
    plan_path: Annotated[
        Path,
        typer.Option("--plan", help="Synchronization plan JSON path."),
    ],
    output: _OutputOption = None,
) -> None:
    """Audit timestamp alignment for the episode named by a plan."""
    bundle = _load(bundle_path, DatasetBundle)
    plan = _load(plan_path, DatasetSynchronizationPlan)
    adapter = ManifestDatasetAdapter(bundle)
    episode_by_id = {item.episode_id: item for item in adapter.episode_manifests()}
    episode = episode_by_id.get(plan.episode_id)
    if episode is None:
        fail(
            f"bundle has no episode named {plan.episode_id!r}",
            code=EXIT_INPUT_ERROR,
        )
    try:
        report = audit_dataset_synchronization(
            plan,
            adapter.dataset_manifest(),
            episode,
            adapter.timing_traces(plan.episode_id),
        )
    except ValueError as exc:
        fail(str(exc), code=EXIT_INPUT_ERROR)
    _emit(report.to_json(), output)
    _exit_for_synchronization(report.disposition)


@app.command()
def audit(
    bundle_path: Annotated[Path, typer.Argument(help="Dataset bundle JSON path.")],
    plan_path: Annotated[
        Path,
        typer.Option("--plan", help="Dataset reliability audit plan JSON path."),
    ],
    output: _OutputOption = None,
) -> None:
    """Audit evidence-qualified labels without rewriting their status."""
    bundle = _load(bundle_path, DatasetBundle)
    plan = _load(plan_path, DatasetReliabilityAuditPlan)
    adapter = ManifestDatasetAdapter(bundle)
    try:
        report = audit_dataset_reliability(
            plan,
            adapter.dataset_manifest(),
            adapter.episode_manifests(),
            adapter.label_assertions(),
        )
    except ValueError as exc:
        fail(str(exc), code=EXIT_INPUT_ERROR)
    _emit(report.to_json(), output)
    _exit_for_dataset_audit(report.disposition)


@app.command()
def split(
    request_path: Annotated[
        Path,
        typer.Argument(help="Dataset split request JSON path."),
    ],
    output: _OutputOption = None,
) -> None:
    """Audit a proposed split for leakage and missing evidence."""
    request = _load(request_path, DatasetSplitRequest)
    try:
        report = audit_dataset_split(
            request.plan,
            request.units,
            request.assignments,
        )
    except ValueError as exc:
        fail(str(exc), code=EXIT_INPUT_ERROR)
    _emit(report.to_json(), output)
    _exit_for_split(report.disposition)


def _load[T](path: Path, artifact_type: type[T]) -> T:
    """Load one strict local artifact or exit with an input error."""
    try:
        return load_json_artifact(path, artifact_type)
    except (ArtifactFormatError, OSError, ValueError) as exc:
        fail(f"{path}: {exc}", code=EXIT_INPUT_ERROR)


def _load_ingestion_plan(
    path: Path,
) -> DatasetIngestionPlan | McapDatasetIngestionPlan | Rosbag2McapIngestionPlan:
    """Load a generic row, MCAP, or rosbag2 ingestion contract."""
    try:
        return load_json_artifact(path, DatasetIngestionPlan)
    except OSError as exc:
        fail(f"{path}: {exc}", code=EXIT_INPUT_ERROR)
    except (ArtifactFormatError, ValueError) as generic_error:
        try:
            return load_json_artifact(path, McapDatasetIngestionPlan)
        except (ArtifactFormatError, OSError, ValueError) as mcap_error:
            try:
                return load_json_artifact(path, Rosbag2McapIngestionPlan)
            except (ArtifactFormatError, OSError, ValueError) as rosbag2_error:
                fail(
                    f"{path}: not a valid dataset, MCAP, or rosbag2 ingestion "
                    f"plan (dataset: {generic_error}; MCAP: {mcap_error}; "
                    f"rosbag2: {rosbag2_error})",
                    code=EXIT_INPUT_ERROR,
                )


def _emit(payload: str, output: Path | None) -> None:
    """Emit canonical JSON to stdout or create a requested output file."""
    if output is None:
        typer.echo(payload)
        return
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
    except OSError as exc:
        fail(f"{output}: {exc}", code=EXIT_INPUT_ERROR)


def _emit_pair(
    bundle_payload: str,
    bundle_path: Path,
    report_payload: str,
    report_path: Path,
) -> None:
    """Create a bundle/report pair without overwriting existing artifacts."""
    _emit_artifacts(
        (
            (bundle_path, bundle_payload),
            (report_path, report_payload),
        )
    )


def _emit_artifacts(artifacts: Sequence[tuple[Path, str]]) -> None:
    """Create new artifacts with preflight and rollback on write errors."""
    paths = tuple(path for path, _ in artifacts)
    resolved_paths = tuple(path.resolve() for path in paths)
    if len(resolved_paths) != len(set(resolved_paths)):
        fail("output paths must differ", code=EXIT_INPUT_ERROR)
    for path in paths:
        if path.exists():
            fail(f"{path}: output already exists", code=EXIT_INPUT_ERROR)
        if not path.parent.is_dir():
            fail(f"{path}: parent directory does not exist", code=EXIT_INPUT_ERROR)
    created: list[Path] = []
    try:
        for path, payload in artifacts:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                created.append(path)
                stream.write(payload)
                stream.write("\n")
    except OSError as exc:
        for path in created:
            path.unlink(missing_ok=True)
        fail(str(exc), code=EXIT_INPUT_ERROR)


def _exit_for_ingestion(disposition: DatasetIngestionDisposition) -> None:
    """Map an ingestion disposition to automation-safe exit status."""
    if disposition is DatasetIngestionDisposition.REVIEW_REQUIRED:
        raise typer.Exit(EXIT_REVIEW_REQUIRED)
    if disposition is DatasetIngestionDisposition.INSUFFICIENT_EVIDENCE:
        raise typer.Exit(EXIT_INSUFFICIENT_EVIDENCE)


def _exit_for_synchronization(
    disposition: SynchronizationDisposition,
) -> None:
    """Map a synchronization disposition to automation-safe exit status."""
    if disposition is SynchronizationDisposition.REVIEW_REQUIRED:
        raise typer.Exit(EXIT_REVIEW_REQUIRED)
    if disposition is SynchronizationDisposition.INSUFFICIENT_EVIDENCE:
        raise typer.Exit(EXIT_INSUFFICIENT_EVIDENCE)


def _exit_for_dataset_audit(disposition: DatasetAuditDisposition) -> None:
    """Map a label-audit disposition to automation-safe exit status."""
    if disposition is DatasetAuditDisposition.REVIEW_REQUIRED:
        raise typer.Exit(EXIT_REVIEW_REQUIRED)
    if disposition is DatasetAuditDisposition.INSUFFICIENT_EVIDENCE:
        raise typer.Exit(EXIT_INSUFFICIENT_EVIDENCE)


def _exit_for_split(disposition: DatasetSplitDisposition) -> None:
    """Map a split-audit disposition to automation-safe exit status."""
    if disposition is DatasetSplitDisposition.REVIEW_REQUIRED:
        raise typer.Exit(EXIT_REVIEW_REQUIRED)
    if disposition is DatasetSplitDisposition.CONTAMINATED:
        raise typer.Exit(EXIT_CONTAMINATED)
    if disposition is DatasetSplitDisposition.INSUFFICIENT_EVIDENCE:
        raise typer.Exit(EXIT_INSUFFICIENT_EVIDENCE)
