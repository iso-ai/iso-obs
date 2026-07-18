"""``iso evidence`` commands for content-addressed SDK reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any

import typer

from iso_obs.evidence import sha256_digest
from iso_obs_cli import api, config
from iso_obs_cli.client import require_api_key
from iso_obs_cli.output import EXIT_API_ERROR, EXIT_INPUT_ERROR, console, fail

app = typer.Typer(help="Submit SDK evidence reports.", no_args_is_help=True)

_SCHEMA_VERSION_PATTERN = re.compile(r"^iso-obs\.[a-z0-9-]+\.v\d+$")


@app.command()
def submit(
    report_path: Annotated[
        Path,
        typer.Argument(help="Path containing an SDK report's to_json() output."),
    ],
    project_id: Annotated[
        str | None,
        typer.Option("--project-id", help="Optional project association."),
    ] = None,
) -> None:
    """Submit a canonical SDK evidence report to Reliability Studio."""
    report_json, schema_version, content_digest = _load_report(report_path)
    api_key = require_api_key()
    try:
        result = api.submit_evidence_report(
            report_json,
            schema_version,
            content_digest,
            api_key=api_key,
            base_url=config.resolve_base_url(),
            project_id=project_id,
        )
    except api.ApiUnreachableError as exc:
        fail(
            f"{exc}. Check your network connection and ISO_OBS_BASE_URL.",
            code=EXIT_API_ERROR,
        )
    except api.ApiError as exc:
        fail(str(exc), code=EXIT_API_ERROR)

    action = "Submitted" if result.created else "Already stored"
    console.print(
        f"{action} evidence report [bold]{result.report_id}[/bold]\n"
        f"schema: {result.schema_version}\n"
        f"digest: {result.content_digest}"
    )
    if result.project_id is not None:
        console.print(f"project: {result.project_id}")


def _load_report(path: Path) -> tuple[str, str, str]:
    """Load, validate, canonicalize, and content-address one report file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        fail(f"could not read {path}: {exc}", code=EXIT_INPUT_ERROR)
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        fail(f"{path} is not a valid SDK report: {exc}", code=EXIT_INPUT_ERROR)
    if not isinstance(document, dict):
        fail(f"{path} must contain a JSON object", code=EXIT_INPUT_ERROR)

    schema_version = _schema_version(document, path)
    report_json = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return report_json, schema_version, sha256_digest(report_json)


def _schema_version(document: dict[str, Any], path: Path) -> str:
    """Resolve the two SDK report-version field conventions."""
    present: list[str] = []
    for field in ("report_schema_version", "schema_version"):
        if field not in document:
            continue
        value = document[field]
        if not isinstance(value, str):
            fail(f"{path} has a non-string schema version", code=EXIT_INPUT_ERROR)
        present.append(value)
    if not present:
        fail(
            f"{path} has no report_schema_version or schema_version",
            code=EXIT_INPUT_ERROR,
        )
    if len(set(present)) > 1:
        fail(f"{path} has conflicting schema versions", code=EXIT_INPUT_ERROR)
    version = present[0]
    if _SCHEMA_VERSION_PATTERN.fullmatch(version) is None:
        fail(
            f"{path} schema version must match iso-obs.<name>.v<N>",
            code=EXIT_INPUT_ERROR,
        )
    return version


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate keys."""
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON field {key!r}")
        document[key] = value
    return document


def _reject_nonfinite(value: str) -> Any:
    """Reject JSON constants that canonical SDK serialization cannot emit."""
    raise ValueError(f"non-finite JSON number {value!r}")
