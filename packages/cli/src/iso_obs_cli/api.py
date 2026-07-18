"""Minimal direct HTTP access for read endpoints the SDK does not expose.

The frozen ``iso_obs`` SDK surface only covers the write side of runs
(create, log events, complete, fail). ``iso run inspect`` needs the read
side, so it issues plain GETs against the same REST API using only the
standard library rather than inventing SDK methods.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from iso_obs_schemas import MetricValue, Run

_TIMEOUT_SECONDS = 30.0


class ApiError(Exception):
    """A request to the iso-obs API failed with an error response."""


class ApiUnreachableError(ApiError):
    """The iso-obs API host could not be reached at all."""


@dataclass(frozen=True, slots=True)
class EvidenceSubmissionResult:
    """Validated receipt for one evidence-report submission."""

    report_id: str
    schema_version: str
    content_digest: str
    project_id: str | None
    created: bool


def _get_json(url: str, api_key: str) -> Any:
    """GET ``url`` with bearer auth and decode the JSON body.

    Args:
        url: Fully qualified URL to fetch.
        api_key: API key sent as a bearer token.

    Returns:
        The decoded JSON document.

    Raises:
        ApiUnreachableError: If the host cannot be reached.
        ApiError: If the server answers with an HTTP error status or a
            non-JSON body.
    """
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ApiError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ApiUnreachableError(f"cannot reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ApiError(f"{url} returned a non-JSON body") from exc


def submit_evidence_report(
    report_json: str,
    schema_version: str,
    content_digest: str,
    *,
    api_key: str,
    base_url: str,
    project_id: str | None = None,
) -> EvidenceSubmissionResult:
    """Submit one canonical SDK evidence report.

    Args:
        report_json: Exact canonical JSON produced from the SDK report.
        schema_version: Top-level report schema version.
        content_digest: SDK-compatible digest of ``report_json``.
        api_key: API key sent as a bearer token.
        base_url: API base URL, including its version prefix.
        project_id: Optional project association.

    Returns:
        Validated receipt and whether the server created a new record.

    Raises:
        ApiUnreachableError: If the host cannot be reached.
        ApiError: If the server rejects the request or returns an invalid receipt.
    """
    url = f"{base_url.rstrip('/')}/evidence/reports"
    body: dict[str, str] = {
        "report_json": report_json,
        "schema_version": schema_version,
        "content_digest": content_digest,
    }
    if project_id is not None:
        body["project_id"] = project_id
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=_TIMEOUT_SECONDS,
        ) as response:
            status = response.status
            document = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ApiError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ApiUnreachableError(f"cannot reach {url}: {exc.reason}") from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ApiError(f"{url} returned a non-JSON body") from exc

    if status not in {200, 201}:
        raise ApiError(f"{url} returned unexpected HTTP {status}")
    if not isinstance(document, dict):
        raise ApiError(f"{url} returned an invalid evidence receipt")
    report_id = _required_receipt_text(document, "id", url)
    returned_version = _required_receipt_text(
        document,
        "schema_version",
        url,
    )
    returned_digest = _required_receipt_text(
        document,
        "content_digest",
        url,
    )
    returned_project = document.get("project_id")
    if returned_project is not None and not isinstance(returned_project, str):
        raise ApiError(f"{url} returned an invalid evidence receipt")
    if returned_version != schema_version or returned_digest != content_digest:
        raise ApiError(f"{url} returned a mismatched evidence receipt")
    if status == 201 and returned_project != project_id:
        raise ApiError(f"{url} returned a mismatched project association")
    return EvidenceSubmissionResult(
        report_id=report_id,
        schema_version=returned_version,
        content_digest=returned_digest,
        project_id=returned_project,
        created=status == 201,
    )


def _required_receipt_text(
    document: dict[str, Any],
    field: str,
    url: str,
) -> str:
    """Read one required nonempty string from an API receipt."""
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ApiError(f"{url} returned an invalid evidence receipt")
    return value


def fetch_run(run_id: str, *, api_key: str, base_url: str) -> Run:
    """Fetch a run by id.

    Args:
        run_id: Typed run id, e.g. ``run_...``.
        api_key: API key sent as a bearer token.
        base_url: API base URL, e.g. ``https://api.iso-obs.com/api/v1``.

    Returns:
        The validated :class:`~iso_obs_schemas.Run`.

    Raises:
        ApiUnreachableError: If the host cannot be reached.
        ApiError: If the server rejects the request.
    """
    data = _get_json(f"{base_url}/runs/{run_id}", api_key)
    return Run.model_validate(data)


def fetch_run_metrics(run_id: str, *, api_key: str, base_url: str) -> list[MetricValue]:
    """Fetch the metric values recorded for a run.

    Args:
        run_id: Typed run id, e.g. ``run_...``.
        api_key: API key sent as a bearer token.
        base_url: API base URL.

    Returns:
        The run's metric values; empty when the server has none (or does not
        expose the metrics resource).

    Raises:
        ApiUnreachableError: If the host cannot be reached.
    """
    try:
        data = _get_json(f"{base_url}/runs/{run_id}/metrics", api_key)
    except ApiUnreachableError:
        raise
    except ApiError:
        # Metrics are supplementary: a server without the metrics resource
        # should not make ``iso run inspect`` fail after the run fetched fine.
        return []
    return [MetricValue.model_validate(item) for item in data]
