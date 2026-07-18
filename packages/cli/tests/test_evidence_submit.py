"""End-to-end tests for ``iso evidence submit``."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iso_obs.evidence import sha256_digest
from iso_obs_cli.main import app

_VERSION = "iso-obs.failure-phenotype-report.v1"
_REPORT_ID = "erep_" + "a" * 24
_PROJECT_ID = "prj_" + "b" * 24


class FakeResponse(io.BytesIO):
    """Context-managed HTTP response containing one JSON document."""

    def __init__(self, status: int, document: dict[str, Any]) -> None:
        """Encode the canned response and expose its HTTP status."""
        super().__init__(json.dumps(document).encode("utf-8"))
        self.status = status

    def __enter__(self) -> FakeResponse:
        """Return the open response."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the response."""
        self.close()


def _write_report(
    path: Path,
    *,
    version_field: str = "report_schema_version",
) -> dict[str, Any]:
    """Write one noncanonical file containing a valid SDK report object."""
    document = {
        version_field: _VERSION,
        "disposition": "novel_candidate",
        "limitations": ["Discovery evidence only."],
    }
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document


def _receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a platform receipt matching one captured request payload."""
    return {
        "id": _REPORT_ID,
        "workspace_id": "ws_" + "c" * 24,
        "project_id": payload.get("project_id"),
        "schema_version": payload["schema_version"],
        "content_digest": payload["content_digest"],
        "disposition": "novel_candidate",
        "submitted_by": "user",
        "created_at": "2026-07-18T00:00:00Z",
    }


@pytest.mark.parametrize(
    ("status", "message"),
    [(201, "Submitted"), (200, "Already stored")],
)
def test_submit_posts_integrity_envelope_and_handles_idempotency(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    message: str,
) -> None:
    """POST canonical JSON with bearer auth for new and existing content."""
    report_path = tmp_path / "report.json"
    document = _write_report(report_path)
    captured: dict[str, Any] = {}

    def open_request(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        """Capture one HTTP request and return its corresponding receipt."""
        payload = json.loads(request.data or b"")
        captured.update(
            request=request,
            timeout=timeout,
            payload=payload,
        )
        receipt = _receipt(payload)
        if status == 200:
            receipt["project_id"] = None
        return FakeResponse(status, receipt)

    monkeypatch.setenv("ISO_OBS_API_KEY", "iso_test")
    monkeypatch.setenv("ISO_OBS_BASE_URL", "https://studio.test/api/v1/")
    monkeypatch.setattr(urllib.request, "urlopen", open_request)

    result = runner.invoke(
        app,
        [
            "evidence",
            "submit",
            str(report_path),
            "--project-id",
            _PROJECT_ID,
        ],
    )

    assert result.exit_code == 0
    assert message in result.output
    assert _REPORT_ID in result.output
    request = captured["request"]
    assert request.full_url == "https://studio.test/api/v1/evidence/reports"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer iso_test"
    assert request.get_header("Content-type") == "application/json"
    payload = captured["payload"]
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert payload == {
        "report_json": canonical,
        "schema_version": _VERSION,
        "content_digest": sha256_digest(canonical),
        "project_id": _PROJECT_ID,
    }


def test_submit_accepts_schema_version_field(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Support reports that use the newer ``schema_version`` convention."""
    report_path = tmp_path / "report.json"
    _write_report(report_path, version_field="schema_version")

    def open_request(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        """Return a matching creation receipt."""
        payload = json.loads(request.data or b"")
        return FakeResponse(201, _receipt(payload))

    monkeypatch.setenv("ISO_OBS_API_KEY", "iso_test")
    monkeypatch.setattr(urllib.request, "urlopen", open_request)

    result = runner.invoke(app, ["evidence", "submit", str(report_path)])

    assert result.exit_code == 0
    assert _VERSION in result.output


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[]", "JSON object"),
        ('{"value":NaN}', "non-finite"),
        (
            '{"schema_version":"iso-obs.a.v1","schema_version":"iso-obs.a.v1"}',
            "duplicate",
        ),
        ('{"disposition":"review"}', "no report_schema_version"),
        (
            '{"report_schema_version":"iso-obs.a.v1",'
            '"schema_version":"iso-obs.b.v1"}',
            "conflicting",
        ),
        ('{"schema_version":"version-one"}', "must match"),
    ],
)
def test_submit_rejects_ambiguous_or_non_sdk_json(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    message: str,
) -> None:
    """Reject malformed evidence locally before authentication or transport."""
    report_path = tmp_path / "report.json"
    report_path.write_text(text, encoding="utf-8")

    def unexpected_request(*args: object, **kwargs: object) -> None:
        """Fail if invalid local input reaches the network."""
        raise AssertionError("network request was not expected")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_request)

    result = runner.invoke(app, ["evidence", "submit", str(report_path)])

    assert result.exit_code == 2
    assert message in " ".join(result.stderr.split())


def test_submit_requires_api_key(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Use the shared credential requirement before attempting submission."""
    report_path = tmp_path / "report.json"
    _write_report(report_path)

    result = runner.invoke(app, ["evidence", "submit", str(report_path)])

    assert result.exit_code == 2
    assert "iso auth login" in result.stderr


def test_submit_maps_http_error_to_api_exit(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return exit code one when the platform rejects a valid request."""
    report_path = tmp_path / "report.json"
    _write_report(report_path)

    def reject_request(*args: object, **kwargs: object) -> None:
        """Raise the same error urllib emits for an HTTP rejection."""
        raise urllib.error.HTTPError(
            "https://studio.test/api/v1/evidence/reports",
            422,
            "Unprocessable Content",
            {},
            None,
        )

    monkeypatch.setenv("ISO_OBS_API_KEY", "iso_test")
    monkeypatch.setattr(urllib.request, "urlopen", reject_request)

    result = runner.invoke(app, ["evidence", "submit", str(report_path)])

    assert result.exit_code == 1
    assert "HTTP 422" in result.stderr


def test_submit_rejects_mismatched_server_receipt(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not confirm success when the server acknowledges different content."""
    report_path = tmp_path / "report.json"
    _write_report(report_path)

    def mismatched_request(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeResponse:
        """Return a receipt for a different content address."""
        payload = json.loads(request.data or b"")
        receipt = _receipt(payload)
        receipt["content_digest"] = sha256_digest("different")
        return FakeResponse(201, receipt)

    monkeypatch.setenv("ISO_OBS_API_KEY", "iso_test")
    monkeypatch.setattr(urllib.request, "urlopen", mismatched_request)

    result = runner.invoke(app, ["evidence", "submit", str(report_path)])

    assert result.exit_code == 1
    assert "mismatched evidence receipt" in " ".join(result.stderr.split())
