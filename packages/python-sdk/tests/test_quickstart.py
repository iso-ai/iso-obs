"""Tests for the two-minute SDK quickstart report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iso_obs.quickstart import write_quickstart_report


def test_quickstart_writes_versioned_boundary_report(tmp_path: Path) -> None:
    """Write an honest three-way report that Studio can import unchanged."""
    output = tmp_path / "failure-boundary-report.json"

    report = write_quickstart_report(output)
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["report_schema_version"] == "iso-obs.failure-boundary-map.v1"
    assert document["disposition"] == "partial"
    assert {cell["disposition"] for cell in document["cells"]} == {
        "reliable_certified",
        "unreliable_certified",
        "unresolved_boundary",
    }
    assert report.content_digest().startswith("sha256:")


def test_quickstart_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    """Preserve an existing report rather than silently replacing evidence."""
    output = tmp_path / "failure-boundary-report.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_quickstart_report(output)
