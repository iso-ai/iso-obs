"""Smoke tests for public SDK examples."""

from __future__ import annotations

import importlib
import json
import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize(
    ("filename", "expected_output"),
    (
        ("causal_perturbation_campaign.py", "harmful_effect_identified"),
        ("failure_boundary_map.py", "unresolved_boundary"),
        ("sim_to_real_transport.py", "limiting strata: rare-low-friction"),
        (
            "mixed_mode_dataset_curation.py",
            "neural candidates remain suggestions, never truth",
        ),
        ("warehouse_release_workflow.py", "release: blocked"),
    ),
)
def test_showcase_example_executes(
    filename: str,
    expected_output: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a showcase example executes and communicates its key result."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    monkeypatch.setattr(sys, "argv", [filename])
    runpy.run_path(str(EXAMPLES_DIR / filename), run_name="__main__")

    assert expected_output in capsys.readouterr().out


def test_warehouse_workflow_writes_immutable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the integrated workflow emits reports and refuses overwrites."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    workflow = importlib.import_module("warehouse_release_workflow")
    reports = workflow.run_studies()
    decision = workflow.decide_release(reports)

    paths = workflow.write_evidence(tmp_path, reports, decision)

    assert {path.name for path in paths} == {
        "causal-perturbation-report.json",
        "failure-boundary-report.json",
        "transportability-report.json",
        "release-decision.json",
    }
    release_payload = json.loads((tmp_path / "release-decision.json").read_text())
    assert release_payload["status"] == "blocked"
    assert len(release_payload["blockers"]) == 4
    causal_payload = json.loads(
        (tmp_path / "causal-perturbation-report.json").read_text()
    )
    assert causal_payload["disposition"] == "harmful_effect_identified"

    with pytest.raises(FileExistsError):
        workflow.write_evidence(tmp_path, reports, decision)


def test_warehouse_workflow_returns_blocking_ci_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify strict mode converts unsupported evidence into a CI failure."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    monkeypatch.setattr(
        sys,
        "argv",
        ["warehouse_release_workflow.py", "--require-ready"],
    )
    workflow = importlib.import_module("warehouse_release_workflow")

    assert workflow.main() == 2
