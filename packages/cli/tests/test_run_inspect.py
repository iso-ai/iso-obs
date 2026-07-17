"""Tests for ``iso run inspect`` rendering and error handling."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from iso_obs_cli import api
from iso_obs_cli.main import app
from iso_obs_schemas import (
    MetricDirection,
    MetricValue,
    MetricValueKind,
    Run,
    RunStatus,
)

_RUN = Run(
    id="run_" + "a" * 24,
    workspace_id="ws_" + "b" * 24,
    project_id="prj_" + "c" * 24,
    suite_execution_id="suite_" + "d" * 24,
    system_version_id="sv_" + "e" * 24,
    environment_version_id="ev_" + "f" * 24,
    scenario_version_id="scv_" + "1" * 24,
    seed=7,
    status=RunStatus.COMPLETED,
)
_METRICS = [
    MetricValue(
        key="success_rate",
        kind=MetricValueKind.SCALAR,
        scalar_value=0.93,
        direction=MetricDirection.MAXIMIZE,
        threshold=0.9,
        passed=True,
    ),
    MetricValue(
        key="recovered",
        kind=MetricValueKind.BOOLEAN,
        boolean_value=False,
        passed=False,
    ),
]


def test_inspect_renders_run_and_metrics(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run inspect`` renders the run fields and the metrics table."""
    monkeypatch.setenv("ISO_OBS_API_KEY", "key_test")
    monkeypatch.setattr(api, "fetch_run", lambda run_id, *, api_key, base_url: _RUN)
    monkeypatch.setattr(
        api, "fetch_run_metrics", lambda run_id, *, api_key, base_url: _METRICS
    )
    result = runner.invoke(app, ["run", "inspect", _RUN.id])
    assert result.exit_code == 0
    assert _RUN.id in result.output
    assert "completed" in result.output
    assert _RUN.system_version_id in result.output
    assert "success_rate" in result.output
    assert "0.93" in result.output
    assert "recovered" in result.output
    assert "yes" in result.output
    assert "no" in result.output


def test_inspect_unreachable_api_exits_1(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable API produces an actionable message and exit code 1."""

    def _raise(run_id: str, *, api_key: str, base_url: str) -> Run:
        raise api.ApiUnreachableError(f"cannot reach {base_url}/runs/{run_id}")

    monkeypatch.setenv("ISO_OBS_API_KEY", "key_test")
    monkeypatch.setattr(api, "fetch_run", _raise)
    result = runner.invoke(app, ["run", "inspect", _RUN.id])
    assert result.exit_code == 1
    assert "cannot reach" in result.stderr


def test_inspect_without_key_exits_2(runner: CliRunner) -> None:
    """``run inspect`` with no configured key fails with exit code 2."""
    result = runner.invoke(app, ["run", "inspect", _RUN.id])
    assert result.exit_code == 2
