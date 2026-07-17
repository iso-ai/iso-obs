"""Tests for ``iso version``."""

from __future__ import annotations

from typer.testing import CliRunner

import iso_obs_cli
from iso_obs_cli.main import app


def test_version_prints_package_version(runner: CliRunner) -> None:
    """``iso version`` prints the package name and version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"iso-obs-cli {iso_obs_cli.__version__}" in result.output
