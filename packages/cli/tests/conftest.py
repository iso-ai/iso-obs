"""Shared fixtures for iso-obs-cli tests.

Every test runs against an isolated ``HOME`` (so the real user config is
never read or written) with the iso-obs environment variables cleared.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOME and cwd to a temp dir and clear iso-obs env vars."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ISO_OBS_API_KEY", raising=False)
    monkeypatch.delenv("ISO_OBS_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def runner() -> CliRunner:
    """Return a CLI runner for invoking the ``iso`` app."""
    return CliRunner()
