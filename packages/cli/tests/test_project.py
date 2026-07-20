"""Tests for ``iso project init``."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from iso_obs_cli.main import app
from iso_obs_schemas import Project

_PROJECT = Project(
    id="prj_" + "c" * 24,
    workspace_id="ws_" + "b" * 24,
    name="lunar-lander",
    slug="lunar-lander",
)


class StubProjects:
    """Records ``create`` calls and returns a canned Project."""

    def __init__(self, response: Project) -> None:
        """Store the canned response and start with no recorded calls."""
        self.response = response
        self.calls: list[tuple[str, str | None]] = []

    def create(self, name: str, *, slug: str | None = None) -> Project:
        self.calls.append((name, slug))
        return self.response


class StubClient:
    """Client stand-in exposing only the ``projects`` resource."""

    def __init__(self, projects: StubProjects) -> None:
        """Attach the stubbed projects resource."""
        self.projects = projects


def test_init_creates_project_and_writes_marker(
    runner: CliRunner, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``project init`` creates the project and writes iso-obs.toml."""
    stub = StubProjects(_PROJECT)
    monkeypatch.setenv("ISO_OBS_API_KEY", "key_test")
    monkeypatch.setattr(
        "iso_obs_cli.client.build_client", lambda api_key: StubClient(stub)
    )
    result = runner.invoke(app, ["project", "init", "--name", "lunar-lander"])
    assert result.exit_code == 0
    assert stub.calls == [("lunar-lander", None)]
    marker = isolated_home / "iso-obs.toml"
    body = marker.read_text(encoding="utf-8")
    assert f'id = "{_PROJECT.id}"' in body
    assert 'name = "lunar-lander"' in body


def test_init_refuses_to_overwrite_marker(
    runner: CliRunner, isolated_home: Path
) -> None:
    """An existing iso-obs.toml is never clobbered."""
    (isolated_home / "iso-obs.toml").write_text("[project]\n", encoding="utf-8")
    result = runner.invoke(app, ["project", "init", "--name", "lunar-lander"])
    assert result.exit_code == 1
    assert "already exists" in result.stderr
