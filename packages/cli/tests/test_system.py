"""Tests for ``iso system register``."""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from iso_obs_cli.main import app
from iso_obs_schemas import SystemType, SystemVersion

_PROJECT_ID = "prj_" + "c" * 24
_SYSTEM_VERSION = SystemVersion(
    id="sv_" + "a" * 24,
    system_id="sys_" + "b" * 24,
    version="1.2.0",
    commit_sha="deadbeef",
    artifact_uri="s3://models/lander-1.2.0.pt",
)


class StubSystems:
    """Records ``register`` calls and returns a canned SystemVersion."""

    def __init__(self, response: SystemVersion) -> None:
        """Store the canned response and start with no recorded calls."""
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def register(
        self,
        project: str,
        name: str,
        version: str,
        artifact_uri: str | None = None,
        source_commit: str | None = None,
        framework: str | None = None,
        system_type: SystemType | str = SystemType.OTHER,
        metadata: dict[str, Any] | None = None,
    ) -> SystemVersion:
        self.calls.append(
            {
                "project": project,
                "name": name,
                "version": version,
                "artifact_uri": artifact_uri,
                "source_commit": source_commit,
                "framework": framework,
                "system_type": system_type,
                "metadata": metadata,
            }
        )
        return self.response


class StubClient:
    """Client stand-in exposing only the ``systems`` resource."""

    def __init__(self, systems: StubSystems) -> None:
        """Attach the stubbed systems resource."""
        self.systems = systems


def test_register_happy_path(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``system register`` calls the SDK and prints the new version id."""
    stub = StubSystems(_SYSTEM_VERSION)
    monkeypatch.setenv("ISO_OBS_API_KEY", "key_test")
    monkeypatch.setattr(
        "iso_obs_cli.client.build_client", lambda api_key: StubClient(stub)
    )
    result = runner.invoke(
        app,
        [
            "system",
            "register",
            "--project",
            _PROJECT_ID,
            "--name",
            "lander-policy",
            "--version",
            "1.2.0",
            "--artifact-uri",
            "s3://models/lander-1.2.0.pt",
            "--source-commit",
            "deadbeef",
            "--framework",
            "torch",
        ],
    )
    assert result.exit_code == 0
    assert _SYSTEM_VERSION.id in result.output
    assert stub.calls == [
        {
            "project": _PROJECT_ID,
            "name": "lander-policy",
            "version": "1.2.0",
            "artifact_uri": "s3://models/lander-1.2.0.pt",
            "source_commit": "deadbeef",
            "framework": "torch",
            "system_type": SystemType.OTHER,
            "metadata": None,
        }
    ]


def test_register_without_key_exits_2(runner: CliRunner) -> None:
    """``system register`` with no configured key fails with exit code 2."""
    result = runner.invoke(
        app,
        [
            "system",
            "register",
            "--project",
            _PROJECT_ID,
            "--name",
            "lander-policy",
            "--version",
            "1.2.0",
        ],
    )
    assert result.exit_code == 2
    assert "no API key configured" in result.stderr
    assert "iso auth login" in result.stderr
