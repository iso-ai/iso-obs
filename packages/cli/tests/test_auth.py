"""Tests for ``iso auth login`` / ``iso auth whoami`` and config storage."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iso_obs_cli import config
from iso_obs_cli.main import app

_KEY = "key_secret1234abcd"


def _config_path(home: Path) -> Path:
    """Return the expected config file path under a redirected HOME."""
    return home / ".config" / "iso-obs" / "config.toml"


def test_login_with_option_stores_key(runner: CliRunner, isolated_home: Path) -> None:
    """``login --api-key`` writes the key to config.toml with mode 600."""
    result = runner.invoke(app, ["auth", "login", "--api-key", _KEY])
    assert result.exit_code == 0
    path = _config_path(isolated_home)
    assert path.is_file()
    assert (path.stat().st_mode & 0o777) == 0o600
    assert f'api_key = "{_KEY}"' in path.read_text(encoding="utf-8")


def test_login_prompts_when_option_omitted(
    runner: CliRunner, isolated_home: Path
) -> None:
    """``login`` without --api-key prompts and stores the entered key."""
    result = runner.invoke(app, ["auth", "login"], input=f"{_KEY}\n")
    assert result.exit_code == 0
    assert config.load_config() == {"api_key": _KEY}


def test_login_rejects_empty_key(runner: CliRunner) -> None:
    """An empty key is rejected with the auth exit code."""
    result = runner.invoke(app, ["auth", "login", "--api-key", "   "])
    assert result.exit_code == 2


def test_whoami_shows_masked_key_and_base_url(
    runner: CliRunner, isolated_home: Path
) -> None:
    """``whoami`` loads the stored key, masks it, and shows the base URL."""
    assert runner.invoke(app, ["auth", "login", "--api-key", _KEY]).exit_code == 0
    result = runner.invoke(app, ["auth", "whoami"])
    assert result.exit_code == 0
    assert "key_...abcd" in result.output
    assert _KEY not in result.output
    assert config.DEFAULT_BASE_URL in result.output


def test_whoami_without_key_exits_2(runner: CliRunner) -> None:
    """``whoami`` with no stored key or env var fails with exit code 2."""
    result = runner.invoke(app, ["auth", "whoami"])
    assert result.exit_code == 2
    assert "iso auth login" in result.stderr
