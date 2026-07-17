"""Configuration handling for the ``iso`` CLI.

Two files are involved:

* ``~/.config/iso-obs/config.toml`` — user-level credentials written by
  ``iso auth login`` (owner-only permissions, since it holds an API secret).
* ``iso-obs.toml`` — a per-project marker written into the working directory
  by ``iso project init`` so later commands can identify the project.

Environment variables (``ISO_OBS_API_KEY``, ``ISO_OBS_BASE_URL``) take
precedence over the stored config so CI can override a developer login
without touching files.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

#: Default API base URL, mirroring the SDK's fallback.
DEFAULT_BASE_URL = "https://api.iso-obs.com/api/v1"

#: Name of the per-project marker file written by ``iso project init``.
PROJECT_FILE_NAME = "iso-obs.toml"

_API_KEY_ENV = "ISO_OBS_API_KEY"
_BASE_URL_ENV = "ISO_OBS_BASE_URL"


def config_file() -> Path:
    """Return the path of the user-level CLI config file.

    Computed on every call rather than at import time so tests (and tools
    that rewrite ``HOME``) see the redirected location.

    Returns:
        Path to ``~/.config/iso-obs/config.toml``.
    """
    return Path.home() / ".config" / "iso-obs" / "config.toml"


def load_config() -> dict[str, str]:
    """Load the user-level config file.

    Returns:
        Mapping of config keys to string values. Missing or malformed files
        yield an empty mapping so a corrupted config degrades to "not logged
        in" instead of a traceback.
    """
    path = config_file()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError:
        return {}
    return {key: value for key, value in data.items() if isinstance(value, str)}


def save_config(values: dict[str, str]) -> None:
    """Write the user-level config file with owner-only permissions.

    Args:
        values: Flat string-valued mapping to persist as TOML.
    """
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f'{key} = "{toml_escape(value)}"\n' for key, value in values.items())
    path.write_text(body, encoding="utf-8")
    # The file holds an API secret: restrict it to the owner after writing.
    path.chmod(0o600)


def write_project_file(path: Path, *, project_id: str, name: str) -> None:
    """Write the per-project ``iso-obs.toml`` marker file.

    Args:
        path: Destination path (normally ``<cwd>/iso-obs.toml``).
        project_id: Typed project id, e.g. ``prj_...``.
        name: Human-readable project name.
    """
    body = (
        "[project]\n"
        f'id = "{toml_escape(project_id)}"\n'
        f'name = "{toml_escape(name)}"\n'
    )
    path.write_text(body, encoding="utf-8")


def toml_escape(value: str) -> str:
    """Escape a string for a basic TOML double-quoted value.

    Args:
        value: Raw string value.

    Returns:
        The value with backslashes and double quotes escaped.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def resolve_api_key() -> str | None:
    """Resolve the API key from the environment or the stored config.

    Returns:
        The API key, or ``None`` when neither source provides one.
    """
    return os.environ.get(_API_KEY_ENV) or load_config().get("api_key")


def resolve_base_url() -> str:
    """Resolve the API base URL from the environment or the default.

    Returns:
        The base URL the CLI (and SDK) will talk to.
    """
    return os.environ.get(_BASE_URL_ENV) or DEFAULT_BASE_URL
