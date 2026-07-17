"""Construction of the ``iso_obs`` SDK client for CLI commands.

The SDK import happens lazily inside the functions below so commands that
never touch the API (``iso version``, ``iso auth login``) do not pay the
import cost, and so tests can substitute a stub client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from iso_obs_cli import config
from iso_obs_cli.output import EXIT_API_ERROR, EXIT_AUTH_ERROR, fail

if TYPE_CHECKING:
    from iso_obs import ReliabilityClient


def require_api_key() -> str:
    """Return the configured API key or exit with an actionable message.

    Returns:
        The API key resolved from the environment or the stored config.

    Raises:
        typer.Exit: With code 2 when no key is configured.
    """
    api_key = config.resolve_api_key()
    if api_key is None:
        fail(
            "no API key configured. Run [bold]iso auth login[/bold] or set "
            "the ISO_OBS_API_KEY environment variable.",
            code=EXIT_AUTH_ERROR,
        )
    return api_key


def build_client(api_key: str) -> ReliabilityClient:
    """Build a :class:`~iso_obs.ReliabilityClient` for the given key.

    Base URL resolution (``ISO_OBS_BASE_URL``, then the default) is
    delegated to the SDK so the CLI and SDK can never disagree.

    Args:
        api_key: API key to authenticate with.

    Returns:
        A configured SDK client.
    """
    from iso_obs import ReliabilityClient

    return ReliabilityClient(api_key=api_key)


def exit_for_client_error(exc: Exception) -> NoReturn:
    """Map an SDK call failure to a friendly message and exit code.

    Args:
        exc: The exception raised by an SDK call.

    Raises:
        typer.Exit: Code 2 for authentication failures, 1 for anything else
            (unreachable API, error responses).
    """
    from iso_obs.exceptions import AuthenticationError

    if isinstance(exc, AuthenticationError):
        fail(
            f"authentication failed: {exc}. Run [bold]iso auth login[/bold] "
            "with a valid API key.",
            code=EXIT_AUTH_ERROR,
        )
    fail(
        f"request to the iso-obs API at {config.resolve_base_url()} failed: "
        f"{exc}. Check your network connection and ISO_OBS_BASE_URL.",
        code=EXIT_API_ERROR,
    )
