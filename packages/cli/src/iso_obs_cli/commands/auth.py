"""``iso auth`` commands: store and inspect CLI credentials."""

from __future__ import annotations

from typing import Annotated

import typer

from iso_obs_cli import api, config
from iso_obs_cli.client import require_api_key
from iso_obs_cli.output import EXIT_AUTH_ERROR, console, fail

app = typer.Typer(help="Manage CLI credentials.", no_args_is_help=True)


@app.command()
def login(
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="API key to store; prompted for if omitted."),
    ] = None,
) -> None:
    """Store an API key in the user-level config file."""
    if api_key is None:
        api_key = typer.prompt("API key", hide_input=True)
    api_key = api_key.strip()
    if not api_key:
        fail("API key must not be empty.", code=EXIT_AUTH_ERROR)
    try:
        api.verify_api_key(api_key=api_key, base_url=config.resolve_base_url())
    except api.ApiUnreachableError as exc:
        fail(f"{exc}. The key was not saved.", code=EXIT_AUTH_ERROR)
    except api.ApiError as exc:
        fail(f"API key verification failed: {exc}", code=EXIT_AUTH_ERROR)

    # Merge instead of overwrite so future config keys survive a re-login.
    values = config.load_config()
    values["api_key"] = api_key
    config.save_config(values)
    console.print(
        "API key verified and activated.\n"
        f"Saved to [bold]{config.config_file()}[/bold] (permissions 600)."
    )


@app.command()
def whoami() -> None:
    """Show the masked API key and the API base URL in use."""
    api_key = require_api_key()
    console.print(f"api key : {mask_key(api_key)}")
    console.print(f"base url: {config.resolve_base_url()}")


def mask_key(api_key: str) -> str:
    """Mask an API key, keeping just enough of it to be identifiable.

    Args:
        api_key: The full API key.

    Returns:
        The first and last four characters with the middle elided, or all
        asterisks when the key is too short to mask safely.
    """
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"
