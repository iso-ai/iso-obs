"""Shared console output and exit-code helpers for the ``iso`` CLI.

Normal output goes to stdout via :data:`console`; errors go to stderr via
:data:`error_console` so scripted consumers can pipe results cleanly.
"""

from __future__ import annotations

from typing import NoReturn

import typer
from rich.console import Console

#: Exit code for API failures (unreachable host, error responses).
EXIT_API_ERROR = 1
#: Exit code for missing or rejected credentials.
EXIT_AUTH_ERROR = 2

console = Console()
error_console = Console(stderr=True)


def fail(message: str, *, code: int) -> NoReturn:
    """Print an error message to stderr and exit the CLI.

    Args:
        message: Human-readable, actionable error message.
        code: Process exit code (:data:`EXIT_API_ERROR` or
            :data:`EXIT_AUTH_ERROR`).

    Raises:
        typer.Exit: Always, carrying ``code``.
    """
    error_console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code)
