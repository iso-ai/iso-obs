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
#: Exit code for usage and local input failures.
EXIT_INPUT_ERROR = 2
#: Exit code for missing or rejected credentials.
EXIT_AUTH_ERROR = EXIT_INPUT_ERROR
#: Exit code for a scientifically valid report requiring review.
EXIT_REVIEW_REQUIRED = 3
#: Exit code for demonstrated contamination.
EXIT_CONTAMINATED = 4
#: Exit code for a report that abstains because evidence is insufficient.
EXIT_INSUFFICIENT_EVIDENCE = 5

console = Console()
error_console = Console(stderr=True)


def fail(message: str, *, code: int) -> NoReturn:
    """Print an error message to stderr and exit the CLI.

    Args:
        message: Human-readable, actionable error message.
        code: Process exit code appropriate to the failed operation.

    Raises:
        typer.Exit: Always, carrying ``code``.
    """
    error_console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code)
