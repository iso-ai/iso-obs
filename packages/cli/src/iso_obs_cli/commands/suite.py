"""``iso suite`` commands for evaluation matrix construction."""

from __future__ import annotations

from typing import Annotated

import typer

from iso_obs_cli import client as client_factory
from iso_obs_cli.output import console

app = typer.Typer(help="Manage evaluation suites.", no_args_is_help=True)


@app.command("create")
def create_suite(
    project: Annotated[str, typer.Option("--project")],
    name: Annotated[str, typer.Option("--name")],
    scenario_version: Annotated[list[str], typer.Option("--scenario-version")],
    seed: Annotated[list[int], typer.Option("--seed")],
) -> None:
    """Create a suite from pinned scenario versions and seeds."""
    client = client_factory.build_client(client_factory.require_api_key())
    try:
        suite = client.suites.create(
            project,
            name,
            scenario_version,
            seed,
        )
    except Exception as exc:
        client_factory.exit_for_client_error(exc)
    console.print(f"Created suite [bold]{name}[/bold] as {suite.id}.")
