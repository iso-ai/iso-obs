"""``iso scenario`` commands for versioned operating conditions."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from iso_obs_cli import client as client_factory
from iso_obs_cli.output import console
from iso_obs_schemas import PerturbationSpec

app = typer.Typer(help="Manage evaluation scenarios.", no_args_is_help=True)


@app.command()
def register(
    environment: Annotated[str, typer.Option("--environment")],
    name: Annotated[str, typer.Option("--name")],
    version: Annotated[str, typer.Option("--version")],
    config_json: Annotated[str, typer.Option("--config-json")] = "{}",
    perturbations_json: Annotated[str, typer.Option("--perturbations-json")] = "[]",
) -> None:
    """Register a scenario and pin a version."""
    perturbations = [
        PerturbationSpec.model_validate(item) for item in json.loads(perturbations_json)
    ]
    client = client_factory.build_client(client_factory.require_api_key())
    try:
        pinned = client.scenarios.register(
            environment,
            name,
            version,
            config=json.loads(config_json),
            perturbations=perturbations,
        )
    except Exception as exc:
        client_factory.exit_for_client_error(exc)
    console.print(
        f"Registered scenario [bold]{name}[/bold] "
        f"({pinned.scenario_id}) version {pinned.id}."
    )
