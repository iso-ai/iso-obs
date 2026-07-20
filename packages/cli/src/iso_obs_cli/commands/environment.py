"""``iso environment`` commands for simulator and benchmark registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from iso_obs_cli import client as client_factory
from iso_obs_cli.output import console
from iso_obs_schemas import EnvironmentRenderManifest

app = typer.Typer(help="Manage evaluation environments.", no_args_is_help=True)


@app.command()
def register(
    project: Annotated[str, typer.Option("--project")],
    name: Annotated[str, typer.Option("--name")],
    version: Annotated[str, typer.Option("--version")],
    image_uri: Annotated[str | None, typer.Option("--image-uri")] = None,
    config_json: Annotated[str, typer.Option("--config-json")] = "{}",
    render_manifest: Annotated[
        Path | None,
        typer.Option(
            "--render-manifest",
            help="JSON manifest containing simulator-declared render geometry.",
        ),
    ] = None,
) -> None:
    """Register an environment and pin a version."""
    client = client_factory.build_client(client_factory.require_api_key())
    try:
        manifest = (
            EnvironmentRenderManifest.model_validate_json(
                render_manifest.read_text(encoding="utf-8")
            )
            if render_manifest is not None
            else None
        )
        pinned = client.environments.register(
            project,
            name,
            version,
            image_uri=image_uri,
            config=json.loads(config_json),
            render_manifest=manifest,
        )
    except Exception as exc:
        client_factory.exit_for_client_error(exc)
    console.print(
        f"Registered environment [bold]{name}[/bold] "
        f"({pinned.environment_id}) version {pinned.id}."
    )
