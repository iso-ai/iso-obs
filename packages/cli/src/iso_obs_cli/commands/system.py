"""``iso system`` commands: register system versions."""

from __future__ import annotations

from typing import Annotated

import typer

from iso_obs_cli import client as client_factory
from iso_obs_cli.output import console

app = typer.Typer(help="Manage systems under evaluation.", no_args_is_help=True)


@app.command()
def register(
    project: Annotated[
        str, typer.Option("--project", help="Project id the system belongs to.")
    ],
    name: Annotated[str, typer.Option("--name", help="System name.")],
    version: Annotated[
        str,
        typer.Option("--version", help="Version label, e.g. a semver or build tag."),
    ],
    artifact_uri: Annotated[
        str | None,
        typer.Option("--artifact-uri", help="Location of the system artifact."),
    ] = None,
    source_commit: Annotated[
        str | None,
        typer.Option("--source-commit", help="Source revision of this build."),
    ] = None,
    framework: Annotated[
        str | None,
        typer.Option("--framework", help="Framework the system is built with."),
    ] = None,
) -> None:
    """Register a system version with Reliability Studio."""
    api_key = client_factory.require_api_key()
    client = client_factory.build_client(api_key)
    try:
        system_version = client.systems.register(
            project,
            name,
            version,
            artifact_uri=artifact_uri,
            source_commit=source_commit,
            framework=framework,
        )
    except Exception as exc:
        client_factory.exit_for_client_error(exc)
    console.print(
        f"Registered system [bold]{name}[/bold] version "
        f"[bold]{system_version.version}[/bold] as {system_version.id}."
    )
    if system_version.artifact_uri:
        console.print(f"artifact: {system_version.artifact_uri}")
    if system_version.commit_sha:
        console.print(f"commit  : {system_version.commit_sha}")
