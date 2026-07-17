"""``iso project`` commands: create and initialize projects."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from iso_obs_cli import client as client_factory
from iso_obs_cli import config
from iso_obs_cli.output import EXIT_API_ERROR, console, fail

app = typer.Typer(help="Manage projects.", no_args_is_help=True)


@app.command()
def init(
    name: Annotated[
        str | None,
        typer.Option("--name", help="Project name; prompted for if omitted."),
    ] = None,
) -> None:
    """Create a project and write ``iso-obs.toml`` in the current directory."""
    if name is None:
        name = typer.prompt("Project name")
    project_file = Path.cwd() / config.PROJECT_FILE_NAME
    if project_file.exists():
        fail(
            f"{project_file} already exists; remove it before initializing "
            "a new project here.",
            code=EXIT_API_ERROR,
        )
    api_key = client_factory.require_api_key()
    client = client_factory.build_client(api_key)
    try:
        project = client.projects.create(name)
    except Exception as exc:
        client_factory.exit_for_client_error(exc)
    config.write_project_file(project_file, project_id=project.id, name=project.name)
    console.print(
        f"Created project [bold]{project.name}[/bold] ({project.id}) and "
        f"wrote [bold]{project_file.name}[/bold]."
    )
