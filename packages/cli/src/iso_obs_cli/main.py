"""Entry point wiring for the ``iso`` command-line interface."""

from __future__ import annotations

import typer

from iso_obs_cli import __version__
from iso_obs_cli.commands import (
    auth,
    dataset,
    environment,
    evidence,
    project,
    run,
    scenario,
    suite,
    system,
)
from iso_obs_cli.output import console

app = typer.Typer(
    name="iso",
    help="Reliability Studio command-line interface.",
    no_args_is_help=True,
)
app.add_typer(auth.app, name="auth")
app.add_typer(project.app, name="project")
app.add_typer(system.app, name="system")
app.add_typer(environment.app, name="environment")
app.add_typer(scenario.app, name="scenario")
app.add_typer(suite.app, name="suite")
app.add_typer(run.app, name="run")
app.add_typer(dataset.app, name="dataset")
app.add_typer(evidence.app, name="evidence")


@app.command()
def version() -> None:
    """Print the installed iso-obs-cli version."""
    console.print(f"iso-obs-cli {__version__}")
