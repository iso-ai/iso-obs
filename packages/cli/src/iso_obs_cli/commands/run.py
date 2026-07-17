"""``iso run`` commands: inspect runs and their metrics."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from iso_obs_cli import api, config
from iso_obs_cli.client import require_api_key
from iso_obs_cli.output import EXIT_API_ERROR, console, fail
from iso_obs_schemas import MetricValue, MetricValueKind, Run

app = typer.Typer(help="Inspect runs.", no_args_is_help=True)

# Placeholder shown for absent optional values in tables.
_EMPTY = "-"


@app.command()
def inspect(
    run_id: Annotated[str, typer.Argument(help="Run id to inspect, e.g. run_9f8c...")],
) -> None:
    """Show a run's details and recorded metrics."""
    api_key = require_api_key()
    base_url = config.resolve_base_url()
    try:
        run = api.fetch_run(run_id, api_key=api_key, base_url=base_url)
        metrics = api.fetch_run_metrics(run_id, api_key=api_key, base_url=base_url)
    except api.ApiUnreachableError as exc:
        fail(
            f"{exc}. Check your network connection and ISO_OBS_BASE_URL.",
            code=EXIT_API_ERROR,
        )
    except api.ApiError as exc:
        fail(str(exc), code=EXIT_API_ERROR)
    console.print(_run_table(run))
    if metrics:
        console.print(_metrics_table(metrics))
    else:
        console.print("No metrics recorded for this run.")


def _run_table(run: Run) -> Table:
    """Render a run's core fields as a two-column table.

    Args:
        run: The run to render.

    Returns:
        A rich table keyed by field name.
    """
    table = Table(title=f"Run {run.id}", show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("status", run.status.value)
    table.add_row("project", run.project_id)
    table.add_row("system version", run.system_version_id)
    table.add_row("environment version", run.environment_version_id)
    table.add_row("scenario version", run.scenario_version_id)
    table.add_row("seed", str(run.seed))
    table.add_row("created", run.created_at.isoformat())
    table.add_row("started", run.started_at.isoformat() if run.started_at else _EMPTY)
    table.add_row(
        "finished", run.finished_at.isoformat() if run.finished_at else _EMPTY
    )
    if run.failure_reason:
        table.add_row("failure reason", run.failure_reason)
    return table


def _metrics_table(metrics: list[MetricValue]) -> Table:
    """Render recorded metric values as a table.

    Args:
        metrics: Metric values recorded for the run.

    Returns:
        A rich table with one row per metric.
    """
    table = Table(title="Metrics")
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_column("unit")
    table.add_column("passed")
    for metric in metrics:
        table.add_row(
            metric.key,
            _format_value(metric),
            metric.unit or _EMPTY,
            _format_passed(metric.passed),
        )
    return table


def _format_value(metric: MetricValue) -> str:
    """Format whichever value variant a metric carries.

    Args:
        metric: The metric value to format.

    Returns:
        A short printable representation of the value.
    """
    if metric.kind is MetricValueKind.SCALAR and metric.scalar_value is not None:
        return f"{metric.scalar_value:g}"
    if metric.kind is MetricValueKind.BOOLEAN and metric.boolean_value is not None:
        return "true" if metric.boolean_value else "false"
    return metric.text_value or _EMPTY


def _format_passed(passed: bool | None) -> str:
    """Format a metric's pass/fail flag.

    Args:
        passed: Whether the metric met its threshold, if it was scored.

    Returns:
        ``yes``/``no`` when scored, a placeholder otherwise.
    """
    if passed is None:
        return _EMPTY
    return "yes" if passed else "no"
