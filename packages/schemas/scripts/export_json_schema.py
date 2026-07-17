"""Export JSON Schema for the public Reliability Studio models.

This runnable module writes the JSON Schema of every public Pydantic model into
``packages/schemas/schemas/json/*.json`` (one file per model). The web app
consumes these files to code-generate TypeScript types, keeping the front-end in
lockstep with the Python contract without hand-maintaining a parallel type set.

Run it from anywhere::

    python -m scripts.export_json_schema
    # or
    python packages/schemas/scripts/export_json_schema.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from iso_obs_schemas import (
    ActionPayload,
    ArtifactRef,
    BaseEvent,
    ConstraintViolationPayload,
    Environment,
    EnvironmentVersion,
    EvaluationSuite,
    Finding,
    Gate,
    Invariant,
    MetricDefinition,
    MetricValue,
    ObservationPayload,
    PerturbationSpec,
    Project,
    ReliabilityReport,
    Run,
    Scenario,
    ScenarioVersion,
    System,
    SystemVersion,
    Workspace,
)

# Models whose JSON Schema is exported for downstream code generation. Order is
# alphabetical for stable, reviewable diffs.
PUBLIC_MODELS: tuple[type[BaseModel], ...] = (
    ActionPayload,
    ArtifactRef,
    BaseEvent,
    ConstraintViolationPayload,
    Environment,
    EnvironmentVersion,
    EvaluationSuite,
    Finding,
    Gate,
    Invariant,
    MetricDefinition,
    MetricValue,
    ObservationPayload,
    PerturbationSpec,
    Project,
    ReliabilityReport,
    Run,
    Scenario,
    ScenarioVersion,
    System,
    SystemVersion,
    Workspace,
)

# Output directory: packages/schemas/schemas/json, resolved relative to this file
# so the script works regardless of the current working directory.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "schemas" / "json"


def export(models: tuple[type[BaseModel], ...], output_dir: Path) -> list[Path]:
    """Write the JSON Schema of each model to ``output_dir``.

    Args:
        models: Pydantic model classes to export.
        output_dir: Directory to write ``<ModelName>.json`` files into; created
            if it does not exist.

    Returns:
        The list of written file paths, in export order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in models:
        schema = model.model_json_schema()
        path = output_dir / f"{model.__name__}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


def main() -> None:
    """Export all public model schemas and report what was written."""
    written = export(PUBLIC_MODELS, OUTPUT_DIR)
    print(f"Wrote {len(written)} JSON Schema files to {OUTPUT_DIR}")
    for path in written:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
