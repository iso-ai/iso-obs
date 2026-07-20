"""Evaluate one pinned world-model slice and stream fidelity evidence."""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pendulum_world_model_workflow import evaluate_slice


def request(method: str, path: str, body: Any | None = None) -> Any:
    """Call the authenticated Reliability Studio API."""
    base_url = os.environ["ISO_OBS_BASE_URL"].rstrip("/")
    payload = json.dumps(body).encode() if body is not None else None
    operation = urllib.request.Request(
        base_url + path,
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['ISO_OBS_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(operation, timeout=30) as response:
        data = response.read()
    return json.loads(data) if data else None


def find_parent_id(
    project_id: str,
    collection: str,
    parent_prefix: str,
    version_id: str,
) -> str:
    """Resolve the parent object for a pinned version."""
    parents = request("GET", f"/projects/{project_id}/{collection}")
    for parent in parents:
        versions = request("GET", f"/{parent_prefix}/{parent['id']}/versions")
        if any(item["id"] == version_id for item in versions):
            return str(parent["id"])
    raise RuntimeError(f"parent not found for {version_id}")


def trace_event(
    run: dict[str, Any],
    event_type: str,
    payload: dict[str, Any],
    *,
    system_id: str,
    environment_id: str,
    scenario_id: str,
    step: int,
) -> dict[str, Any]:
    """Build one valid event for the pinned platform run."""
    return {
        "event_id": f"evt_{uuid.uuid4().hex[:24]}",
        "workspace_id": run["workspace_id"],
        "project_id": run["project_id"],
        "suite_id": run["suite_execution_id"],
        "run_id": run["id"],
        "system_id": system_id,
        "system_version": run["system_version_id"],
        "environment_id": environment_id,
        "environment_version": run["environment_version_id"],
        "scenario_id": scenario_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "step": step,
        "event_type": event_type,
        "source": "pendulum-world-model-adapter",
        "payload": payload,
        "attributes": {"seed": run["seed"]},
    }


def main() -> None:
    """Evaluate a serialized baseline or candidate against one condition."""
    started = time.perf_counter()
    run = request("GET", f"/runs/{os.environ['ISO_RUN_ID']}")
    scenario = json.loads(os.environ["ISO_SCENARIO_CONFIG"])
    system = json.loads(os.environ["ISO_SYSTEM_CONTEXT"])
    config = scenario.get("config", {})
    model_directory = Path(os.environ["ISO_WORLD_MODEL_DIR"])
    model_name = (
        "candidate-mlp-model.json"
        if "candidate" in system.get("version", "")
        else "baseline-linear-model.json"
    )
    with (model_directory / model_name).open(encoding="utf-8") as stream:
        model = json.load(stream)
    one_step, rollout, sample_count = evaluate_slice(
        model,
        slice_id=str(config["slice_id"]),
        mass=float(config["payload_mass"]),
        damping=float(config["joint_damping"]),
        seed=int(run["seed"]),
        episodes=int(config.get("episodes", 20)),
        horizon=int(config.get("horizon_steps", 25)),
    )
    threshold = float(config["rollout_rmse_threshold"])
    passed = rollout <= threshold
    system_id = find_parent_id(
        run["project_id"], "systems", "systems", run["system_version_id"]
    )
    environment_id = find_parent_id(
        run["project_id"],
        "environments",
        "environments",
        run["environment_version_id"],
    )
    scenarios = request("GET", f"/environments/{environment_id}/scenarios")
    scenario_id = next(
        item["id"]
        for item in scenarios
        if any(
            version["id"] == run["scenario_version_id"]
            for version in request("GET", f"/scenarios/{item['id']}/versions")
        )
    )
    metric_specs = (
        ("one_step_rmse", one_step, 0.08),
        ("rollout_rmse_25_step", rollout, threshold),
        (
            "evaluation_wall_time_ms",
            (time.perf_counter() - started) * 1000.0,
            2_000.0,
        ),
    )
    events = [
        trace_event(
            run,
            "metric.recorded",
            {
                "key": key,
                "kind": "scalar",
                "scalar_value": value,
                "unit": "milliseconds" if key.endswith("_ms") else "rmse",
                "direction": "minimize",
                "threshold": limit,
                "passed": value <= limit,
            },
            system_id=system_id,
            environment_id=environment_id,
            scenario_id=scenario_id,
            step=index,
        )
        for index, (key, value, limit) in enumerate(metric_specs, start=1)
    ]
    if not passed:
        events.append(
            trace_event(
                run,
                "constraint.violated",
                {
                    "constraint": "maximum_25_step_rollout_rmse",
                    "observed_value": rollout,
                    "threshold": threshold,
                    "acceptance_operator": "lte",
                    "severity": "critical",
                    "first_occurrence": True,
                },
                system_id=system_id,
                environment_id=environment_id,
                scenario_id=scenario_id,
                step=len(events) + 1,
            )
        )
    events.append(
        trace_event(
            run,
            "metric.recorded",
            {
                "key": "evaluation_samples",
                "kind": "scalar",
                "scalar_value": float(sample_count),
                "unit": "transitions",
                "direction": "maximize",
                "passed": True,
            },
            system_id=system_id,
            environment_id=environment_id,
            scenario_id=scenario_id,
            step=len(events) + 1,
        )
    )
    request("POST", f"/runs/{run['id']}/events:batch", events)


if __name__ == "__main__":
    main()
