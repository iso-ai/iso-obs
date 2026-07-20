"""Replay Modal GPU evaluation observations into one pinned Studio run."""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    """Resolve the parent object for one pinned version."""
    parents = request("GET", f"/projects/{project_id}/{collection}")
    for parent in parents:
        versions = request("GET", f"/{parent_prefix}/{parent['id']}/versions")
        if any(item["id"] == version_id for item in versions):
            return str(parent["id"])
    raise RuntimeError(f"parent not found for {version_id}")


def main() -> None:
    """Submit the exact Modal-observed slice metrics to a queued Studio run."""
    started = time.perf_counter()
    run = request("GET", f"/runs/{os.environ['ISO_RUN_ID']}")
    scenario = json.loads(os.environ["ISO_SCENARIO_CONFIG"])
    slice_id = str(scenario["config"]["slice_id"])
    report_path = Path(os.environ["ISO_GPU_WORLD_MODEL_REPORT"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    observation = next(
        item for item in report["evaluation_slices"] if item["slice_id"] == slice_id
    )
    system_id = find_parent_id(
        run["project_id"],
        "systems",
        "systems",
        run["system_version_id"],
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
    metrics = (
        ("one_step_rmse", float(observation["one_step_rmse"]), 0.08),
        (
            "rollout_rmse_25_step",
            float(observation["rollout_rmse"]),
            float(observation["rollout_rmse_threshold"]),
        ),
        (
            "gpu_training_duration_seconds",
            float(report["duration_seconds"]),
            900.0,
        ),
        (
            "gpu_peak_memory_mb",
            float(report["peak_memory_mb"]),
            24_576.0,
        ),
    )
    now = datetime.now(UTC).isoformat()
    events = [
        {
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
            "timestamp": now,
            "monotonic_ns": time.monotonic_ns(),
            "step": index,
            "event_type": "metric.recorded",
            "source": "modal-l4-world-model-evaluation",
            "payload": {
                "key": key,
                "kind": "scalar",
                "scalar_value": value,
                "unit": (
                    "seconds"
                    if key.endswith("_seconds")
                    else "megabytes" if key.endswith("_mb") else "rmse"
                ),
                "direction": "minimize",
                "threshold": threshold,
                "passed": value <= threshold,
            },
            "attributes": {
                "seed": run["seed"],
                "training_report_digest": report["model_content_digest"],
                "evidence_mode": "precomputed-on-modal-l4",
            },
        }
        for index, (key, value, threshold) in enumerate(metrics, start=1)
    ]
    events.append(
        {
            **events[-1],
            "event_id": f"evt_{uuid.uuid4().hex[:24]}",
            "monotonic_ns": time.monotonic_ns(),
            "step": len(events) + 1,
            "payload": {
                "key": "evaluation_replay_wall_time_ms",
                "kind": "scalar",
                "scalar_value": (time.perf_counter() - started) * 1000.0,
                "unit": "milliseconds",
                "direction": "minimize",
                "passed": True,
            },
        }
    )
    request("POST", f"/runs/{run['id']}/events:batch", events)


if __name__ == "__main__":
    main()
