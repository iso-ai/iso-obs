"""Execute one pinned warehouse run and stream scored evidence to Studio."""

from __future__ import annotations

import json
import os
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from typing import Any


def request(method: str, path: str, body: Any | None = None) -> Any:
    """Call the authenticated platform API."""
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


def event(
    run: dict[str, Any],
    event_type: str,
    payload: dict[str, Any],
    *,
    system_id: str,
    environment_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    """Build one valid trace event for the pinned run."""
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
        "step": 1,
        "event_type": event_type,
        "source": "warehouse-platform-adapter",
        "payload": payload,
        "attributes": {"seed": run["seed"]},
    }


def main() -> None:
    """Score one deterministic warehouse simulation assignment."""
    run = request("GET", f"/runs/{os.environ['ISO_RUN_ID']}")
    scenario = json.loads(os.environ["ISO_SCENARIO_CONFIG"])
    system = json.loads(os.environ["ISO_SYSTEM_CONTEXT"])
    config = scenario.get("config", {})
    candidate = "candidate" in system.get("version", "")
    visibility = float(config.get("visibility", 1.0))
    latency_ms = float(config.get("latency_ms", 0.0))
    score = 1.0 - (1.0 - visibility) * 0.35 - latency_ms / 500.0
    if candidate:
        score -= (1.0 - visibility) * 0.45 + latency_ms / 700.0
    score = max(0.0, min(1.0, score))
    passed = score >= 0.65
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
    events = [
        event(
            run,
            "metric.recorded",
            {
                "key": "task_success_score",
                "kind": "scalar",
                "scalar_value": score,
                "unit": "ratio",
                "direction": "maximize",
                "threshold": 0.65,
                "passed": passed,
            },
            system_id=system_id,
            environment_id=environment_id,
            scenario_id=scenario_id,
        )
    ]
    if not passed:
        events.append(
            event(
                run,
                "constraint.violated",
                {
                    "constraint": "minimum_task_success",
                    "observed_value": score,
                    "threshold": 0.65,
                    "acceptance_operator": "gte",
                    "severity": "critical",
                    "first_occurrence": True,
                },
                system_id=system_id,
                environment_id=environment_id,
                scenario_id=scenario_id,
            )
        )
    request("POST", f"/runs/{run['id']}/events:batch", events)


if __name__ == "__main__":
    main()
