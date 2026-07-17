"""Shared test doubles for the iso-obs SDK test suite.

This lives in its own module (not conftest.py) with a package-unique basename:
every workspace member's tests/ dir lands on sys.path, so a generic module
name here would shadow — or be shadowed by — another member's.

Everything network-shaped goes through :class:`httpx.MockTransport`, so the
suite never opens a socket. ``make_client`` builds a real
:class:`~iso_obs.ReliabilityClient` and swaps its transport for one backed by
the given handler with zero backoff, exercising the production retry and
error-mapping code paths hermetically.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from iso_obs import ReliabilityClient
from iso_obs._transport import Transport
from iso_obs_schemas import IdPrefix, generate_id

BASE_URL = "https://api.test/api/v1"


def make_run_payload(**overrides: Any) -> dict[str, Any]:
    """Build a valid ``Run`` JSON payload as the API would return it.

    Args:
        **overrides: Field values overriding the generated defaults.

    Returns:
        A dict that validates as :class:`iso_obs_schemas.Run`, with the
        resolved definition ids echoed in ``metadata`` per the run-creation
        contract.
    """
    payload: dict[str, Any] = {
        "id": generate_id(IdPrefix.RUN),
        "workspace_id": generate_id(IdPrefix.WORKSPACE),
        "project_id": generate_id(IdPrefix.PROJECT),
        "suite_execution_id": generate_id(IdPrefix.EVALUATION_SUITE),
        "system_version_id": generate_id(IdPrefix.SYSTEM_VERSION),
        "environment_version_id": generate_id(IdPrefix.ENVIRONMENT_VERSION),
        "scenario_version_id": generate_id(IdPrefix.SCENARIO_VERSION),
        "seed": 42,
        "status": "running",
        "metadata": {
            "system_id": generate_id(IdPrefix.SYSTEM),
            "environment_id": generate_id(IdPrefix.ENVIRONMENT),
            "scenario_id": generate_id(IdPrefix.SCENARIO),
        },
    }
    payload.update(overrides)
    return payload


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> ReliabilityClient:
    """Build a client whose transport is backed by a mock handler.

    Args:
        handler: Handler callable wrapped in an ``httpx.MockTransport``.

    Returns:
        A client with retries enabled but zero backoff delay.
    """
    mock = httpx.MockTransport(handler)
    client = ReliabilityClient(api_key="key_test", base_url=BASE_URL)
    client._transport = Transport(
        api_key="key_test",
        base_url=BASE_URL,
        timeout=5.0,
        http_transport=mock,
        backoff_base=0.0,
    )
    return client


class FakeApi:
    """In-memory Reliability Studio API double.

    Records event batches and lifecycle calls so tests can assert on exactly
    what reached the server. ``fail_batch_requests`` makes the first N
    ``events:batch`` HTTP requests return 500, which exercises both the
    transport retry and the RunContext buffering/spill behavior.
    """

    def __init__(
        self,
        run_payload: dict[str, Any] | None = None,
        fail_batch_requests: int = 0,
    ) -> None:
        """Initialize the double.

        Args:
            run_payload: Run JSON returned by the create-run route.
            fail_batch_requests: Number of ``events:batch`` requests to fail
                with a 500 before accepting batches.
        """
        self.run_payload = run_payload or make_run_payload()
        self.fail_batch_requests = fail_batch_requests
        self.batch_request_count = 0
        self.batches: list[list[dict[str, Any]]] = []
        self.completed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    @property
    def run_id(self) -> str:
        """Id of the run this double serves."""
        run_id = self.run_payload["id"]
        assert isinstance(run_id, str)
        return run_id

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Route a request to the matching in-memory handler.

        Args:
            request: The intercepted HTTP request.

        Returns:
            The canned response for the route.
        """
        path = request.url.path
        if request.method == "POST" and path.endswith("/runs"):
            return httpx.Response(200, json=self.run_payload)
        if path.endswith("/events:batch"):
            self.batch_request_count += 1
            if self.batch_request_count <= self.fail_batch_requests:
                return httpx.Response(500, json={"detail": "ingest unavailable"})
            body = json.loads(request.content)
            self.batches.append(body["events"])
            return httpx.Response(202, json={"accepted": len(body["events"])})
        if path.endswith("/complete"):
            self.completed.append(path.rsplit("/", 2)[-2])
            return httpx.Response(200, json=self.run_payload)
        if path.endswith("/fail"):
            body = json.loads(request.content)
            self.failed.append((path.rsplit("/", 2)[-2], body["reason"]))
            return httpx.Response(200, json=self.run_payload)
        return httpx.Response(404, json={"detail": f"no route for {path}"})
