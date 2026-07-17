"""The Reliability Studio API client.

:class:`ReliabilityClient` is the entry point of the SDK. It resolves
credentials and the API root (explicit arguments first, then the
``ISO_OBS_API_KEY`` / ``ISO_OBS_BASE_URL`` environment variables), owns the
retrying HTTP transport, and exposes the API as three resource namespaces:

* ``client.projects`` — create and list projects.
* ``client.systems`` — register system versions.
* ``client.runs`` — create runs, stream trace events, and finish runs.

Resource objects resolve the transport through the client at call time rather
than capturing it, so a test (or an advanced caller) can swap
``client._transport`` once and every namespace follows.
"""

from __future__ import annotations

import os
from typing import Any

from iso_obs_schemas import BaseEvent, PerturbationSpec, Project, Run, SystemVersion

from ._transport import Transport
from .exceptions import AuthenticationError

# Default API root, used when neither the base_url argument nor the
# ISO_OBS_BASE_URL environment variable is set.
DEFAULT_BASE_URL = "https://api.iso-obs.com/api/v1"


class _Resource:
    """Shared plumbing for the API resource namespaces."""

    def __init__(self, client: ReliabilityClient) -> None:
        """Bind the namespace to its owning client.

        Args:
            client: The client whose transport this namespace routes through.
        """
        self._client = client

    @property
    def _transport(self) -> Transport:
        """Return the client's current transport, resolved late by design."""
        return self._client._transport


class ProjectsResource(_Resource):
    """Operations on projects (``/projects``)."""

    def create(self, name: str) -> Project:
        """Create a project.

        Args:
            name: Human-readable project name.

        Returns:
            The created :class:`~iso_obs_schemas.Project`.
        """
        data = self._transport.request("POST", "/projects", json={"name": name})
        return Project.model_validate(data)

    def list(self) -> list[Project]:
        """List the projects visible to the API key's workspace.

        Returns:
            All visible :class:`~iso_obs_schemas.Project` objects.
        """
        data = self._transport.request("GET", "/projects")
        return [Project.model_validate(item) for item in data]


class SystemsResource(_Resource):
    """Operations on systems under evaluation (``/systems``)."""

    def register(
        self,
        project: str,
        name: str,
        version: str,
        artifact_uri: str | None = None,
        source_commit: str | None = None,
        framework: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SystemVersion:
        """Register a system version, creating the system if needed.

        Args:
            project: Project the system belongs to (name or id).
            name: Human-readable system name.
            version: Version label, e.g. a semver or build tag.
            artifact_uri: Location of the system artifact (weights/image).
            source_commit: Source revision that produced this build.
            framework: Framework the system is built with, e.g. ``"jax"``.
            metadata: Free-form metadata stored with the version.

        Returns:
            The pinned :class:`~iso_obs_schemas.SystemVersion`.
        """
        body = {
            "project": project,
            "name": name,
            "version": version,
            "artifact_uri": artifact_uri,
            "source_commit": source_commit,
            "framework": framework,
            "metadata": metadata or {},
        }
        data = self._transport.request("POST", "/systems:register", json=body)
        return SystemVersion.model_validate(data)


class RunsResource(_Resource):
    """Operations on evaluation runs (``/runs``)."""

    def create(
        self,
        project: str,
        system_version: str,
        environment: str,
        scenario: str,
        seed: int,
        perturbations: list[PerturbationSpec] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Run:
        """Create a run of one system version against one scenario.

        Args:
            project: Project the run belongs to (name or id).
            system_version: System version under evaluation (label or id).
            environment: Environment to run in (name or id).
            scenario: Scenario to evaluate (name or id).
            seed: Random seed for the run.
            perturbations: Perturbations to apply during the run.
            metadata: Free-form metadata stored with the run.

        Returns:
            The created :class:`~iso_obs_schemas.Run`. The API echoes the
            resolved ``system_id``, ``environment_id``, and ``scenario_id`` in
            ``Run.metadata`` so trace events can reference them.
        """
        body = {
            "project": project,
            "system_version": system_version,
            "environment": environment,
            "scenario": scenario,
            "seed": seed,
            "perturbations": [
                spec.model_dump(mode="json") for spec in perturbations or []
            ],
            "metadata": metadata or {},
        }
        data = self._transport.request("POST", "/runs", json=body)
        return Run.model_validate(data)

    def log_events(self, run_id: str, events: list[BaseEvent]) -> None:
        """Append a batch of trace events to a run.

        Args:
            run_id: Run the events belong to.
            events: Events to append, in emission order.
        """
        body = {"events": [event.model_dump(mode="json") for event in events]}
        self._transport.request("POST", f"/runs/{run_id}/events:batch", json=body)

    def complete(self, run_id: str) -> None:
        """Mark a run as completed.

        Args:
            run_id: Run to complete.
        """
        self._transport.request("POST", f"/runs/{run_id}/complete")

    def fail(self, run_id: str, reason: str) -> None:
        """Mark a run as failed.

        Args:
            run_id: Run to fail.
            reason: Human-readable failure reason.
        """
        self._transport.request("POST", f"/runs/{run_id}/fail", json={"reason": reason})


class ReliabilityClient:
    """Client for the Reliability Studio control-plane API.

    Attributes:
        projects: Project operations.
        systems: System registration operations.
        runs: Run lifecycle and trace-event operations.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: API key; falls back to the ``ISO_OBS_API_KEY``
                environment variable.
            base_url: API root; falls back to ``ISO_OBS_BASE_URL``, then to
                :data:`DEFAULT_BASE_URL`.
            timeout: Per-request timeout in seconds.

        Raises:
            AuthenticationError: If no API key is provided and
                ``ISO_OBS_API_KEY`` is unset.
        """
        resolved_key = api_key or os.environ.get("ISO_OBS_API_KEY")
        if not resolved_key:
            raise AuthenticationError(
                "no API key: pass api_key or set the ISO_OBS_API_KEY "
                "environment variable"
            )
        resolved_base = (
            base_url or os.environ.get("ISO_OBS_BASE_URL") or DEFAULT_BASE_URL
        )
        self._transport = Transport(
            api_key=resolved_key, base_url=resolved_base, timeout=timeout
        )
        self.projects = ProjectsResource(self)
        self.systems = SystemsResource(self)
        self.runs = RunsResource(self)
