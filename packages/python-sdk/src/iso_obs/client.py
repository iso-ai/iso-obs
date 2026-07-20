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
import re
from typing import TYPE_CHECKING, Any

from iso_obs_schemas import (
    BaseEvent,
    Environment,
    EnvironmentRenderManifest,
    EnvironmentVersion,
    EvaluationSuite,
    PerturbationSpec,
    Project,
    Run,
    Scenario,
    ScenarioVersion,
    System,
    SystemType,
    SystemVersion,
)

from ._transport import Transport
from .exceptions import AuthenticationError

if TYPE_CHECKING:
    from .run import RunContext

# Default API root, used when neither the base_url argument nor the
# ISO_OBS_BASE_URL environment variable is set.
DEFAULT_BASE_URL = "https://reliability-studio-5cmy6.ondigitalocean.app/api/v1"


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

    def create(
        self,
        name: str,
        *,
        slug: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Project:
        """Create a project.

        Args:
            name: Human-readable project name.
            slug: URL-safe project slug. Derived from ``name`` when omitted.
            description: Optional project description.
            metadata: Free-form project metadata.

        Returns:
            The created :class:`~iso_obs_schemas.Project`.
        """
        resolved_slug = slug or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not resolved_slug:
            resolved_slug = "project"
        data = self._transport.request(
            "POST",
            "/projects",
            json={
                "name": name,
                "slug": resolved_slug,
                "description": description,
                "metadata": metadata or {},
            },
        )
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
        system_type: SystemType | str = SystemType.OTHER,
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
            system_type: Broad category of the evaluated system.
            metadata: Free-form metadata stored with the version.

        Returns:
            The pinned :class:`~iso_obs_schemas.SystemVersion`.
        """
        systems_data = self._transport.request("GET", f"/projects/{project}/systems")
        systems = [System.model_validate(item) for item in systems_data]
        system = next((item for item in systems if item.name == name), None)
        if system is None:
            created = self._transport.request(
                "POST",
                f"/projects/{project}/systems",
                json={
                    "name": name,
                    "system_type": SystemType(system_type).value,
                    "description": None,
                    "metadata": {"framework": framework} if framework else {},
                },
            )
            system = System.model_validate(created)
        version_metadata = dict(metadata or {})
        if framework is not None:
            version_metadata.setdefault("framework", framework)
        body = {
            "version": version,
            "artifact_uri": artifact_uri,
            "commit_sha": source_commit,
            "metadata": version_metadata,
        }
        data = self._transport.request(
            "POST", f"/systems/{system.id}/versions", json=body
        )
        return SystemVersion.model_validate(data)


class EnvironmentsResource(_Resource):
    """Operations on simulator, benchmark, and rig environments."""

    def register(
        self,
        project_id: str,
        name: str,
        version: str,
        *,
        image_uri: str | None = None,
        config: dict[str, Any] | None = None,
        render_manifest: EnvironmentRenderManifest | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EnvironmentVersion:
        """Register an environment and pin one immutable version.

        Args:
            project_id: Project that owns the environment.
            name: Stable environment name.
            version: Immutable build label.
            image_uri: Optional simulator image reference.
            config: Frozen simulator configuration.
            render_manifest: Exact declared geometry rendered by paid Studio plans.
            description: Optional environment description.
            metadata: Additional structured metadata.

        Returns:
            The pinned environment version.
        """
        data = self._transport.request("GET", f"/projects/{project_id}/environments")
        environments = [Environment.model_validate(item) for item in data]
        environment = next((item for item in environments if item.name == name), None)
        if environment is None:
            created = self._transport.request(
                "POST",
                f"/projects/{project_id}/environments",
                json={
                    "name": name,
                    "description": description,
                    "metadata": metadata or {},
                },
            )
            environment = Environment.model_validate(created)
        frozen_config = dict(config or {})
        if render_manifest is not None:
            frozen_config["render_manifest"] = render_manifest.model_dump(mode="json")
        version_data = self._transport.request(
            "POST",
            f"/environments/{environment.id}/versions",
            json={
                "version": version,
                "image_uri": image_uri,
                "config": frozen_config,
                "metadata": metadata or {},
            },
        )
        return EnvironmentVersion.model_validate(version_data)


class ScenariosResource(_Resource):
    """Operations on versioned evaluation scenarios."""

    def register(
        self,
        environment_id: str,
        name: str,
        version: str,
        *,
        perturbations: list[PerturbationSpec] | None = None,
        config: dict[str, Any] | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScenarioVersion:
        """Register a scenario and pin one immutable version."""
        data = self._transport.request(
            "GET", f"/environments/{environment_id}/scenarios"
        )
        scenarios = [Scenario.model_validate(item) for item in data]
        scenario = next((item for item in scenarios if item.name == name), None)
        if scenario is None:
            created = self._transport.request(
                "POST",
                f"/environments/{environment_id}/scenarios",
                json={
                    "name": name,
                    "description": description,
                    "metadata": metadata or {},
                },
            )
            scenario = Scenario.model_validate(created)
        version_data = self._transport.request(
            "POST",
            f"/scenarios/{scenario.id}/versions",
            json={
                "version": version,
                "perturbations": [
                    item.model_dump(mode="json") for item in perturbations or []
                ],
                "config": config or {},
                "metadata": metadata or {},
            },
        )
        return ScenarioVersion.model_validate(version_data)


class SuitesResource(_Resource):
    """Operations on evaluation suites."""

    def create(
        self,
        project_id: str,
        name: str,
        scenario_version_ids: list[str],
        seeds: list[int],
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationSuite:
        """Create an evaluation suite from pinned scenarios and seeds."""
        data = self._transport.request(
            "POST",
            f"/projects/{project_id}/suites",
            json={
                "name": name,
                "description": description,
                "scenario_version_ids": scenario_version_ids,
                "seeds": seeds,
                "metadata": metadata or {},
            },
        )
        return EvaluationSuite.model_validate(data)


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
        self.environments = EnvironmentsResource(self)
        self.scenarios = ScenariosResource(self)
        self.suites = SuitesResource(self)
        self.runs = RunsResource(self)

    def run(
        self,
        *,
        project: str,
        system_version: str,
        environment: str,
        scenario: str,
        seed: int,
        perturbations: list[PerturbationSpec] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunContext:
        """Create a context for one instrumented evaluation run.

        This convenience method is equivalent to constructing
        :class:`iso_obs.run.RunContext` directly. It does not contact the API
        until the returned context is entered.

        Args:
            project: Project the run belongs to (name or id).
            system_version: System version under evaluation (label or id).
            environment: Environment to run in (name or id).
            scenario: Scenario to evaluate (name or id).
            seed: Random seed for the run.
            perturbations: Perturbations applied during the run.
            metadata: Free-form run metadata.

        Returns:
            A context manager that creates, traces, and finishes the run.
        """
        # Import locally to avoid a module-level cycle: RunContext uses this
        # client type to send lifecycle and event requests.
        from .run import RunContext

        return RunContext(
            client=self,
            project=project,
            system_version=system_version,
            environment=environment,
            scenario=scenario,
            seed=seed,
            perturbations=perturbations,
            metadata=metadata,
        )
