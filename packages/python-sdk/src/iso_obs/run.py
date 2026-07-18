"""Run lifecycle instrumentation for the iso-obs SDK.

:class:`RunContext` is the primary way to trace a run. Entering the context
creates the run through the API; the ``log_*`` methods buffer
:class:`~iso_obs_schemas.BaseEvent` records; the buffer is flushed in batches
of :data:`BATCH_SIZE` events and again on exit. If the body raises, the run is
marked failed and the exception re-raised; otherwise it is marked completed.

Offline resilience: a flush that still fails after the transport's retries
leaves the buffer intact, so the events are re-attempted on the next flush or
on close. If the final flush on close also fails, the buffered events are first
spilled to ``.iso-obs/pending-events.jsonl`` (one JSON event per line, appended,
relative to the current working directory) and the error is then raised, so no
telemetry is silently lost.

Event stamping: :class:`~iso_obs_schemas.BaseEvent` requires the definition ids
(``system_id``, ``environment_id``, ``scenario_id``) that the ``Run`` model
itself does not carry, so the run-creation endpoint echoes those resolved ids
in ``Run.metadata``; :class:`RunContext` reads them from there.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from iso_obs_schemas import (
    ActionPayload,
    ArtifactRef,
    BaseEvent,
    EventType,
    IdPrefix,
    ObservationPayload,
    PerturbationSpec,
    Run,
    generate_id,
)

from .client import ReliabilityClient
from .exceptions import ApiError

# Number of buffered events that triggers a flush mid-run; flushes also send
# events in chunks of this size.
BATCH_SIZE = 500

# Where unsent events are spilled on a final flush failure, relative to the
# current working directory.
SPILL_PATH = Path(".iso-obs") / "pending-events.jsonl"

# Value of BaseEvent.source for everything this module emits.
_SOURCE = "sdk"


def _as_payload_dict(value: Any) -> dict[str, Any]:
    """Coerce a logged value into the dict shape the payload models expect.

    Args:
        value: The observation/action/state the caller logged. Mappings pass
            through; anything else is wrapped under a ``"value"`` key so the
            payload stays a JSON object.

    Returns:
        A dict representation of ``value``.
    """
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


class RunContext:
    """Context manager that creates, traces, and finishes a run.

    Example:
        >>> with RunContext(
        ...     client=client,
        ...     project="robot-arm",
        ...     system_version="policy-v17",
        ...     environment="warehouse-v4",
        ...     scenario="obstructed-pick",
        ...     seed=42,
        ... ) as run:
        ...     run.log_observation({"joint_pos": [0.0]})
    """

    def __init__(
        self,
        *,
        client: ReliabilityClient,
        project: str,
        system_version: str,
        environment: str,
        scenario: str,
        seed: int,
        perturbations: list[PerturbationSpec] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the context; no API call happens until ``__enter__``.

        Args:
            client: Client used for run creation, event batches, and finish.
            project: Project the run belongs to (name or id).
            system_version: System version under evaluation (label or id).
            environment: Environment to run in (name or id).
            scenario: Scenario to evaluate (name or id).
            seed: Random seed for the run, exposed back via :attr:`seed`.
            perturbations: Perturbations applied during the run.
            metadata: Free-form run metadata.
        """
        self._client = client
        self._project = project
        self._system_version = system_version
        self._environment = environment
        self._scenario = scenario
        self._seed = seed
        self._perturbations = list(perturbations or [])
        self._metadata = dict(metadata or {})
        self._run: Run | None = None
        self._buffer: list[BaseEvent] = []
        # Step index of the current environment step; -1 until the first
        # observation arrives, then incremented per observation.
        self._step = -1

    @property
    def seed(self) -> int:
        """Random seed of the run, for seeding the environment/policy."""
        return self._seed

    def __enter__(self) -> Self:
        """Create the run through the API and start tracing.

        Returns:
            This context, ready to log events.
        """
        self._run = self._client.runs.create(
            project=self._project,
            system_version=self._system_version,
            environment=self._environment,
            scenario=self._scenario,
            seed=self._seed,
            perturbations=self._perturbations,
            metadata=self._metadata,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Flush remaining events and finish the run.

        Never suppresses a body exception (returns ``None``).

        Args:
            exc_type: Exception type raised in the body, if any.
            exc: Exception instance raised in the body, if any.
            tb: Traceback of the exception, if any.

        Raises:
            ApiError: On the clean-exit path, if the final flush fails even
                after spilling the buffer to :data:`SPILL_PATH`.
        """
        run = self._require_run()
        if exc_type is None:
            self._flush(raise_on_failure=True)
            self._client.runs.complete(run.id)
            return
        # Failure path: salvage what we can, but never mask the user's
        # exception — flush errors are spilled, and a failing fail() call is
        # suppressed for the same reason.
        self._flush(raise_on_failure=False)
        if self._buffer:
            self._spill()
        with contextlib.suppress(ApiError):
            self._client.runs.fail(run.id, reason=f"{exc_type.__name__}: {exc}")

    def log_observation(self, obs: Any) -> None:
        """Record the observation delivered to the system this step.

        Advances the internal step counter: each observation starts a new
        environment step, and subsequent events are stamped with it.

        Args:
            obs: The observation; mappings are logged as-is, other values are
                wrapped under a ``"value"`` key.
        """
        self._step += 1
        payload = ObservationPayload(observation=_as_payload_dict(obs)).model_dump()
        self._emit(EventType.OBSERVATION_RECEIVED, payload)

    def log_action(self, action: Any, latency_ms: float | None = None) -> None:
        """Record the action the system emitted for the current step.

        Args:
            action: The action; mappings are logged as-is, other values are
                wrapped under a ``"value"`` key.
            latency_ms: Time the system took to produce the action, in
                milliseconds.
        """
        payload = ActionPayload(
            action=_as_payload_dict(action), latency_ms=latency_ms
        ).model_dump()
        self._emit(EventType.ACTION_EMITTED, payload)

    def log_metric(self, name: str, value: float) -> None:
        """Record a scalar metric sample for the current step.

        Args:
            name: Metric name, e.g. ``"reward"``.
            value: Scalar value of the sample.
        """
        self._emit(EventType.METRIC_RECORDED, {"name": name, "value": float(value)})

    def log_state(self, state: Any) -> None:
        """Record a ground-truth state snapshot for the current step.

        Args:
            state: The state; mappings are logged as-is, other values are
                wrapped under a ``"value"`` key.
        """
        self._emit(EventType.STATE_RECORDED, {"state": _as_payload_dict(state)})

    def log_artifact(self, path_or_uri: str | os.PathLike[str]) -> None:
        """Record an artifact produced by the run (video, plot, log file).

        Args:
            path_or_uri: Local path or storage URI of the artifact.
        """
        ref = ArtifactRef(uri=str(path_or_uri), kind="file")
        self._emit(EventType.ARTIFACT_CREATED, {"artifact": ref.model_dump()})

    def step(
        self,
        *,
        observation: Any,
        action: Any,
        reward: float | None = None,
        state: Any | None = None,
        metrics: Mapping[str, float] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Record one ordered environment interaction.

        The observation advances the step counter once. Every event emitted
        afterward is stamped with that same step in this deterministic order:
        action, optional state, optional reward, then custom metrics in mapping
        iteration order.

        Args:
            observation: Observation delivered to the evaluated system.
            action: Action emitted by the evaluated system.
            reward: Optional scalar reward for the transition.
            state: Optional privileged environment-state snapshot.
            metrics: Additional scalar measurements for the transition.
            latency_ms: Time the system took to produce the action.

        Raises:
            ValueError: If both ``reward`` and a custom ``"reward"`` metric
                are provided.
            RuntimeError: If the run context has not been entered.
        """
        resolved_metrics = dict(metrics or {})
        if reward is not None and "reward" in resolved_metrics:
            raise ValueError(
                "reward was provided twice; remove metrics['reward'] or "
                "the reward argument"
            )
        self._require_run()
        self.log_observation(observation)
        self.log_action(action, latency_ms=latency_ms)
        if state is not None:
            self.log_state(state)
        if reward is not None:
            self.log_metric("reward", reward)
        for name, value in resolved_metrics.items():
            self.log_metric(name, value)

    def flush(self) -> None:
        """Synchronously deliver every currently buffered event.

        Flushing does not complete the run. If delivery fails after transport
        retries, pending events are written to :data:`SPILL_PATH` before the
        API error is re-raised.

        Raises:
            ApiError: If the buffered events cannot be delivered.
            RuntimeError: If the run context has not been entered.
        """
        self._flush(raise_on_failure=True)

    def _require_run(self) -> Run:
        """Return the created run, enforcing context-manager usage.

        Returns:
            The run created on ``__enter__``.

        Raises:
            RuntimeError: If the context was never entered.
        """
        if self._run is None:
            raise RuntimeError(
                "RunContext must be entered with 'with' before logging events"
            )
        return self._run

    def _resolved_id(self, key: str) -> str:
        """Read a definition id the API echoed in ``Run.metadata``.

        Args:
            key: Metadata key, one of ``"system_id"``, ``"environment_id"``,
                or ``"scenario_id"``.

        Returns:
            The id string stored under ``key``.

        Raises:
            ValueError: If the run's metadata does not carry the id, meaning
                the server did not honor the run-creation contract.
        """
        run = self._require_run()
        value = run.metadata.get(key)
        if not isinstance(value, str):
            raise ValueError(
                f"run {run.id} metadata is missing '{key}'; the run-creation "
                "endpoint must echo resolved definition ids in Run.metadata"
            )
        return value

    def _emit(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """Build a trace event, buffer it, and flush when the batch is full.

        Args:
            event_type: Kind of event being recorded.
            payload: Event-type-specific body.
        """
        run = self._require_run()
        event = BaseEvent(
            event_id=generate_id(IdPrefix.EVENT),
            workspace_id=run.workspace_id,
            project_id=run.project_id,
            suite_id=run.suite_execution_id,
            run_id=run.id,
            system_id=self._resolved_id("system_id"),
            system_version=run.system_version_id,
            environment_id=self._resolved_id("environment_id"),
            environment_version=run.environment_version_id,
            scenario_id=self._resolved_id("scenario_id"),
            monotonic_ns=time.monotonic_ns(),
            step=self._step if self._step >= 0 else None,
            event_type=event_type,
            source=_SOURCE,
            payload=payload,
        )
        self._buffer.append(event)
        if len(self._buffer) >= BATCH_SIZE:
            self._flush(raise_on_failure=False)

    def _flush(self, *, raise_on_failure: bool) -> None:
        """Send buffered events in :data:`BATCH_SIZE` chunks.

        Args:
            raise_on_failure: When ``False`` (mid-run flushes), a failure
                leaves the buffer intact for the next attempt. When ``True``
                (final flush on close), a failure spills the buffer to
                :data:`SPILL_PATH` and re-raises.

        Raises:
            ApiError: If a batch fails and ``raise_on_failure`` is set.
        """
        run = self._require_run()
        while self._buffer:
            batch = self._buffer[:BATCH_SIZE]
            try:
                self._client.runs.log_events(run.id, batch)
            except ApiError:
                if raise_on_failure:
                    self._spill()
                    raise
                return
            del self._buffer[: len(batch)]

    def _spill(self) -> None:
        """Persist the buffer to the local spill file and clear it.

        Events are appended to :data:`SPILL_PATH` as one JSON object per line
        so a later process can replay them.
        """
        path = Path.cwd() / SPILL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for event in self._buffer:
                handle.write(event.model_dump_json() + "\n")
        self._buffer.clear()
