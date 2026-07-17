# iso-obs — the Reliability Studio Python SDK

Instrument evaluation runs of autonomous systems — policies, controllers,
agents — and stream their traces to [Reliability Studio](https://iso-obs.com).
Every run preserves seed, versions, observations, actions, metrics, and
artifacts, so failures are reproducible and every claim is traceable.

## Installation

```bash
pip install iso-obs
```

Requires Python 3.12+.

## Quickstart

```python
from iso_obs import ReliabilityClient
from iso_obs.run import RunContext

client = ReliabilityClient()  # reads ISO_OBS_API_KEY from the environment

with RunContext(
    client=client,
    project="robot-arm",
    system_version="policy-v17",
    environment="warehouse-v4",
    scenario="obstructed-pick",
    seed=42,
) as run:
    obs = env.reset(seed=run.seed)
    while True:
        run.log_observation(obs)
        action = policy(obs)
        run.log_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        run.log_metric("reward", reward)
        if terminated or truncated:
            break
    run.log_artifact("replay.mp4")
```

On clean exit the buffered events are flushed and the run is marked
**completed**; if the body raises, the run is marked **failed** (with the
exception as the reason) and the exception is re-raised.

## Configuration

| Setting | Argument | Environment variable | Default |
|---|---|---|---|
| API key | `api_key` | `ISO_OBS_API_KEY` | — (required) |
| API root | `base_url` | `ISO_OBS_BASE_URL` | `https://api.iso-obs.com/api/v1` |
| Timeout | `timeout` | — | `30.0` seconds |

`ReliabilityClient` raises `iso_obs.exceptions.AuthenticationError` when no
API key can be resolved.

## Client API

```python
client.projects.create(name)            # -> Project
client.projects.list()                  # -> list[Project]
client.systems.register(project, name, version,
                        artifact_uri=None, source_commit=None,
                        framework=None, metadata=None)  # -> SystemVersion
client.runs.create(project, system_version, environment, scenario, seed,
                   perturbations=None, metadata=None)   # -> Run
client.runs.log_events(run_id, events)  # POST /runs/{id}/events:batch
client.runs.complete(run_id)
client.runs.fail(run_id, reason)
```

All returned objects are the Pydantic models from
[`iso-obs-schemas`](../schemas), the shared contract package.

## Reliability semantics

- **Retries** — every request is attempted up to 3 times on `429`, `5xx`, and
  transport errors, with exponential backoff.
- **Typed errors** — failures raise `ApiError` subclasses from
  `iso_obs.exceptions`: `AuthenticationError`, `NotFoundError`,
  `RateLimitError`, `ServerError`.
- **Event batching** — `RunContext` buffers events and flushes in batches of
  500, plus a final flush on exit.
- **Offline resilience** — if a flush still fails after retries, the buffer is
  preserved and re-attempted on the next flush/close. If the final flush on
  close also fails, the pending events are appended to
  `.iso-obs/pending-events.jsonl` (one JSON event per line, relative to the
  working directory) before the error is raised, so no telemetry is lost. An
  exception inside the run body is never masked by flush errors.

## Custom metrics

Declare project-specific metrics with the `metric` decorator; the declaration
is picked up by suite tooling:

```python
from iso_obs.metrics import metric

@metric("tracking_error", "minimize")
def tracking_error(trace) -> float:
    return max(abs(step["error"]) for step in trace)
```

## Custom environments

Bring your own simulator by implementing the adapter ABC:

```python
from iso_obs.environments import ReliabilityEnvironment

class MySimulator(ReliabilityEnvironment):
    def reset(self, scenario, seed): ...
    def observe(self): ...
    def step(self, action): ...
    def get_state(self): ...
    def collect_artifacts(self): ...
    def close(self): ...
```

## Development

```bash
# from packages/python-sdk
PYTHONPATH=src:../schemas/src python -m pytest
```

Licensed under [Apache-2.0](./LICENSE).
