# iso-obs-schemas

The **authoritative shared schema contract** for [Reliability Studio](https://iso-obs.com)
by Iso AI. This package defines the Pydantic v2 models that every other part of
the system is built against:

- the **Python SDK** (`iso_obs`) emits traces shaped by these models,
- the **API** (`apps/api`) validates and persists them,
- the **web app** (`apps/web`) code-generates its TypeScript types from the JSON
  Schema exported here.

If a shape changes here, it changes everywhere. Treat this package as the single
source of truth and change it deliberately.

- PyPI name: `iso-obs-schemas`
- Import name: `iso_obs_schemas`
- Python: `>=3.12`

## Install

```bash
pip install iso-obs-schemas
# for development (tests):
pip install "iso-obs-schemas[dev]"
```

## What's inside

| Module | Contents |
| --- | --- |
| `ids` | Typed, prefixed id convention (`IdPrefix`, `generate_id`, validated id aliases) |
| `enums` | Shared `StrEnum` vocabularies (`EventType`, `RunStatus`, `FailureCategory`, `Severity`, `SystemType`, `PerturbationFamily`, `MetricDirection`, `WorkspaceRole`) |
| `core` | The eight core objects and their versions, plus `Run` |
| `events` | `BaseEvent` trace schema and typed payload models |
| `metrics` | `MetricValue`, `MetricDefinition`, `Invariant`, `Gate` |
| `reports` | `Finding`, `ReliabilityReport` |

```python
from iso_obs_schemas import Run, BaseEvent, RunStatus, generate_id, IdPrefix

run = Run(
    id=generate_id(IdPrefix.RUN),
    workspace_id=generate_id(IdPrefix.WORKSPACE),
    project_id=generate_id(IdPrefix.PROJECT),
    suite_execution_id=generate_id(IdPrefix.EVALUATION_SUITE),
    system_version_id=generate_id(IdPrefix.SYSTEM_VERSION),
    environment_version_id=generate_id(IdPrefix.ENVIRONMENT_VERSION),
    scenario_version_id=generate_id(IdPrefix.SCENARIO_VERSION),
    seed=7,
    status=RunStatus.COMPLETED,
)
```

## Typed id conventions

Every persistent object carries a typed, prefixed string id (e.g.
`run_9f8c2a...`). The prefix names the object kind, so ids are self-describing in
logs, traces, and URLs, and the API can reject a mismatched id at the edge. The
prefix set is **closed** — adding a kind means adding an `IdPrefix` member and a
matching validated alias in `ids.py`.

| Prefix | Object |
| --- | --- |
| `ws_` | Workspace |
| `prj_` | Project |
| `sys_` | System |
| `sv_` | SystemVersion |
| `env_` | Environment |
| `ev_` | EnvironmentVersion |
| `scn_` | Scenario |
| `scv_` | ScenarioVersion |
| `suite_` | EvaluationSuite / suite execution |
| `run_` | Run |
| `evt_` | Trace event |
| `fail_` | Failure |
| `cmp_` | Comparison |
| `rpt_` | Report |
| `key_` | API key |
| `wh_` | Webhook |

The id aliases are validated: assigning a `ws_...` id to a field typed as a
`ProjectId` raises a `ValidationError`.

## Exporting JSON Schema

The web app codegen consumes JSON Schema for the public models. Regenerate it
into `schemas/json/` with:

```bash
python -m scripts.export_json_schema
```

## Tests

```bash
pytest
```
