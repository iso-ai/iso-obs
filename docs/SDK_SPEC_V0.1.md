# iso-obs SDK specification — v0.1

**Status:** Draft implementation contract

**Product:** Iso AI Reliability Studio

**Package:** `iso-obs`

**Python:** 3.12+

**API:** `/api/v1`

## 1. Purpose

`iso-obs` is the customer-side execution plane for Reliability Studio. It
connects an autonomous system to its environment, executes reproducible
evaluations, captures evidence-grade traces, evaluates local metrics and
invariants, and delivers only authorized telemetry and artifacts to the hosted
control plane.

The SDK is not:

- a generic application-performance monitoring library;
- a simulator or model-training framework;
- the hosted suite scheduler, comparison engine, or report generator;
- an excuse to move customer models or source code into Iso infrastructure.

The v0.1 product loop is:

```text
register system
→ load one scenario
→ execute one local run
→ capture and ingest evidence
→ calculate metrics and invariant results
→ reproduce the run
```

## 2. Design principles

1. **Evaluation must outlive telemetry failures.** A control-plane outage must
   not invalidate or block a customer evaluation.
2. **Every conclusion must resolve to evidence.** Metrics, violations, and
   failures reference the events and artifacts that support them.
3. **Reproduction is part of the contract.** Seeds, immutable versions,
   configuration digests, runtime provenance, and artifact checksums are
   captured for every completed run.
4. **Customer environments remain native.** Adapters preserve simulator
   lifecycles instead of forcing them into a large framework hierarchy.
5. **Privacy is explicit.** Data collection, upload, retention, and training
   eligibility are declared policies, never implicit behavior.
6. **Deterministic analysis precedes AI interpretation.** AI-generated
   statements are hypotheses grounded in structured results and cited evidence.
7. **The common path is small.** An engineer can instrument an existing
   evaluation loop in fewer than ten meaningful lines.

## 3. Package boundaries

```text
iso_obs/
├── client.py          # authenticated control-plane resources
├── run.py             # run context and trace capture
├── environments.py    # simulator protocol and step result
├── scenarios.py       # scenario loading, validation, and digesting
├── suites.py          # local run-matrix expansion
├── perturbations.py   # perturbation protocol and built-in composition
├── metrics.py         # metric declarations and deterministic evaluation
├── invariants.py      # constrained temporal invariant evaluation
├── artifacts.py       # artifact policy, hashing, and upload preparation
├── spool.py           # durable local event and artifact queue
├── provenance.py      # runtime and source provenance collection
├── policies.py        # collection, redaction, retention, and consent
├── runner.py          # local evaluation runtime
└── exceptions.py      # stable public exception hierarchy
```

The `iso_obs_schemas` package owns shared wire models and generated JSON
schemas. The `iso_obs_cli` package is a thin, machine-readable interface over
the SDK and must not contain a second execution implementation.

## 4. Public API

### 4.1 Client

```python
from iso_obs import ReliabilityClient

client = ReliabilityClient(
    api_key=None,       # ISO_OBS_API_KEY
    base_url=None,      # ISO_OBS_BASE_URL
    timeout=30.0,
)
```

Required resource namespaces:

- `client.projects`
- `client.systems`
- `client.environments`
- `client.scenarios`
- `client.suites`
- `client.runs`
- `client.artifacts`

The v0.1 SDK may expose only resources supported by the current Reliability API,
but namespace additions must not require changing the constructor.

### 4.2 Run context

The ergonomic API is `client.run(...)`; direct `RunContext(...)` construction
remains supported.

```python
with client.run(
    project="robot-arm",
    system_version="policy-v17",
    environment="warehouse-v4",
    scenario="obstructed-pick",
    seed=42,
) as run:
    observation = env.reset(seed=run.seed)

    while not done:
        action = policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        run.step(
            observation=observation,
            action=action,
            reward=reward,
            state=info.get("state"),
        )
        observation = next_observation
        done = terminated or truncated
```

The explicit methods remain available:

- `run.log_observation(...)`
- `run.log_action(...)`
- `run.log_state(...)`
- `run.log_metric(...)`
- `run.log_event(...)`
- `run.log_artifact(...)`
- `run.flush()`

`run.step(...)` is convenience composition, not a distinct wire event. It emits
the same ordered typed events as the explicit methods.

### 4.3 Environment protocol

Structural typing is preferred over mandatory inheritance:

```python
class ReliabilityEnvironment(Protocol):
    def reset(self, scenario: ScenarioDefinition, seed: int) -> object: ...
    def observe(self) -> object: ...
    def step(self, action: object) -> StepResult: ...
    def get_state(self) -> object | None: ...
    def collect_artifacts(self) -> Iterable[Artifact]: ...
    def close(self) -> None: ...
```

```python
@dataclass(frozen=True, slots=True)
class StepResult:
    observation: object
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, object] = field(default_factory=dict)
```

An abstract base class may remain as an optional compatibility helper. Runtime
acceptance is based on the protocol.

## 5. Scenario format

Scenario files are versioned YAML or JSON documents:

```yaml
schema_version: "1.0"
name: obstructed-pick
environment: warehouse-v4
initial_conditions:
  object_type: carton
  object_pose: {x: 0.42, y: -0.18, z: 0.04}
goal:
  target_bin: bin_3
constraints:
  - name: max_gripper_force
    expression: gripper_force <= 18
    severity: critical
perturbations:
  - family: observation_noise
    name: wrist-camera-occlusion
    target: wrist_camera
    parameters:
      fraction: 0.20
metadata: {}
```

Loading a scenario:

1. validates the schema version;
2. rejects unknown required semantics;
3. canonicalizes the document;
4. computes a SHA-256 content digest;
5. preserves the original document as reproducibility evidence.

Environment-specific configuration remains under typed extension fields rather
than being promoted into the universal scenario schema.

## 6. Local execution

The local runner owns one evaluation process and drives:

```text
validate configuration
→ establish run and provenance
→ reset environment deterministically
→ execute system/environment loop
→ apply perturbations at declared boundaries
→ emit ordered evidence
→ evaluate streaming invariants
→ collect artifacts
→ finalize metrics
→ durably record terminal intent
→ synchronize with Reliability
→ close environment exactly once
```

The environment and evaluated system are user-provided callables. The SDK does
not import or deserialize customer model artifacts unless an adapter explicitly
does so.

Cancellation is cooperative in v0.1. The runner checks a local cancellation
token between steps and before artifact collection.

## 7. Suite expansion

A local suite definition expands deterministically into a run matrix:

```text
system versions × scenarios × perturbation variants × seeds × repetitions
```

v0.1 supports:

- fixed perturbations;
- evenly spaced sweeps;
- Cartesian grids;
- seeded random samples.

Boundary search, adaptive search, and Bayesian optimization are deferred. Matrix
expansion must be inspectable without execution:

```bash
iso suite plan reliability/release.yaml --json
```

The same suite document and seed must always produce the same ordered matrix.

## 8. Trace contract

### 8.1 Ordering

Every event carries:

- a globally unique typed event ID;
- workspace, project, suite, run, system, environment, and scenario IDs;
- a timezone-aware UTC timestamp;
- a process monotonic timestamp;
- an optional environment step;
- a source;
- a trace-schema version;
- a typed payload or forward-compatible extension payload.

Ordering within one producer is the append order, disambiguated by
`monotonic_ns`. Future multi-process producers require an explicit producer ID
and per-producer sequence number before claiming total ordering.

### 8.2 Payload policy

Inline payloads are limited by configurable serialized byte size. Images,
video, point clouds, tensors, large arrays, and logs above that limit become
content-addressed artifacts.

Unsupported Python objects are not stringified silently. Serialization either:

- uses a registered encoder;
- replaces the value according to an explicit redaction policy; or
- raises a documented serialization exception before the event is accepted.

### 8.3 Idempotency

Event IDs are generated once and survive every retry and spool replay. The
server treats duplicate event IDs within a workspace as the same event.

Terminal run operations are also retry-safe. A locally recorded completion or
failure intent can be synchronized after process restart.

## 9. Durable delivery

The durable hybrid delivery model is mandatory for v0.1:

1. Events are appended to a local durable spool before they are acknowledged to
   the caller as captured.
2. A bounded in-process queue feeds background batches.
3. Successful server acknowledgement marks spool records deliverable for
   compaction.
4. Retryable failures use bounded exponential backoff with jitter.
5. Non-retryable failures remain inspectable and require policy or user action.
6. Process restart resumes incomplete batches without generating new event IDs.

The spool should use SQLite or another transactional single-file store. JSONL is
acceptable only as a migration input from the current implementation; it cannot
provide atomic acknowledgement, safe concurrent access, or efficient replay.

Default behavior when the spool reaches its configured limit is to stop
accepting additional telemetry and surface `SpoolFullError`. Dropping evidence
silently is forbidden. An explicit lossy policy may be configured for
high-frequency, non-critical samples.

## 10. Metrics and invariants

Metrics are deterministic, versioned functions over a trace or streaming state.
Every metric result records:

- metric key and version;
- value and unit;
- direction;
- evaluation time;
- implementation digest where available;
- supporting event IDs or trace range;
- threshold result, if evaluated.

v0.1 metric outputs are scalar, boolean, text/category, event count, and bounded
time series.

The constrained invariant language supports:

- `always`;
- `eventually`;
- `until`;
- `at_termination`;
- `within`;
- numeric and boolean comparisons;
- access to an allowlisted trace-state namespace.

No arbitrary Python evaluation is permitted for YAML expressions. Python
invariants are trusted local code and must be declared separately.

Invariant violations emit structured events containing the rule version,
severity, observed value, threshold, first occurrence, and evidence range.

## 11. Artifacts

An artifact record contains:

- logical name and kind;
- media type;
- byte size;
- SHA-256 checksum;
- creation timestamp;
- producing step or event;
- local path or storage URI;
- collection and retention policy;
- upload status.

The SDK never uploads a local artifact merely because its path was logged.
Upload requires the active artifact policy to authorize its kind, size, and
source. Large uploads use server-issued short-lived URLs and resumable or
multipart transfer.

## 12. Provenance and reproduction

The SDK collects, when available and authorized:

- SDK and schema versions;
- Python implementation and version;
- operating system and architecture;
- source commit and dirty-worktree indicator;
- system artifact digest;
- environment/container version;
- scenario digest;
- suite digest;
- random seeds;
- dependency lockfile digest;
- metric and invariant versions;
- perturbation configuration;
- start and finish timestamps.

A reproduction bundle is a manifest, not a copy of customer source or model
weights. It references authorized artifacts and contains the exact scenario,
configuration digests, versions, and command needed to recreate the run.

## 13. Privacy, consent, and Reliability Intelligence

Collection policy and model-training consent are separate concepts.

```python
DataPolicy(
    capture_observations=True,
    capture_actions=True,
    capture_state=False,
    capture_logs=False,
    artifact_allowlist={"video", "plot"},
    redactors=[...],
)

TrainingPolicy(
    training_allowed=False,
    cross_tenant_training_allowed=False,
    allowed_purposes=frozenset(),
)
```

Defaults:

- no training on customer data;
- no cross-tenant training;
- no privileged simulator state collection unless enabled;
- no implicit environment-variable or filesystem capture;
- no credential values in events, artifacts, logs, or exception messages.

Training eligibility travels with the data and cannot be broadened by a
downstream service. Revocation and deletion are control-plane responsibilities,
but the SDK must emit stable policy identifiers so affected data can be found.

To preserve future Reliability Intelligence opportunities, v0.1 traces
distinguish:

- observed facts;
- deterministic calculated findings;
- human annotations;
- AI-generated hypotheses.

Each annotation or hypothesis records its author or model, timestamp,
confidence where applicable, and evidence references. v0.1 does not train or
fine-tune a model.

## 14. CLI contract

Required v0.1 commands:

```text
iso auth login
iso auth status
iso doctor
iso project init
iso system register
iso env validate
iso scenario validate
iso suite plan
iso suite run
iso run inspect
iso run reproduce
iso spool status
iso spool retry
iso spool export
iso schemas export
```

Every inspection and execution command supports `--json`. Machine-readable
output is versioned and written to stdout; diagnostics go to stderr. Exit codes:

- `0`: success;
- `2`: invalid user configuration or input;
- `3`: authentication/authorization failure;
- `4`: control-plane or transport failure;
- `5`: evaluation failure;
- `6`: reliability gate failure;
- `7`: local spool or artifact failure.

This interface is the initial agent-native integration boundary. An MCP surface
may wrap the same application services later; it must not create a parallel
behavioral contract.

## 15. Failure behavior

Public exceptions derive from `IsoObsError` and are divided into:

- configuration and validation errors;
- authentication and authorization errors;
- transport and API errors;
- serialization and policy errors;
- spool and artifact errors;
- adapter and evaluation errors.

An exception raised by customer evaluation code remains the primary exception.
Telemetry, finalization, and failure-reporting errors are attached as notes or
locally recorded diagnostics and must not mask it.

On clean exit, terminal synchronization failure does not rewrite a successfully
executed evaluation as failed. The local state is `completed_pending_sync` until
delivery succeeds.

## 16. Security requirements

- API keys are accepted only explicitly or through documented environment
  variables and are never serialized.
- HTTP uses TLS except for explicitly configured loopback development URLs.
- Redirects do not forward authorization to a different origin.
- Upload URLs are short-lived and scoped to one artifact.
- Local spool and artifact permissions are owner-only where the OS supports it.
- Sensitive-field redaction occurs before durable persistence.
- Deserializing scenario and suite files never executes arbitrary code.
- Checksums are verified before marking an artifact synchronized.

Local spool encryption at rest is recommended but not mandatory for v0.1; the
SDK must document this clearly and support placing the spool on an encrypted
volume.

## 17. Performance targets

- Instrumentation adds less than 1 ms p95 per scalar event before serialization
  of user payloads on a typical developer workstation.
- No network I/O occurs on the simulation thread in background mode.
- Default in-memory buffering is bounded.
- The client sustains at least 1,000 small events per second into the local
  spool.
- Batch size and flush interval are configurable.
- Large artifact hashing and upload run outside the simulation step path.

Benchmarks must report payload size, hardware, Python version, durability mode,
and percentile distribution.

## 18. Compatibility

- API paths are versioned independently from trace schemas.
- Trace envelope breaking changes increment `trace_version`.
- Scenario and suite files carry `schema_version`.
- Additive optional fields remain backward compatible.
- Unknown event types can be stored and forwarded even if the local SDK cannot
  interpret their payload.
- Deprecations warn for at least one minor release before removal prior to 1.0,
  unless a security issue requires immediate removal.

Cross-repository compatibility follows `docs/COORDINATION.md`.

## 19. v0.1 milestones

### M1 — Contract and ergonomic capture

- Align shared schemas with the current Reliability API.
- Add `client.run(...)` and `run.step(...)`.
- Add explicit public `flush()` and stable exceptions.
- Add captured HTTP contract fixtures.

### M2 — Durable delivery

- Replace final-failure JSONL spilling with a transactional spool.
- Preserve event IDs and terminal intents across restart.
- Add retry inspection and CLI spool commands.
- Prove bounded-memory behavior.

### M3 — Scenario and local runner

- Load, validate, canonicalize, and digest scenario YAML/JSON.
- Introduce the structural environment protocol and `StepResult`.
- Execute one deterministic local run.
- Produce a reproduction manifest.

### M4 — Evidence evaluation

- Version metric results and evidence references.
- Evaluate constrained streaming invariants.
- Emit typed violation events.
- Enforce inline payload and artifact policies.

### M5 — Vertical-slice integration

- Ship a Gymnasium reference example.
- Run against a real Reliability test server.
- Survive an induced mid-run control-plane outage.
- Synchronize the completed run after restart.
- Display the trace and derived metric in Reliability.

Suite matrices and perturbation sweeps begin after M5 unless required by the
reference demo.

## 20. v0.1 acceptance criteria

v0.1 is complete when all of the following are demonstrated:

1. A new user instruments the reference environment in fewer than ten
   meaningful lines beyond its existing loop.
2. The same scenario, versions, configuration, and seed produce the same run
   plan and reproduction manifest.
3. A network outage during execution does not stop the environment loop or lose
   accepted telemetry.
4. Killing and restarting the process preserves pending events and terminal
   intent.
5. Duplicate delivery creates no duplicate events.
6. A large observation is handled according to explicit policy rather than
   silently bloating or dropping the trace.
7. A metric and an invariant violation link to supporting evidence.
8. Unauthorized artifact kinds and sensitive fields never enter the durable
   spool.
9. The CLI exposes deterministic JSON output suitable for a coding agent.
10. Reliability ingests and displays the run produced by the public SDK.
11. Ruff, Black, strict mypy, unit tests, package builds, and cross-repository
    contract tests pass.
12. No captured customer data is eligible for model training by default.

## 21. Explicitly deferred

- Model training or fine-tuning;
- autonomous causal or root-cause claims;
- adaptive/Bayesian perturbation search;
- proprietary simulation;
- hosted compute scheduling in the SDK;
- real-time video streaming;
- production hardware control;
- formal verification;
- TypeScript SDK;
- broad simulator-adapter catalog;
- transparent cross-tenant learning.

These are future capabilities, not omissions from the product vision. v0.1
creates the reproducible evidence and safe execution substrate they require.
