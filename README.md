<div align="center">

# ∂ &nbsp;iso-obs

**The official SDK for Reliability Studio** — evaluation and failure analysis for
physical AI and autonomous systems, by [Iso AI](https://iso-obs.com).

```bash
pip install iso-obs
```

[Documentation](https://iso-obs.com/docs) · [Quickstart](#quickstart) · [Report an issue](https://github.com/iso-ai/iso-obs/issues/new/choose)

</div>

---

Reliability Studio answers one question about an autonomous system:

> What changed, under which conditions, why did behavior change, and can this system still be trusted?

This repository contains everything you need to connect your system to it:

| Package | Import | What it is |
|---|---|---|
| `iso-obs` | `iso_obs` | Python SDK — clients, run contexts, metrics, environment adapters |
| `iso-obs-cli` | `iso_obs_cli` | The `iso` command-line interface |
| `iso-obs-schemas` | `iso_obs_schemas` | The shared trace/event/config contract (Pydantic v2 + JSON Schema) |

All code here is licensed under [Apache-2.0](./LICENSE). The Reliability Studio
platform itself (web app, API, execution infrastructure) is a hosted product — your
models and simulator stay in your environment; the SDK sends only the telemetry and
artifacts you authorize.

The implementation target and cross-repository ownership rules are documented
in [the SDK v0.1 specification](./docs/SDK_SPEC_V0.1.md) and
[the coordination contract](./docs/COORDINATION.md).

## Examples

- [Warehouse picking](./examples/warehouse-picking/README.md) — a self-contained
  physical-AI evaluation with observation delay/noise, actuator degradation,
  safety constraints, and baseline/candidate policies.
- [Gymnasium CartPole](./examples/gymnasium-cartpole/README.md) — instrumentation
  of an existing control loop with sensor noise and delayed actions.

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

Bring your own environment — Isaac Sim, MuJoCo, Gazebo, CARLA, PyBullet, Gymnasium,
log replay, or anything that can implement a six-method adapter:

```python
from iso_obs.environments import ReliabilityEnvironment

class WarehouseEnvironment(ReliabilityEnvironment):
    def reset(self, scenario, seed): ...
    def observe(self): ...
    def step(self, action): ...
    def get_state(self): ...
    def collect_artifacts(self): ...
    def close(self): ...
```

## CLI

```bash
iso auth login
iso project init
iso system register --project robot-arm --name pick-policy --version 2026.07.16
iso run inspect run_01J...
```

## Getting help & reporting problems

- **Bug in the SDK or CLI?** [Open a bug report](https://github.com/iso-ai/iso-obs/issues/new/choose)
- **Usage question?** Use the [SDK question template](https://github.com/iso-ai/iso-obs/issues/new/choose) or [Discussions](https://github.com/iso-ai/iso-obs/discussions)
- **Security issue?** Follow [SECURITY.md](./SECURITY.md) — never a public issue
- **Docs gap?** There's a template for that too

Every issue is triaged. We treat friction in this SDK as a product bug.

## Contributing

We welcome contributions — see [CONTRIBUTING.md](./CONTRIBUTING.md) for setup, coding
standards (PEP 8 / Black / Google docstrings / mypy strict), and the PR process.

## License

[Apache-2.0](./LICENSE) © 2026 Iso AI
