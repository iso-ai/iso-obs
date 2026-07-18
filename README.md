<div align="center">

# ∂ &nbsp;iso-obs

**The Python SDK and CLI for Reliability Studio**

Capture failures, reproduce their conditions, test causal mechanisms, map
operating boundaries, validate sim-to-real transfer, and turn resolved
failures into regression evidence.

```bash
pip install iso-obs
```

[SDK guide](./packages/python-sdk/docs/index.md) ·
[Integration walkthrough](./packages/python-sdk/docs/integration-workflow.md) ·
[Examples](./packages/python-sdk/examples/README.md) ·
[Report an issue](https://github.com/iso-ai/iso-obs/issues/new/choose)

</div>

---

Reliability Studio is the failure intelligence layer for autonomous and
AI-driven systems. It helps engineering teams answer:

- Where does this system fail?
- Under which conditions does it fail?
- Which change caused the behavior?
- Does simulator evidence transfer to reality?
- Did the next version actually fix the problem?

This public repository contains the Apache-2.0 client surface:

| Package | Import or command | Purpose |
|---|---|---|
| `iso-obs` | `iso_obs` | Capture, evidence analysis, replay, regression, and release-assurance SDK |
| `iso-obs-cli` | `iso` | Authentication, project operations, and evidence submission |
| `iso-obs-schemas` | `iso_obs_schemas` | Versioned public trace and API contracts |

The hosted API and Reliability Studio application are separate. Your model
and simulator remain in your environment; you choose which evidence and
artifacts to submit.

## Start with a complete workflow

The
[warehouse release walkthrough](./packages/python-sdk/docs/integration-workflow.md)
shows how to connect a simulator or evaluation harness, translate domain
outcomes into reliability evidence, run three complementary analyses, emit
content-addressed reports, and use the result in CI.

```bash
git clone https://github.com/iso-ai/iso-obs.git
cd iso-obs
uv sync --package iso-obs
uv run python \
  packages/python-sdk/examples/warehouse_release_workflow.py \
  --output-dir evidence
```

The included evidence intentionally blocks release. That is the useful
result: observation latency has a supported harmful effect, the visibility
map contains unresolved and certified-unreliable regions, and a rare
low-friction condition fails sim-to-real validation even though the aggregate
discrepancy looks acceptable.

## Instrument an existing control loop

```python
from iso_obs import ReliabilityClient

client = ReliabilityClient()  # reads ISO_OBS_API_KEY

with client.run(
    project="warehouse-amr",
    system_version="navigation-policy-v18",
    environment="warehouse-twin-v4.2",
    scenario="blind-intersection",
    seed=42,
) as run:
    observation = env.reset(seed=run.seed)
    while True:
        action = policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        run.step(
            observation=observation,
            action=action,
            reward=reward,
            state=info.get("state"),
            metrics={
                "minimum_stopping_margin_m": info["minimum_stopping_margin_m"],
            },
        )
        observation = next_observation
        if terminated or truncated:
            break
    run.log_artifact("replay.mp4")
```

The same pattern works with Isaac Sim, MuJoCo, Gazebo, CARLA, PyBullet,
Gymnasium, world-model rollouts, recorded logs, and internal evaluation
harnesses.

## Submit a report

Every scientific report supports canonical JSON and a stable content digest:

```python
from pathlib import Path

Path("report.json").write_text(report.to_json(), encoding="utf-8")
print(report.content_digest())
```

```bash
iso evidence submit report.json --project-id YOUR_PROJECT_ID
```

Submissions are idempotent on report content.

## What makes the evidence rigorous?

The SDK preserves:

- preregistered plans and material-effect thresholds;
- uncertainty intervals, prediction sets, and unresolved regions;
- separate discovery and confirmatory evidence roles;
- simultaneous error control across families of claims;
- noncompensatory decisions for rare critical conditions;
- model, simulator, dataset, environment, and artifact provenance;
- explicit limitations and operating envelopes.

Read [Scientific method and claim discipline](./packages/python-sdk/docs/scientific-method.md)
before using evidence for a release or safety decision.

## Getting help

- [SDK bug or documentation gap](https://github.com/iso-ai/iso-obs/issues/new/choose)
- [Usage question](https://github.com/iso-ai/iso-obs/discussions)
- Security issue: follow [SECURITY.md](./SECURITY.md), never a public issue

## License

Apache-2.0 © 2026 Iso AI
