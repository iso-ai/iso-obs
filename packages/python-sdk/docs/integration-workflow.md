# Integrate Reliability Studio into an autonomy workflow

This walkthrough follows one candidate navigation policy from simulator runs
to a release decision. It is designed to show where the SDK belongs in an
existing engineering system—not require you to replace that system.

The scenario is an autonomous mobile robot approaching a partially occluded
warehouse intersection. The release team needs to answer three different
questions:

1. Does added observation latency cause failures?
2. Where does visibility degradation become unreliable?
3. Does braking behavior in the simulator match physical anchors?

Those questions require separate evidence designs. The workflow keeps their
claims separate, then combines their dispositions with a noncompensatory
release policy.

## Where the SDK fits

```text
simulator / logs / world model / real tests
                    |
                    v
        domain-to-evidence adapters
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
      causal     boundary    sim-to-real
       report      report       report
        +-----------+-----------+
                    |
                    v
        release decision + reports
                    |
             +------+------+
             |             |
             v             v
        CI release gate  Reliability Studio
```

Reliability Studio does not dictate your simulator, policy framework, or data
store. You translate the outputs you already have into typed evidence records.

## 1. Install the SDK

From PyPI:

```bash
pip install iso-obs
```

From this repository:

```bash
uv sync --package iso-obs
```

## 2. Instrument the evaluation loop

Use `ReliabilityClient.run` around an existing control loop. Record the state
and metrics that define a failure; avoid sending raw data that your study does
not need.

```python
from iso_obs import ReliabilityClient

client = ReliabilityClient()

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
```

For ROS 2 recordings, use the MCAP or rosbag2 ingestion examples instead of
rewriting the control loop.

## 3. Add narrow domain adapters

The functions named `build_observations` in the example studies are the seams
you replace:

| Your existing output | SDK evidence record | Stable matching identity |
|---|---|---|
| Baseline and perturbed replay result | `PairedPerturbationObservation` | Same initial state, seed, and random stream |
| Fixed trial at an operating anchor | `BoundaryObservation` | Declared coordinate and sample index |
| Matched simulator and physical measurement | `PairedAnchorObservation` | Same commanded state and anchor ID |

Keep conversion code narrow. Do not infer missing pair identities, silently
sort timestamps, or mix exploratory labels into confirmatory evidence.

The supplied fixture builders create deterministic evidence so the full
example runs without a simulator. In your project, load your trial records and
construct the same SDK types.

## 4. Declare plans before inspecting results

Each study has a plan that fixes:

- the target population and system artifact;
- the failure or outcome definition;
- intervention or pairing rules;
- material engineering thresholds;
- sample counts and error allocation;
- limitations and the permitted evidence use.

Start from:

- [`causal_perturbation_campaign.py`](../examples/causal_perturbation_campaign.py)
- [`failure_boundary_map.py`](../examples/failure_boundary_map.py)
- [`sim_to_real_transport.py`](../examples/sim_to_real_transport.py)

Replace the domain values. Do not tune thresholds after seeing the candidate
system's results.

## 5. Run the integrated release workflow

```bash
uv run python \
  packages/python-sdk/examples/warehouse_release_workflow.py \
  --output-dir evidence
```

The example produces:

```text
evidence/
├── causal-perturbation-report.json
├── failure-boundary-report.json
├── transportability-report.json
└── release-decision.json
```

Each scientific report has its own schema version, limitations, and content
digest. The release decision references conclusions; it does not collapse the
reports into a scientifically meaningless universal score.

## 6. Understand the blocked result

The included candidate is intentionally not ready:

- 120 ms observation latency has a supported material harmful effect;
- the visibility map contains an unresolved boundary;
- the declared visibility envelope also contains a certified unreliable
  region;
- a rare low-friction stratum exceeds its sim-to-real discrepancy limit.

The overall sim-to-real discrepancy looks acceptable because the common floor
stratum represents 98% of exposure. The local failure still blocks release.
This prevents a frequent condition from compensating for a rare critical one.

The next experiments are explicit:

1. reduce latency sensitivity and rerun matched replays;
2. collect fixed-design samples inside the unresolved visibility band;
3. resolve the blackout failure or exclude it from the operating envelope;
4. improve low-friction physics or narrow the supported operating envelope.

## 7. Use the result in CI

After connecting the builders to your evidence, make readiness required:

```bash
uv run python \
  packages/python-sdk/examples/warehouse_release_workflow.py \
  --output-dir "evidence-${GITHUB_SHA}" \
  --require-ready
```

The command exits with status `2` when any critical requirement blocks
release. Upload the evidence directory as a CI artifact even when the gate
fails; it explains the decision and makes the exact reports reproducible.

## 8. Submit reports to the platform

```bash
iso evidence submit evidence/causal-perturbation-report.json \
  --project-id YOUR_PROJECT_ID
iso evidence submit evidence/failure-boundary-report.json \
  --project-id YOUR_PROJECT_ID
iso evidence submit evidence/transportability-report.json \
  --project-id YOUR_PROJECT_ID
```

The CLI uses `ISO_OBS_API_KEY`. Submission is idempotent on report content, so
retries do not create duplicate evidence.

## 9. Turn confirmed failures into regression assets

Once a failure is independently supported:

1. compile its environment, input, timing, and artifact identities into a
   replay capsule;
2. reproduce the failure under the declared acceptance criteria;
3. minimize the failure conditions without losing the outcome;
4. build a regression pack containing the canonical case, neighborhood,
   boundary probes, and positive and negative controls;
5. use independent confirmatory executions for release gating.

See [`failure_replay_capsule.py`](../examples/failure_replay_capsule.py) and
the regression and promotion sections of the [SDK guide](index.md).

## Adaptation checklist

Before using this workflow for a real decision, verify:

- [ ] System, model, simulator, and environment versions are immutable.
- [ ] Failure definitions and thresholds were chosen before analysis.
- [ ] Matched comparisons truly share their declared context.
- [ ] Critical operating strata cannot compensate for one another.
- [ ] Real anchors cover the operating envelope you claim.
- [ ] Learned labels retain prediction and calibration provenance.
- [ ] Discovery and confirmation datasets are separated.
- [ ] Reports, replay artifacts, and CI outputs are retained by digest.
