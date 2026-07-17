# Warehouse picking reliability evaluation

This example evaluates two versions of a robotic picking policy under nominal
conditions and controlled camera/actuator perturbations. It is deliberately
self-contained: the simulator uses only the Python standard library, while the
evaluation trace is sent through `iso-obs`.

The example demonstrates:

- deterministic scenario execution from a seed;
- candidate-versus-baseline policy behavior;
- observation noise, observation latency, and actuator degradation;
- observations, actions, privileged state, latency, and scalar metrics;
- safety-constraint evidence;
- a local JSON reproduction artifact for every run.

## Run it

Register the two system versions and the environment/scenarios in Reliability,
then run the same matrix for each policy:

```bash
export ISO_OBS_API_KEY="..."

for policy in baseline candidate; do
  for scenario in nominal stressed; do
    uv run python examples/warehouse-picking/warehouse_evaluation.py \
      --project warehouse-arm \
      --system-version "pick-policy-${policy}" \
      --environment warehouse-v4 \
      --scenario "$scenario" \
      --scenario-file \
        "examples/warehouse-picking/scenarios/${scenario}.json" \
      --policy "$policy" \
      --seeds 11 23 42 71 101
  done
done
```

Expected behavior:

- Both policies usually succeed in `nominal`.
- The candidate is faster in nominal conditions.
- Under delayed/noisy observations and actuator degradation, the candidate
  overshoots more often and breaches the force constraint.
- The baseline is slower but degrades more gracefully.

That creates the intended Reliability Studio story: the aggregate nominal score
looks favorable, while the perturbed evaluation reveals a deployment-blocking
regression.

Run artifacts are written under `.iso-obs/examples/warehouse-picking/`. The SDK
records the artifact reference; artifact upload will become automatic when the
v0.1 artifact-policy milestone lands.
