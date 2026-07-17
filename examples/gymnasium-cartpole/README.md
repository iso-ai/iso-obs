# Gymnasium CartPole robustness evaluation

This example shows how to instrument an existing Gymnasium loop without
replacing the simulator. It compares two small policies while applying
observation noise and action delay outside Gymnasium.

Install the optional example dependency:

```bash
uv pip install "gymnasium[classic-control]>=1.0"
```

Run a candidate across multiple deterministic seeds:

```bash
export ISO_OBS_API_KEY="..."

uv run python examples/gymnasium-cartpole/cartpole_evaluation.py \
  --project control-benchmarks \
  --system-version cartpole-candidate \
  --environment gymnasium-cartpole-v1 \
  --scenario noisy-delayed-balance \
  --policy candidate \
  --noise-std 0.04 \
  --action-delay 2 \
  --seeds 1 2 3 5 8 13 21
```

The script records every observation, emitted action, reward, pole-angle safety
margin, and final episode result. Running the same command with
`--policy baseline` creates a directly comparable set of traces.

This is an integration example, not a claim that CartPole represents a
production physical-AI workload. Its purpose is to make the control loop and
failure boundary easy to inspect before connecting a proprietary simulator.
