# Reliability Studio SDK examples

These examples show how Reliability Studio fits into an engineering workflow.
They are not benchmark toys or disconnected calls: each one declares a
question, adapts domain evidence, computes a bounded conclusion, and preserves
the report for replay, comparison, CI, or platform submission.

## Start here: one candidate, one release decision

The best entry point is the
[`warehouse_release_workflow.py`](warehouse_release_workflow.py) walkthrough.
It combines three complementary studies for an autonomous mobile robot:

```text
matched simulator replays ──> causal perturbation report ──┐
fixed operating anchors ────> failure boundary report ─────┼─> CI decision
simulator + physical pairs ─> transportability report ─────┘
```

Run it from the repository root:

```bash
uv sync --package iso-obs
uv run python \
  packages/python-sdk/examples/warehouse_release_workflow.py \
  --output-dir evidence
```

Then follow the
[step-by-step integration guide](../docs/integration-workflow.md) to replace
the deterministic fixture builders with data from your simulator, logs,
world-model rollouts, or physical tests.

The included candidate intentionally blocks release. A useful reliability
system must expose unsupported claims and next experiments—not manufacture a
passing demo.

## Choose the engineering question

| Question | Executable study | What enters | What comes out |
|---|---|---|---|
| Did a controlled change cause harm? | [`causal_perturbation_campaign.py`](causal_perturbation_campaign.py) | Matched baseline and intervention outcomes | Simultaneous effect intervals and a noncompensatory campaign disposition |
| Where is behavior known to be safe or unsafe? | [`failure_boundary_map.py`](failure_boundary_map.py) | Fixed trials at predeclared operating anchors | Certified reliable, unresolved, and unreliable regions |
| Does simulator evidence transfer to reality? | [`sim_to_real_transport.py`](sim_to_real_transport.py) | Matched simulation and physical anchors by critical stratum | Local and overall discrepancy bounds plus limiting strata |
| Can a learned detector help curate mixed-mode data? | [`mixed_mode_dataset_curation.py`](mixed_mode_dataset_curation.py) | Synchronized modalities, detector lineage, prediction sets, and human evidence | Reviewable transition candidates and an auditable label ledger |
| Can a confirmed failure be reproduced? | [`failure_replay_capsule.py`](failure_replay_capsule.py) | Dataset, synchronization, simulator, recording, and claim evidence | A content-addressed replay capsule with explicit fidelity |
| Can robotics recordings retain source lineage? | [`mcap_ros2_ingestion.py`](mcap_ros2_ingestion.py) and [`rosbag2_split_ingestion.py`](rosbag2_split_ingestion.py) | MCAP or split rosbag2 recordings | Dataset manifests and recording evidence without silent repair |
| Can customer-hosted GPU training retain auditable lineage? | [`modal_gpu_world_model_workflow.py`](modal_gpu_world_model_workflow.py) | A generated trajectory manifest and a single customer-owned Modal L4 | A checkpoint, compute observations, sliced rollout evaluation, and versioned training evidence |

The Modal example keeps provider and Studio credentials on opposite sides of
the integration boundary. After configuring a short-lived Modal profile, run:

```bash
MODAL_PROFILE=iso-gpu-e2e \
  uv run --with modal modal run \
  packages/python-sdk/examples/modal_gpu_world_model_workflow.py \
  --output-dir evidence/modal-gpu
```

The function is capped at one L4, one container, no function retries, and a
15-minute timeout. Image hydration may still be attempted more than once by
the provider, so monitor and abort repeated startup failures instead of
assuming `retries=0` covers container initialization.

## The integration seam

You do not need to replace your simulator. Replace each example's
`build_observations` function with a narrow adapter over records you already
produce:

```python
def build_observations(plan):
    records = trial_store.load(study_id=plan.plan_id)
    return tuple(
        PairedPerturbationObservation(
            plan_content_digest=plan.content_digest(),
            contrast_id=record.intervention_id,
            pair_id=record.pair_id,
            matched_context_digest=record.initial_state_digest,
            baseline_evidence_digest=record.baseline_digest,
            perturbed_evidence_digest=record.perturbed_digest,
            baseline_failed=record.baseline.minimum_margin_m < 0.0,
            perturbed_failed=record.perturbed.minimum_margin_m < 0.0,
        )
        for record in records
    )
```

The important requirement is not the storage technology. It is preserving the
identity and assumptions behind the comparison. A pair must really be matched;
an anchor must really be predeclared; a physical observation must really
correspond to its simulator observation.

## A report is an evidence artifact

Every scientific report can be serialized and content-addressed:

```python
from pathlib import Path

Path("report.json").write_text(report.to_json(), encoding="utf-8")
print(report.content_digest())
```

Use the digest as the identity in experiment tracking, artifact storage, code
review, and release records. Submit the same JSON to Reliability Studio:

```bash
iso evidence submit report.json --project-id YOUR_PROJECT_ID
```

## How to read a result

Every rigorous workflow separates:

1. **Plan** — the question, estimand, thresholds, sample design, and decision
   rule chosen before inspecting results.
2. **Evidence** — observations plus matching, timing, version, and provenance
   identities.
3. **Uncertainty object** — an interval, prediction set, identified set, or
   unresolved map region.
4. **Disposition** — the strongest conclusion currently supported, including
   `review_required`, `partial`, and `not_supported`.
5. **Claim boundary** — populations, environments, versions, and operating
   conditions outside which the result must not be used.

Do not branch on a disposition alone. Inspect the uncertainty, limiting
strata, evidence scope, assumptions, and limitations that justify it.

## Discovery is not confirmation

The mixed-mode example intentionally keeps neural detector output in
`suggested` status. Learned representations can propose transitions and
failure labels at scale, but model output becomes trusted evidence only after
the declared review or adjudication process.

Use exploratory datasets to discover failure hypotheses. Use independently
designated confirmatory evidence to certify fixes and gate releases.

## From failure to regression

A production loop usually looks like this:

```text
capture failure
  -> reproduce exact conditions
  -> test causal perturbations
  -> map neighboring failure boundary
  -> minimize the counterexample
  -> build canonical + neighborhood + control cases
  -> execute independent regression evidence
  -> gate the next release
```

Reliability Studio keeps the evidence chain intact across that loop. A
reproducible failure is not automatically a causal explanation, and a
regression candidate is not automatically a passed release gate.

## Before adapting an example

- Define the operational failure from measurable signals.
- Choose material thresholds from engineering or safety requirements.
- Preserve stable system, simulator, environment, dataset, and run identities.
- Keep critical strata noncompensatory.
- Retain unresolved regions for targeted evidence collection.
- Record learned predictions separately from adjudicated labels.
- Store the report JSON and its digest with the build that produced it.

The bundled evidence is deterministic so examples run locally and in CI. Its
conclusions are illustrative; the analysis rules and report contracts are the
production SDK behavior.
