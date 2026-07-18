# Use the Reliability Studio Python SDK

The SDK helps you turn failures into structured engineering evidence. It is
organized around questions—not a single opaque reliability score.

## Install and create a report

```bash
pip install iso-obs
```

Most evidence workflows follow the same conceptual shape (the linked
examples use the concrete plan and analysis types):

```python
plan = StudyPlan(...)       # declare the question and decision rule
evidence = (...)            # attach observations and provenance
report = analyze(plan, evidence)

print(report.disposition)
print(report.to_json())
print(report.content_digest())
```

The exact plan, evidence, and analysis types depend on the question. Start
from a complete [executable example](../examples/README.md), then substitute
your system's thresholds, strata, and evidence.

If you are integrating an existing simulator or evaluation harness, use the
[warehouse autonomy workflow](integration-workflow.md). It walks from control
loop instrumentation through typed evidence, report generation, CI gating,
platform submission, and regression promotion.

## Navigate by engineering question

| You need to know… | SDK capability | Result |
|---|---|---|
| Where two executions first differ | divergence analysis | Earliest supported divergence with aligned trace evidence |
| Whether a change caused harm | causal perturbations and counterfactuals | Effect intervals under a declared causal design |
| Where behavior becomes unreliable | failure boundaries and surfaces | Certified reliable, unresolved, and unreliable regions |
| Which failures recur together | failure phenotypes | Auditable groups with membership evidence |
| Whether dataset labels mix operating modes | synchronization, mixed-mode detection, and dataset reliability | Candidate transitions, label lineage, and audit findings |
| Whether simulation transfers to deployment | sim-to-real and transportability | Stratum-specific support with limiting conditions |
| Whether a failure is reproducible | replay capsules and replication | Content-addressed evidence and replay outcomes |
| Whether a fix is ready to ship | regression and promotion | Preregistered gates backed by confirmatory evidence |

## Interpret a result correctly

A report is more than pass or fail. Read:

- **disposition** — the strongest conclusion currently justified;
- **interval or set** — uncertainty that must accompany the point estimate;
- **scope** — versions, populations, environments, and conditions covered;
- **limitations** — known restrictions on interpretation;
- **content digest** — the stable identity of the complete report.

An unresolved or unsupported result is useful. It identifies the next
experiment or evidence gap without overstating what is known.

## Submit evidence to Reliability Studio

Save any SDK report's JSON, then submit it with the CLI:

```python
from pathlib import Path

Path("report.json").write_text(report.to_json(), encoding="utf-8")
```

```bash
iso evidence submit report.json --project-id YOUR_PROJECT_ID
```

The CLI reads `ISO_OBS_API_KEY`, uses the report schema version and content
digest, and submits idempotently. Re-submitting identical content refers to
the same evidence report.

## Build a defensible workflow

Read [Scientific method and claim discipline](scientific-method.md) before
using a report for release or safety decisions. It explains why the SDK
preserves uncertainty, separates discovery from confirmation, and prevents a
strong aggregate result from compensating for a failed critical condition.
