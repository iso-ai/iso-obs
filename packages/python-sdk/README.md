# iso-obs — test whether autonomous systems are ready to deploy

Use the same reliability workflow for simulator evaluations, learned world
models, robot policies, controllers, and autonomy tests. `iso-obs` finds where
apparently successful behavior stops holding, preserves the evidence, and turns
reproduced failures into regression targets before deployment. Every run keeps
its seed, versions, observations, actions, metrics, and artifacts so results are
inspectable rather than reduced to a single success-rate claim.

## Installation

```bash
pip install iso-obs
```

Requires Python 3.12+.

## Two-minute local report

Generate a versioned failure-boundary report without an account or API key:

```bash
pip install iso-obs
python -m iso_obs.quickstart
```

The command writes `failure-boundary-report.json` and prints its disposition,
schema version, and content digest. The included trials are deterministic demo
evidence for learning the workflow; replace them with measurements from your
robot or simulator before making a production reliability claim.

## Publish the report and activate Studio

The local report answers a concrete question: **where does behavior stop being
reliable under a controlled perturbation?** Publish that exact artifact to keep
it reviewable and turn confirmed failures into regression evidence:

1. [Create a Reliability Studio API key](https://iso-obs.studio/settings).
2. Validate and store it: `iso auth login --api-key <YOUR_KEY>`.
3. Submit the report: `iso evidence submit failure-boundary-report.json`.
4. Open [Evidence in Studio](https://iso-obs.studio/evidence).

`iso auth login` verifies the key before saving it. An invalid or unreachable
credential is not written to disk.

## Learn by example

Start with the [SDK guide](https://iso-obs.com/docs/sdk) for a question-driven
tour of the package, or browse the
[executable examples](https://github.com/iso-ai/iso-obs/tree/main/packages/python-sdk/examples)
to see complete reliability studies:

- isolate a causal failure mechanism with matched perturbations;
- map certified reliable, unresolved, and unreliable operating regions;
- prevent aggregate metrics from hiding a rare sim-to-real failure;
- curate mixed-mode datasets without treating neural predictions as truth;
- preserve failures as replay capsules and promote them into regression tests.

The
[scientific-method guide](https://github.com/iso-ai/iso-obs/blob/main/packages/python-sdk/docs/scientific-method.md)
explains the abstention, uncertainty, multiplicity, and evidence-scope rules
behind these workflows.

For a complete simulator-to-CI journey, follow the
[warehouse autonomy integration walkthrough](https://github.com/iso-ai/iso-obs/blob/main/packages/python-sdk/docs/integration-workflow.md).

## Quickstart

```python
from iso_obs import ReliabilityClient

client = ReliabilityClient()  # reads ISO_OBS_API_KEY from the environment

with client.run(
    project="robot-arm",
    system_version="policy-v17",
    environment="warehouse-v4",
    scenario="obstructed-pick",
    seed=42,
) as run:
    obs = env.reset(seed=run.seed)
    while True:
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        run.step(
            observation=obs,
            action=action,
            reward=reward,
            state=info.get("state"),
        )
        obs = next_obs
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
| API root | `base_url` | `ISO_OBS_BASE_URL` | `https://reliability-studio-5cmy6.ondigitalocean.app/api/v1` |
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
client.run(project=..., system_version=..., environment=...,
           scenario=..., seed=...)  # -> RunContext
```

All returned objects are the Pydantic models from
[`iso-obs-schemas`](https://pypi.org/project/iso-obs-schemas/), the shared
contract package.

## Reliability semantics

- **Retries** — every request is attempted up to 3 times on `429`, `5xx`, and
  transport errors, with exponential backoff.
- **Typed errors** — failures raise `ApiError` subclasses from
  `iso_obs.exceptions`: `AuthenticationError`, `NotFoundError`,
  `RateLimitError`, `ServerError`.
- **Event batching** — `RunContext` buffers events and flushes in batches of
  500, plus a final flush on exit. Call `run.flush()` to synchronously deliver
  buffered evidence without completing the run.
- **Ordered steps** — `run.step(...)` records one observation/action
  interaction with optional state, reward, custom metrics, and action latency.
  Every emitted event shares one step index in deterministic order.
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

## Behavioral divergence

Compare paired baseline and candidate traces with explicit, signal-specific
tolerances and persistence requirements:

```python
from iso_obs.divergence import (
    PairingAssessment,
    SignalSpec,
    TraceStep,
    compare_traces,
)

pairing = PairingAssessment.assess(
    same_scenario_version=True,
    same_environment_version=True,
    same_seed=True,
    same_perturbation_realization=True,
)
report = compare_traces(
    baseline_steps,
    candidate_steps,
    signals=[
        SignalSpec(
            path="gripper_force_n",
            category="control",
            absolute_tolerance=0.5,
            relative_tolerance=0.05,
            persistence_steps=3,
        )
    ],
    pairing=pairing,
)
```

The report distinguishes the first numerical difference from the first
sustained meaningful divergence, retains supporting event IDs, records
unmatched or missing evidence, and always states that observed divergence alone
does not establish causality. The initial analyzer uses exact environment-step
alignment; more permissive temporal alignment will expose its own quality and
warping diagnostics rather than silently forcing traces to match.

Build analysis-ready steps from normalized run events with explicit extraction
rules:

```python
from iso_obs.trace import EventSignalSpec, extract_trace_steps
from iso_obs_schemas import EventType

extraction = extract_trace_steps(
    events,
    signals=[
        EventSignalSpec(
            signal="gripper_force_n",
            event_type=EventType.METRIC_RECORDED,
            value_path="value",
            where={"name": "gripper_force_n"},
        )
    ],
)
```

Extraction rejects mixed runs, duplicate event IDs, non-numeric values, missing
payload paths, and ambiguous within-step multiplicity. Its diagnostics preserve
empty steps and count unmatched or unstepped evidence so missing data cannot be
mistaken for zero.

For a pre-specified scalar outcome measured across matched seeds, quantify the
candidate-minus-baseline effect and its uncertainty:

```python
from iso_obs.divergence import PairingQuality
from iso_obs.replication import PairedObservation, analyze_paired_effect

report = analyze_paired_effect(
    [
        PairedObservation(
            pair_id=f"seed-{seed}",
            baseline=baseline_results[seed],
            candidate=candidate_results[seed],
            pairing_quality=PairingQuality.EXACT_REPLAY,
        )
        for seed in matched_seeds
    ]
)
```

The report includes the paired mean and median difference, sample standard
deviation, standardized mean difference when defined, a deterministic
percentile-bootstrap confidence interval, and a two-sided paired sign-flip
randomization p-value. Exact randomization is used for small samples; larger
samples use seeded Monte Carlo with a plus-one correction. Weakly paired runs
are rejected, pair ordering cannot change the result, and the report explicitly
states its causal and multiple-comparison limitations.

## Failure and simulation evidence

Package a failure as a deterministic, portable artifact while keeping the
credibility of the simulated world explicit:

```python
from iso_obs.evidence import (
    EvidenceLevel,
    FailureEvidenceBundle,
    SimulationEvidenceManifest,
    sha256_digest,
)

simulation = SimulationEvidenceManifest(
    simulator_name="warehouse-twin",
    simulator_version="4.2.0",
    model_type="hybrid",
    artifact_digest=sha256_digest("warehouse-twin:4.2.0"),
    intended_use="Low-speed indoor mobile-robot navigation",
    validity_envelope=(...),
    uncertainty_sources=(...),
    verification_evidence=(...),
    calibration_evidence=(...),
    validation_evidence=(...),
    known_omissions=("Tire wear is not modeled.",),
)

failure = FailureEvidenceBundle(
    failure_category="unsafe_stop_margin",
    scenario_family="blind_intersection",
    scenario_version="3",
    perturbation_family="observation_latency",
    invariant_id="minimum-stopping-distance",
    invariant_version="2",
    signal_group=("stopping_margin_m", "velocity_mps"),
    execution_phase="closed_loop_control",
    component="navigation_stack",
    evidence_level=EvidenceLevel.SUSTAINED_DIVERGENCE,
    summary="Stopping margin diverged before the monitor activated.",
    affected_system_versions=("policy-v17",),
    evidence_event_ids=("evt-101", "evt-102"),
    reproduction=reproduction_manifest,
    simulation=simulation,
    analyses=(divergence_provenance,),
    limitations=("Observed divergence does not establish causality.",),
)
```

`failure.content_digest()` identifies the exact serialized evidence.
`failure.failure_fingerprint()` groups observations by stable mechanism factors
and deliberately excludes summaries, event IDs, seeds, and affected versions.
Stronger evidence labels require matching analysis provenance, so replicated
or intervention-supported claims cannot be created from a trace comparison
alone. Simulator verification, calibration, validation, uncertainty sources,
real-world anchors, validity ranges, and known omissions remain separate rather
than being collapsed into a misleading credibility score.

## Failure-derived regression packs

Turn a failure bundle into a simulator-neutral regression definition that tests
the failure neighborhood instead of memorizing one trajectory:

```python
from iso_obs.evidence import sha256_digest
from iso_obs.regression import (
    GateCriterion,
    GateMethod,
    RegressionPack,
    SeedPanel,
    SeedStrategy,
    evaluate_regression_pack,
)

pack = RegressionPack(
    pack_id="blind-intersection-stop-margin",
    pack_version="1",
    source_failure_content_digest=failure.content_digest(),
    source_failure_fingerprint=failure.failure_fingerprint(),
    simulation_manifest_digest=simulation.content_digest(),
    cases=(
        canonical_reproduction,
        neighborhood_probe,
        boundary_probe,
        negative_control,
        positive_control,
    ),
    seed_panel=SeedPanel(
        seeds=tuple(range(100)),
        strategy=SeedStrategy.COMMON_RANDOM_NUMBERS,
        random_stream_digest=sha256_digest("matched-random-streams"),
    ),
    gate_criteria=(
        GateCriterion(
            name="failure-neighborhood",
            case_ids=case_ids,
            method=GateMethod.WILSON_LOWER_BOUND,
            minimum_expected_outcome_rate=0.95,
            minimum_trials=500,
            confidence_level=0.95,
        ),
    ),
    limitations=("Validated only for low-speed indoor navigation.",),
)

report = evaluate_regression_pack(pack, completed_observations)
assert report.passed
```

Every pack requires exactly one canonical reproduction plus neighborhood,
boundary, negative-control, and positive-control cases. Numeric invariant
oracles are versioned, simulator assets are content-addressed, and stochastic
tests declare exact replay, common-random-number, or independent seed strategy.
Release gates can require exact conformance or a one-sided Wilson lower
confidence bound. A perfect result from too few trials therefore remains
insufficient evidence. Evaluation requires exactly one observation for every
declared case and seed and states its simulation-bias, scope, and regression
overfitting limitations.

## Simulation execution planning

Compile a regression pack for a physics simulator, learned world model, hybrid
twin, log replay system, or hardware-in-the-loop adapter:

```python
from iso_obs.simulation import (
    BackendType,
    EvidenceUse,
    ExecutionIntent,
    SimulationAdapterManifest,
    SimulationCapability,
    compile_simulation_plan,
)

adapter = SimulationAdapterManifest(
    adapter_name="warehouse-isaac",
    adapter_version="1.2.0",
    backend_type=BackendType.PHYSICS_SIMULATOR,
    simulator_name="Isaac Sim",
    simulator_version="5",
    simulation_evidence_manifest_digest=simulation.content_digest(),
    capabilities=(
        SimulationCapability.SEEDED_RESET,
        SimulationCapability.PARAMETER_OVERRIDE,
        SimulationCapability.ARTIFACT_LOADING,
        SimulationCapability.RANDOM_STREAM_CONTROL,
        SimulationCapability.INVARIANT_SIGNAL_EXPORT,
    ),
    supported_parameters=("floor_friction", "observation_latency_ms"),
    supported_artifacts=("scenario",),
    max_parallelism=8,
    limitations=("Contact severity is not validated.",),
)

plan = compile_simulation_plan(
    pack,
    adapter,
    intent=ExecutionIntent.REPRODUCTION,
    evidence_use=EvidenceUse.CONFIRMATORY,
)
```

Compilation performs a preflight check for missing capabilities, unsupported
parameters and artifacts, and simulation-evidence mismatches before compute is
spent. It then expands the complete case-by-seed matrix into content-addressed
work items and deterministic parallel batches. Reproduction requires the
simulation evidence linked by the pack; cross-backend validation records the
explicit difference.

Learned-world-model screening requires uncertainty quantification and
out-of-distribution detection. Surrogate screening is always discovery-only
and is marked ineligible for release-gate evidence, preventing a world model
from certifying its own generated failures without independent confirmation.

## Execution-result evaluation

Adapters return an identity-bound result for every planned work item:

```python
from iso_obs.execution import (
    SimulationRunResult,
    SimulationRunStatus,
    evaluate_simulation_campaign,
    simulation_work_item_digest,
)

result = SimulationRunResult(
    work_item_id=item.work_item_id,
    work_item_digest=simulation_work_item_digest(item),
    run_id=run.id,
    status=SimulationRunStatus.COMPLETED,
    trace=analysis_ready_trace,
    model_diagnostics=world_model_diagnostics,
    artifact_digests=(replay_digest,),
)

report = evaluate_simulation_campaign(
    plan,
    pack,
    tuple(completed_results),
)
```

Invariant oracles are evaluated over contiguous trace steps with their declared
persistence. Simulator failures, timeouts, missing signals, missing results,
duplicate results, and work-item identity mismatches make the experiment
incomplete rather than counting as system passes or failures.

Backends that declare learned-model uncertainty and OOD capabilities must
return calibrated rollout diagnostics. Out-of-domain results remain available
for failure discovery but block release evidence. A regression gate is
calculated only when the entire plan is complete, identity-verified, in-domain,
and marked for confirmatory use.

## Multiplicity and adaptive failure search

Control false discoveries across large signal and scenario searches while
keeping exploratory and confirmatory evidence separate:

```python
from iso_obs.multiplicity import (
    AnalysisMode,
    MultiplicityMethod,
    MultiplicityPlan,
    adjust_hypothesis_family,
)

plan = MultiplicityPlan(
    family_id="warehouse-failure-signals",
    mode=AnalysisMode.HELD_OUT_CONFIRMATION,
    method=MultiplicityMethod.HOLM,
    target_error_rate=0.05,
    analysis_dataset_digest=confirmation_dataset_digest,
    selection_dataset_digest=discovery_dataset_digest,
    preregistration_digest=preregistration_digest,
    planned_hypothesis_ids=tuple(planned_hypothesis_ids),
)

report = adjust_hypothesis_family(plan, held_out_test_results)
```

Benjamini-Hochberg and Benjamini-Yekutieli control false discovery rate;
Holm and Bonferroni control family-wise error. Every adjusted result preserves
its effect estimate, sample size, analysis digest, and bounded evidence label.
Exploratory selections are never relabeled as confirmed findings.

`AdaptiveSearchLedger` content-addresses every proposed scenario, proposal
policy, information set, score, and observed outcome. Adaptive searches may
produce exploratory signals, but preregistered or held-out confirmation rejects
hypotheses adapted on the analysis data. Multiplicity adjustment controls only
its declared statistical error criterion; it does not remove simulator,
measurement, model-form, selection, or causal-identification bias.

## Counterfactual replay and minimal counterexamples

Test a pre-specified mechanism by holding the scenario, initial state, seed,
random stream, simulator evidence, and nuisance configuration fixed while
applying one versioned intervention:

```python
from iso_obs.counterfactual import (
    CounterfactualDesign,
    CounterfactualEstimand,
    EffectDirection,
    analyze_counterfactual_design,
)

design = CounterfactualDesign(
    design_id="restore-brake-controller",
    design_version="1",
    preregistration_digest=preregistration_digest,
    simulation_evidence_manifest_digest=simulation_evidence_digest,
    intervention=intervention,
    estimand=CounterfactualEstimand(
        outcome_name="minimum time to collision",
        summary_statistic="minimum over episode",
        expected_direction=EffectDirection.INCREASE,
        minimum_meaningful_effect=0.5,
        unit="seconds",
    ),
    pairs=counterfactual_pairs,
)

report = analyze_counterfactual_design(design, paired_observations)
```

The analyzer reuses exact paired replay inference, a deterministic bootstrap,
and a sign-flip randomization test. Its strongest result is a replicated
intervention association within the simulator validity envelope—not proof of
a real-world causal mechanism. The meaningful-effect threshold and direction
are part of the content-addressed, preregistered design.

`assess_minimal_counterexample` selects the smallest failure-reproducing
condition set among executed ablations. It grants a `one_minimal` label only
when every single-condition removal has been run and stops reproduction. This
is local minimality over executed trials, not global minimality or causality.

## Evidence-qualified regression promotion

Promote a reproduced failure into a durable regression asset while verifying
that every artifact belongs to the same evidence chain:

```python
from iso_obs.promotion import (
    PromotionIntent,
    RegressionPromotionRequest,
    review_regression_promotion,
)

request = RegressionPromotionRequest(
    request_id="promote-blind-intersection",
    request_version="1",
    intent=PromotionIntent.RELEASE_GATE_CANDIDATE,
    source_failure_content_digest=failure_bundle.content_digest(),
    counterfactual_design_digest=counterfactual_design.content_digest(),
    counterfactual_report_digest=counterfactual_report.content_digest(),
    minimal_counterexample_report_digest=minimal_report.content_digest(),
    regression_pack_digest=regression_pack.content_digest(),
    condition_bindings=condition_parameter_bindings,
    limitations=("Requires execution against the proposed system version.",),
)

review = review_regression_promotion(
    request,
    source_failure=failure_bundle,
    counterfactual_design=counterfactual_design,
    counterfactual_report=counterfactual_report,
    minimal_counterexample=minimal_report,
    regression_pack=regression_pack,
)
```

Discovery regressions require coherent, content-addressed provenance but may
retain unresolved mechanism questions. Release-gate candidates additionally
require a replicated meaningful intervention association, a verified
1-minimal counterexample, simulation validation evidence, a real-world anchor,
and a canonical case that expects the proposed fix to satisfy its invariants.

`release_gate_candidate_ready` qualifies a pack for execution. It is not a
release decision: the complete simulation campaign must still pass every gate,
remain inside the declared validity envelope, and preserve uncertainty and
sim-to-real limitations.

## Scoped release assurance dossiers

Assemble known-failure coverage, completed campaigns, independent challenge
evidence, and unresolved counterarguments without reducing them to a composite
confidence score:

```python
from iso_obs.assurance import (
    ReleaseAssuranceDossier,
    evaluate_release_assurance,
)

dossier = ReleaseAssuranceDossier(
    dossier_id="policy-v18-warehouse-release",
    dossier_version="1",
    release_scope=release_scope,
    policy=assurance_policy,
    regression_evidence=known_failure_evidence_links,
    challenge_evidence=independent_challenge_evidence,
    defeaters=declared_defeaters,
    limitations=("Human release authority remains accountable.",),
)

report = evaluate_release_assurance(
    dossier,
    promotion_reviews=promotion_reviews,
    campaign_reports=completed_campaign_reports,
)
```

Every failure fingerprint required by the preregistered policy must link to an
evidence-qualified promotion and a release-eligible campaign whose regression
gate passed. A failed campaign, failed challenge, or open critical defeater
blocks support and cannot be averaged away by strength elsewhere.

Independent passing challenge evidence is required for support within scope.
Inconclusive challenges and open material defeaters yield a restricted
disposition. The dossier content-addresses the exact system artifacts,
operational domain, simulator validity envelope, and risk policy to prevent a
result from being silently generalized.

`supported_within_scope` is technical decision support—not deployment
authorization, a safety guarantee, or evidence outside the declared envelope.

## Anytime-valid post-deployment surveillance

Tie production monitoring to the exact scoped assurance report and detect
evidence that a failure, intervention, or scope-exit rate exceeds its accepted
bound:

```python
from iso_obs.surveillance import (
    AlternativeRate,
    SequentialMonitoringPlan,
    SignalMonitoringSpec,
    evaluate_sequential_surveillance,
)

failure_signal = SignalMonitoringSpec(
    signal_id="safety-intervention",
    description="A safety monitor overrides the autonomous controller.",
    outcome_definition_digest=outcome_definition_digest,
    maximum_acceptable_event_rate=0.001,
    alternatives=(
        AlternativeRate(event_probability=0.005, weight=0.5),
        AlternativeRate(event_probability=0.01, weight=0.5),
    ),
    allocated_error_rate=0.025,
)

plan = SequentialMonitoringPlan(
    plan_id="policy-v18-field-monitoring",
    plan_version="1",
    assurance_report_digest=assurance_report.content_digest(),
    release_scope_digest=assurance_report.release_scope_digest,
    family_error_rate=0.05,
    signals=(failure_signal, scope_exit_signal),
    limitations=("Events depend on validated production labeling.",),
)

report = evaluate_sequential_surveillance(
    plan,
    assurance_report,
    ordered_observations,
)
```

Each signal uses a weighted mixture of preregistered likelihood-ratio
e-processes. Its threshold may be checked after every observation while
retaining anytime-valid false-alarm control under the declared conditional
event-rate null. Allocated per-signal error rates and the union bound control
the family without assuming signals are independent.

A historical threshold crossing remains visible even if later observations
reduce the current evidence value. `no_threshold_crossing` means continue
monitoring—not that the system is proven safe. A crossing escalates review of
the scoped assurance claim; it does not automatically identify causality,
authorize a shutdown, or prescribe corrective action.

## Model-guided experiment selection

Fit a compact Bayesian reliability model to exploratory simulation outcomes
and select the next operating region under an explicit compute budget:

```python
from iso_obs.experiment_selection import (
    BayesianSelectorConfig,
    ExperimentRegion,
    ExperimentSelectionObservation,
    ExperimentSelectionPlan,
    fit_bayesian_experiment_selector,
    recommend_next_experiment,
)

plan = ExperimentSelectionPlan(
    plan_id="policy-v18-failure-search",
    plan_version="1",
    target_population_digest=target_population_digest,
    partition_definition_digest=partition_digest,
    simulation_manifest_digest=simulation_evidence_digest,
    system_artifact_digest=system_artifact_digest,
    outcome_definition_digest=failure_oracle_digest,
    total_execution_budget=1_000.0,
    regions=(
        ExperimentRegion(
            region_id="nominal",
            condition_definition_digest=nominal_digest,
            target_population_mass=0.95,
            severity_weight=1.0,
            expected_execution_cost=1.0,
            minimum_exploration_count=2,
            maximum_experiment_count=100,
        ),
        ExperimentRegion(
            region_id="occluded-crossing",
            condition_definition_digest=occluded_digest,
            target_population_mass=0.05,
            severity_weight=20.0,
            expected_execution_cost=2.0,
            minimum_exploration_count=5,
            maximum_experiment_count=250,
        ),
    ),
    config=BayesianSelectorConfig(),
    limitations=("Declared regions do not share statistical strength.",),
)

state = fit_bayesian_experiment_selector(plan, exploratory_observations)
recommendation = recommend_next_experiment(plan, state)
next_region = recommendation.selected_region_id
```

The v1 model learns an independent Beta-Bernoulli failure-rate posterior for
each declared region. Required coverage precedes optimization. Thereafter, the
acquisition score combines severity-weighted posterior failure probability,
posterior uncertainty, exact expected variance reduction, target-population
mass, and expected execution cost. Every recommendation includes the full
score decomposition and exact fitted-state digest.

Adaptive observations remain discovery-only. After the selector identifies a
consequential region, define a new fixed sampling design before using additional
observations for confirmatory failure-rate or release claims. The initial model
does not generalize between regions; that limitation makes it an interpretable
baseline for future shared-representation and continuous-boundary models.

## Failure-phenotype classification

Classify trace-derived failure signatures against a reviewed phenotype
taxonomy without forcing every failure into a known category:

```python
from iso_obs.failure_phenotypes import (
    FailurePhenotypeExample,
    FailurePhenotypePlan,
    FailurePhenotypeQuery,
    PhenotypeFeature,
    PhenotypeTrainingSplit,
    classify_failure_phenotype,
    fit_failure_phenotype_model,
)

plan = FailurePhenotypePlan(
    plan_id="policy-v18-failure-phenotypes",
    plan_version="1",
    feature_extraction_digest=feature_extractor_digest,
    target_population_digest=target_population_digest,
    phenotype_taxonomy_digest=reviewed_taxonomy_digest,
    features=(
        PhenotypeFeature(
            feature_id="time-to-collision-at-intervention",
            definition_digest=ttc_feature_digest,
        ),
        PhenotypeFeature(
            feature_id="peak-lateral-error",
            definition_digest=lateral_error_feature_digest,
        ),
    ),
    phenotype_ids=("late-braking", "steering-oscillation"),
    miscoverage_rate=0.05,
    minimum_fit_examples_per_phenotype=20,
    minimum_calibration_examples_per_phenotype=20,
    limitations=("Validated for low-speed warehouse failures only.",),
)

fit_example = FailurePhenotypeExample(
    plan_content_digest=plan.content_digest(),
    evidence_digest=trace_evidence_digest,
    independence_unit_id="incident-0182",
    phenotype_id="late-braking",
    split=PhenotypeTrainingSplit.FIT,
    feature_values=(0.42, 0.08),
)

model = fit_failure_phenotype_model(
    plan,
    independently_sampled_fit_and_calibration_examples,
)
query = FailurePhenotypeQuery(
    plan_content_digest=plan.content_digest(),
    evidence_digest=new_trace_evidence_digest,
    query_id="incident-0419",
    feature_values=(0.39, 0.11),
)
report = classify_failure_phenotype(plan, model, query)
```

The model robustly scales features using fit-only medians and median absolute
deviations, learns one median prototype per phenotype, and calculates
class-conditional conformal p-values from held-out calibration examples. The
result is a prediction set:

- A singleton set yields `assigned`.
- Multiple supported phenotypes yield `ambiguous`.
- An empty set yields `novel_candidate`.

Each row must represent a unique independent incident or scenario unit; the
SDK rejects repeated units across fitting and calibration. Calibration evidence
does not influence feature scaling or prototype fitting. Every candidate
includes the complete feature-level distance decomposition, and every model,
query, and report is content-addressed.

The conformal coverage target depends on exchangeability between calibration
and future examples within each phenotype. It is not guaranteed after
unmeasured distribution shift, taxonomy error, or dependent sampling.
`novel_candidate` is a review and reproduction trigger—not proof of a new
failure mechanism. Phenotype outputs remain discovery-only and cannot
authorize a release decision.

## Sim-to-real validity assessment

Test whether simulation and real-world anchors agree within predeclared,
region-specific engineering tolerances:

```python
from iso_obs.sim_to_real import (
    SimToRealValidityPlan,
    TransferDomain,
    TransferFeature,
    TransferObservation,
    TransferRegion,
    assess_sim_to_real_validity,
)

plan = SimToRealValidityPlan(
    plan_id="policy-v18-sim-to-real-validity",
    plan_version="1",
    target_population_digest=target_population_digest,
    simulation_evidence_digest=simulation_evidence_digest,
    real_world_anchor_digest=real_anchor_dataset_digest,
    system_artifact_digest=system_artifact_digest,
    failure_outcome_definition_digest=failure_oracle_digest,
    features=(
        TransferFeature(
            feature_id="normalized-stopping-distance",
            definition_digest=stopping_distance_feature_digest,
            lower_bound=0.0,
            upper_bound=1.0,
            maximum_acceptable_mean_gap=0.05,
        ),
    ),
    regions=(
        TransferRegion(
            region_id="nominal",
            condition_definition_digest=nominal_condition_digest,
            target_population_mass=0.9,
            minimum_simulation_samples=1_000,
            minimum_real_world_samples=1_000,
        ),
        TransferRegion(
            region_id="occluded-crossing",
            condition_definition_digest=occluded_condition_digest,
            target_population_mass=0.1,
            minimum_simulation_samples=2_000,
            minimum_real_world_samples=500,
        ),
    ),
    maximum_acceptable_failure_rate_gap=0.01,
    familywise_error_rate=0.05,
    limitations=("Validated for low-speed warehouse operation only.",),
)

real_anchor = TransferObservation(
    plan_content_digest=plan.content_digest(),
    evidence_digest=real_trace_digest,
    independence_unit_id="incident-0419",
    domain=TransferDomain.REAL_WORLD,
    region_id="occluded-crossing",
    feature_values=(0.42,),
    failed=False,
)

report = assess_sim_to_real_validity(
    plan,
    (*simulation_observations, *real_world_anchor_observations),
)
```

For every region, the SDK separately evaluates each bounded feature-mean gap
and the failure-rate gap. Distribution-free Hoeffding bounds are allocated
with Bonferroni correction across the complete planned family of claims. This
controls familywise error without assuming claims or features are independent.

A claim is `supported_within_tolerance` only when its entire simultaneous
confidence interval lies inside the engineering tolerance. It is
`shift_exceeds_tolerance` only when the interval lies outside that tolerance.
Overlap is `inconclusive`; a non-significant difference is never relabeled as
equivalence.

Regional aggregation is non-compensatory. One detected shift blocks overall
support, and one under-sampled region produces `insufficient_evidence`.
Supported target-population mass remains visible but cannot average away a
narrow failure. Every observation is bound to the simulator evidence,
real-world anchor set, target population, system artifact, feature definitions,
and failure oracle.

The resulting evidence is confirmatory only when the content-addressed plan was
fixed before observing outcomes. Content addressing alone does not prove
preregistration. Support means equivalence within the declared mean and
failure-rate tolerances—not that simulation is reality, that every relevant
variable was measured, or that a release is authorized.

## Causal perturbation-response assessment

Estimate how controlled perturbations change failure probability relative to
matched baseline replays:

```python
from iso_obs.causal_perturbations import (
    CausalPerturbationPlan,
    PairedPerturbationObservation,
    PerturbationContrast,
    assess_causal_perturbations,
)

plan = CausalPerturbationPlan(
    plan_id="policy-v18-perturbation-effects",
    plan_version="1",
    target_population_digest=target_population_digest,
    simulation_evidence_digest=simulation_evidence_digest,
    system_artifact_digest=system_artifact_digest,
    failure_outcome_definition_digest=failure_oracle_digest,
    intervention_protocol_digest=intervention_protocol_digest,
    pairing_protocol_digest=matched_replay_protocol_digest,
    contrasts=(
        PerturbationContrast(
            contrast_id="rain-intensity-plus-20-percent",
            perturbation_spec_digest=rain_perturbation_digest,
            factor_id="rain-intensity",
            magnitude=0.2,
            unit="normalized",
            minimum_material_failure_rate_change=0.01,
            minimum_pair_count=1_000,
        ),
    ),
    familywise_error_rate=0.05,
    limitations=("Validated for low-speed warehouse operation only.",),
)

paired_result = PairedPerturbationObservation(
    plan_content_digest=plan.content_digest(),
    contrast_id="rain-intensity-plus-20-percent",
    pair_id="scenario-seed-0419",
    matched_context_digest=matched_context_digest,
    baseline_evidence_digest=baseline_trace_digest,
    perturbed_evidence_digest=rain_trace_digest,
    baseline_failed=False,
    perturbed_failed=True,
)

report = assess_causal_perturbations(plan, paired_observations)
```

The estimand is the paired failure-rate change: perturbed minus baseline. Each
pair must preserve the same exogenous replay context while changing only the
declared intervention. The SDK rejects duplicate pairs, perturbed-evidence
reuse, inconsistent shared baselines, and replay contexts presented as
multiple independent units.

Simultaneous paired Hoeffding bounds use Bonferroni allocation across the
complete perturbation family. A contrast is `causal_increase_supported` only
when its lower confidence bound exceeds the predeclared material-effect
threshold. A decrease uses the symmetric upper-bound rule. Equivalence requires
the entire interval to lie inside the material-effect band; overlap remains
`inconclusive`.

Campaign aggregation is non-compensatory: one harmful contrast blocks an
otherwise benign family. Sufficiently sampled contrasts are ranked
deterministically by their conservative harmful lower bound so teams can
prioritize reproduction and mitigation.

Causal interpretation is conditional on intervention isolation, consistency,
no interference, representative paired contexts, and a pairing protocol fixed
before outcomes were observed. The result does not identify the downstream
mechanism, establish real-world transport, or authorize release.

## Continuous failure-boundary mapping

Map where a system transitions from reliable to unreliable across continuous
operating conditions without silently interpolating through evidence gaps:

```python
from iso_obs.failure_boundaries import (
    BoundaryAnchor,
    BoundaryDimension,
    BoundaryObservation,
    FailureBoundaryPlan,
    fit_failure_boundary_model,
    map_failure_boundary,
)

plan = FailureBoundaryPlan(
    plan_id="policy-v18-friction-speed-boundary",
    plan_version="1",
    target_population_digest=target_population_digest,
    simulation_evidence_digest=simulation_evidence_digest,
    system_artifact_digest=system_artifact_digest,
    failure_outcome_definition_digest=failure_oracle_digest,
    coordinate_extraction_digest=coordinate_extractor_digest,
    dimensions=(
        BoundaryDimension(
            dimension_id="surface-friction",
            definition_digest=friction_definition_digest,
            lower_bound=0.1,
            upper_bound=1.0,
            distance_weight=1.0,
            grid_points=(0.1, 0.3, 0.5, 0.7, 1.0),
        ),
        BoundaryDimension(
            dimension_id="speed-meters-per-second",
            definition_digest=speed_definition_digest,
            lower_bound=0.0,
            upper_bound=5.0,
            distance_weight=2.0,
            grid_points=(0.0, 1.0, 2.5, 4.0, 5.0),
        ),
    ),
    anchors=(
        BoundaryAnchor(
            anchor_id="low-friction-high-speed",
            coordinates=(0.1, 5.0),
            planned_sample_count=2_000,
        ),
        BoundaryAnchor(
            anchor_id="nominal-friction-low-speed",
            coordinates=(1.0, 1.0),
            planned_sample_count=2_000,
        ),
    ),
    maximum_acceptable_failure_probability=0.01,
    failure_probability_lipschitz_constant=0.5,
    familywise_error_rate=0.05,
    maximum_grid_cell_count=1_000,
    limitations=("Coordinates cover the validated warehouse envelope.",),
)

model = fit_failure_boundary_model(plan, fixed_anchor_observations)
boundary_map = map_failure_boundary(plan, model)
```

The SDK computes simultaneous fixed-sample Hoeffding intervals at every
predeclared anchor. It then propagates those intervals using the declared
Lipschitz limit on failure-probability change under weighted normalized
coordinate distance.

Propagation uses the farthest point from each anchor to each cell—not the cell
center. Therefore:

- `reliable_certified` means the uniform upper bound is below the failure
  threshold for every point in that cell.
- `unreliable_certified` means the uniform lower bound is above the threshold
  for every point in that cell.
- `unresolved_boundary` means available evidence cannot certify either side.

The model detects anchor intervals that contradict the declared Lipschitz
constant and returns `assumptions_violated` without producing a map. Missing
fixed samples return `insufficient_evidence`; sampling beyond the frozen count
returns `design_violated`, preventing optional continuation from being
presented as fixed-design inference.

Reliable, unreliable, and unresolved geometric volumes are reported
separately. Geometric volume is not operational probability mass and cannot be
used as one without an independently justified target distribution.

The Lipschitz constant is a substantive engineering assumption. Anchor
consistency can falsify an undersized value but cannot prove smoothness between
anchors. Boundary certification remains conditional on that assumption,
coordinate validity, independent trials, and a plan fixed before outcomes.

## Stratified failure-surface estimation

Estimate operational failure probability without allowing common conditions
to hide a narrow, safety-critical region:

```python
from iso_obs.failure_surface import (
    FailureSurfacePlan,
    FailureSurfaceStratum,
    estimate_failure_surface,
)

rare_glare = FailureSurfaceStratum(
    stratum_id="low-friction-oncoming-glare",
    condition_definition_digest=condition_digest,
    target_population_mass=0.002,
    planned_sample_count=5_000,
    maximum_acceptable_failure_rate=0.001,
    allocated_error_rate=0.005,
)

plan = FailureSurfacePlan(
    plan_id="policy-v18-warehouse-surface",
    plan_version="1",
    target_population_digest=exposure_population_digest,
    partition_definition_digest=partition_digest,
    simulation_manifest_digest=simulation_evidence_digest,
    system_artifact_digest=system_artifact_digest,
    outcome_definition_digest=failure_oracle_digest,
    maximum_acceptable_overall_failure_rate=0.0001,
    family_error_rate=0.05,
    strata=(common_conditions, rare_glare, boundary_conditions),
    limitations=("Target weights come from the current exposure model.",),
)

report = estimate_failure_surface(plan, fixed_design_observations)
```

Rare strata may be deliberately oversampled. Their empirical rates are mapped
back to the target operating population using preregistered stratum masses.
Two-sided Hoeffding intervals use allocated error rates and cover all strata
simultaneously through the union bound; their weighted sum bounds the overall
target-population failure probability.

The design is non-compensatory: `within_limits` requires both the overall upper
bound and every stratum upper bound to meet their declared limits. A rare
stratum whose lower bound exceeds its local limit produces `exceeds_limits`
even when the weighted average looks favorable.

Confirmatory results require the exact fixed sample count in every stratum.
Incomplete cells expose their raw counts but withhold confidence intervals and
surface conclusions, preventing optional stopping and coverage gaps from being
presented as finished evidence.

## Simulator-to-real transport validation

Validate a simulator against matched real-world anchors for a specific metric,
intended use, and operating envelope:

```python
from iso_obs.transportability import (
    TransportValidationPlan,
    TransportStratum,
    assess_transport_applicability,
    validate_transportability,
)

low_friction = TransportStratum(
    stratum_id="low-friction",
    condition_definition_digest=condition_digest,
    target_population_mass=0.02,
    planned_pair_count=2_000,
    maximum_acceptable_mean_expanded_discrepancy=0.1,
    maximum_possible_expanded_discrepancy=1.0,
    allocated_error_rate=0.01,
)

plan = TransportValidationPlan(
    plan_id="warehouse-twin-stopping-margin",
    plan_version="1",
    simulator_manifest_digest=simulation_manifest_digest,
    real_anchor_dataset_digest=anchor_dataset_digest,
    pairing_protocol_digest=pairing_protocol_digest,
    target_population_digest=target_population_digest,
    partition_definition_digest=partition_digest,
    metric_name="stopping margin",
    metric_unit="m",
    outcome_definition_digest=metric_definition_digest,
    maximum_acceptable_overall_mean_expanded_discrepancy=0.05,
    family_error_rate=0.05,
    anchor_envelope=validated_ranges,
    discrepancy_sources=declared_discrepancy_sources,
    physics_evidence=physics_of_failure_evidence,
    strata=(common_conditions, low_friction),
    limitations=("Anchors use the production sensor calibration.",),
)

report = validate_transportability(plan, matched_anchor_observations)
applicability = assess_transport_applicability(
    plan,
    report,
    requested_operating_conditions,
)
```

The outcome is conservative expanded absolute discrepancy: the paired absolute
simulator/real residual plus declared simulation and measurement uncertainty.
Fixed-sample Hoeffding bounds cover all strata simultaneously under their
allocated error rates. Target-population weights recover the operational mean
when rare anchor strata are deliberately oversampled.

Support is non-compensatory. Every stratum and the weighted overall upper bound
must satisfy their limits. A rare cell whose lower bound exceeds its local
limit prevents support even when the aggregate looks favorable. Known
unquantified discrepancy sources yield `inconclusive`; an observation above
its preregistered maximum possible discrepancy yields
`assumptions_violated` and invalidates inference.

`supported_within_anchor_envelope` is never permission to extrapolate.
Applicability requires an exact, unit-checked operating point inside every
validated range. Changes to the simulator, real measurement protocol, metric,
target population, partition, system, or envelope require new evidence.

## Local dataset reliability workflow

Package one immutable dataset version for repeatable local inspection and
auditing:

```python
from iso_obs.dataset_io import (
    DATASET_BUNDLE_SCHEMA_VERSION,
    DatasetBundle,
    ManifestDatasetAdapter,
    inspect_dataset_bundle,
    load_json_artifact,
)

bundle = DatasetBundle(
    schema_version=DATASET_BUNDLE_SCHEMA_VERSION,
    manifest=dataset_manifest,
    episodes=episode_manifests,
    label_assertions=label_assertions,
    timing_traces=source_order_timing_traces,
)

inspection = inspect_dataset_bundle(bundle)
adapter = ManifestDatasetAdapter(bundle)

# Round-trip strict canonical JSON. Unknown fields, duplicate object keys,
# non-finite numbers, and cross-manifest references are rejected.
restored = load_json_artifact("dataset-bundle.json", DatasetBundle)
assert restored.content_digest() == bundle.content_digest()
```

The bundle is an evidence carrier, not a quality score. Its inspection reports
structural counts and missingness only. Use `audit_dataset_reliability`,
`audit_dataset_synchronization`, and `audit_dataset_split` for the corresponding
scientific claims. Timing arrays retain source ordering; the loader does not
sort samples, impute labels, or infer missing clocks.

Convert row-oriented JSONL or Parquet trajectories through an explicit,
content-addressed mapping:

```python
from iso_obs.dataset_ingestion import (
    DATASET_INGESTION_PLAN_SCHEMA_VERSION,
    ChannelIngestionRule,
    DatasetIngestionPlan,
    DatasetSourceFormat,
    SourceField,
    TimestampUnit,
    ingest_dataset_source,
)

plan = DatasetIngestionPlan(
    schema_version=DATASET_INGESTION_PLAN_SCHEMA_VERSION,
    plan_id="warehouse-trajectory-ingestion",
    plan_version="1",
    source_format=DatasetSourceFormat.JSONL,
    dataset_id="warehouse-picks",
    dataset_version="2026-07",
    source_uri="s3://evidence/warehouse-picks.jsonl",
    license_id="LicenseRef-internal",
    scope=dataset_scope,
    episode_id_field=SourceField(("episode_id",)),
    independence_unit_id_field=SourceField(("run_id",)),
    timestamp_field=SourceField(("timestamp_ns",)),
    timestamp_unit=TimestampUnit.NANOSECONDS,
    channel_rules=(
        ChannelIngestionRule(
            channel_id="camera-front",
            modality_id="camera",
            clock_id="camera-clock",
            payload_field=SourceField(("observation", "camera_uri")),
            timestamp_field=SourceField(("camera_timestamp_ns",)),
            timestamp_uncertainty_seconds=0.0005,
        ),
        ChannelIngestionRule(
            channel_id="joint-state",
            modality_id="robot-state",
            clock_id="controller",
            payload_field=SourceField(("observation", "joint_state")),
            timestamp_uncertainty_seconds=0.0001,
        ),
    ),
)

result = ingest_dataset_source(plan, "warehouse-picks.jsonl")
bundle = result.bundle
ingestion_report = result.report
```

The report links the exact source-file digest, mapping-plan digest, manifest
digest, and output-bundle digest. Sparse channels retain explicit missing
payload counts. A declared required channel with no samples yields
`review_required`; an empty source yields `insufficient_evidence`. Binary
Parquet payloads are hashed with byte counts rather than copied into the
bundle. Install Parquet support with `pip install "iso-obs[parquet]"`.

### MCAP and ROS 2 bags

MCAP uses a dedicated plan because pub/sub messages are sparse topic records,
not wide table rows:

```python
from iso_obs.dataset_mcap import (
    MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    McapDatasetIngestionPlan,
    McapEpisodeWindow,
    McapPayloadMode,
    McapTimestampSource,
    McapTopicRule,
    ingest_mcap_source,
)

plan = McapDatasetIngestionPlan(
    schema_version=MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    plan_id="warehouse-rosbag-ingestion",
    plan_version="1",
    dataset_id="warehouse-robot-bags",
    dataset_version="2026-07",
    source_uri="s3://evidence/robot-a-run-42.mcap",
    license_id="LicenseRef-internal",
    scope=dataset_scope,
    topic_rules=(
        McapTopicRule(
            topic="/camera/front/image_raw",
            channel_id="camera-front",
            modality_id="camera",
            clock_id="recorder-clock",
            payload_mode=McapPayloadMode.RAW_BYTES,
            timestamp_source=McapTimestampSource.LOG_TIME,
            timestamp_uncertainty_seconds=0.0005,
            expected_message_encoding="cdr",
            expected_schema_name="sensor_msgs/msg/Image",
            expected_schema_encoding="ros2msg",
        ),
        McapTopicRule(
            topic="/joint_states",
            channel_id="joint-state",
            modality_id="robot-state",
            clock_id="ros-header-clock",
            payload_mode=McapPayloadMode.ROS2_DECODED,
            timestamp_source=McapTimestampSource.ROS_HEADER,
            timestamp_uncertainty_seconds=0.0001,
            expected_message_encoding="cdr",
            expected_schema_name="sensor_msgs/msg/JointState",
            expected_schema_encoding="ros2msg",
        ),
    ),
    episode_windows=(
        McapEpisodeWindow(
            episode_id="pick-0042",
            independence_unit_id="physical-run-42",
            start_time_ns=1_721_305_000_000_000_000,
            end_time_ns=1_721_305_030_000_000_000,
            source_split="unassigned",
        ),
    ),
    episode_window_timestamp_source=McapTimestampSource.LOG_TIME,
)

result = ingest_mcap_source(plan, "robot-a-run-42.mcap")
```

Install raw/JSON MCAP support with `pip install "iso-obs[mcap]"`. Decoded ROS 2
payloads and header timestamps require `pip install "iso-obs[mcap-ros2]"`.
Decoding uses the official schema-aware MCAP decoder and requires embedded
`ros2msg` definitions. IDL-only CDR recordings can still be ingested with
`RAW_BYTES`, but the SDK will not pretend they were decoded or extract a header
timestamp.

The adapter uses the full-stream, CRC-validating MCAP reader with bounded
record sizes and `log_time_order=False`, preserving physical file order. Topic
rules enforce expected message encoding, schema name, and schema encoding.
Episode membership is explicit and half-open; messages outside declared
windows remain outside the dataset rather than being assigned heuristically.

### Split rosbag2 recordings

Rosbag2 recordings may contain multiple MCAP files governed by
`metadata.yaml`. Wrap the same MCAP plan instead of globbing those files:

```python
from iso_obs.dataset_rosbag2 import (
    ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    Rosbag2McapIngestionPlan,
    ingest_rosbag2_mcap_source,
)

recording_plan = Rosbag2McapIngestionPlan(
    schema_version=ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION,
    mcap_plan=plan,
    limitations=(
        "Recorder host clock synchronization was not independently measured.",
    ),
)

result = ingest_rosbag2_mcap_source(recording_plan, "robot-a-run-42/")
bundle = result.bundle
ingestion_report = result.report
recording_evidence = result.recording_evidence
```

Install support with `pip install "iso-obs[rosbag2]"`. Persist both the dataset
ingestion report and `recording_evidence`; the latter records the exact
metadata digest, ordered relative file paths and content digests, byte counts,
declared versus observed per-file statistics, and topic-level type,
serialization, and message-count evidence.

The metadata is parsed as bounded UTF-8 YAML with duplicate keys and aliases
forbidden. Only rosbag2 metadata versions 5 through 9 using uncompressed
external MCAP files are accepted. Paths must be canonical, recording-relative,
and contained within the recording directory.

The SDK verifies metadata claims against CRC-validated physical MCAP streams.
Count, timing, topic, schema, or encoding disagreement yields
`review_required`; it does not rewrite the metadata. Metadata file order is
preserved, physical order is preserved within each file, and messages are not
globally sorted. Split-boundary overlaps—including repeated transient-local
messages—remain explicit and are never silently deduplicated.

### Failure replay capsules

Turn one recorded failure interval into a portable simulator replay definition
without silently equating reconstructability with label truth:

```python
from iso_obs.replay_capsule import (
    FailureClaimQualification,
    ReplayFidelity,
    compile_failure_replay_capsule,
)

capsule = compile_failure_replay_capsule(
    replay_request,
    bundle=dataset_bundle,
    synchronization=synchronization_report,
    adapter=simulation_adapter_manifest,
    recording_evidence=rosbag2_recording_evidence,
)

if capsule.fidelity is ReplayFidelity.EXACT_REPLAY_READY:
    print("Declared state, inputs, timing, transforms, and controls replay.")

if (
    capsule.failure_claim_qualification
    is FailureClaimQualification.REVIEW_REQUIRED
):
    print("Reconstruction may be exact, but the failure label needs review.")
```

The versioned `iso-obs.failure-replay-request.v1` request binds the exact
dataset bundle, episode, label assertions, extraction interval, reference
clock, simulator and adapter manifests, channel-to-artifact transformations,
initial-state snapshot, stochastic controls, and configuration digests. The
compiler verifies those identities and emits
`iso-obs.failure-replay-capsule.v1`.

Replay fidelity has three intentionally conservative states:

- `exact_replay_ready`: no reconstruction-affecting findings;
- `approximate_replay_only`: discovery use with explicit timing, state,
  interpolation, or stochastic-control limitations;
- `insufficient_evidence`: at least one blocking reconstruction gap.

`failure_claim_qualification` is a separate axis. A suggested or disputed
label does not weaken an otherwise exact reconstruction claim, but it prevents
`build_canonical_regression_case` from promoting the replay into a canonical
regression. Promotion requires both exact replay and support for the selected
failure claim. This avoids gating on a reproducible interval whose asserted
failure mode is still ambiguous.

For a file-based end-to-end workflow, see
[`examples/failure_replay_capsule.py`](examples/failure_replay_capsule.py).

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
