"""Paired counterfactual replay and minimal-counterexample evidence.

The counterfactual protocol changes one declared intervention while holding
replay controls fixed within each pair. Its strongest label is a replicated
intervention association inside the represented simulator validity envelope;
it does not establish a real-world causal mechanism.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .divergence import PairingQuality
from .evidence import NamedDigest, _canonical_json, sha256_digest
from .replication import (
    PairedAnalysisConfig,
    PairedEffectReport,
    PairedObservation,
    analyze_paired_effect,
)


class InterventionScope(StrEnum):
    """System region in which an intervention is applied."""

    ENVIRONMENT = "environment"
    OBSERVATION = "observation"
    ACTION = "action"
    COMPONENT = "component"
    TIMING = "timing"
    STATE = "state"
    SOFTWARE = "software"


class InterventionOperation(StrEnum):
    """Atomic operation performed on an intervention target."""

    SET = "set"
    REMOVE = "remove"
    REPLACE = "replace"
    CLAMP = "clamp"
    DELAY = "delay"
    DISABLE = "disable"


class EffectDirection(StrEnum):
    """Preregistered direction of an intervention effect."""

    INCREASE = "increase"
    DECREASE = "decrease"


class CounterfactualEvidenceLevel(StrEnum):
    """Strongest bounded claim supported by a counterfactual report."""

    INCONCLUSIVE = "inconclusive"
    DIRECTIONALLY_CONSISTENT = "directionally_consistent"
    REPLICATED_INTERVENTION_ASSOCIATION = "replicated_intervention_association"


@dataclass(frozen=True, slots=True)
class ReplayControl:
    """Immutable inputs held fixed within one counterfactual replay pair."""

    scenario_id: str
    scenario_version: str
    simulation_manifest_digest: str
    initial_state_digest: str
    seed: int
    random_stream_digest: str
    nuisance_configuration_digests: tuple[NamedDigest, ...] = ()

    def __post_init__(self) -> None:
        """Validate and canonicalize replay controls."""
        _require_text(self.scenario_id, "scenario ID")
        _require_text(self.scenario_version, "scenario version")
        for value, label in (
            (self.simulation_manifest_digest, "simulation manifest"),
            (self.initial_state_digest, "initial state"),
            (self.random_stream_digest, "random stream"),
        ):
            NamedDigest(label, value)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("replay seed must be an integer")
        _require_unique(
            (item.name for item in self.nuisance_configuration_digests),
            "nuisance configuration names",
        )
        object.__setattr__(
            self,
            "nuisance_configuration_digests",
            tuple(
                sorted(
                    self.nuisance_configuration_digests,
                    key=lambda item: item.name,
                )
            ),
        )

    def content_digest(self) -> str:
        """Calculate the exact replay-control content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(_canonical_json(self))


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """Versioned, atomic change applied during counterfactual replay."""

    intervention_id: str
    mechanism_hypothesis_id: str
    scope: InterventionScope
    operation: InterventionOperation
    target: str
    baseline_value_digest: str
    intervention_value_digest: str
    rationale: str
    start_step: int | None = None
    end_step: int | None = None

    def __post_init__(self) -> None:
        """Validate intervention identity, values, and optional schedule."""
        for value, label in (
            (self.intervention_id, "intervention ID"),
            (self.mechanism_hypothesis_id, "mechanism hypothesis ID"),
            (self.target, "intervention target"),
            (self.rationale, "intervention rationale"),
        ):
            _require_text(value, label)
        object.__setattr__(self, "scope", InterventionScope(self.scope))
        object.__setattr__(
            self,
            "operation",
            InterventionOperation(self.operation),
        )
        NamedDigest("baseline intervention value", self.baseline_value_digest)
        NamedDigest(
            "counterfactual intervention value",
            self.intervention_value_digest,
        )
        if self.baseline_value_digest == self.intervention_value_digest:
            raise ValueError("intervention must change the target value")
        if (self.start_step is None) != (self.end_step is None):
            raise ValueError(
                "intervention start_step and end_step must be supplied together"
            )
        if self.start_step is not None:
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (self.start_step, self.end_step)
            ):
                raise ValueError("intervention steps must be integers")
            if self.start_step < 0 or (self.end_step or 0) < self.start_step:
                raise ValueError(
                    "intervention steps must form a non-negative ordered range"
                )

    def content_digest(self) -> str:
        """Calculate the exact intervention content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(_canonical_json(self))


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    """One controlled replay pair declared before analysis."""

    pair_id: str
    control: ReplayControl
    source_failure_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate pair identity and source-failure provenance."""
        _require_text(self.pair_id, "counterfactual pair ID")
        _require_unique_text(
            self.source_failure_event_ids,
            "source failure event IDs",
        )
        object.__setattr__(
            self,
            "source_failure_event_ids",
            tuple(sorted(self.source_failure_event_ids)),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualEstimand:
    """Pre-specified scalar outcome and practically meaningful effect."""

    outcome_name: str
    summary_statistic: str
    expected_direction: EffectDirection
    minimum_meaningful_effect: float
    unit: str

    def __post_init__(self) -> None:
        """Validate the estimand and meaningful-effect threshold."""
        for value, label in (
            (self.outcome_name, "outcome name"),
            (self.summary_statistic, "summary statistic"),
            (self.unit, "estimand unit"),
        ):
            _require_text(value, label)
        object.__setattr__(
            self,
            "expected_direction",
            EffectDirection(self.expected_direction),
        )
        threshold = float(self.minimum_meaningful_effect)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("minimum_meaningful_effect must be finite and positive")
        object.__setattr__(self, "minimum_meaningful_effect", threshold)


@dataclass(frozen=True, slots=True)
class CounterfactualDesign:
    """Preregistered paired design for one isolated intervention."""

    design_id: str
    design_version: str
    preregistration_digest: str
    simulation_evidence_manifest_digest: str
    intervention: InterventionSpec
    estimand: CounterfactualEstimand
    pairs: tuple[CounterfactualPair, ...]
    analysis_config: PairedAnalysisConfig = PairedAnalysisConfig()

    def __post_init__(self) -> None:
        """Validate and canonicalize the complete experimental design."""
        _require_text(self.design_id, "counterfactual design ID")
        _require_text(self.design_version, "counterfactual design version")
        NamedDigest("preregistration", self.preregistration_digest)
        NamedDigest(
            "simulation evidence manifest",
            self.simulation_evidence_manifest_digest,
        )
        if len(self.pairs) < 2:
            raise ValueError("counterfactual design requires at least two pairs")
        _require_unique((item.pair_id for item in self.pairs), "pair IDs")
        object.__setattr__(
            self,
            "pairs",
            tuple(sorted(self.pairs, key=lambda item: item.pair_id)),
        )

    def to_json(self) -> str:
        """Serialize the design to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact design content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class CounterfactualObservation:
    """Observed scalar outcomes from one declared replay pair."""

    pair_id: str
    baseline: float
    intervention: float
    baseline_run_id: str
    intervention_run_id: str
    replay_control_digest: str
    intervention_spec_digest: str

    def __post_init__(self) -> None:
        """Validate observation identity, values, and content references."""
        for value, label in (
            (self.pair_id, "counterfactual observation pair ID"),
            (self.baseline_run_id, "baseline run ID"),
            (self.intervention_run_id, "intervention run ID"),
        ):
            _require_text(value, label)
        if self.baseline_run_id == self.intervention_run_id:
            raise ValueError("baseline and intervention run IDs must be different")
        baseline = float(self.baseline)
        intervention = float(self.intervention)
        if not math.isfinite(baseline) or not math.isfinite(intervention):
            raise ValueError("counterfactual outcomes must be finite")
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "intervention", intervention)
        NamedDigest("replay control", self.replay_control_digest)
        NamedDigest("intervention specification", self.intervention_spec_digest)


@dataclass(frozen=True, slots=True)
class CounterfactualEvidenceReport:
    """Effect estimate and bounded evidence from paired intervention replay."""

    design_content_digest: str
    intervention_id: str
    mechanism_hypothesis_id: str
    estimand: CounterfactualEstimand
    paired_effect: PairedEffectReport
    directionally_consistent: bool
    meaningfully_supported: bool
    evidence_level: CounterfactualEvidenceLevel
    limitations: tuple[str, ...]
    report_schema_version: str = "iso-obs.counterfactual-evidence-report.v1"

    def to_json(self) -> str:
        """Serialize the report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def analyze_counterfactual_design(
    design: CounterfactualDesign,
    observations: Sequence[CounterfactualObservation],
) -> CounterfactualEvidenceReport:
    """Analyze one complete, identity-verified counterfactual design.

    Args:
        design: Preregistered paired intervention design.
        observations: Exactly one result for every declared pair.

    Returns:
        Paired effect report with a bounded intervention-evidence label.

    Raises:
        ValueError: If observations are incomplete, duplicated, or do not
            match the declared replay controls and intervention.
    """
    ordered = tuple(sorted(observations, key=lambda item: item.pair_id))
    observed_ids = tuple(item.pair_id for item in ordered)
    _require_unique(observed_ids, "counterfactual observation pair IDs")
    planned_by_id = {item.pair_id: item for item in design.pairs}
    planned_ids = set(planned_by_id)
    supplied_ids = set(observed_ids)
    if supplied_ids != planned_ids:
        missing = sorted(planned_ids - supplied_ids)
        unknown = sorted(supplied_ids - planned_ids)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(
            "counterfactual observations must exactly match design pairs ("
            + "; ".join(details)
            + ")"
        )

    intervention_digest = design.intervention.content_digest()
    paired_observations = []
    for observation in ordered:
        planned = planned_by_id[observation.pair_id]
        if observation.replay_control_digest != planned.control.content_digest():
            raise ValueError(
                f"pair '{observation.pair_id}' replay control digest mismatch"
            )
        if observation.intervention_spec_digest != intervention_digest:
            raise ValueError(
                f"pair '{observation.pair_id}' intervention digest mismatch"
            )
        paired_observations.append(
            PairedObservation(
                pair_id=observation.pair_id,
                baseline=observation.baseline,
                candidate=observation.intervention,
                pairing_quality=PairingQuality.EXACT_REPLAY,
            )
        )

    effect = analyze_paired_effect(
        paired_observations,
        config=design.analysis_config,
    )
    directionally_consistent, meaningfully_supported = _classify_effect(
        design.estimand,
        effect,
    )
    alpha = 1.0 - design.analysis_config.confidence_level
    replicated = meaningfully_supported and effect.randomization_p_value <= alpha
    if replicated:
        level = CounterfactualEvidenceLevel.REPLICATED_INTERVENTION_ASSOCIATION
    elif directionally_consistent:
        level = CounterfactualEvidenceLevel.DIRECTIONALLY_CONSISTENT
    else:
        level = CounterfactualEvidenceLevel.INCONCLUSIVE
    limitations = (
        "The result supports a replicated intervention association only "
        "inside the represented simulator validity envelope; it does not "
        "establish a real-world causal mechanism.",
        "Intervention isolation depends on complete replay controls, simulator "
        "fidelity, measurement validity, and the absence of interference or "
        "carryover between runs.",
        "Common random numbers reduce paired variance but cannot remove "
        "common-mode simulator or model-form bias.",
        "Testing multiple mechanisms or outcomes requires declared "
        "multiplicity control and preferably held-out confirmation.",
    )
    return CounterfactualEvidenceReport(
        design_content_digest=design.content_digest(),
        intervention_id=design.intervention.intervention_id,
        mechanism_hypothesis_id=(design.intervention.mechanism_hypothesis_id),
        estimand=design.estimand,
        paired_effect=effect,
        directionally_consistent=directionally_consistent,
        meaningfully_supported=meaningfully_supported,
        evidence_level=level,
        limitations=limitations,
    )


def _classify_effect(
    estimand: CounterfactualEstimand,
    effect: PairedEffectReport,
) -> tuple[bool, bool]:
    """Classify direction and practical support for a paired effect.

    Args:
        estimand: Pre-specified expected direction and effect threshold.
        effect: Estimated intervention-minus-baseline effect.

    Returns:
        Directional-consistency and meaningful-support flags.
    """
    interval = effect.confidence_interval
    threshold = estimand.minimum_meaningful_effect
    if estimand.expected_direction is EffectDirection.INCREASE:
        return effect.mean_difference > 0.0, interval.lower >= threshold
    return effect.mean_difference < 0.0, interval.upper <= -threshold


@dataclass(frozen=True, slots=True)
class FailureCondition:
    """One content-addressed condition considered during minimization."""

    condition_id: str
    value_digest: str
    category: str

    def __post_init__(self) -> None:
        """Validate condition identity and content."""
        _require_text(self.condition_id, "failure condition ID")
        _require_text(self.category, "failure condition category")
        NamedDigest("failure condition value", self.value_digest)


@dataclass(frozen=True, slots=True)
class ConditionSetTrial:
    """One executed reproduction attempt using a retained condition set."""

    retained_condition_ids: tuple[str, ...]
    failure_reproduced: bool
    run_id: str
    evidence_digest: str

    def __post_init__(self) -> None:
        """Validate and canonicalize an executed condition-set trial."""
        _require_unique_text(
            self.retained_condition_ids,
            "retained condition IDs",
            required=False,
        )
        if not isinstance(self.failure_reproduced, bool):
            raise ValueError("failure_reproduced must be a boolean")
        _require_text(self.run_id, "condition-set trial run ID")
        NamedDigest("condition-set trial evidence", self.evidence_digest)
        object.__setattr__(
            self,
            "retained_condition_ids",
            tuple(sorted(self.retained_condition_ids)),
        )


@dataclass(frozen=True, slots=True)
class MinimalCounterexampleReport:
    """Smallest reproduced condition set supported by executed trials."""

    source_condition_ids: tuple[str, ...]
    retained_condition_ids: tuple[str, ...]
    eliminated_condition_ids: tuple[str, ...]
    one_minimal: bool
    untested_single_removals: tuple[str, ...]
    reproducing_run_ids: tuple[str, ...]
    trial_evidence_digests: tuple[str, ...]
    limitations: tuple[str, ...]
    report_schema_version: str = "iso-obs.minimal-counterexample-report.v1"

    def to_json(self) -> str:
        """Serialize the report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def assess_minimal_counterexample(
    conditions: Sequence[FailureCondition],
    trials: Sequence[ConditionSetTrial],
) -> MinimalCounterexampleReport:
    """Find and locally verify a minimal reproduced condition set.

    The result is 1-minimal only when removing each retained condition has
    been executed and prevents reproduction. It is not a proof of global
    minimality over untested conditions or of a causal mechanism.

    Args:
        conditions: Declared conditions in the source failure.
        trials: Executed reproduction attempts over subsets of conditions.

    Returns:
        Deterministic minimal-counterexample report.

    Raises:
        ValueError: If provenance is incomplete, duplicated, or inconsistent.
    """
    if not conditions:
        raise ValueError("minimal-counterexample analysis requires conditions")
    condition_ids = tuple(sorted(item.condition_id for item in conditions))
    _require_unique(condition_ids, "source condition IDs")
    if not trials:
        raise ValueError("minimal-counterexample analysis requires trials")
    ordered_trials = tuple(
        sorted(
            trials,
            key=lambda item: (
                len(item.retained_condition_ids),
                item.retained_condition_ids,
                item.run_id,
            ),
        )
    )
    _require_unique(
        (item.run_id for item in ordered_trials),
        "condition-set trial run IDs",
    )
    _require_unique(
        (item.retained_condition_ids for item in ordered_trials),
        "executed retained condition sets",
    )
    declared = set(condition_ids)
    for trial in ordered_trials:
        unknown = sorted(set(trial.retained_condition_ids) - declared)
        if unknown:
            raise ValueError(
                "condition-set trial contains unknown conditions: " + ", ".join(unknown)
            )
    full_trial = next(
        (
            trial
            for trial in ordered_trials
            if trial.retained_condition_ids == condition_ids
        ),
        None,
    )
    if full_trial is None or not full_trial.failure_reproduced:
        raise ValueError("the full source condition set must have a reproducing trial")
    reproducing = tuple(trial for trial in ordered_trials if trial.failure_reproduced)
    selected = reproducing[0]
    trial_by_conditions = {item.retained_condition_ids: item for item in ordered_trials}
    untested = []
    necessary = True
    for condition_id in selected.retained_condition_ids:
        removal = tuple(
            item for item in selected.retained_condition_ids if item != condition_id
        )
        removal_trial = trial_by_conditions.get(removal)
        if removal_trial is None:
            untested.append(condition_id)
            necessary = False
        elif removal_trial.failure_reproduced:
            necessary = False
    one_minimal = necessary and not untested
    retained = selected.retained_condition_ids
    eliminated = tuple(sorted(declared - set(retained)))
    limitations = (
        "The selected set is the smallest reproduced set among executed "
        "trials; unexecuted subsets may be smaller.",
        "A 1-minimal label means every tested single-condition removal stopped "
        "reproduction. It does not prove global minimality or causality.",
        "Reproduction remains conditional on simulator validity, replay "
        "fidelity, failure-oracle validity, and declared nuisance controls.",
    )
    return MinimalCounterexampleReport(
        source_condition_ids=condition_ids,
        retained_condition_ids=retained,
        eliminated_condition_ids=eliminated,
        one_minimal=one_minimal,
        untested_single_removals=tuple(sorted(untested)),
        reproducing_run_ids=tuple(sorted(item.run_id for item in reproducing)),
        trial_evidence_digests=tuple(
            sorted(item.evidence_digest for item in ordered_trials)
        ),
        limitations=limitations,
    )


def _require_text(value: str, label: str) -> None:
    """Require non-empty text.

    Args:
        value: Candidate text.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_unique(values: Iterable[object], label: str) -> None:
    """Require unique values.

    Args:
        values: Candidate values.
        label: Human-readable collection label.
    """
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} must be unique")


def _require_unique_text(
    values: Sequence[str],
    label: str,
    *,
    required: bool = True,
) -> None:
    """Require unique non-empty text values.

    Args:
        values: Candidate text values.
        label: Human-readable collection label.
        required: Whether at least one value is required.
    """
    if required and not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_text(value, label)
    _require_unique(values, label)
