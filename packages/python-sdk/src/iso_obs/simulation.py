"""Compile regression packs into auditable simulator execution plans.

The compiler expands every regression case and seed into a deterministic work
item, checks adapter capabilities before compute is spent, and records whether
results are intended for reproduction, cross-backend validation, or surrogate
screening. Learned-world-model screening requires uncertainty and
out-of-distribution capabilities and is never marked as release-gate evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest
from .regression import (
    CaseRole,
    ExpectedOutcome,
    InvariantOracle,
    RegressionCase,
    RegressionPack,
    ScenarioParameter,
    SeedStrategy,
)


class BackendType(StrEnum):
    """Kind of environment executing a regression plan."""

    PHYSICS_SIMULATOR = "physics_simulator"
    WORLD_MODEL = "world_model"
    HYBRID_TWIN = "hybrid_twin"
    LOG_REPLAY = "log_replay"
    HARDWARE_IN_THE_LOOP = "hardware_in_the_loop"


class SimulationCapability(StrEnum):
    """Behavior an adapter can guarantee to the execution compiler."""

    SEEDED_RESET = "seeded_reset"
    PARAMETER_OVERRIDE = "parameter_override"
    ARTIFACT_LOADING = "artifact_loading"
    RANDOM_STREAM_CONTROL = "random_stream_control"
    INVARIANT_SIGNAL_EXPORT = "invariant_signal_export"
    PRIVILEGED_STATE_EXPORT = "privileged_state_export"
    DETERMINISTIC_TIME_CONTROL = "deterministic_time_control"
    SNAPSHOT_RESTORE = "snapshot_restore"
    PERTURBATION_SCHEDULING = "perturbation_scheduling"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    OUT_OF_DISTRIBUTION_DETECTION = "out_of_distribution_detection"


class ExecutionIntent(StrEnum):
    """Scientific purpose of an execution plan."""

    REPRODUCTION = "reproduction"
    CROSS_BACKEND_VALIDATION = "cross_backend_validation"
    SURROGATE_SCREENING = "surrogate_screening"


class EvidenceUse(StrEnum):
    """Whether plan results may enter confirmatory release evidence."""

    CONFIRMATORY = "confirmatory"
    DISCOVERY_ONLY = "discovery_only"


class CompatibilityIssueCode(StrEnum):
    """Machine-readable reason an adapter cannot execute a pack faithfully."""

    MISSING_CAPABILITY = "missing_capability"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    UNSUPPORTED_ARTIFACT = "unsupported_artifact"
    EVIDENCE_MANIFEST_MISMATCH = "evidence_manifest_mismatch"
    INVALID_INTENT = "invalid_intent"


@dataclass(frozen=True, slots=True)
class SimulationAdapterManifest:
    """Versioned capabilities and evidence identity for one adapter."""

    adapter_name: str
    adapter_version: str
    backend_type: BackendType
    simulator_name: str
    simulator_version: str
    simulation_evidence_manifest_digest: str
    capabilities: tuple[SimulationCapability, ...]
    supported_parameters: tuple[str, ...]
    supported_artifacts: tuple[str, ...]
    accepts_arbitrary_parameters: bool = False
    accepts_arbitrary_artifacts: bool = False
    max_parallelism: int = 1
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and canonicalize adapter identity and capabilities."""
        for value, label in (
            (self.adapter_name, "adapter name"),
            (self.adapter_version, "adapter version"),
            (self.simulator_name, "simulator name"),
            (self.simulator_version, "simulator version"),
        ):
            _require_text(value, label)
        object.__setattr__(self, "backend_type", BackendType(self.backend_type))
        NamedDigest(
            "simulation evidence manifest",
            self.simulation_evidence_manifest_digest,
        )
        resolved_capabilities = tuple(
            SimulationCapability(item) for item in self.capabilities
        )
        _require_unique(
            (item.value for item in resolved_capabilities),
            "adapter capabilities",
        )
        _require_unique_text(
            self.supported_parameters,
            "supported parameters",
            required=False,
        )
        _require_unique_text(
            self.supported_artifacts,
            "supported artifacts",
            required=False,
        )
        _require_unique_text(
            self.limitations,
            "adapter limitations",
            required=False,
        )
        if not isinstance(self.accepts_arbitrary_parameters, bool):
            raise ValueError("accepts_arbitrary_parameters must be a boolean")
        if not isinstance(self.accepts_arbitrary_artifacts, bool):
            raise ValueError("accepts_arbitrary_artifacts must be a boolean")
        if (
            isinstance(self.max_parallelism, bool)
            or not isinstance(self.max_parallelism, int)
            or self.max_parallelism < 1
        ):
            raise ValueError("max_parallelism must be a positive integer")
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(resolved_capabilities, key=lambda item: item.value)),
        )
        object.__setattr__(
            self,
            "supported_parameters",
            tuple(sorted(self.supported_parameters)),
        )
        object.__setattr__(
            self,
            "supported_artifacts",
            tuple(sorted(self.supported_artifacts)),
        )
        object.__setattr__(
            self,
            "limitations",
            tuple(sorted(self.limitations)),
        )

    def to_json(self) -> str:
        """Serialize the adapter manifest to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact adapter-manifest content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One deterministic incompatibility between a pack and adapter."""

    code: CompatibilityIssueCode
    subject: str
    affected_case_ids: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        """Validate and canonicalize an incompatibility."""
        object.__setattr__(self, "code", CompatibilityIssueCode(self.code))
        _require_text(self.subject, "compatibility issue subject")
        _require_text(self.detail, "compatibility issue detail")
        _require_unique_text(
            self.affected_case_ids,
            "affected case IDs",
            required=False,
        )
        object.__setattr__(
            self,
            "affected_case_ids",
            tuple(sorted(self.affected_case_ids)),
        )


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Capability assessment performed before plan compilation."""

    pack_id: str
    adapter_name: str
    intent: ExecutionIntent
    evidence_use: EvidenceUse
    required_capabilities: tuple[SimulationCapability, ...]
    issues: tuple[CompatibilityIssue, ...]
    compatible: bool
    report_schema_version: str = "iso-obs.simulation-compatibility-report.v1"

    def to_json(self) -> str:
        """Serialize the compatibility report to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact compatibility-report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


class IncompatibleSimulationAdapterError(ValueError):
    """Raised when an adapter cannot execute a regression pack faithfully."""

    def __init__(self, report: CompatibilityReport) -> None:
        """Store the structured compatibility report.

        Args:
            report: Complete preflight compatibility diagnostics.
        """
        self.report = report
        subjects = ", ".join(item.subject for item in report.issues)
        super().__init__(f"simulation adapter is incompatible: {subjects}")


@dataclass(frozen=True, slots=True)
class SimulationWorkItem:
    """One deterministic case-by-seed simulator execution."""

    work_item_id: str
    pack_id: str
    pack_version: str
    pack_content_digest: str
    adapter_manifest_digest: str
    intent: ExecutionIntent
    evidence_use: EvidenceUse
    case_id: str
    role: CaseRole
    scenario_id: str
    scenario_version: str
    seed: int
    seed_strategy: SeedStrategy
    random_stream_digest: str | None
    parameters: tuple[ScenarioParameter, ...]
    artifact_digests: tuple[NamedDigest, ...]
    oracles: tuple[InvariantOracle, ...]
    expected_outcome: ExpectedOutcome
    source_evidence_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """Deterministic group bounded by adapter parallelism."""

    batch_index: int
    work_items: tuple[SimulationWorkItem, ...]


@dataclass(frozen=True, slots=True)
class SimulationExecutionPlan:
    """Content-addressed expansion of a pack for one adapter."""

    plan_version: str
    pack_id: str
    pack_version: str
    pack_content_digest: str
    adapter_manifest: SimulationAdapterManifest
    intent: ExecutionIntent
    evidence_use: EvidenceUse
    eligible_for_release_gate: bool
    required_capabilities: tuple[SimulationCapability, ...]
    work_items: tuple[SimulationWorkItem, ...]
    batches: tuple[ExecutionBatch, ...]
    limitations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize the execution plan to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact execution-plan content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def assess_adapter_compatibility(
    pack: RegressionPack,
    adapter: SimulationAdapterManifest,
    *,
    intent: ExecutionIntent,
    evidence_use: EvidenceUse,
) -> CompatibilityReport:
    """Assess whether an adapter can execute a pack without silent weakening.

    Args:
        pack: Failure-derived regression definition.
        adapter: Declared backend capabilities.
        intent: Scientific purpose of the execution.
        evidence_use: Intended evidentiary use of the results.

    Returns:
        Structured required capabilities and incompatibilities.
    """
    resolved_intent = ExecutionIntent(intent)
    resolved_evidence_use = EvidenceUse(evidence_use)
    required = _required_capabilities(pack, adapter, resolved_intent)
    issues: list[CompatibilityIssue] = []
    supported = set(adapter.capabilities)
    all_case_ids = tuple(item.case_id for item in pack.cases)

    for capability in required:
        if capability not in supported:
            issues.append(
                CompatibilityIssue(
                    code=CompatibilityIssueCode.MISSING_CAPABILITY,
                    subject=capability.value,
                    affected_case_ids=all_case_ids,
                    detail=(
                        f"Execution requires {capability.value}, but the "
                        "adapter does not declare it."
                    ),
                )
            )

    if not adapter.accepts_arbitrary_parameters:
        supported_parameters = set(adapter.supported_parameters)
        parameter_cases = _subjects_by_case(
            (
                parameter.name,
                case.case_id,
            )
            for case in pack.cases
            for parameter in case.parameters
            if parameter.name not in supported_parameters
        )
        for parameter, case_ids in parameter_cases.items():
            issues.append(
                CompatibilityIssue(
                    code=CompatibilityIssueCode.UNSUPPORTED_PARAMETER,
                    subject=parameter,
                    affected_case_ids=case_ids,
                    detail=f"Adapter does not support parameter '{parameter}'.",
                )
            )

    if not adapter.accepts_arbitrary_artifacts:
        supported_artifacts = set(adapter.supported_artifacts)
        artifact_cases = _subjects_by_case(
            (
                artifact.name,
                case.case_id,
            )
            for case in pack.cases
            for artifact in case.artifact_digests
            if artifact.name not in supported_artifacts
        )
        for artifact, case_ids in artifact_cases.items():
            issues.append(
                CompatibilityIssue(
                    code=CompatibilityIssueCode.UNSUPPORTED_ARTIFACT,
                    subject=artifact,
                    affected_case_ids=case_ids,
                    detail=f"Adapter does not support artifact '{artifact}'.",
                )
            )

    if (
        resolved_intent is ExecutionIntent.REPRODUCTION
        and adapter.simulation_evidence_manifest_digest
        != pack.simulation_manifest_digest
    ):
        issues.append(
            CompatibilityIssue(
                code=CompatibilityIssueCode.EVIDENCE_MANIFEST_MISMATCH,
                subject="simulation_evidence_manifest_digest",
                affected_case_ids=all_case_ids,
                detail=(
                    "Reproduction requires the simulation evidence manifest "
                    "linked by the regression pack."
                ),
            )
        )

    if (
        resolved_intent is ExecutionIntent.SURROGATE_SCREENING
        and resolved_evidence_use is EvidenceUse.CONFIRMATORY
    ):
        issues.append(
            CompatibilityIssue(
                code=CompatibilityIssueCode.INVALID_INTENT,
                subject="surrogate_screening_confirmatory",
                affected_case_ids=all_case_ids,
                detail=(
                    "Surrogate screening is discovery-only and cannot be used "
                    "as confirmatory release evidence."
                ),
            )
        )
    if (
        resolved_intent is ExecutionIntent.SURROGATE_SCREENING
        and adapter.backend_type
        not in {BackendType.WORLD_MODEL, BackendType.HYBRID_TWIN}
    ):
        issues.append(
            CompatibilityIssue(
                code=CompatibilityIssueCode.INVALID_INTENT,
                subject="surrogate_screening_backend",
                affected_case_ids=all_case_ids,
                detail=(
                    "Surrogate screening requires a world-model or hybrid-twin "
                    "backend."
                ),
            )
        )

    resolved_issues = tuple(
        sorted(
            issues,
            key=lambda item: (
                item.code.value,
                item.subject,
                item.affected_case_ids,
            ),
        )
    )
    return CompatibilityReport(
        pack_id=pack.pack_id,
        adapter_name=adapter.adapter_name,
        intent=resolved_intent,
        evidence_use=resolved_evidence_use,
        required_capabilities=required,
        issues=resolved_issues,
        compatible=not resolved_issues,
    )


def compile_simulation_plan(
    pack: RegressionPack,
    adapter: SimulationAdapterManifest,
    *,
    intent: ExecutionIntent,
    evidence_use: EvidenceUse,
) -> SimulationExecutionPlan:
    """Compile a compatible regression pack into deterministic work items.

    Args:
        pack: Failure-derived regression definition.
        adapter: Versioned backend capability manifest.
        intent: Scientific purpose of the execution.
        evidence_use: Intended evidentiary use of the results.

    Returns:
        A deterministic, content-addressed execution plan.

    Raises:
        IncompatibleSimulationAdapterError: If preflight detects any
            unsupported or scientifically invalid requirement.
    """
    report = assess_adapter_compatibility(
        pack,
        adapter,
        intent=intent,
        evidence_use=evidence_use,
    )
    if not report.compatible:
        raise IncompatibleSimulationAdapterError(report)

    pack_digest = pack.content_digest()
    adapter_digest = adapter.content_digest()
    work_items = tuple(
        _work_item(
            pack,
            case=case,
            seed=seed,
            pack_digest=pack_digest,
            adapter_digest=adapter_digest,
            intent=report.intent,
            evidence_use=report.evidence_use,
        )
        for case in pack.cases
        for seed in pack.seed_panel.seeds
    )
    batches = tuple(
        ExecutionBatch(
            batch_index=index // adapter.max_parallelism,
            work_items=work_items[index : index + adapter.max_parallelism],
        )
        for index in range(0, len(work_items), adapter.max_parallelism)
    )
    release_eligible = (
        report.evidence_use is EvidenceUse.CONFIRMATORY
        and report.intent is not ExecutionIntent.SURROGATE_SCREENING
    )
    limitations = [
        "The plan proves declared adapter compatibility, not simulator "
        "fidelity or correctness; consult the linked simulation evidence.",
        "Execution results inherit every validity-envelope limit, uncertainty "
        "source, and known omission in the simulation evidence manifest.",
    ]
    if report.intent is ExecutionIntent.CROSS_BACKEND_VALIDATION:
        limitations.append(
            "Cross-backend results measure transfer and disagreement; they "
            "are not exact reproductions of the source simulator."
        )
    if not release_eligible:
        limitations.append(
            "This plan is discovery-only and its results are not eligible for "
            "release-gate evidence without independent confirmation."
        )
    return SimulationExecutionPlan(
        plan_version="iso-obs.simulation-execution.v1",
        pack_id=pack.pack_id,
        pack_version=pack.pack_version,
        pack_content_digest=pack_digest,
        adapter_manifest=adapter,
        intent=report.intent,
        evidence_use=report.evidence_use,
        eligible_for_release_gate=release_eligible,
        required_capabilities=report.required_capabilities,
        work_items=work_items,
        batches=batches,
        limitations=tuple(limitations),
    )


def _required_capabilities(
    pack: RegressionPack,
    adapter: SimulationAdapterManifest,
    intent: ExecutionIntent,
) -> tuple[SimulationCapability, ...]:
    """Infer capabilities required for faithful plan execution.

    Args:
        pack: Regression definition being compiled.
        adapter: Target adapter manifest.
        intent: Scientific purpose of the execution.

    Returns:
        Sorted unique capability requirements.
    """
    required = {
        SimulationCapability.SEEDED_RESET,
        SimulationCapability.INVARIANT_SIGNAL_EXPORT,
    }
    if any(case.parameters for case in pack.cases):
        required.add(SimulationCapability.PARAMETER_OVERRIDE)
    if any(case.artifact_digests for case in pack.cases):
        required.add(SimulationCapability.ARTIFACT_LOADING)
    if pack.seed_panel.strategy is SeedStrategy.COMMON_RANDOM_NUMBERS:
        required.add(SimulationCapability.RANDOM_STREAM_CONTROL)
    if (
        intent is ExecutionIntent.SURROGATE_SCREENING
        or adapter.backend_type is BackendType.WORLD_MODEL
    ):
        required.update(
            {
                SimulationCapability.UNCERTAINTY_QUANTIFICATION,
                SimulationCapability.OUT_OF_DISTRIBUTION_DETECTION,
            }
        )
    return tuple(sorted(required, key=lambda item: item.value))


def _work_item(
    pack: RegressionPack,
    *,
    case: RegressionCase,
    seed: int,
    pack_digest: str,
    adapter_digest: str,
    intent: ExecutionIntent,
    evidence_use: EvidenceUse,
) -> SimulationWorkItem:
    """Build one deterministic execution work item.

    Args:
        pack: Source regression pack.
        case: Case being expanded.
        seed: Declared execution seed.
        pack_digest: Exact source-pack digest.
        adapter_digest: Exact target-adapter digest.
        intent: Scientific purpose of the execution.
        evidence_use: Intended evidentiary use.

    Returns:
        Fully traceable case-by-seed work item.
    """
    identity = _canonical_json(
        {
            "adapter_manifest_digest": adapter_digest,
            "case_id": case.case_id,
            "evidence_use": evidence_use.value,
            "intent": intent.value,
            "pack_content_digest": pack_digest,
            "seed": seed,
            "work_item_schema": "iso-obs.simulation-work-item.v1",
        }
    )
    work_item_id = "work_" + sha256_digest(identity).split(":", 1)[1][:32]
    return SimulationWorkItem(
        work_item_id=work_item_id,
        pack_id=pack.pack_id,
        pack_version=pack.pack_version,
        pack_content_digest=pack_digest,
        adapter_manifest_digest=adapter_digest,
        intent=intent,
        evidence_use=evidence_use,
        case_id=case.case_id,
        role=case.role,
        scenario_id=case.scenario_id,
        scenario_version=case.scenario_version,
        seed=seed,
        seed_strategy=pack.seed_panel.strategy,
        random_stream_digest=pack.seed_panel.random_stream_digest,
        parameters=case.parameters,
        artifact_digests=case.artifact_digests,
        oracles=case.oracles,
        expected_outcome=case.expected_outcome,
        source_evidence_event_ids=case.source_evidence_event_ids,
    )


def _subjects_by_case(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Group unsupported subjects by affected regression case.

    Args:
        pairs: Iterable of ``(subject, case_id)`` pairs.

    Returns:
        Sorted case IDs keyed by sorted subject insertion order.
    """
    grouped: dict[str, set[str]] = {}
    for subject, case_id in pairs:
        grouped.setdefault(subject, set()).add(case_id)
    return {subject: tuple(sorted(grouped[subject])) for subject in sorted(grouped)}


def _require_text(value: str, label: str) -> None:
    """Require a non-empty string.

    Args:
        value: Candidate text.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_unique(values: Iterable[str], label: str) -> None:
    """Require unique values from a sequence or generator.

    Args:
        values: Candidate values.
        label: Human-readable collection label.
    """
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{label} must be unique")


def _require_unique_text(
    values: tuple[str, ...],
    label: str,
    *,
    required: bool = True,
) -> None:
    """Require non-empty, unique strings in a tuple.

    Args:
        values: Candidate string tuple.
        label: Human-readable collection label.
        required: Whether at least one value is required.
    """
    if required and not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_text(value, label)
    _require_unique(values, label)
