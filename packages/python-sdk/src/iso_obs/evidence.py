"""Deterministic evidence artifacts for autonomous-system failures.

This module distinguishes evidence about a system failure from evidence about
the simulated world that produced it. Simulator verification, calibration,
validation, uncertainty, validity limits, and omissions remain separate rather
than being collapsed into one credibility score.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class EvidenceLevel(StrEnum):
    """Strongest claim justified by a failure evidence bundle."""

    OBSERVED_DIFFERENCE = "observed_difference"
    SUSTAINED_DIVERGENCE = "sustained_divergence"
    OUTCOME_ASSOCIATION = "outcome_association"
    REPLICATED_ASSOCIATION = "replicated_association"
    INTERVENTION_SUPPORTED = "intervention_supported"


class AnalysisKind(StrEnum):
    """Kind of analysis recorded in provenance."""

    TRACE_COMPARISON = "trace_comparison"
    SUSTAINED_DIVERGENCE = "sustained_divergence"
    OUTCOME_ASSOCIATION = "outcome_association"
    REPLICATION = "replication"
    INTERVENTION = "intervention"


class SimulationModelType(StrEnum):
    """Primary source of environment behavior."""

    PHYSICS_BASED = "physics_based"
    DATA_DRIVEN = "data_driven"
    HYBRID = "hybrid"
    LOG_REPLAY = "log_replay"


class UncertaintyKind(StrEnum):
    """Source of uncertainty in a simulation result."""

    ALEATORIC = "aleatoric"
    PARAMETRIC_EPISTEMIC = "parametric_epistemic"
    MODEL_FORM_EPISTEMIC = "model_form_epistemic"
    NUMERICAL = "numerical"
    MEASUREMENT = "measurement"


@dataclass(frozen=True, slots=True)
class NamedDigest:
    """Content digest with a stable semantic name."""

    name: str
    digest: str

    def __post_init__(self) -> None:
        """Validate the digest name and SHA-256 representation."""
        _require_text(self.name, "digest name")
        _require_digest(self.digest, f"digest '{self.name}'")


@dataclass(frozen=True, slots=True)
class ValidityRange:
    """Validated range for one simulation operating condition."""

    parameter: str
    lower: float
    upper: float
    unit: str

    def __post_init__(self) -> None:
        """Validate a finite, ordered operating range."""
        _require_text(self.parameter, "validity parameter")
        _require_text(self.unit, "validity range unit")
        lower = float(self.lower)
        upper = float(self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("validity range bounds must be finite")
        if lower > upper:
            raise ValueError("validity range lower bound must not exceed upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class UncertaintySource:
    """One declared source of simulation-result uncertainty."""

    name: str
    kind: UncertaintyKind
    characterization: str
    quantified: bool

    def __post_init__(self) -> None:
        """Validate and canonicalize an uncertainty declaration."""
        _require_text(self.name, "uncertainty source name")
        _require_text(self.characterization, "uncertainty characterization")
        if not isinstance(self.quantified, bool):
            raise ValueError("uncertainty quantified must be a boolean")
        object.__setattr__(self, "kind", UncertaintyKind(self.kind))


@dataclass(frozen=True, slots=True)
class SimulationEvidenceManifest:
    """Evidence and limitations for the world used in an evaluation."""

    simulator_name: str
    simulator_version: str
    model_type: SimulationModelType
    artifact_digest: str
    intended_use: str
    validity_envelope: tuple[ValidityRange, ...]
    uncertainty_sources: tuple[UncertaintySource, ...]
    verification_evidence: tuple[NamedDigest, ...] = ()
    calibration_evidence: tuple[NamedDigest, ...] = ()
    validation_evidence: tuple[NamedDigest, ...] = ()
    real_world_anchors: tuple[NamedDigest, ...] = ()
    known_omissions: tuple[str, ...] = ()
    solver_name: str | None = None
    solver_version: str | None = None
    time_step_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate identity, evidence, uncertainty, and execution metadata."""
        _require_text(self.simulator_name, "simulator name")
        _require_text(self.simulator_version, "simulator version")
        _require_text(self.intended_use, "simulator intended use")
        _require_digest(self.artifact_digest, "simulator artifact digest")
        object.__setattr__(self, "model_type", SimulationModelType(self.model_type))
        _require_unique(
            (item.parameter for item in self.validity_envelope),
            "validity parameters",
        )
        object.__setattr__(
            self,
            "validity_envelope",
            tuple(sorted(self.validity_envelope, key=lambda item: item.parameter)),
        )
        _require_unique(
            (item.name for item in self.uncertainty_sources),
            "uncertainty sources",
        )
        object.__setattr__(
            self,
            "uncertainty_sources",
            tuple(sorted(self.uncertainty_sources, key=lambda item: item.name)),
        )
        for label, evidence in (
            ("verification evidence", self.verification_evidence),
            ("calibration evidence", self.calibration_evidence),
            ("validation evidence", self.validation_evidence),
            ("real-world anchors", self.real_world_anchors),
        ):
            _require_unique((item.name for item in evidence), label)
        for field_name in (
            "verification_evidence",
            "calibration_evidence",
            "validation_evidence",
            "real_world_anchors",
        ):
            evidence = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                tuple(sorted(evidence, key=lambda item: item.name)),
            )
        _require_unique_text(self.known_omissions, "known omissions", required=False)
        object.__setattr__(self, "known_omissions", tuple(sorted(self.known_omissions)))
        if (self.solver_name is None) != (self.solver_version is None):
            raise ValueError("solver name and version must be supplied together")
        if self.solver_name is not None:
            _require_text(self.solver_name, "solver name")
            _require_text(self.solver_version or "", "solver version")
        if self.time_step_seconds is not None:
            time_step = float(self.time_step_seconds)
            if not math.isfinite(time_step) or time_step <= 0.0:
                raise ValueError("time_step_seconds must be finite and positive")
            object.__setattr__(self, "time_step_seconds", time_step)

    def to_json(self) -> str:
        """Serialize the manifest to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact manifest content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class ReproductionManifest:
    """Inputs required to reconstruct or replay a failure."""

    scenario_id: str
    scenario_version: str
    environment_id: str
    environment_version: str
    system_versions: tuple[str, ...]
    seed: int | None
    perturbation_realization_digest: str | None
    sdk_version: str
    schema_version: str
    configuration_digests: tuple[NamedDigest, ...]
    trace_digest: str
    command: str

    def __post_init__(self) -> None:
        """Validate replay identity and immutable input references."""
        for value, label in (
            (self.scenario_id, "scenario ID"),
            (self.scenario_version, "scenario version"),
            (self.environment_id, "environment ID"),
            (self.environment_version, "environment version"),
            (self.sdk_version, "SDK version"),
            (self.schema_version, "schema version"),
            (self.command, "reproduction command"),
        ):
            _require_text(value, label)
        _require_unique_text(self.system_versions, "system versions")
        object.__setattr__(
            self,
            "system_versions",
            tuple(sorted(self.system_versions)),
        )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")
        if self.perturbation_realization_digest is not None:
            _require_digest(
                self.perturbation_realization_digest,
                "perturbation realization digest",
            )
        _require_unique(
            (item.name for item in self.configuration_digests),
            "configuration digests",
        )
        object.__setattr__(
            self,
            "configuration_digests",
            tuple(sorted(self.configuration_digests, key=lambda item: item.name)),
        )
        _require_digest(self.trace_digest, "trace digest")


@dataclass(frozen=True, slots=True)
class AnalysisProvenance:
    """Versioned, reproducible provenance for one analysis result."""

    kind: AnalysisKind
    analyzer: str
    analyzer_version: str
    input_digests: tuple[NamedDigest, ...]
    configuration_digest: str
    method: str
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate analysis identity and deterministic configuration."""
        object.__setattr__(self, "kind", AnalysisKind(self.kind))
        _require_text(self.analyzer, "analyzer name")
        _require_text(self.analyzer_version, "analyzer version")
        _require_text(self.method, "analysis method")
        if not self.input_digests:
            raise ValueError("analysis provenance requires at least one input")
        _require_unique(
            (item.name for item in self.input_digests),
            "analysis input digests",
        )
        object.__setattr__(
            self,
            "input_digests",
            tuple(sorted(self.input_digests, key=lambda item: item.name)),
        )
        _require_digest(self.configuration_digest, "analysis configuration digest")
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)
        ):
            raise ValueError("analysis random_seed must be an integer or None")


@dataclass(frozen=True, slots=True)
class FailureEvidenceBundle:
    """Content-addressed evidence supporting a bounded failure claim."""

    failure_category: str
    scenario_family: str
    scenario_version: str
    perturbation_family: str
    invariant_id: str
    invariant_version: str
    signal_group: tuple[str, ...]
    execution_phase: str
    component: str
    evidence_level: EvidenceLevel
    summary: str
    affected_system_versions: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    reproduction: ReproductionManifest
    simulation: SimulationEvidenceManifest
    analyses: tuple[AnalysisProvenance, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate failure identity, traceability, and claim support."""
        for value, label in (
            (self.failure_category, "failure category"),
            (self.scenario_family, "scenario family"),
            (self.scenario_version, "scenario version"),
            (self.perturbation_family, "perturbation family"),
            (self.invariant_id, "invariant ID"),
            (self.invariant_version, "invariant version"),
            (self.execution_phase, "execution phase"),
            (self.component, "component"),
            (self.summary, "failure summary"),
        ):
            _require_text(value, label)
        object.__setattr__(self, "evidence_level", EvidenceLevel(self.evidence_level))
        _require_unique_text(self.signal_group, "signal group")
        object.__setattr__(self, "signal_group", tuple(sorted(self.signal_group)))
        _require_unique_text(
            self.affected_system_versions,
            "affected system versions",
        )
        object.__setattr__(
            self,
            "affected_system_versions",
            tuple(sorted(self.affected_system_versions)),
        )
        _require_unique_text(self.evidence_event_ids, "evidence event IDs")
        object.__setattr__(
            self,
            "evidence_event_ids",
            tuple(sorted(self.evidence_event_ids)),
        )
        _require_unique_text(self.limitations, "limitations")
        _require_unique(
            (
                f"{item.kind.value}:{item.analyzer}:{item.analyzer_version}"
                for item in self.analyses
            ),
            "analysis provenance entries",
        )
        object.__setattr__(
            self,
            "analyses",
            tuple(
                sorted(
                    self.analyses,
                    key=lambda item: (
                        item.kind.value,
                        item.analyzer,
                        item.analyzer_version,
                    ),
                )
            ),
        )
        required = _required_analysis(self.evidence_level)
        if required not in {item.kind for item in self.analyses}:
            raise ValueError(
                f"evidence level {self.evidence_level.value} requires "
                f"{required.value} analysis provenance"
            )

    def to_json(self) -> str:
        """Serialize the complete bundle to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate a digest identifying the exact evidence artifact.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())

    def failure_fingerprint(self) -> str:
        """Calculate a stable fingerprint for the failure mechanism.

        The fingerprint excludes summaries, event IDs, affected versions,
        analyses, and reproduction details. It represents a failure family
        rather than one observation.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        identity = {
            "component": self.component,
            "execution_phase": self.execution_phase,
            "failure_category": self.failure_category,
            "fingerprint_schema": "iso-obs.failure-fingerprint.v1",
            "invariant_id": self.invariant_id,
            "invariant_version": self.invariant_version,
            "perturbation_family": self.perturbation_family,
            "scenario_family": self.scenario_family,
            "scenario_version": self.scenario_version,
            "signal_group": sorted(self.signal_group),
        }
        return sha256_digest(_canonical_json(identity))


def sha256_digest(content: str | bytes) -> str:
    """Calculate a normalized SHA-256 digest.

    Args:
        content: UTF-8 text or raw bytes to hash.

    Returns:
        A digest in ``sha256:<lowercase hex>`` form.
    """
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _required_analysis(level: EvidenceLevel) -> AnalysisKind:
    """Return the analysis needed to justify an evidence level.

    Args:
        level: Proposed strongest evidence claim.

    Returns:
        Analysis kind that must be present in provenance.
    """
    return {
        EvidenceLevel.OBSERVED_DIFFERENCE: AnalysisKind.TRACE_COMPARISON,
        EvidenceLevel.SUSTAINED_DIVERGENCE: AnalysisKind.SUSTAINED_DIVERGENCE,
        EvidenceLevel.OUTCOME_ASSOCIATION: AnalysisKind.OUTCOME_ASSOCIATION,
        EvidenceLevel.REPLICATED_ASSOCIATION: AnalysisKind.REPLICATION,
        EvidenceLevel.INTERVENTION_SUPPORTED: AnalysisKind.INTERVENTION,
    }[level]


def _canonical_json(value: object) -> str:
    """Serialize supported evidence values to deterministic JSON.

    Args:
        value: Evidence dataclass or JSON-compatible value.

    Returns:
        Compact JSON with sorted keys and no insignificant whitespace.
    """
    return json.dumps(
        _canonicalize(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonicalize(value: object) -> Any:
    """Convert immutable evidence objects into JSON-compatible values.

    Args:
        value: Value nested within an evidence artifact.

    Returns:
        A JSON-compatible value with deterministic mapping order.

    Raises:
        TypeError: If the value is unsupported.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _require_text(value: str, label: str) -> None:
    """Require a non-empty string.

    Args:
        value: Candidate text.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_digest(value: str, label: str) -> None:
    """Require a normalized SHA-256 digest.

    Args:
        value: Candidate digest.
        label: Human-readable field label.
    """
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must use sha256:<64 lowercase hex>")


def _require_unique(values: Iterable[str], label: str) -> None:
    """Require unique values.

    Args:
        values: Text values to check.
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
    """Require non-empty, unique strings in a sequence.

    Args:
        values: Candidate string sequence.
        label: Human-readable collection label.
        required: Whether at least one value is required.
    """
    if required and not values:
        raise ValueError(f"{label} must not be empty")
    for value in values:
        _require_text(value, label)
    _require_unique(values, label)
