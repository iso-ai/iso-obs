"""Calibrated change-point contracts for mixed-mode dataset discovery.

Statistical, neural, and hybrid detectors implement a common adapter and emit
candidate mode sets over intervals. Detector outputs remain suggestions: this
module validates their provenance and scope, preserves calibration intervals,
and can translate supported candidates into the dataset label ledger without
promoting them to ground truth.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .dataset_reliability import (
    DatasetLabelAssertion,
    EvidenceRelation,
    LabelEvidence,
    LabelStatus,
    ModeFamily,
)
from .dataset_synchronization import (
    DatasetSynchronizationReport,
    SynchronizationDisposition,
)
from .evidence import NamedDigest, _canonical_json, sha256_digest
from .simulation import EvidenceUse

MIXED_MODE_DETECTION_REPORT_SCHEMA_VERSION = "iso-obs.mixed-mode-detection-report.v1"


class DetectorModelFamily(StrEnum):
    """Implementation family of a mixed-mode detector."""

    STATISTICAL = "statistical"
    NEURAL = "neural"
    HYBRID = "hybrid"


class ModeDetectionDisposition(StrEnum):
    """Strongest claim supported by one detector evaluation."""

    CANDIDATES_IDENTIFIED = "candidates_identified"
    NO_CANDIDATE_WITHIN_SCOPE = "no_candidate_within_scope"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class ModeDetectorManifest:
    """Content-addressed identity and training scope of a detector."""

    model_id: str
    model_version: str
    model_family: DetectorModelFamily
    artifact_digest: str
    architecture_digest: str
    feature_extractor_digest: str
    training_dataset_digests: tuple[str, ...]
    target_population_digest: str
    mode_family: ModeFamily
    label_namespace: str
    calibration_evidence_digest: str | None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate model provenance, target construct, and calibration."""
        _require_text(self.model_id, "mode detector model ID")
        _require_text(self.model_version, "mode detector model version")
        for digest, label in (
            (self.artifact_digest, "mode detector artifact"),
            (self.architecture_digest, "mode detector architecture"),
            (self.feature_extractor_digest, "mode detector feature extractor"),
            (self.target_population_digest, "mode detector target population"),
        ):
            NamedDigest(label, digest)
        training_digests = tuple(sorted(set(self.training_dataset_digests)))
        if not training_digests:
            raise ValueError("mode detector requires training dataset provenance")
        for digest in training_digests:
            NamedDigest("mode detector training dataset", digest)
        _require_text(self.label_namespace, "mode detector label namespace")
        if self.calibration_evidence_digest is not None:
            NamedDigest(
                "mode detector calibration evidence",
                self.calibration_evidence_digest,
            )
        limitations = _unique_text(
            self.limitations,
            "mode detector limitations",
            required=False,
        )
        object.__setattr__(
            self,
            "model_family",
            DetectorModelFamily(self.model_family),
        )
        object.__setattr__(
            self,
            "training_dataset_digests",
            training_digests,
        )
        object.__setattr__(self, "mode_family", ModeFamily(self.mode_family))
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the detector manifest to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact detector manifest digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class ModeDetectionWindow:
    """Synchronized episode interval supplied to a detector."""

    window_id: str
    episode_id: str
    synchronization_report_digest: str
    start_seconds: float
    end_seconds: float
    modality_artifact_digests: tuple[NamedDigest, ...]
    feature_artifact_digest: str

    def __post_init__(self) -> None:
        """Validate temporal scope and immutable input artifacts."""
        _require_text(self.window_id, "mode detection window ID")
        _require_text(self.episode_id, "mode detection episode ID")
        NamedDigest(
            "dataset synchronization report",
            self.synchronization_report_digest,
        )
        start, end = _closed_interval(
            self.start_seconds,
            self.end_seconds,
            "mode detection window",
        )
        if start == end:
            raise ValueError("mode detection window must have positive duration")
        modalities = tuple(
            sorted(self.modality_artifact_digests, key=lambda item: item.name)
        )
        if not modalities:
            raise ValueError("mode detection window requires modality artifacts")
        _require_unique(
            (item.name for item in modalities),
            "mode detection modality artifact names",
        )
        NamedDigest("mode detection feature artifact", self.feature_artifact_digest)
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "modality_artifact_digests", modalities)

    def to_json(self) -> str:
        """Serialize the detection window to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact detection-window digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class ModeInterval:
    """Finite temporal interval used as mode context."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        """Validate interval bounds."""
        start, end = _closed_interval(
            self.start_seconds,
            self.end_seconds,
            "mode context",
        )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True, slots=True)
class ModeSupportInterval:
    """Calibrated support interval for one candidate mode."""

    mode_value: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate candidate identity and probability bounds."""
        _require_text(self.mode_value, "candidate mode value")
        lower = _unit_interval(self.lower, "candidate support lower bound")
        upper = _unit_interval(self.upper, "candidate support upper bound")
        if lower > upper:
            raise ValueError("candidate support lower bound must not exceed upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class ModeChangePointCandidate:
    """One uncertain regime boundary with before/after prediction sets."""

    candidate_id: str
    change_interval: ModeInterval
    before_interval: ModeInterval
    after_interval: ModeInterval
    before_prediction_set: tuple[str, ...]
    after_prediction_set: tuple[str, ...]
    before_support: tuple[ModeSupportInterval, ...]
    after_support: tuple[ModeSupportInterval, ...]
    evidence_digest: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate temporal ordering, sets, support, and provenance."""
        _require_text(self.candidate_id, "mode change candidate ID")
        if (
            self.before_interval.end_seconds > self.change_interval.start_seconds
            or self.change_interval.end_seconds > self.after_interval.start_seconds
        ):
            raise ValueError("mode change context intervals must be temporally ordered")
        before_set = _unique_text(
            self.before_prediction_set,
            "before prediction set",
            required=False,
        )
        after_set = _unique_text(
            self.after_prediction_set,
            "after prediction set",
            required=False,
        )
        before_support = _canonical_support(
            self.before_support,
            "before candidate support",
        )
        after_support = _canonical_support(
            self.after_support,
            "after candidate support",
        )
        if not set(before_set).issubset({item.mode_value for item in before_support}):
            raise ValueError("before prediction set lacks support intervals")
        if not set(after_set).issubset({item.mode_value for item in after_support}):
            raise ValueError("after prediction set lacks support intervals")
        NamedDigest("mode change evidence", self.evidence_digest)
        limitations = _unique_text(
            self.limitations,
            "mode change candidate limitations",
            required=False,
        )
        object.__setattr__(self, "before_prediction_set", before_set)
        object.__setattr__(self, "after_prediction_set", after_set)
        object.__setattr__(self, "before_support", before_support)
        object.__setattr__(self, "after_support", after_support)
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True, slots=True)
class ModeDetectorOutput:
    """Content-addressed raw output from one detector execution."""

    model_manifest_digest: str
    detection_window_digest: str
    execution_digest: str
    candidates: tuple[ModeChangePointCandidate, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate execution provenance and canonical candidate order."""
        NamedDigest("mode detector manifest", self.model_manifest_digest)
        NamedDigest("mode detection window", self.detection_window_digest)
        NamedDigest("mode detector execution", self.execution_digest)
        candidates = tuple(sorted(self.candidates, key=lambda item: item.candidate_id))
        _require_unique(
            (item.candidate_id for item in candidates),
            "mode change candidate IDs",
        )
        limitations = _unique_text(
            self.limitations,
            "mode detector output limitations",
            required=False,
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize detector output to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact detector-output digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class MixedModeDetectionReport:
    """Versioned, scope-bound review artifact for detector candidates."""

    schema_version: str
    model_manifest_digest: str
    detection_window_digest: str
    detector_output_digest: str
    synchronization_report_digest: str
    episode_id: str
    mode_family: ModeFamily
    label_namespace: str
    disposition: ModeDetectionDisposition
    candidates: tuple[ModeChangePointCandidate, ...]
    evidence_use: EvidenceUse
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate report contract, canonical candidates, and limitations."""
        if self.schema_version != MIXED_MODE_DETECTION_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported mixed-mode detection schema version")
        for digest, label in (
            (self.model_manifest_digest, "mode detector manifest"),
            (self.detection_window_digest, "mode detection window"),
            (self.detector_output_digest, "mode detector output"),
            (
                self.synchronization_report_digest,
                "dataset synchronization report",
            ),
        ):
            NamedDigest(label, digest)
        _require_text(self.episode_id, "mixed-mode report episode ID")
        _require_text(self.label_namespace, "mixed-mode label namespace")
        candidates = tuple(sorted(self.candidates, key=lambda item: item.candidate_id))
        _require_unique(
            (item.candidate_id for item in candidates),
            "mixed-mode report candidate IDs",
        )
        evidence_use = EvidenceUse(self.evidence_use)
        if evidence_use is not EvidenceUse.DISCOVERY_ONLY:
            raise ValueError("mixed-mode detection report must be discovery-only")
        limitations = _unique_text(
            self.limitations,
            "mixed-mode detection report limitations",
            required=False,
        )
        object.__setattr__(self, "mode_family", ModeFamily(self.mode_family))
        object.__setattr__(
            self,
            "disposition",
            ModeDetectionDisposition(self.disposition),
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "evidence_use", evidence_use)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact report content digest."""
        return sha256_digest(self.to_json())


class MixedModeDetector(ABC):
    """Adapter implemented by statistical, neural, or hybrid detectors."""

    @abstractmethod
    def manifest(self) -> ModeDetectorManifest:
        """Return immutable model, training, scope, and calibration provenance."""

    @abstractmethod
    def detect(self, window: ModeDetectionWindow) -> ModeDetectorOutput:
        """Detect candidate regime boundaries in one synchronized window.

        Args:
            window: Content-addressed synchronized modality and feature inputs.

        Returns:
            Candidate intervals and calibrated before/after prediction sets.
        """


def evaluate_mixed_mode_detection(
    manifest: ModeDetectorManifest,
    window: ModeDetectionWindow,
    synchronization_report: DatasetSynchronizationReport,
    output: ModeDetectorOutput,
) -> MixedModeDetectionReport:
    """Validate detector output against synchronization and model provenance.

    Args:
        manifest: Detector identity, training sources, target, and calibration.
        window: Exact synchronized input window.
        synchronization_report: Timing evidence supporting the input window.
        output: Candidate change points emitted by the detector.

    Returns:
        A discovery-only report that abstains when timing or calibration is weak.

    Raises:
        ValueError: If immutable identities or candidate temporal scope conflict.
    """
    if window.episode_id != synchronization_report.episode_id:
        raise ValueError("detection window and synchronization episode differ")
    if window.synchronization_report_digest != synchronization_report.content_digest():
        raise ValueError("detection window references a different sync report")
    if output.model_manifest_digest != manifest.content_digest():
        raise ValueError("detector output references a different model manifest")
    if output.detection_window_digest != window.content_digest():
        raise ValueError("detector output references a different detection window")
    for candidate in output.candidates:
        if (
            candidate.before_interval.start_seconds < window.start_seconds
            or candidate.after_interval.end_seconds > window.end_seconds
        ):
            raise ValueError("mode change candidate falls outside detection window")

    population_matches = (
        manifest.target_population_digest
        == synchronization_report.scope.target_population_digest
    )
    if (
        synchronization_report.disposition
        is not SynchronizationDisposition.ALIGNED_WITHIN_SCOPE
        or manifest.calibration_evidence_digest is None
        or not population_matches
    ):
        disposition = ModeDetectionDisposition.INSUFFICIENT_EVIDENCE
    elif output.candidates:
        disposition = ModeDetectionDisposition.CANDIDATES_IDENTIFIED
    else:
        disposition = ModeDetectionDisposition.NO_CANDIDATE_WITHIN_SCOPE
    limitations = _unique_text(
        (
            *manifest.limitations,
            *output.limitations,
            *synchronization_report.limitations,
            *(
                ()
                if population_matches
                else (
                    "Detector and synchronized input target populations differ; "
                    "transport is unsupported.",
                )
            ),
            "Detected boundaries and mode sets are suggestions, not ground truth.",
            "No candidate within scope is not evidence that no latent mode exists.",
            "Change-point timing is bounded by intervals and synchronization "
            "uncertainty.",
        ),
        "mixed-mode detection report limitations",
    )
    return MixedModeDetectionReport(
        schema_version=MIXED_MODE_DETECTION_REPORT_SCHEMA_VERSION,
        model_manifest_digest=manifest.content_digest(),
        detection_window_digest=window.content_digest(),
        detector_output_digest=output.content_digest(),
        synchronization_report_digest=synchronization_report.content_digest(),
        episode_id=window.episode_id,
        mode_family=manifest.mode_family,
        label_namespace=manifest.label_namespace,
        disposition=disposition,
        candidates=output.candidates,
        evidence_use=EvidenceUse.DISCOVERY_ONLY,
        limitations=limitations,
    )


def suggested_label_assertions(
    report: MixedModeDetectionReport,
    manifest: ModeDetectorManifest,
) -> tuple[DatasetLabelAssertion, ...]:
    """Translate supported candidates into explicitly suggested label assertions.

    Args:
        report: Evaluated mixed-mode detector artifact.
        manifest: Detector provenance referenced by the report.

    Returns:
        Before/after temporal assertions that remain machine suggestions.

    Raises:
        ValueError: If the report is unsupported or references another model.
    """
    if report.model_manifest_digest != manifest.content_digest():
        raise ValueError("mixed-mode report references a different detector")
    if report.disposition is ModeDetectionDisposition.INSUFFICIENT_EVIDENCE:
        raise ValueError("insufficient detector evidence cannot create assertions")
    if report.disposition is ModeDetectionDisposition.NO_CANDIDATE_WITHIN_SCOPE:
        return ()

    assertions: list[DatasetLabelAssertion] = []
    for candidate in report.candidates:
        evidence = LabelEvidence(
            evidence_id=f"{candidate.candidate_id}:detector-output",
            evidence_digest=candidate.evidence_digest,
            independent_source_id=manifest.model_id,
            relation=EvidenceRelation.DERIVED_FROM,
            description=(
                "Machine-derived mixed-mode candidate; independent review required."
            ),
        )
        for side, interval, prediction_set in (
            ("before", candidate.before_interval, candidate.before_prediction_set),
            ("after", candidate.after_interval, candidate.after_prediction_set),
        ):
            assertions.append(
                DatasetLabelAssertion(
                    assertion_id=f"{candidate.candidate_id}:{side}",
                    episode_id=report.episode_id,
                    label_namespace=report.label_namespace,
                    mode_family=report.mode_family,
                    start_seconds=interval.start_seconds,
                    end_seconds=interval.end_seconds,
                    candidate_values=prediction_set,
                    status=LabelStatus.SUGGESTED,
                    evidence=(evidence,),
                    method_digest=manifest.content_digest(),
                    limitations=(
                        *candidate.limitations,
                        "Machine suggestion; not independently corroborated.",
                    ),
                )
            )
    return tuple(sorted(assertions, key=lambda item: item.assertion_id))


def _canonical_support(
    values: Sequence[ModeSupportInterval],
    label: str,
) -> tuple[ModeSupportInterval, ...]:
    """Validate and canonicalize mode-support intervals."""
    support = tuple(sorted(values, key=lambda item: item.mode_value))
    _require_unique((item.mode_value for item in support), label)
    return support


def _closed_interval(
    lower: float,
    upper: float,
    label: str,
) -> tuple[float, float]:
    """Validate a finite closed interval."""
    lower_value = _finite(lower, f"{label} lower bound")
    upper_value = _finite(upper, f"{label} upper bound")
    if lower_value > upper_value:
        raise ValueError(f"{label} lower bound must not exceed upper")
    return lower_value, upper_value


def _unit_interval(value: float, label: str) -> float:
    """Validate and return a probability-like value in [0, 1]."""
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _finite(value: float, label: str) -> float:
    """Validate and return a finite number."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_text(value: str, label: str) -> str:
    """Validate a required nonempty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _unique_text(
    values: Iterable[str],
    label: str,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    """Validate, deduplicate, and sort a collection of strings."""
    normalized = tuple(_require_text(value, label) for value in values)
    if required and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(normalized))


def _require_unique(values: Iterable[str], label: str) -> None:
    """Require a collection of identifiers to contain no duplicates."""
    collected = tuple(values)
    if len(set(collected)) != len(collected):
        raise ValueError(f"{label} must be unique")
