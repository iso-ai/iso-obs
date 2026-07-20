"""Auditable training evidence for learned world models.

Reliability Studio does not train customer models. This module preserves the
lineage, compute use, learning trajectory, and sliced rollout evaluation
produced by an external CPU or customer-hosted GPU training job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from .evidence import NamedDigest, _canonical_json, sha256_digest

WORLD_MODEL_TRAINING_PLAN_SCHEMA_VERSION = "iso-obs.world-model-training-plan.v1"
WORLD_MODEL_TRAINING_REPORT_SCHEMA_VERSION = "iso-obs.world-model-training-report.v1"


class WorldModelTrainingDisposition(StrEnum):
    """Strongest training conclusion supported by the declared evaluation."""

    ACCEPTABLE_FOR_DECLARED_EVALUATION = "acceptable_for_declared_evaluation"
    NEEDS_IMPROVEMENT = "needs_improvement"
    INVALID_TRAINING_EVIDENCE = "invalid_training_evidence"


@dataclass(frozen=True, slots=True)
class WorldModelTrainingPlan:
    """Content-addressed declaration of a world-model training job."""

    schema_version: str
    plan_id: str
    plan_version: str
    dataset_content_digest: str
    model_family: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    compute_backend: str
    accelerator: str
    random_seed: int
    epochs: int
    evaluation_horizon_steps: int
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate stable identities, dimensions, and training bounds."""
        if self.schema_version != WORLD_MODEL_TRAINING_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported world-model training plan schema version")
        for text_value, label in (
            (self.plan_id, "training plan ID"),
            (self.plan_version, "training plan version"),
            (self.model_family, "model family"),
            (self.compute_backend, "compute backend"),
            (self.accelerator, "accelerator"),
        ):
            _require_text(text_value, label)
        NamedDigest("training dataset", self.dataset_content_digest)
        _require_unique_text(self.feature_names, "feature names")
        _require_unique_text(self.target_names, "target names")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("training random seed must be an integer")
        for count_value, label in (
            (self.epochs, "training epochs"),
            (self.evaluation_horizon_steps, "evaluation horizon"),
        ):
            if (
                isinstance(count_value, bool)
                or not isinstance(count_value, int)
                or count_value <= 0
            ):
                raise ValueError(f"{label} must be a positive integer")
        object.__setattr__(
            self,
            "limitations",
            _unique_limitations(self.limitations),
        )

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Return the plan's content address."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class TrainingEpoch:
    """Observed loss values at one completed training epoch."""

    epoch: int
    training_loss: float
    validation_loss: float

    def __post_init__(self) -> None:
        """Validate epoch identity and finite nonnegative losses."""
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int):
            raise ValueError("training epoch must be an integer")
        if self.epoch <= 0:
            raise ValueError("training epoch must be positive")
        for error_value, label in (
            (self.training_loss, "training loss"),
            (self.validation_loss, "validation loss"),
        ):
            _nonnegative_finite(error_value, label)


@dataclass(frozen=True, slots=True)
class WorldModelEvaluationSlice:
    """One held-out operating slice for one-step and rollout fidelity."""

    slice_id: str
    sample_count: int
    horizon_steps: int
    one_step_rmse: float
    rollout_rmse: float
    rollout_rmse_threshold: float
    in_training_support: bool
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate slice identity, counts, errors, and support declaration."""
        _require_text(self.slice_id, "evaluation slice ID")
        for count_value, label in (
            (self.sample_count, "evaluation sample count"),
            (self.horizon_steps, "evaluation horizon"),
        ):
            if (
                isinstance(count_value, bool)
                or not isinstance(count_value, int)
                or count_value <= 0
            ):
                raise ValueError(f"{label} must be a positive integer")
        for error_value, label in (
            (self.one_step_rmse, "one-step RMSE"),
            (self.rollout_rmse, "rollout RMSE"),
            (self.rollout_rmse_threshold, "rollout RMSE threshold"),
        ):
            _nonnegative_finite(error_value, label)
        if not isinstance(self.in_training_support, bool):
            raise ValueError("training-support declaration must be boolean")
        object.__setattr__(
            self,
            "limitations",
            _unique_limitations(self.limitations),
        )

    @property
    def passed(self) -> bool:
        """Return whether rollout error meets the declared slice threshold."""
        return self.rollout_rmse <= self.rollout_rmse_threshold


@dataclass(frozen=True, slots=True)
class WorldModelTrainingReport:
    """Versioned evidence emitted by an external world-model training job."""

    schema_version: str
    plan_content_digest: str
    dataset_content_digest: str
    model_content_digest: str
    compute_backend: str
    accelerator: str
    duration_seconds: float
    peak_memory_mb: float
    parameter_count: int
    epochs: tuple[TrainingEpoch, ...]
    evaluation_slices: tuple[WorldModelEvaluationSlice, ...]
    baseline_model_content_digest: str | None
    baseline_rollout_rmse: float | None
    candidate_rollout_rmse: float
    disposition: WorldModelTrainingDisposition
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate provenance, compute observations, curves, and disposition."""
        if self.schema_version != WORLD_MODEL_TRAINING_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported world-model training report schema version")
        for label, digest in (
            ("training plan", self.plan_content_digest),
            ("training dataset", self.dataset_content_digest),
            ("trained model", self.model_content_digest),
        ):
            NamedDigest(label, digest)
        if self.baseline_model_content_digest is not None:
            NamedDigest("baseline model", self.baseline_model_content_digest)
        _require_text(self.compute_backend, "training compute backend")
        _require_text(self.accelerator, "training accelerator")
        for value, label in (
            (self.duration_seconds, "training duration"),
            (self.peak_memory_mb, "peak memory"),
            (self.candidate_rollout_rmse, "candidate rollout RMSE"),
        ):
            _nonnegative_finite(value, label)
        if self.baseline_rollout_rmse is not None:
            _nonnegative_finite(
                self.baseline_rollout_rmse,
                "baseline rollout RMSE",
            )
        if (
            isinstance(self.parameter_count, bool)
            or not isinstance(self.parameter_count, int)
            or self.parameter_count <= 0
        ):
            raise ValueError("parameter count must be a positive integer")
        epochs = tuple(sorted(self.epochs, key=lambda item: item.epoch))
        if not epochs or len({item.epoch for item in epochs}) != len(epochs):
            raise ValueError("training report requires unique observed epochs")
        slices = tuple(sorted(self.evaluation_slices, key=lambda item: item.slice_id))
        if not slices or len({item.slice_id for item in slices}) != len(slices):
            raise ValueError("training report requires unique evaluation slices")
        disposition = WorldModelTrainingDisposition(self.disposition)
        if (
            disposition
            is WorldModelTrainingDisposition.ACCEPTABLE_FOR_DECLARED_EVALUATION
        ):
            if any(not item.passed for item in slices):
                raise ValueError(
                    "acceptable training evidence requires every declared slice to pass"
                )
            if any(not item.in_training_support for item in slices):
                raise ValueError(
                    "out-of-support slices cannot support an acceptable disposition"
                )
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "evaluation_slices", slices)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "limitations",
            _unique_limitations(self.limitations),
        )

    @property
    def rollout_error_reduction(self) -> float | None:
        """Return relative rollout-error reduction against the baseline."""
        if self.baseline_rollout_rmse in (None, 0.0):
            return None
        return (
            self.baseline_rollout_rmse - self.candidate_rollout_rmse
        ) / self.baseline_rollout_rmse

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Return the report's content address."""
        return sha256_digest(self.to_json())


def _require_text(value: str, label: str) -> None:
    """Require a nonempty text value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty")


def _require_unique_text(values: tuple[str, ...], label: str) -> None:
    """Require a nonempty tuple of unique nonempty strings."""
    if not values:
        raise ValueError(f"{label} must not be empty")
    normalized = tuple(value.strip() for value in values)
    if any(not value for value in normalized) or len(set(normalized)) != len(values):
        raise ValueError(f"{label} must contain unique nonempty values")


def _nonnegative_finite(value: float, label: str) -> None:
    """Require a finite nonnegative numeric observation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")


def _unique_limitations(values: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize and deduplicate limitations without changing first order."""
    normalized = tuple(value.strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError("limitations must be nonempty strings")
    return tuple(dict.fromkeys(normalized))
