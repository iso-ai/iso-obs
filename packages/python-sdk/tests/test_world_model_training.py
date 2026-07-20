"""Tests for versioned world-model training evidence."""

from __future__ import annotations

import pytest

from iso_obs.evidence import sha256_digest
from iso_obs.world_model_training import (
    WORLD_MODEL_TRAINING_PLAN_SCHEMA_VERSION,
    WORLD_MODEL_TRAINING_REPORT_SCHEMA_VERSION,
    TrainingEpoch,
    WorldModelEvaluationSlice,
    WorldModelTrainingDisposition,
    WorldModelTrainingPlan,
    WorldModelTrainingReport,
)


def plan() -> WorldModelTrainingPlan:
    """Build a compact deterministic training plan."""
    return WorldModelTrainingPlan(
        schema_version=WORLD_MODEL_TRAINING_PLAN_SCHEMA_VERSION,
        plan_id="pendulum-dynamics",
        plan_version="1",
        dataset_content_digest=sha256_digest("dataset"),
        model_family="mlp-transition-model",
        feature_names=("sin_theta", "cos_theta", "omega", "torque"),
        target_names=("next_sin_theta", "next_cos_theta", "next_omega"),
        compute_backend="python",
        accelerator="cpu",
        random_seed=7,
        epochs=20,
        evaluation_horizon_steps=25,
    )


def report() -> WorldModelTrainingReport:
    """Build valid training evidence with a measured baseline."""
    training_plan = plan()
    return WorldModelTrainingReport(
        schema_version=WORLD_MODEL_TRAINING_REPORT_SCHEMA_VERSION,
        plan_content_digest=training_plan.content_digest(),
        dataset_content_digest=training_plan.dataset_content_digest,
        model_content_digest=sha256_digest("candidate"),
        compute_backend="python",
        accelerator="cpu",
        duration_seconds=1.5,
        peak_memory_mb=42.0,
        parameter_count=131,
        epochs=(
            TrainingEpoch(epoch=1, training_loss=0.4, validation_loss=0.5),
            TrainingEpoch(epoch=20, training_loss=0.04, validation_loss=0.05),
        ),
        evaluation_slices=(
            WorldModelEvaluationSlice(
                slice_id="nominal",
                sample_count=100,
                horizon_steps=25,
                one_step_rmse=0.04,
                rollout_rmse=0.12,
                rollout_rmse_threshold=0.2,
                in_training_support=True,
            ),
        ),
        baseline_model_content_digest=sha256_digest("baseline"),
        baseline_rollout_rmse=0.3,
        candidate_rollout_rmse=0.12,
        disposition=(WorldModelTrainingDisposition.ACCEPTABLE_FOR_DECLARED_EVALUATION),
    )


def test_training_report_round_trips_and_calculates_reduction() -> None:
    """Canonical evidence is stable and derives only the declared comparison."""
    result = report()

    assert result.content_digest() == sha256_digest(result.to_json())
    assert result.rollout_error_reduction == pytest.approx(0.6)
    assert result.evaluation_slices[0].passed


def test_acceptable_disposition_rejects_failed_or_unsupported_slices() -> None:
    """A passing headline cannot hide a failed or out-of-support slice."""
    result = report()
    failed = WorldModelEvaluationSlice(
        slice_id="mass-shift",
        sample_count=100,
        horizon_steps=25,
        one_step_rmse=0.1,
        rollout_rmse=0.4,
        rollout_rmse_threshold=0.2,
        in_training_support=False,
    )

    with pytest.raises(ValueError, match="every declared slice"):
        WorldModelTrainingReport(
            schema_version=result.schema_version,
            plan_content_digest=result.plan_content_digest,
            dataset_content_digest=result.dataset_content_digest,
            model_content_digest=result.model_content_digest,
            compute_backend=result.compute_backend,
            accelerator=result.accelerator,
            duration_seconds=result.duration_seconds,
            peak_memory_mb=result.peak_memory_mb,
            parameter_count=result.parameter_count,
            epochs=result.epochs,
            evaluation_slices=(failed,),
            baseline_model_content_digest=result.baseline_model_content_digest,
            baseline_rollout_rmse=result.baseline_rollout_rmse,
            candidate_rollout_rmse=result.candidate_rollout_rmse,
            disposition=(
                WorldModelTrainingDisposition.ACCEPTABLE_FOR_DECLARED_EVALUATION
            ),
        )


def test_training_plan_rejects_duplicate_features() -> None:
    """Feature identity must be unambiguous for model replay."""
    with pytest.raises(ValueError, match="unique"):
        WorldModelTrainingPlan(
            schema_version=WORLD_MODEL_TRAINING_PLAN_SCHEMA_VERSION,
            plan_id="bad",
            plan_version="1",
            dataset_content_digest=sha256_digest("dataset"),
            model_family="mlp",
            feature_names=("state", "state"),
            target_names=("next_state",),
            compute_backend="python",
            accelerator="cpu",
            random_seed=1,
            epochs=1,
            evaluation_horizon_steps=1,
        )
