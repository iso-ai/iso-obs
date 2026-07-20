"""Train and evaluate compact trajectory world models on CPU.

The workflow generates versioned pendulum trajectories from declared reference
dynamics, trains a linear baseline and a small nonlinear transition model, and
writes content-addressed training evidence for Reliability Studio. The
reference dynamics are a simulator, not physical hardware evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from iso_obs.dataset_ingestion import (
    DATASET_INGESTION_PLAN_SCHEMA_VERSION,
    ChannelIngestionRule,
    DatasetIngestionPlan,
    DatasetSourceFormat,
    SourceField,
    TimestampUnit,
    ingest_dataset_source,
)
from iso_obs.dataset_reliability import DatasetEvidenceRole, DatasetScope
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

DT_SECONDS = 0.05
GRAVITY = 9.81
LENGTH_METERS = 1.0
MAX_SPEED = 8.0
FEATURE_NAMES = (
    "sin_theta",
    "cos_theta",
    "angular_velocity",
    "torque",
    "payload_mass",
    "joint_damping",
)
TARGET_NAMES = ("next_sin_theta", "next_cos_theta", "next_angular_velocity")


def reference_step(
    state: tuple[float, float],
    torque: float,
    *,
    mass: float,
    damping: float,
) -> tuple[float, float]:
    """Advance the declared pendulum reference dynamics by one time step."""
    theta, omega = state
    acceleration = (
        3.0 * GRAVITY / (2.0 * LENGTH_METERS) * math.sin(theta)
        + 3.0 * torque / (mass * LENGTH_METERS**2)
        - damping * omega
    )
    next_omega = max(-MAX_SPEED, min(MAX_SPEED, omega + DT_SECONDS * acceleration))
    next_theta = math.atan2(
        math.sin(theta + DT_SECONDS * next_omega),
        math.cos(theta + DT_SECONDS * next_omega),
    )
    return next_theta, next_omega


def features(
    state: tuple[float, float],
    torque: float,
    mass: float,
    damping: float,
) -> list[float]:
    """Build the declared transition-model feature vector."""
    theta, omega = state
    return [math.sin(theta), math.cos(theta), omega, torque, mass, damping]


def targets(state: tuple[float, float]) -> list[float]:
    """Build the next-state target without an angular wrap discontinuity."""
    theta, omega = state
    return [math.sin(theta), math.cos(theta), omega]


def generate_dataset(
    path: Path,
    *,
    seed: int,
    episodes: int,
    steps: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    """Generate deterministic domain-randomized trajectories and JSONL evidence."""
    rng = random.Random(seed)
    train_x: list[list[float]] = []
    train_y: list[list[float]] = []
    validation_x: list[list[float]] = []
    validation_y: list[list[float]] = []
    with path.open("x", encoding="utf-8") as stream:
        for episode in range(episodes):
            split = "validation" if episode % 5 == 0 else "train"
            mass = rng.uniform(0.8, 1.2)
            damping = rng.uniform(0.03, 0.12)
            state = (rng.uniform(-math.pi, math.pi), rng.uniform(-2.0, 2.0))
            for step in range(steps):
                torque = rng.uniform(-2.0, 2.0)
                next_state = reference_step(
                    state,
                    torque,
                    mass=mass,
                    damping=damping,
                )
                x = features(state, torque, mass, damping)
                y = targets(next_state)
                if split == "train":
                    train_x.append(x)
                    train_y.append(y)
                else:
                    validation_x.append(x)
                    validation_y.append(y)
                row = {
                    "episode_id": f"episode-{episode:04d}",
                    "physical_run_id": f"reference-sim-{episode:04d}",
                    "timestamp_us": int((episode * steps + step) * DT_SECONDS * 1e6),
                    "source_split": split,
                    "observation": {
                        "state": [state[0], state[1]],
                        "action": torque,
                        "dynamics_context": {
                            "payload_mass": mass,
                            "joint_damping": damping,
                        },
                        "next_state": [next_state[0], next_state[1]],
                    },
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                state = next_state
    return train_x, train_y, validation_x, validation_y


def dataset_plan(source_uri: str) -> DatasetIngestionPlan:
    """Declare the trajectory source mapping and its scientific limitations."""
    return DatasetIngestionPlan(
        schema_version=DATASET_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="pendulum-world-model-trajectories",
        plan_version="1",
        source_format=DatasetSourceFormat.JSONL,
        dataset_id="pendulum-domain-randomized-trajectories",
        dataset_version="2026-07-19",
        source_uri=source_uri,
        license_id="LicenseRef-Iso-generated-reference-dynamics",
        scope=DatasetScope(
            domain_namespace="continuous-control/pendulum",
            target_population_digest=sha256_digest(
                "simulated-pendulum-mass-0.8-1.2-damping-0.03-0.12"
            ),
            collection_protocol_digest=sha256_digest(
                "uniform-state-and-torque-reference-dynamics-v1"
            ),
            modality_ids=("state", "action", "dynamics-context", "next-state"),
            evidence_roles=(DatasetEvidenceRole.REPRESENTATION_LEARNING,),
            limitations=(
                "Trajectories come from declared reference dynamics, not hardware.",
                "The dataset does not establish sim-to-real transport.",
            ),
        ),
        episode_id_field=SourceField(("episode_id",)),
        independence_unit_id_field=SourceField(("physical_run_id",)),
        timestamp_field=SourceField(("timestamp_us",)),
        timestamp_unit=TimestampUnit.MICROSECONDS,
        channel_rules=(
            ChannelIngestionRule(
                channel_id="state",
                modality_id="state",
                clock_id="reference-sim",
                payload_field=SourceField(("observation", "state")),
            ),
            ChannelIngestionRule(
                channel_id="action",
                modality_id="action",
                clock_id="reference-sim",
                payload_field=SourceField(("observation", "action")),
            ),
            ChannelIngestionRule(
                channel_id="dynamics-context",
                modality_id="dynamics-context",
                clock_id="reference-sim",
                payload_field=SourceField(("observation", "dynamics_context")),
            ),
            ChannelIngestionRule(
                channel_id="next-state",
                modality_id="next-state",
                clock_id="reference-sim",
                payload_field=SourceField(("observation", "next_state")),
            ),
        ),
        source_split_field=SourceField(("source_split",)),
        limitations=(
            "Validation episodes are held out by episode identity.",
            "No physical measurements are included.",
        ),
    )


def moments(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    """Calculate per-dimension mean and nonzero standard deviation."""
    width = len(rows[0])
    means = [sum(row[index] for row in rows) / len(rows) for index in range(width)]
    deviations = [
        math.sqrt(sum((row[index] - means[index]) ** 2 for row in rows) / len(rows))
        for index in range(width)
    ]
    return means, [value if value > 1e-9 else 1.0 for value in deviations]


def normalize(
    rows: list[list[float]],
    means: list[float],
    deviations: list[float],
) -> list[list[float]]:
    """Normalize a matrix using fixed training statistics."""
    return [
        [(value - means[index]) / deviations[index] for index, value in enumerate(row)]
        for row in rows
    ]


def mse(predicted: list[list[float]], expected: list[list[float]]) -> float:
    """Return mean squared error over all rows and output dimensions."""
    return sum(
        (prediction - target) ** 2
        for predicted_row, expected_row in zip(predicted, expected, strict=True)
        for prediction, target in zip(predicted_row, expected_row, strict=True)
    ) / (len(expected) * len(expected[0]))


def linear_predict(model: dict[str, Any], row: list[float]) -> list[float]:
    """Predict one normalized target with a trained affine model."""
    return [
        model["bias"][output]
        + sum(
            model["weights"][output][index] * value for index, value in enumerate(row)
        )
        for output in range(len(model["bias"]))
    ]


def train_linear(
    train_x: list[list[float]],
    train_y: list[list[float]],
    validation_x: list[list[float]],
    validation_y: list[list[float]],
    *,
    epochs: int,
) -> tuple[dict[str, Any], tuple[TrainingEpoch, ...]]:
    """Train an affine transition baseline with deterministic batch descent."""
    output_width = len(train_y[0])
    input_width = len(train_x[0])
    weights = [[0.0] * input_width for _ in range(output_width)]
    bias = [0.0] * output_width
    curve: list[TrainingEpoch] = []
    learning_rate = 0.08
    for epoch in range(1, epochs + 1):
        grad_w = [[0.0] * input_width for _ in range(output_width)]
        grad_b = [0.0] * output_width
        for row, target in zip(train_x, train_y, strict=True):
            prediction = [
                bias[output]
                + sum(weights[output][index] * value for index, value in enumerate(row))
                for output in range(output_width)
            ]
            for output in range(output_width):
                error = prediction[output] - target[output]
                grad_b[output] += error
                for index, value in enumerate(row):
                    grad_w[output][index] += error * value
        scale = 2.0 / (len(train_x) * output_width)
        for output in range(output_width):
            bias[output] -= learning_rate * scale * grad_b[output]
            for index in range(input_width):
                weights[output][index] -= learning_rate * scale * grad_w[output][index]
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            model = {"weights": weights, "bias": bias}
            curve.append(
                TrainingEpoch(
                    epoch=epoch,
                    training_loss=mse(
                        [linear_predict(model, row) for row in train_x],
                        train_y,
                    ),
                    validation_loss=mse(
                        [linear_predict(model, row) for row in validation_x],
                        validation_y,
                    ),
                )
            )
    return {"kind": "linear", "weights": weights, "bias": bias}, tuple(curve)


def mlp_predict(model: dict[str, Any], row: list[float]) -> list[float]:
    """Predict one normalized target with a single-hidden-layer MLP."""
    hidden = [
        math.tanh(
            model["b1"][unit]
            + sum(model["w1"][unit][index] * value for index, value in enumerate(row))
        )
        for unit in range(len(model["b1"]))
    ]
    return [
        model["b2"][output]
        + sum(model["w2"][output][unit] * value for unit, value in enumerate(hidden))
        for output in range(len(model["b2"]))
    ]


def train_mlp(
    train_x: list[list[float]],
    train_y: list[list[float]],
    validation_x: list[list[float]],
    validation_y: list[list[float]],
    *,
    seed: int,
    epochs: int,
    hidden_width: int,
) -> tuple[dict[str, Any], tuple[TrainingEpoch, ...]]:
    """Train a compact nonlinear transition model with deterministic SGD."""
    rng = random.Random(seed)
    input_width = len(train_x[0])
    output_width = len(train_y[0])
    scale = 1.0 / math.sqrt(input_width)
    model: dict[str, Any] = {
        "kind": "mlp",
        "w1": [
            [rng.uniform(-scale, scale) for _ in range(input_width)]
            for _ in range(hidden_width)
        ],
        "b1": [0.0] * hidden_width,
        "w2": [
            [rng.uniform(-scale, scale) for _ in range(hidden_width)]
            for _ in range(output_width)
        ],
        "b2": [0.0] * output_width,
    }
    learning_rate = 0.012
    order = list(range(len(train_x)))
    curve: list[TrainingEpoch] = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        for row_index in order:
            row = train_x[row_index]
            target = train_y[row_index]
            hidden_pre = [
                model["b1"][unit]
                + sum(
                    model["w1"][unit][index] * value for index, value in enumerate(row)
                )
                for unit in range(hidden_width)
            ]
            hidden = [math.tanh(value) for value in hidden_pre]
            prediction = [
                model["b2"][output]
                + sum(
                    model["w2"][output][unit] * hidden[unit]
                    for unit in range(hidden_width)
                )
                for output in range(output_width)
            ]
            output_error = [
                prediction[output] - target[output] for output in range(output_width)
            ]
            hidden_error = [
                (1.0 - hidden[unit] ** 2)
                * sum(
                    output_error[output] * model["w2"][output][unit]
                    for output in range(output_width)
                )
                for unit in range(hidden_width)
            ]
            for output in range(output_width):
                for unit in range(hidden_width):
                    model["w2"][output][unit] -= (
                        learning_rate * output_error[output] * hidden[unit]
                    )
                model["b2"][output] -= learning_rate * output_error[output]
            for unit in range(hidden_width):
                for index, value in enumerate(row):
                    model["w1"][unit][index] -= (
                        learning_rate * hidden_error[unit] * value
                    )
                model["b1"][unit] -= learning_rate * hidden_error[unit]
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            curve.append(
                TrainingEpoch(
                    epoch=epoch,
                    training_loss=mse(
                        [mlp_predict(model, row) for row in train_x],
                        train_y,
                    ),
                    validation_loss=mse(
                        [mlp_predict(model, row) for row in validation_x],
                        validation_y,
                    ),
                )
            )
    return model, tuple(curve)


def attach_normalization(
    model: dict[str, Any],
    x_mean: list[float],
    x_std: list[float],
    y_mean: list[float],
    y_std: list[float],
) -> dict[str, Any]:
    """Attach immutable training normalization to a model artifact."""
    return {
        **model,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def predict_state(
    model: dict[str, Any],
    state: tuple[float, float],
    torque: float,
    mass: float,
    damping: float,
) -> tuple[float, float]:
    """Predict one physical state from a serialized model artifact."""
    raw = features(state, torque, mass, damping)
    row = [
        (value - model["x_mean"][index]) / model["x_std"][index]
        for index, value in enumerate(raw)
    ]
    normalized = (
        linear_predict(model, row)
        if model["kind"] == "linear"
        else mlp_predict(model, row)
    )
    prediction = [
        normalized[index] * model["y_std"][index] + model["y_mean"][index]
        for index in range(len(normalized))
    ]
    norm = math.hypot(prediction[0], prediction[1])
    sin_theta = prediction[0] / norm if norm > 1e-9 else 0.0
    cos_theta = prediction[1] / norm if norm > 1e-9 else 1.0
    return math.atan2(sin_theta, cos_theta), max(
        -MAX_SPEED,
        min(MAX_SPEED, prediction[2]),
    )


def evaluate_slice(
    model: dict[str, Any],
    *,
    slice_id: str,
    mass: float,
    damping: float,
    seed: int,
    episodes: int = 20,
    horizon: int = 25,
) -> tuple[float, float, int]:
    """Measure one-step and open-loop rollout RMSE on reference dynamics."""
    rng = random.Random(seed)
    one_step_squared = 0.0
    rollout_squared = 0.0
    values = 0
    for _ in range(episodes):
        reference = (rng.uniform(-math.pi, math.pi), rng.uniform(-2.0, 2.0))
        predicted = reference
        for _ in range(horizon):
            torque = rng.uniform(-2.0, 2.0)
            reference_next = reference_step(
                reference,
                torque,
                mass=mass,
                damping=damping,
            )
            one_step = predict_state(
                model,
                reference,
                torque,
                mass,
                damping,
            )
            predicted = predict_state(
                model,
                predicted,
                torque,
                mass,
                damping,
            )
            for observed, expected in zip(
                targets(one_step),
                targets(reference_next),
                strict=True,
            ):
                one_step_squared += (observed - expected) ** 2
            for observed, expected in zip(
                targets(predicted),
                targets(reference_next),
                strict=True,
            ):
                rollout_squared += (observed - expected) ** 2
            values += 3
            reference = reference_next
    return (
        math.sqrt(one_step_squared / values),
        math.sqrt(rollout_squared / values),
        values // 3,
    )


def parameter_count(model: dict[str, Any]) -> int:
    """Count trainable scalar parameters in a serialized model."""
    if model["kind"] == "linear":
        return len(model["bias"]) + sum(len(row) for row in model["weights"])
    return (
        len(model["b1"])
        + len(model["b2"])
        + sum(len(row) for row in model["w1"])
        + sum(len(row) for row in model["w2"])
    )


def peak_memory_mb() -> float:
    """Return peak resident memory in MiB across macOS and Linux semantics."""
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return maximum / divisor


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact without overwriting prior evidence."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
        stream.write("\n")


def main() -> None:
    """Generate data, train both models, evaluate slices, and emit evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=60)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = arguments.output_dir / "pendulum-trajectories.jsonl"
    train_x, train_y, validation_x, validation_y = generate_dataset(
        dataset_path,
        seed=arguments.seed,
        episodes=arguments.episodes,
        steps=arguments.steps,
    )
    ingestion = ingest_dataset_source(
        dataset_plan(dataset_path.resolve().as_uri()),
        dataset_path,
    )
    write_json(
        arguments.output_dir / "dataset-bundle.json",
        json.loads(ingestion.bundle.to_json()),
    )
    write_json(
        arguments.output_dir / "dataset-ingestion-report.json",
        json.loads(ingestion.report.to_json()),
    )

    x_mean, x_std = moments(train_x)
    y_mean, y_std = moments(train_y)
    train_x_normalized = normalize(train_x, x_mean, x_std)
    train_y_normalized = normalize(train_y, y_mean, y_std)
    validation_x_normalized = normalize(validation_x, x_mean, x_std)
    validation_y_normalized = normalize(validation_y, y_mean, y_std)

    start = time.perf_counter()
    baseline, baseline_curve = train_linear(
        train_x_normalized,
        train_y_normalized,
        validation_x_normalized,
        validation_y_normalized,
        epochs=arguments.epochs,
    )
    candidate, candidate_curve = train_mlp(
        train_x_normalized,
        train_y_normalized,
        validation_x_normalized,
        validation_y_normalized,
        seed=arguments.seed,
        epochs=arguments.epochs,
        hidden_width=16,
    )
    duration_seconds = time.perf_counter() - start
    baseline = attach_normalization(baseline, x_mean, x_std, y_mean, y_std)
    candidate = attach_normalization(candidate, x_mean, x_std, y_mean, y_std)
    baseline_json = json.dumps(baseline, separators=(",", ":"), sort_keys=True)
    candidate_json = json.dumps(candidate, separators=(",", ":"), sort_keys=True)
    baseline_digest = sha256_digest(baseline_json)
    candidate_digest = sha256_digest(candidate_json)
    write_json(arguments.output_dir / "baseline-linear-model.json", baseline)
    write_json(arguments.output_dir / "candidate-mlp-model.json", candidate)

    slice_specs = (
        ("nominal", 1.0, 0.075, True),
        ("heavy-payload", 1.2, 0.075, True),
        ("high-friction", 1.0, 0.12, True),
        ("compound-edge", 1.2, 0.12, True),
        ("out-of-support", 1.45, 0.18, False),
    )
    baseline_slices: list[WorldModelEvaluationSlice] = []
    candidate_slices: list[WorldModelEvaluationSlice] = []
    for index, (slice_id, mass, damping, in_support) in enumerate(slice_specs):
        baseline_one, baseline_rollout, count = evaluate_slice(
            baseline,
            slice_id=slice_id,
            mass=mass,
            damping=damping,
            seed=arguments.seed + index,
        )
        candidate_one, candidate_rollout, _ = evaluate_slice(
            candidate,
            slice_id=slice_id,
            mass=mass,
            damping=damping,
            seed=arguments.seed + index,
        )
        threshold = 0.6 if in_support else 0.25
        baseline_slices.append(
            WorldModelEvaluationSlice(
                slice_id=slice_id,
                sample_count=count,
                horizon_steps=25,
                one_step_rmse=baseline_one,
                rollout_rmse=baseline_rollout,
                rollout_rmse_threshold=threshold,
                in_training_support=in_support,
            )
        )
        candidate_slices.append(
            WorldModelEvaluationSlice(
                slice_id=slice_id,
                sample_count=count,
                horizon_steps=25,
                one_step_rmse=candidate_one,
                rollout_rmse=candidate_rollout,
                rollout_rmse_threshold=threshold,
                in_training_support=in_support,
            )
        )

    baseline_rollout_rmse = sum(
        item.rollout_rmse for item in baseline_slices[:-1]
    ) / len(baseline_slices[:-1])
    candidate_rollout_rmse = sum(
        item.rollout_rmse for item in candidate_slices[:-1]
    ) / len(candidate_slices[:-1])
    training_plan = WorldModelTrainingPlan(
        schema_version=WORLD_MODEL_TRAINING_PLAN_SCHEMA_VERSION,
        plan_id="pendulum-transition-world-model",
        plan_version="1",
        dataset_content_digest=ingestion.bundle.content_digest(),
        model_family="single-hidden-layer-mlp",
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        compute_backend="python-stdlib",
        accelerator="cpu",
        random_seed=arguments.seed,
        epochs=arguments.epochs,
        evaluation_horizon_steps=25,
        limitations=(
            "Reference dynamics are simulated and do not establish physical transport.",
            "The compact MLP is an integration example, not a frontier world model.",
        ),
    )
    report = WorldModelTrainingReport(
        schema_version=WORLD_MODEL_TRAINING_REPORT_SCHEMA_VERSION,
        plan_content_digest=training_plan.content_digest(),
        dataset_content_digest=training_plan.dataset_content_digest,
        model_content_digest=candidate_digest,
        compute_backend="python-stdlib",
        accelerator="cpu",
        duration_seconds=duration_seconds,
        peak_memory_mb=peak_memory_mb(),
        parameter_count=parameter_count(candidate),
        epochs=candidate_curve,
        evaluation_slices=tuple(candidate_slices),
        baseline_model_content_digest=baseline_digest,
        baseline_rollout_rmse=baseline_rollout_rmse,
        candidate_rollout_rmse=candidate_rollout_rmse,
        disposition=WorldModelTrainingDisposition.NEEDS_IMPROVEMENT,
        limitations=(
            "The out-of-support slice is discovery-only.",
            "Hardware-in-the-loop or physical anchors are required before "
            "transport claims.",
        ),
    )
    write_json(
        arguments.output_dir / "world-model-training-plan.json",
        json.loads(training_plan.to_json()),
    )
    write_json(
        arguments.output_dir / "world-model-training-report.json",
        json.loads(report.to_json()),
    )
    write_json(
        arguments.output_dir / "workflow-summary.json",
        {
            "baseline_curve": [asdict(item) for item in baseline_curve],
            "candidate_curve": [asdict(item) for item in candidate_curve],
            "baseline_rollout_rmse": baseline_rollout_rmse,
            "candidate_rollout_rmse": candidate_rollout_rmse,
            "rollout_error_reduction": report.rollout_error_reduction,
            "duration_seconds": duration_seconds,
            "train_rows": len(train_x),
            "validation_rows": len(validation_x),
        },
    )
    print(report.to_json())


if __name__ == "__main__":
    main()
