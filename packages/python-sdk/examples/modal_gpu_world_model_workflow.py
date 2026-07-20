# mypy: ignore-errors
"""Train and evaluate a trajectory world model on a customer-hosted Modal GPU.

The remote function owns only generated simulator trajectories and returns a
checkpoint plus bounded training observations. The caller constructs the
versioned Iso evidence locally, so Modal never receives Studio credentials.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "iso-world-model-gpu-e2e"
DATASET_SPEC = {
    "generator": "declared-pendulum-reference-dynamics-v2",
    "seed": 20260719,
    "training_transitions": 131_072,
    "validation_transitions": 16_384,
    "mass_range": [0.8, 1.2],
    "damping_range": [0.03, 0.12],
    "torque_range": [-2.0, 2.0],
    "dt_seconds": 0.05,
    "evidence_scope": "simulator-only",
}
FEATURE_NAMES = (
    "sin_theta",
    "cos_theta",
    "angular_velocity",
    "torque",
    "payload_mass",
    "joint_damping",
)
TARGET_NAMES = ("delta_sin_theta", "delta_cos_theta", "delta_angular_velocity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .pip_install("torch==2.4.1")
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    cpu=2.0,
    memory=8192,
    timeout=900,
    retries=0,
    scaledown_window=30,
)
def train_on_l4() -> dict[str, Any]:
    """Train one residual dynamics network and return auditable observations."""
    import math
    import time

    import torch

    torch.manual_seed(DATASET_SPEC["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Modal scheduled the function without a CUDA device")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    def generate(count: int, seed_offset: int) -> tuple[Any, Any]:
        """Generate declared reference transitions directly on the GPU."""
        generator = torch.Generator(device=device)
        generator.manual_seed(DATASET_SPEC["seed"] + seed_offset)
        theta = (
            torch.rand(count, generator=generator, device=device) * 2.0 - 1.0
        ) * math.pi
        omega = torch.rand(count, generator=generator, device=device) * 4.0 - 2.0
        torque = torch.rand(count, generator=generator, device=device) * 4.0 - 2.0
        mass = torch.rand(count, generator=generator, device=device) * 0.4 + 0.8
        damping = torch.rand(count, generator=generator, device=device) * 0.09 + 0.03
        acceleration = (
            3.0 * 9.81 / 2.0 * torch.sin(theta) + 3.0 * torque / mass - damping * omega
        )
        next_omega = torch.clamp(omega + 0.05 * acceleration, -8.0, 8.0)
        next_theta = theta + 0.05 * next_omega
        features = torch.stack(
            (torch.sin(theta), torch.cos(theta), omega, torque, mass, damping),
            dim=1,
        )
        current = torch.stack(
            (torch.sin(theta), torch.cos(theta), omega),
            dim=1,
        )
        target = torch.stack(
            (torch.sin(next_theta), torch.cos(next_theta), next_omega),
            dim=1,
        )
        return features, target - current

    class ResidualDynamics(torch.nn.Module):
        """Residual transition network used for the GPU integration test."""

        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(6, 512),
                torch.nn.SiLU(),
                torch.nn.Linear(512, 512),
                torch.nn.SiLU(),
                torch.nn.Linear(512, 512),
                torch.nn.SiLU(),
                torch.nn.Linear(512, 3),
            )

        def forward(self, inputs: Any) -> Any:
            """Predict the next-state residual."""
            return self.network(inputs)

    train_x, train_y = generate(DATASET_SPEC["training_transitions"], 0)
    validation_x, validation_y = generate(
        DATASET_SPEC["validation_transitions"],
        1,
    )
    feature_mean = train_x.mean(dim=0)
    feature_scale = train_x.std(dim=0).clamp_min(1e-6)
    target_mean = train_y.mean(dim=0)
    target_scale = train_y.std(dim=0).clamp_min(1e-6)
    normalized_train_x = (train_x - feature_mean) / feature_scale
    normalized_train_y = (train_y - target_mean) / target_scale
    normalized_validation_x = (validation_x - feature_mean) / feature_scale
    normalized_validation_y = (validation_y - target_mean) / target_scale

    model = ResidualDynamics().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    batch_size = 4096
    epoch_rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, 31):
        permutation = torch.randperm(
            normalized_train_x.shape[0],
            generator=torch.Generator(device=device).manual_seed(
                DATASET_SPEC["seed"] + epoch
            ),
            device=device,
        )
        model.train()
        for start in range(0, normalized_train_x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            prediction = model(normalized_train_x[indices])
            loss = torch.nn.functional.mse_loss(
                prediction,
                normalized_train_y[indices],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == 30:
            model.eval()
            with torch.no_grad():
                training_loss = torch.nn.functional.mse_loss(
                    model(normalized_train_x),
                    normalized_train_y,
                )
                validation_loss = torch.nn.functional.mse_loss(
                    model(normalized_validation_x),
                    normalized_validation_y,
                )
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "training_loss": float(training_loss.item()),
                    "validation_loss": float(validation_loss.item()),
                }
            )

    def predict(current: Any, torque: Any, mass: float, damping: float) -> Any:
        """Advance the learned residual transition model."""
        feature_rows = torch.stack(
            (
                torch.sin(current[:, 0]),
                torch.cos(current[:, 0]),
                current[:, 1],
                torque,
                torch.full_like(torque, mass),
                torch.full_like(torque, damping),
            ),
            dim=1,
        )
        normalized = (feature_rows - feature_mean) / feature_scale
        delta = model(normalized) * target_scale + target_mean
        encoded = torch.stack(
            (
                torch.sin(current[:, 0]),
                torch.cos(current[:, 0]),
                current[:, 1],
            ),
            dim=1,
        )
        next_encoded = encoded + delta
        next_theta = torch.atan2(next_encoded[:, 0], next_encoded[:, 1])
        return torch.stack((next_theta, next_encoded[:, 2]), dim=1)

    def reference(current: Any, torque: Any, mass: float, damping: float) -> Any:
        """Advance the declared reference dynamics."""
        acceleration = (
            3.0 * 9.81 / 2.0 * torch.sin(current[:, 0])
            + 3.0 * torque / mass
            - damping * current[:, 1]
        )
        next_omega = torch.clamp(current[:, 1] + 0.05 * acceleration, -8.0, 8.0)
        next_theta = current[:, 0] + 0.05 * next_omega
        return torch.stack((next_theta, next_omega), dim=1)

    def encoded_state_error(predicted: Any, expected: Any) -> Any:
        """Measure state error without an angular wrap discontinuity."""
        predicted_encoded = torch.stack(
            (
                torch.sin(predicted[:, 0]),
                torch.cos(predicted[:, 0]),
                predicted[:, 1],
            ),
            dim=1,
        )
        expected_encoded = torch.stack(
            (
                torch.sin(expected[:, 0]),
                torch.cos(expected[:, 0]),
                expected[:, 1],
            ),
            dim=1,
        )
        return torch.mean((predicted_encoded - expected_encoded) ** 2)

    slices = (
        ("nominal", 1.0, 0.075, True, 0.15),
        ("heavy-payload", 1.2, 0.075, True, 0.15),
        ("high-friction", 1.0, 0.12, True, 0.15),
        ("compound-edge", 1.2, 0.12, True, 0.15),
        ("out-of-support", 1.45, 0.18, False, 0.25),
    )
    evaluation_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for index, (slice_id, mass, damping, in_support, threshold) in enumerate(
            slices
        ):
            generator = torch.Generator(device=device)
            generator.manual_seed(DATASET_SPEC["seed"] + 100 + index)
            state = torch.stack(
                (
                    (torch.rand(128, generator=generator, device=device) * 2 - 1)
                    * math.pi,
                    torch.rand(128, generator=generator, device=device) * 4 - 2,
                ),
                dim=1,
            )
            predicted = state.clone()
            expected = state.clone()
            squared_one_step = 0.0
            squared_rollout = 0.0
            for _ in range(25):
                torque = (
                    torch.rand(
                        128,
                        generator=generator,
                        device=device,
                    )
                    * 4
                    - 2
                )
                expected_next = reference(expected, torque, mass, damping)
                predicted_next = predict(predicted, torque, mass, damping)
                reference_from_predicted = reference(
                    predicted,
                    torque,
                    mass,
                    damping,
                )
                squared_one_step += float(
                    encoded_state_error(
                        predicted_next,
                        reference_from_predicted,
                    )
                )
                squared_rollout += float(
                    encoded_state_error(predicted_next, expected_next)
                )
                expected = expected_next
                predicted = predicted_next
            evaluation_rows.append(
                {
                    "slice_id": slice_id,
                    "sample_count": 128 * 25,
                    "horizon_steps": 25,
                    "one_step_rmse": math.sqrt(squared_one_step / 25),
                    "rollout_rmse": math.sqrt(squared_rollout / 25),
                    "rollout_rmse_threshold": threshold,
                    "in_training_support": in_support,
                }
            )

    duration_seconds = time.perf_counter() - started
    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "dataset_spec": DATASET_SPEC,
    }
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    return {
        "accelerator": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "duration_seconds": duration_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epoch_rows,
        "evaluation_slices": evaluation_rows,
        "checkpoint_b64": base64.b64encode(stream.getvalue()).decode("ascii"),
    }


@app.local_entrypoint()
def main(
    output_dir: str,
    baseline_rollout_rmse: float = 0.1990567,
) -> None:
    """Run the capped GPU job and write versioned local evidence artifacts."""
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

    artifact_directory = Path(output_dir)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    result = train_on_l4.remote()
    checkpoint = base64.b64decode(result.pop("checkpoint_b64"))
    checkpoint_path = artifact_directory / "modal-l4-world-model.pt"
    checkpoint_path.write_bytes(checkpoint)

    dataset_manifest = {
        "schema_version": "iso-obs.generated-dataset-manifest.v1",
        **DATASET_SPEC,
        "limitations": [
            "Trajectories are generated from declared simulator dynamics.",
            "No physical measurements or sim-to-real anchors are included.",
        ],
    }
    dataset_json = json.dumps(
        dataset_manifest,
        separators=(",", ":"),
        sort_keys=True,
    )
    dataset_digest = sha256_digest(dataset_json)
    model_digest = sha256_digest(checkpoint)
    plan = WorldModelTrainingPlan(
        schema_version=WORLD_MODEL_TRAINING_PLAN_SCHEMA_VERSION,
        plan_id="modal-l4-pendulum-world-model",
        plan_version="1",
        dataset_content_digest=dataset_digest,
        model_family="three-hidden-layer-residual-mlp",
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
        compute_backend="modal-pytorch",
        accelerator=str(result["accelerator"]),
        random_seed=DATASET_SPEC["seed"],
        epochs=30,
        evaluation_horizon_steps=25,
        limitations=(
            "Customer-hosted Modal GPU; Iso does not custody Modal credentials.",
            "Simulator-only training does not establish physical transport.",
        ),
    )
    slices = tuple(
        WorldModelEvaluationSlice(**row) for row in result["evaluation_slices"]
    )
    in_support_rollout = sum(
        item.rollout_rmse for item in slices if item.in_training_support
    ) / sum(item.in_training_support for item in slices)
    disposition = (
        WorldModelTrainingDisposition.ACCEPTABLE_FOR_DECLARED_EVALUATION
        if all(item.passed and item.in_training_support for item in slices)
        else WorldModelTrainingDisposition.NEEDS_IMPROVEMENT
    )
    report = WorldModelTrainingReport(
        schema_version=WORLD_MODEL_TRAINING_REPORT_SCHEMA_VERSION,
        plan_content_digest=plan.content_digest(),
        dataset_content_digest=dataset_digest,
        model_content_digest=model_digest,
        compute_backend="modal-pytorch",
        accelerator=str(result["accelerator"]),
        duration_seconds=float(result["duration_seconds"]),
        peak_memory_mb=float(result["peak_memory_mb"]),
        parameter_count=int(result["parameter_count"]),
        epochs=tuple(TrainingEpoch(**row) for row in result["epochs"]),
        evaluation_slices=slices,
        baseline_model_content_digest=None,
        baseline_rollout_rmse=baseline_rollout_rmse,
        candidate_rollout_rmse=in_support_rollout,
        disposition=disposition,
        limitations=(
            "The CPU baseline used the same declared dynamics but a smaller dataset.",
            "Cross-hardware differences are associative, not causal.",
            "The out-of-support slice remains discovery evidence.",
        ),
    )
    artifacts = {
        "generated-dataset-manifest.json": dataset_manifest,
        "world-model-training-plan.json": json.loads(plan.to_json()),
        "world-model-training-report.json": json.loads(report.to_json()),
        "workflow-summary.json": {
            **result,
            "dataset_content_digest": dataset_digest,
            "model_content_digest": model_digest,
            "plan_content_digest": plan.content_digest(),
            "candidate_rollout_rmse": in_support_rollout,
            "rollout_error_reduction": report.rollout_error_reduction,
        },
    }
    for name, payload in artifacts.items():
        (artifact_directory / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(report.to_json())
