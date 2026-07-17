"""Evaluate warehouse-picking policies under controlled perturbations."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from iso_obs import ReliabilityClient
from iso_obs.run import RunContext


@dataclass(frozen=True, slots=True)
class Scenario:
    """Configuration for one deterministic warehouse-picking scenario."""

    name: str
    max_steps: int
    target_position: float
    object_mass_kg: float
    max_gripper_force_n: float
    observation_noise_std: float
    observation_delay_steps: int
    actuator_scale: float


@dataclass(frozen=True, slots=True)
class Observation:
    """Policy-visible state for one simulation step."""

    arm_position: float
    arm_velocity: float
    object_position: float
    object_secured: bool


@dataclass(frozen=True, slots=True)
class Action:
    """Command emitted by the picking policy."""

    acceleration: float
    gripper_force_n: float


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Terminal measurements for one evaluation episode."""

    success: bool
    steps: int
    peak_force_n: float
    max_position_error: float
    force_violation: bool


class PickingPolicy:
    """Simple controller with intentionally different robustness profiles."""

    def __init__(self, version: str) -> None:
        """Initialize the requested policy version.

        Args:
            version: Either ``"baseline"`` or ``"candidate"``.

        Raises:
            ValueError: If the version is unknown.
        """
        if version not in {"baseline", "candidate"}:
            raise ValueError(f"unknown policy version: {version}")
        self.version = version

    def act(self, observation: Observation, target: float) -> Action:
        """Produce a control action from the current observation.

        Args:
            observation: Noisy, possibly delayed policy observation.
            target: Desired object position.

        Returns:
            Acceleration and grip-force command.
        """
        error = target - observation.arm_position
        if self.version == "baseline":
            acceleration = 2.4 * error - 1.35 * observation.arm_velocity
            force = 12.5 + min(abs(error) * 2.0, 2.5)
        else:
            acceleration = 4.4 * error - 0.75 * observation.arm_velocity
            force = 15.5 + min(abs(error) * 4.0, 5.0)
        return Action(acceleration=acceleration, gripper_force_n=force)


class WarehouseSimulator:
    """Small deterministic dynamics model for reliability demonstrations."""

    def __init__(self, scenario: Scenario, seed: int) -> None:
        """Initialize simulator state and perturbation sources.

        Args:
            scenario: Scenario and perturbation configuration.
            seed: Seed controlling all stochastic behavior.
        """
        self.scenario = scenario
        self._rng = random.Random(seed)
        self._position = 0.0
        self._velocity = 0.0
        self._object_position = 0.0
        self._object_secured = False
        self._step = 0
        self._history: deque[Observation] = deque(
            maxlen=scenario.observation_delay_steps + 1
        )

    def observe(self) -> Observation:
        """Return the current noisy and delayed policy observation."""
        noisy = Observation(
            arm_position=self._position
            + self._rng.gauss(0.0, self.scenario.observation_noise_std),
            arm_velocity=self._velocity
            + self._rng.gauss(0.0, self.scenario.observation_noise_std),
            object_position=self._object_position,
            object_secured=self._object_secured,
        )
        self._history.append(noisy)
        if len(self._history) <= self.scenario.observation_delay_steps:
            return self._history[0]
        return self._history[-(self.scenario.observation_delay_steps + 1)]

    def step(self, action: Action) -> None:
        """Advance the simulated arm and object by one control step.

        Args:
            action: Policy command before actuator degradation.
        """
        dt = 0.05
        applied_acceleration = action.acceleration * self.scenario.actuator_scale
        self._velocity = 0.97 * self._velocity + applied_acceleration * dt
        self._position += self._velocity * dt
        grasp_distance = abs(self._position - self._object_position)
        if grasp_distance < 0.08 and action.gripper_force_n >= 10.0:
            self._object_secured = True
        if self._object_secured:
            self._object_position = self._position
        self._step += 1

    def state(self) -> dict[str, float | bool | int]:
        """Return privileged simulator state for failure analysis."""
        return {
            "step": self._step,
            "true_arm_position": self._position,
            "true_arm_velocity": self._velocity,
            "true_object_position": self._object_position,
            "object_secured": self._object_secured,
        }


def load_scenario(path: Path) -> Scenario:
    """Load a scenario document into the local simulation model.

    Args:
        path: JSON scenario path.

    Returns:
        Validated scenario values used by this example.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    perturbations = document["perturbations"]
    return Scenario(
        name=str(document["name"]),
        max_steps=int(document["max_steps"]),
        target_position=float(document["target_position"]),
        object_mass_kg=float(document["object_mass_kg"]),
        max_gripper_force_n=float(document["max_gripper_force_n"]),
        observation_noise_std=float(perturbations["observation_noise_std"]),
        observation_delay_steps=int(perturbations["observation_delay_steps"]),
        actuator_scale=float(perturbations["actuator_scale"]),
    )


def artifact_path(policy: str, scenario: str, seed: int) -> Path:
    """Return the local evidence-artifact path for an episode.

    Args:
        policy: Evaluated policy version.
        scenario: Evaluated scenario name.
        seed: Evaluation seed.

    Returns:
        Path under the SDK's local example state directory.
    """
    directory = Path(".iso-obs/examples/warehouse-picking")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{policy}-{scenario}-seed-{seed}.json"


def evaluate_episode(
    client: ReliabilityClient,
    *,
    project: str,
    system_version: str,
    environment: str,
    scenario_name: str,
    scenario: Scenario,
    policy_name: str,
    seed: int,
) -> EpisodeResult:
    """Execute, trace, and summarize one picking episode.

    Args:
        client: Authenticated Reliability Studio client.
        project: Project name or ID.
        system_version: Registered system-version label or ID.
        environment: Registered environment label or ID.
        scenario_name: Registered scenario label or ID.
        scenario: Local scenario configuration.
        policy_name: Local policy implementation to execute.
        seed: Deterministic episode seed.

    Returns:
        Terminal episode measurements.
    """
    simulator = WarehouseSimulator(scenario, seed)
    policy = PickingPolicy(policy_name)
    peak_force = 0.0
    max_error = 0.0
    force_violation = False
    success = False
    step = -1

    with RunContext(
        client=client,
        project=project,
        system_version=system_version,
        environment=environment,
        scenario=scenario_name,
        seed=seed,
    ) as run:
        for _step in range(scenario.max_steps):
            observation = simulator.observe()
            started = time.perf_counter_ns()
            action = policy.act(observation, scenario.target_position)
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            simulator.step(action)

            error = abs(
                scenario.target_position - simulator.state()["true_object_position"]
            )
            assert isinstance(error, float)
            peak_force = max(peak_force, action.gripper_force_n)
            max_error = max(max_error, error)
            violated = action.gripper_force_n > scenario.max_gripper_force_n
            force_violation = force_violation or violated

            run.log_observation(asdict(observation))
            run.log_action(asdict(action), latency_ms=latency_ms)
            run.log_state(simulator.state())
            run.log_metric("position_error", error)
            run.log_metric("gripper_force_n", action.gripper_force_n)
            run.log_metric("force_limit_violated", float(violated))

            secured = bool(simulator.state()["object_secured"])
            if secured and error < 0.06:
                success = True
                break

        result = EpisodeResult(
            success=success and not force_violation,
            steps=step + 1,
            peak_force_n=peak_force,
            max_position_error=max_error,
            force_violation=force_violation,
        )
        run.log_metric("success", float(result.success))
        run.log_metric("episode_steps", float(result.steps))

        evidence_path = artifact_path(policy_name, scenario.name, seed)
        evidence = {
            "policy": policy_name,
            "system_version": system_version,
            "scenario": asdict(scenario),
            "seed": seed,
            "result": asdict(result),
            "final_state": simulator.state(),
        }
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run.log_artifact(evidence_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argument sequence.

    Returns:
        Parsed command namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--system-version", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scenario-file", required=True, type=Path)
    parser.add_argument("--policy", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested evaluation matrix.

    Args:
        argv: Optional explicit argument sequence.

    Returns:
        Zero when all runs complete, otherwise a non-zero process code.
    """
    args = parse_args(argv)
    scenario = load_scenario(args.scenario_file)
    client = ReliabilityClient()
    results: list[EpisodeResult] = []
    for seed in args.seeds:
        result = evaluate_episode(
            client,
            project=args.project,
            system_version=args.system_version,
            environment=args.environment,
            scenario_name=args.scenario,
            scenario=scenario,
            policy_name=args.policy,
            seed=seed,
        )
        results.append(result)
        print(
            json.dumps(
                {"policy": args.policy, "seed": seed, **asdict(result)},
                sort_keys=True,
            )
        )
    success_rate = sum(result.success for result in results) / len(results)
    print(json.dumps({"summary": {"success_rate": success_rate}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
