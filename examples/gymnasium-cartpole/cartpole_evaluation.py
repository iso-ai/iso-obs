"""Evaluate Gymnasium CartPole policies under noise and action delay."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from collections.abc import Sequence

import gymnasium as gym  # type: ignore[import-not-found]

from iso_obs import ReliabilityClient
from iso_obs.run import RunContext


def choose_action(policy: str, observation: Sequence[float]) -> int:
    """Choose an action from a transparent baseline or candidate policy.

    Args:
        policy: Policy version label.
        observation: Cart position, velocity, angle, and angular velocity.

    Returns:
        Gymnasium discrete action.
    """
    _, cart_velocity, pole_angle, angular_velocity = observation
    if policy == "baseline":
        score = pole_angle + 0.45 * angular_velocity
    else:
        score = 1.35 * pole_angle + 0.12 * angular_velocity + 0.08 * cart_velocity
    return int(score > 0.0)


def perturb_observation(
    observation: Sequence[float],
    *,
    rng: random.Random,
    noise_std: float,
) -> list[float]:
    """Apply deterministic Gaussian sensor noise.

    Args:
        observation: Raw environment observation.
        rng: Seeded randomness source.
        noise_std: Standard deviation applied independently to each field.

    Returns:
        Perturbed observation values.
    """
    return [float(value) + rng.gauss(0.0, noise_std) for value in observation]


def evaluate_episode(
    client: ReliabilityClient,
    *,
    project: str,
    system_version: str,
    environment: str,
    scenario: str,
    policy: str,
    seed: int,
    noise_std: float,
    action_delay: int,
) -> dict[str, float | int | bool]:
    """Execute and trace one perturbed CartPole episode.

    Args:
        client: Authenticated Reliability Studio client.
        project: Reliability project name or ID.
        system_version: Registered policy-version label or ID.
        environment: Registered environment label or ID.
        scenario: Registered scenario label or ID.
        policy: Local policy implementation.
        seed: Gymnasium and perturbation seed.
        noise_std: Observation-noise standard deviation.
        action_delay: Number of steps before an emitted action is applied.

    Returns:
        Episode-level measurements.
    """
    env = gym.make("CartPole-v1")
    raw_observation, _ = env.reset(seed=seed)
    rng = random.Random(seed)
    delayed_actions: deque[int] = deque([0] * action_delay)
    total_reward = 0.0
    peak_angle = 0.0
    terminated = False
    truncated = False
    steps = 0

    try:
        with RunContext(
            client=client,
            project=project,
            system_version=system_version,
            environment=environment,
            scenario=scenario,
            seed=seed,
        ) as run:
            while not (terminated or truncated):
                observation = perturb_observation(
                    raw_observation,
                    rng=rng,
                    noise_std=noise_std,
                )
                started = time.perf_counter_ns()
                emitted_action = choose_action(policy, observation)
                latency_ms = (time.perf_counter_ns() - started) / 1_000_000
                delayed_actions.append(emitted_action)
                applied_action = delayed_actions.popleft()

                run.log_observation({"cartpole": observation})
                run.log_action(
                    {
                        "emitted": emitted_action,
                        "applied": applied_action,
                        "delay_steps": action_delay,
                    },
                    latency_ms=latency_ms,
                )

                raw_observation, reward, terminated, truncated, _ = env.step(
                    applied_action
                )
                steps += 1
                total_reward += float(reward)
                pole_angle = abs(float(raw_observation[2]))
                peak_angle = max(peak_angle, pole_angle)
                run.log_metric("reward", float(reward))
                run.log_metric("pole_angle_rad", pole_angle)
                run.log_metric(
                    "angle_safety_margin_rad",
                    max(0.0, 0.2095 - pole_angle),
                )

            success = bool(truncated and not terminated)
            run.log_metric("episode_return", total_reward)
            run.log_metric("episode_steps", float(steps))
            run.log_metric("success", float(success))
    finally:
        env.close()

    return {
        "seed": seed,
        "success": success,
        "steps": steps,
        "episode_return": total_reward,
        "peak_angle_rad": peak_angle,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argument sequence.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--system-version", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policy", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--action-delay", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested CartPole evaluation matrix.

    Args:
        argv: Optional explicit argument sequence.

    Returns:
        Zero after all episodes complete.
    """
    args = parse_args(argv)
    client = ReliabilityClient()
    results = [
        evaluate_episode(
            client,
            project=args.project,
            system_version=args.system_version,
            environment=args.environment,
            scenario=args.scenario,
            policy=args.policy,
            seed=seed,
            noise_std=args.noise_std,
            action_delay=args.action_delay,
        )
        for seed in args.seeds
    ]
    for result in results:
        print(json.dumps(result, sort_keys=True))
    successes = sum(bool(result["success"]) for result in results)
    print(
        json.dumps(
            {
                "summary": {
                    "episodes": len(results),
                    "success_rate": successes / len(results),
                }
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
