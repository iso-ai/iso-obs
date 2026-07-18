"""Tests for deterministic paired inference across repeated runs."""

from __future__ import annotations

import math

import pytest

from iso_obs.divergence import PairingQuality
from iso_obs.replication import (
    PairedAnalysisConfig,
    PairedObservation,
    RandomizationMethod,
    analyze_paired_effect,
)


def observation(
    pair_id: str,
    difference: float,
    *,
    quality: PairingQuality = PairingQuality.EXACT_REPLAY,
) -> PairedObservation:
    """Build a paired observation with a requested difference."""
    return PairedObservation(
        pair_id=pair_id,
        baseline=10.0,
        candidate=10.0 + difference,
        pairing_quality=quality,
    )


def config(**overrides: int | float) -> PairedAnalysisConfig:
    """Build a fast deterministic test configuration."""
    values: dict[str, int | float] = {
        "bootstrap_resamples": 1_000,
        "randomization_resamples": 1_000,
        "random_seed": 42,
    }
    values.update(overrides)
    return PairedAnalysisConfig(**values)  # type: ignore[arg-type]


def test_zero_effect_is_reported_without_inventing_standardization() -> None:
    report = analyze_paired_effect(
        [observation(f"seed-{seed}", 0.0) for seed in range(4)],
        config=config(),
    )

    assert report.mean_difference == 0.0
    assert report.median_difference == 0.0
    assert report.sample_standard_deviation == 0.0
    assert report.standardized_mean_difference is None
    assert report.confidence_interval.lower == 0.0
    assert report.confidence_interval.upper == 0.0
    assert report.randomization_p_value == 1.0


def test_exact_sign_flip_test_and_effect_estimates() -> None:
    report = analyze_paired_effect(
        [
            observation("seed-1", 1.0),
            observation("seed-2", 2.0),
            observation("seed-3", 3.0),
            observation("seed-4", 4.0),
        ],
        config=config(),
    )

    assert report.pair_count == 4
    assert report.mean_difference == 2.5
    assert report.median_difference == 2.5
    assert report.sample_standard_deviation == pytest.approx(1.2909944487)
    assert report.standardized_mean_difference == pytest.approx(1.9364916731)
    assert report.randomization_method is RandomizationMethod.EXACT
    assert report.randomization_samples == 16
    assert report.randomization_p_value == 0.125
    assert report.confidence_interval.lower <= report.mean_difference
    assert report.confidence_interval.upper >= report.mean_difference


def test_input_order_does_not_change_seeded_report() -> None:
    observations = [
        observation("seed-c", 3.0),
        observation("seed-a", 1.0),
        observation("seed-b", 2.0),
    ]

    forward = analyze_paired_effect(observations, config=config())
    reverse = analyze_paired_effect(
        list(reversed(observations)),
        config=config(),
    )

    assert forward == reverse


def test_swapping_baseline_candidate_reverses_effect_not_p_value() -> None:
    observations = [
        observation("seed-1", 1.0),
        observation("seed-2", 2.0),
        observation("seed-3", 3.0),
    ]
    swapped = [
        PairedObservation(
            pair_id=item.pair_id,
            baseline=item.candidate,
            candidate=item.baseline,
            pairing_quality=item.pairing_quality,
        )
        for item in observations
    ]

    forward = analyze_paired_effect(observations, config=config())
    reverse = analyze_paired_effect(swapped, config=config())

    assert reverse.mean_difference == -forward.mean_difference
    assert reverse.median_difference == -forward.median_difference
    assert reverse.standardized_mean_difference == (
        -forward.standardized_mean_difference
    )
    assert reverse.confidence_interval.lower == pytest.approx(
        -forward.confidence_interval.upper
    )
    assert reverse.confidence_interval.upper == pytest.approx(
        -forward.confidence_interval.lower
    )
    assert reverse.randomization_p_value == forward.randomization_p_value


def test_large_sample_uses_deterministic_monte_carlo_with_correction() -> None:
    observations = [
        observation(f"seed-{index:02d}", 0.5 + index / 100.0) for index in range(17)
    ]

    first = analyze_paired_effect(observations, config=config())
    second = analyze_paired_effect(observations, config=config())

    assert first == second
    assert first.randomization_method is RandomizationMethod.MONTE_CARLO
    assert first.randomization_samples == 1_000
    assert first.randomization_p_value >= 1 / 1_001


def test_pairing_quality_counts_are_explicit() -> None:
    report = analyze_paired_effect(
        [
            observation("exact", 1.0),
            observation(
                "seed-matched",
                1.5,
                quality=PairingQuality.SEED_MATCHED,
            ),
        ],
        config=config(),
    )

    assert report.pairing_quality_counts == (
        (PairingQuality.EXACT_REPLAY, 1),
        (PairingQuality.SEED_MATCHED, 1),
    )


@pytest.mark.parametrize(
    "quality",
    [
        PairingQuality.CONFIGURATION_MATCHED,
        PairingQuality.UNPAIRED,
    ],
)
def test_weakly_paired_observation_is_rejected(
    quality: PairingQuality,
) -> None:
    with pytest.raises(ValueError, match="paired inference requires"):
        observation("weak-pair", 1.0, quality=quality)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_observation_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        PairedObservation(
            pair_id="seed-1",
            baseline=value,
            candidate=1.0,
            pairing_quality=PairingQuality.EXACT_REPLAY,
        )


def test_duplicate_pair_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate pair IDs"):
        analyze_paired_effect(
            [
                observation("seed-1", 1.0),
                observation("seed-1", 2.0),
            ],
            config=config(),
        )


def test_fewer_than_two_pairs_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        analyze_paired_effect([observation("seed-1", 1.0)], config=config())


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("confidence_level", 1.0, "between zero and one"),
        ("bootstrap_resamples", 999, "at least 1000"),
        ("randomization_resamples", 999, "at least 1000"),
        ("exact_randomization_max_pairs", 21, "between 1 and 20"),
    ],
)
def test_invalid_analysis_configuration_is_rejected(
    name: str,
    value: int | float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config(**{name: value})


def test_report_states_interpretive_limitations() -> None:
    report = analyze_paired_effect(
        [observation("seed-1", 1.0), observation("seed-2", 2.0)],
        config=config(),
    )

    assert len(report.limitations) == 4
    assert "exchangeable under sign reversal" in report.limitations[1]
    assert "does not establish causality" in report.limitations[1]
    assert "multiplicity control" in report.limitations[2]
    assert "fewer than 10 pairs" in report.limitations[3]


def test_string_pairing_quality_is_canonicalized() -> None:
    item = PairedObservation(
        pair_id="seed-1",
        baseline=1,
        candidate=2,
        pairing_quality="exact_replay",  # type: ignore[arg-type]
    )

    assert item.baseline == 1.0
    assert item.candidate == 2.0
    assert item.pairing_quality is PairingQuality.EXACT_REPLAY


@pytest.mark.parametrize(
    "name",
    [
        "bootstrap_resamples",
        "randomization_resamples",
        "exact_randomization_max_pairs",
        "random_seed",
    ],
)
def test_non_integer_configuration_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="must be integers"):
        config(**{name: 1_000.5})
