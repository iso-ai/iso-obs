"""Reproducible paired inference across repeated evaluation runs.

This module summarizes one pre-specified scalar outcome across matched
baseline/candidate runs. It reports effect estimates and uncertainty rather
than treating statistical significance as the result. The paired sign-flip
test evaluates a sharp no-effect null; it does not establish causality or
correct for selecting an outcome after inspecting the data.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .divergence import PairingQuality
from .evidence import _canonical_json, sha256_digest


class RandomizationMethod(StrEnum):
    """Method used to estimate the paired sign-flip p-value."""

    EXACT = "exact_sign_flip"
    MONTE_CARLO = "monte_carlo_sign_flip"


@dataclass(frozen=True, slots=True)
class PairedObservation:
    """One matched baseline/candidate scalar outcome."""

    pair_id: str
    baseline: float
    candidate: float
    pairing_quality: PairingQuality

    def __post_init__(self) -> None:
        """Validate pair identity, values, and experimental control."""
        if not self.pair_id.strip():
            raise ValueError("pair_id must not be empty")
        baseline = float(self.baseline)
        candidate = float(self.candidate)
        quality = PairingQuality(self.pairing_quality)
        if not math.isfinite(baseline) or not math.isfinite(candidate):
            raise ValueError(f"pair '{self.pair_id}' contains non-finite values")
        if quality not in {
            PairingQuality.EXACT_REPLAY,
            PairingQuality.SEED_MATCHED,
        }:
            raise ValueError(
                f"pair '{self.pair_id}' has quality "
                f"{quality.value}; paired inference requires "
                "exact_replay or seed_matched runs"
            )
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "pairing_quality", quality)


@dataclass(frozen=True, slots=True)
class PairedAnalysisConfig:
    """Deterministic resampling configuration for paired inference."""

    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    randomization_resamples: int = 10_000
    exact_randomization_max_pairs: int = 16
    random_seed: int = 0

    def __post_init__(self) -> None:
        """Validate statistical configuration bounds."""
        if (
            not math.isfinite(self.confidence_level)
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be between zero and one")
        integer_fields = {
            "bootstrap_resamples": self.bootstrap_resamples,
            "randomization_resamples": self.randomization_resamples,
            "exact_randomization_max_pairs": self.exact_randomization_max_pairs,
            "random_seed": self.random_seed,
        }
        invalid_integer_fields = [
            name
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int)
        ]
        if invalid_integer_fields:
            names = ", ".join(invalid_integer_fields)
            raise ValueError(f"configuration fields must be integers: {names}")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("bootstrap_resamples must be at least 1000")
        if self.randomization_resamples < 1_000:
            raise ValueError("randomization_resamples must be at least 1000")
        if not 1 <= self.exact_randomization_max_pairs <= 20:
            raise ValueError("exact_randomization_max_pairs must be between 1 and 20")


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Percentile-bootstrap confidence interval for a paired mean."""

    level: float
    lower: float
    upper: float
    method: str
    resamples: int


@dataclass(frozen=True, slots=True)
class PairedEffectReport:
    """Effect, uncertainty, and diagnostics for one paired scalar outcome."""

    pair_count: int
    mean_difference: float
    median_difference: float
    sample_standard_deviation: float
    standardized_mean_difference: float | None
    confidence_interval: ConfidenceInterval
    randomization_p_value: float
    randomization_method: RandomizationMethod
    randomization_samples: int
    pairing_quality_counts: tuple[tuple[PairingQuality, int], ...]
    random_seed: int
    limitations: tuple[str, ...]
    report_schema_version: str = "iso-obs.paired-effect-report.v1"

    def to_json(self) -> str:
        """Serialize estimates and diagnostics to canonical JSON.

        Returns:
            Stable compact JSON with sorted object keys.
        """
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact paired-effect report content digest.

        Returns:
            A prefixed lowercase SHA-256 digest.
        """
        return sha256_digest(self.to_json())


def analyze_paired_effect(
    observations: Sequence[PairedObservation],
    *,
    config: PairedAnalysisConfig | None = None,
) -> PairedEffectReport:
    """Estimate a candidate-minus-baseline effect across matched runs.

    Observations are sorted by ``pair_id`` before every calculation, making the
    report invariant to caller iteration order. The confidence interval uses a
    deterministic percentile bootstrap over pairs. The two-sided randomization
    p-value uses all sign assignments for small samples and seeded Monte Carlo
    sign flips for larger samples.

    Args:
        observations: Matched baseline/candidate scalar outcomes.
        config: Optional deterministic resampling configuration.

    Returns:
        Paired effect estimates, uncertainty, randomization result, and
        interpretive limitations.

    Raises:
        ValueError: If fewer than two pairs or duplicate pair IDs are supplied.
    """
    resolved_config = config or PairedAnalysisConfig()
    ordered = tuple(sorted(observations, key=lambda item: item.pair_id))
    if len(ordered) < 2:
        raise ValueError("paired analysis requires at least two observations")
    pair_ids = [item.pair_id for item in ordered]
    duplicates = sorted(
        {pair_id for pair_id in pair_ids if pair_ids.count(pair_id) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate pair IDs: {', '.join(duplicates)}")

    differences = tuple(item.candidate - item.baseline for item in ordered)
    mean_difference = statistics.fmean(differences)
    median_difference = float(statistics.median(differences))
    sample_standard_deviation = statistics.stdev(differences)
    standardized = (
        mean_difference / sample_standard_deviation
        if sample_standard_deviation > 0.0
        else None
    )
    confidence_interval = _bootstrap_interval(
        differences,
        config=resolved_config,
    )
    (
        randomization_p_value,
        randomization_method,
        randomization_samples,
    ) = _randomization_test(
        differences,
        config=resolved_config,
    )
    quality_counts = Counter(item.pairing_quality for item in ordered)
    pairing_quality_counts = tuple(
        (quality, quality_counts[quality])
        for quality in PairingQuality
        if quality_counts[quality]
    )
    limitations = [
        "The interval quantifies repeated-pair sampling uncertainty; it "
        "does not include simulator, measurement, or model-form bias.",
        "The sign-flip test assumes paired differences are exchangeable under "
        "sign reversal and does not establish causality.",
        "This analysis assumes one pre-specified outcome. Exploratory analysis "
        "across multiple signals requires multiplicity control and independent "
        "validation.",
    ]
    if len(ordered) < 10:
        limitations.append(
            "The sample contains fewer than 10 pairs; percentile-bootstrap "
            "coverage may be unstable and should be interpreted cautiously."
        )
    return PairedEffectReport(
        pair_count=len(ordered),
        mean_difference=mean_difference,
        median_difference=median_difference,
        sample_standard_deviation=sample_standard_deviation,
        standardized_mean_difference=standardized,
        confidence_interval=confidence_interval,
        randomization_p_value=randomization_p_value,
        randomization_method=randomization_method,
        randomization_samples=randomization_samples,
        pairing_quality_counts=pairing_quality_counts,
        random_seed=resolved_config.random_seed,
        limitations=tuple(limitations),
    )


def _bootstrap_interval(
    differences: Sequence[float],
    *,
    config: PairedAnalysisConfig,
) -> ConfidenceInterval:
    """Calculate a deterministic percentile-bootstrap interval.

    Args:
        differences: Candidate-minus-baseline paired differences.
        config: Resampling configuration.

    Returns:
        Percentile confidence interval for the paired mean.
    """
    rng = random.Random(config.random_seed)
    sample_size = len(differences)
    means = sorted(
        statistics.fmean(
            differences[rng.randrange(sample_size)] for _ in range(sample_size)
        )
        for _ in range(config.bootstrap_resamples)
    )
    tail_probability = (1.0 - config.confidence_level) / 2.0
    return ConfidenceInterval(
        level=config.confidence_level,
        lower=_quantile(means, tail_probability),
        upper=_quantile(means, 1.0 - tail_probability),
        method="paired_percentile_bootstrap",
        resamples=config.bootstrap_resamples,
    )


def _randomization_test(
    differences: Sequence[float],
    *,
    config: PairedAnalysisConfig,
) -> tuple[float, RandomizationMethod, int]:
    """Calculate a two-sided paired sign-flip randomization p-value.

    Args:
        differences: Candidate-minus-baseline paired differences.
        config: Randomization configuration.

    Returns:
        P-value, method, and number of sampled sign assignments.
    """
    observed = abs(statistics.fmean(differences))
    tolerance = 1e-15
    if len(differences) <= config.exact_randomization_max_pairs:
        statistics_under_null = (
            abs(
                statistics.fmean(
                    sign * difference
                    for sign, difference in zip(
                        signs,
                        differences,
                        strict=True,
                    )
                )
            )
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        )
        extreme_count = 0
        sample_count = 0
        for statistic in statistics_under_null:
            sample_count += 1
            if statistic >= observed - tolerance:
                extreme_count += 1
        return (
            extreme_count / sample_count,
            RandomizationMethod.EXACT,
            sample_count,
        )

    rng = random.Random(config.random_seed)
    extreme_count = 0
    for _ in range(config.randomization_resamples):
        statistic = abs(
            statistics.fmean(
                difference if rng.getrandbits(1) else -difference
                for difference in differences
            )
        )
        if statistic >= observed - tolerance:
            extreme_count += 1
    # The plus-one correction prevents a Monte Carlo estimate of exactly zero.
    return (
        (extreme_count + 1) / (config.randomization_resamples + 1),
        RandomizationMethod.MONTE_CARLO,
        config.randomization_resamples,
    )


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Interpolate a quantile from sorted numeric values.

    Args:
        sorted_values: Ascending numeric values.
        probability: Quantile probability in ``[0, 1]``.

    Returns:
        Linearly interpolated quantile.
    """
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )
