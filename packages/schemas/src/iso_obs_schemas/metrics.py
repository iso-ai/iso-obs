"""Metrics, invariants, and gates for the Reliability Studio contract.

This module models the *measurement* layer:

* :class:`MetricValue` — a single measured value (scalar, boolean, or text),
  optionally scored against a threshold.
* :class:`MetricDefinition` — the declaration of a metric a suite computes.
* :class:`Invariant` — a property that must hold during a run, evaluated over a
  scope with an enforcement mode.
* :class:`Gate` — a pass/fail policy composed of metric requirements plus an
  optional regression requirement against a baseline.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .core import SchemaModel
from .enums import MetricDirection, Severity
from .ids import SuiteExecutionId


class MetricValueKind(StrEnum):
    """Value type carried by a :class:`MetricValue`."""

    SCALAR = "scalar"
    BOOLEAN = "boolean"
    TEXT = "text"


class InvariantScope(StrEnum):
    """Granularity at which an :class:`Invariant` is evaluated."""

    STEP = "step"
    EPISODE = "episode"
    RUN = "run"


class InvariantMode(StrEnum):
    """How an :class:`Invariant` violation is treated.

    * ``hard`` — a violation fails the run.
    * ``soft`` — a violation is recorded but does not fail the run.
    * ``shadow`` — evaluated for observability only, never affects outcomes.
    """

    HARD = "hard"
    SOFT = "soft"
    SHADOW = "shadow"


class ComparisonOperator(StrEnum):
    """Comparison operator used by a :class:`GateRequirement`."""

    GTE = "gte"
    LTE = "lte"
    GT = "gt"
    LT = "lt"
    EQ = "eq"
    NE = "ne"


class MetricValue(SchemaModel):
    """A single measured metric value.

    Exactly one of ``scalar_value``, ``boolean_value``, or ``text_value`` is set,
    selected by ``kind``. Scoring fields (``direction``, ``threshold``,
    ``passed``) apply to scalar and boolean metrics.
    """

    key: str = Field(description="Metric key this value corresponds to.")
    kind: MetricValueKind = Field(description="Which value variant is populated.")
    scalar_value: float | None = Field(
        default=None, description="Numeric value when kind is 'scalar'."
    )
    boolean_value: bool | None = Field(
        default=None, description="Boolean value when kind is 'boolean'."
    )
    text_value: str | None = Field(
        default=None, description="Text value when kind is 'text'."
    )
    unit: str | None = Field(default=None, description="Unit of a scalar value.")
    direction: MetricDirection | None = Field(
        default=None, description="Direction in which this metric improves."
    )
    threshold: float | None = Field(
        default=None, description="Threshold this value was scored against."
    )
    passed: bool | None = Field(
        default=None, description="Whether the value met its threshold, if scored."
    )

    @model_validator(mode="after")
    def _check_variant(self) -> MetricValue:
        """Ensure the populated value matches the declared ``kind``.

        Returns:
            The validated model.

        Raises:
            ValueError: If the value field for ``kind`` is missing.
        """
        expected = {
            MetricValueKind.SCALAR: self.scalar_value,
            MetricValueKind.BOOLEAN: self.boolean_value,
            MetricValueKind.TEXT: self.text_value,
        }[self.kind]
        if expected is None:
            raise ValueError(f"kind '{self.kind}' requires its matching value field")
        return self


class MetricDefinition(SchemaModel):
    """Declaration of a metric computed for a suite's runs."""

    key: str = Field(description="Stable machine key, e.g. 'success_rate'.")
    name: str = Field(description="Human-readable metric name.")
    kind: MetricValueKind = Field(description="Value type this metric produces.")
    description: str | None = None
    unit: str | None = Field(default=None, description="Unit of scalar values.")
    direction: MetricDirection = Field(
        default=MetricDirection.MAXIMIZE,
        description="Direction in which the metric improves.",
    )
    default_threshold: float | None = Field(
        default=None, description="Default pass/fail threshold, if any."
    )


class Invariant(SchemaModel):
    """A property that must hold while a run executes.

    The ``expression`` is a boolean predicate over trace state written in the
    platform's invariant expression language; this contract stores it as text
    and does not evaluate it.
    """

    name: str = Field(description="Human-readable invariant name.")
    expression: str = Field(description="Boolean predicate over trace state.")
    scope: InvariantScope = Field(
        default=InvariantScope.STEP, description="Granularity of evaluation."
    )
    severity: Severity = Field(
        default=Severity.CRITICAL, description="Severity of a violation."
    )
    mode: InvariantMode = Field(
        default=InvariantMode.HARD, description="How a violation is treated."
    )


class GateRequirement(SchemaModel):
    """A single metric threshold that a gate requires to pass."""

    metric: str = Field(description="Metric key the requirement is evaluated on.")
    operator: ComparisonOperator = Field(description="Comparison to apply.")
    value: float = Field(description="Right-hand side of the comparison.")


class RegressionRequirement(SchemaModel):
    """A limit on how much a metric may regress versus a baseline."""

    metric: str = Field(description="Metric key checked for regression.")
    baseline_suite_execution_id: SuiteExecutionId | None = Field(
        default=None,
        description="Suite execution to compare against; latest if omitted.",
    )
    max_absolute_regression: float | None = Field(
        default=None, description="Maximum allowed absolute drop from baseline."
    )
    max_relative_regression: float | None = Field(
        default=None,
        description="Maximum allowed fractional drop from baseline, in [0, 1].",
    )


class Gate(SchemaModel):
    """A pass/fail policy applied to a run or suite execution.

    A gate passes when every metric requirement is met and, if present, the
    regression requirement holds.
    """

    name: str = Field(description="Human-readable gate name.")
    requirements: list[GateRequirement] = Field(
        default_factory=list, description="Metric thresholds that must all pass."
    )
    regression: RegressionRequirement | None = Field(
        default=None, description="Optional regression check against a baseline."
    )
