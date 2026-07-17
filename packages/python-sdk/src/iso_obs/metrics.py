"""Custom metric registration for the iso-obs SDK.

The :func:`metric` decorator declares a user-defined scalar metric: it attaches
a :class:`~iso_obs_schemas.MetricDefinition` to the decorated function (under
``__iso_obs_metric__``) and records it in a process-wide registry that suite
tooling reads via :func:`registered_metrics`. The function itself is returned
unchanged, so decorating never alters runtime behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from iso_obs_schemas import MetricDefinition, MetricDirection, MetricValueKind

_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True)
class RegisteredMetric:
    """A custom metric declared through the :func:`metric` decorator.

    Attributes:
        definition: Contract-level declaration of the metric.
        compute: The user function that computes the metric's value.
    """

    definition: MetricDefinition
    compute: Callable[..., Any]


# Process-wide registry of custom metrics, keyed by metric name.
_REGISTRY: dict[str, RegisteredMetric] = {}


def metric(name: str, direction: MetricDirection | str) -> Callable[[_F], _F]:
    """Declare the decorated function as a custom scalar metric.

    Args:
        name: Stable metric key, e.g. ``"tracking_error"``.
        direction: Direction in which the metric improves, as a
            :class:`~iso_obs_schemas.MetricDirection` or its string value
            (``"maximize"`` / ``"minimize"``).

    Returns:
        A decorator that registers the function and returns it unchanged.

    Example:
        >>> @metric("tracking_error", "minimize")
        ... def tracking_error(trace: list[dict[str, float]]) -> float:
        ...     return max(abs(step["error"]) for step in trace)
    """
    resolved = MetricDirection(direction)

    def decorator(func: _F) -> _F:
        """Register ``func`` under ``name`` and return it unchanged.

        Args:
            func: The metric computation function.

        Returns:
            ``func``, with its definition attached as ``__iso_obs_metric__``.
        """
        definition = MetricDefinition(
            key=name,
            name=name,
            kind=MetricValueKind.SCALAR,
            direction=resolved,
        )
        _REGISTRY[name] = RegisteredMetric(definition=definition, compute=func)
        func.__iso_obs_metric__ = definition  # type: ignore[attr-defined]
        return func

    return decorator


def registered_metrics() -> dict[str, RegisteredMetric]:
    """Return a snapshot of every metric declared via :func:`metric`.

    Returns:
        Mapping from metric name to its registration, copied so callers
        cannot mutate the registry.
    """
    return dict(_REGISTRY)
