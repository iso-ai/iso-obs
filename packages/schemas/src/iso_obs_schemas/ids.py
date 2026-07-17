"""Typed identifier conventions for Reliability Studio objects.

Every persistent object in Reliability Studio carries a *typed, prefixed* string
identifier, for example ``ws_a1b2c3d4`` for a workspace. The leading prefix names
the kind of object the id refers to, which makes ids self-describing in logs,
traces, URLs, and error messages, and lets the API reject a mismatched id at the
edge instead of failing deep inside a query.

This module is the single source of truth for that convention. ``core.py``,
``events.py``, ``metrics.py``, and ``reports.py`` all import the id aliases
defined here so that an id used in one schema references the exact same validated
type everywhere it appears.

The prefix set is closed. Adding a new object kind means adding a member to
:class:`IdPrefix` and a matching type alias below, never inventing an ad-hoc
prefix at a call site.
"""

from __future__ import annotations

import secrets
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator

# Separator between the kind prefix and the random token, e.g. ``run_<token>``.
_SEPARATOR = "_"

# Number of random bytes in a generated token; 12 bytes -> 24 hex characters,
# comfortably collision-resistant for object ids while staying URL-friendly.
_DEFAULT_TOKEN_BYTES = 12


class IdPrefix(StrEnum):
    """Canonical prefix for each identifiable Reliability Studio object kind.

    The value of each member is the literal prefix (without the trailing
    separator) used to build and validate ids of that kind.
    """

    WORKSPACE = "ws"
    PROJECT = "prj"
    SYSTEM = "sys"
    SYSTEM_VERSION = "sv"
    ENVIRONMENT = "env"
    ENVIRONMENT_VERSION = "ev"
    SCENARIO = "scn"
    SCENARIO_VERSION = "scv"
    EVALUATION_SUITE = "suite"
    RUN = "run"
    EVENT = "evt"
    FAILURE = "fail"
    COMPARISON = "cmp"
    REPORT = "rpt"
    API_KEY = "key"
    WEBHOOK = "wh"


def generate_id(
    prefix: IdPrefix | str, *, token_bytes: int = _DEFAULT_TOKEN_BYTES
) -> str:
    """Generate a new random prefixed id.

    Args:
        prefix: Object kind, given either as an :class:`IdPrefix` member or as
            the bare prefix string (e.g. ``"run"``).
        token_bytes: Number of random bytes in the token; the hex encoding is
            twice this many characters.

    Returns:
        A new id of the form ``"<prefix>_<hex_token>"``, e.g. ``"run_9f8c...".``
    """
    prefix_str = prefix.value if isinstance(prefix, IdPrefix) else prefix
    return f"{prefix_str}{_SEPARATOR}{secrets.token_hex(token_bytes)}"


def prefix_of(value: str) -> str:
    """Return the kind prefix of a prefixed id.

    Args:
        value: A prefixed id such as ``"prj_abc123"``.

    Returns:
        The prefix portion, e.g. ``"prj"``.

    Raises:
        ValueError: If ``value`` contains no separator.
    """
    prefix, sep, _ = value.partition(_SEPARATOR)
    if not sep:
        raise ValueError(f"'{value}' is not a valid prefixed id")
    return prefix


def _prefix_validator(prefix: IdPrefix) -> AfterValidator:
    """Build a Pydantic ``AfterValidator`` that enforces a specific id prefix.

    Args:
        prefix: The required object-kind prefix.

    Returns:
        An ``AfterValidator`` usable inside an ``Annotated`` id type alias.
    """
    expected = f"{prefix.value}{_SEPARATOR}"

    def _validate(value: str) -> str:
        """Reject ids that do not carry the expected prefix or lack a token."""
        if not value.startswith(expected):
            raise ValueError(f"expected id with prefix '{expected}', got '{value}'")
        if len(value) <= len(expected):
            raise ValueError(f"id '{value}' is missing its token suffix")
        return value

    return AfterValidator(_validate)


# Validated id type aliases (PEP 695). Each is a plain ``str`` at runtime but is
# checked to carry the correct prefix whenever it is parsed through a Pydantic
# model.
type WorkspaceId = Annotated[str, _prefix_validator(IdPrefix.WORKSPACE)]
type ProjectId = Annotated[str, _prefix_validator(IdPrefix.PROJECT)]
type SystemId = Annotated[str, _prefix_validator(IdPrefix.SYSTEM)]
type SystemVersionId = Annotated[str, _prefix_validator(IdPrefix.SYSTEM_VERSION)]
type EnvironmentId = Annotated[str, _prefix_validator(IdPrefix.ENVIRONMENT)]
type EnvironmentVersionId = Annotated[
    str, _prefix_validator(IdPrefix.ENVIRONMENT_VERSION)
]
type ScenarioId = Annotated[str, _prefix_validator(IdPrefix.SCENARIO)]
type ScenarioVersionId = Annotated[str, _prefix_validator(IdPrefix.SCENARIO_VERSION)]
type EvaluationSuiteId = Annotated[str, _prefix_validator(IdPrefix.EVALUATION_SUITE)]
# A suite *execution* is one run of an evaluation suite. It shares the ``suite_``
# namespace with the suite definition it instantiates.
type SuiteExecutionId = Annotated[str, _prefix_validator(IdPrefix.EVALUATION_SUITE)]
type RunId = Annotated[str, _prefix_validator(IdPrefix.RUN)]
type EventId = Annotated[str, _prefix_validator(IdPrefix.EVENT)]
type FailureId = Annotated[str, _prefix_validator(IdPrefix.FAILURE)]
type ComparisonId = Annotated[str, _prefix_validator(IdPrefix.COMPARISON)]
type ReportId = Annotated[str, _prefix_validator(IdPrefix.REPORT)]
type ApiKeyId = Annotated[str, _prefix_validator(IdPrefix.API_KEY)]
type WebhookId = Annotated[str, _prefix_validator(IdPrefix.WEBHOOK)]
