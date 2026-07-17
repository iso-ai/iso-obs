"""Typed exceptions raised by the iso-obs SDK.

Every error surfaced by the SDK derives from :class:`ApiError`, so callers can
catch one type to handle any API failure while still being able to branch on
the specific condition (bad credentials, missing object, throttling, server
fault). Transport-level failures — DNS, connection, timeout — are reported as a
bare :class:`ApiError` with ``status_code`` of ``None`` because no HTTP status
was ever received.
"""

from __future__ import annotations


class ApiError(Exception):
    """Base class for every error raised by the iso-obs API client.

    Attributes:
        status_code: HTTP status code of the failed response, or ``None`` when
            the failure happened below HTTP (e.g. a connection error).
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the failure.
            status_code: HTTP status code of the failed response, if any.
        """
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(ApiError):
    """The request was rejected for missing or invalid credentials (401/403).

    Also raised locally by :class:`iso_obs.ReliabilityClient` when no API key
    is provided and ``ISO_OBS_API_KEY`` is unset.
    """


class NotFoundError(ApiError):
    """The referenced object does not exist (404)."""


class RateLimitError(ApiError):
    """The request was throttled (429) and retries were exhausted."""


class ServerError(ApiError):
    """The API failed with a 5xx status and retries were exhausted."""
