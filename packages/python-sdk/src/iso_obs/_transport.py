"""HTTP transport for the iso-obs SDK.

Wraps a single :class:`httpx.Client` with the retry and error-mapping policy
shared by every API call:

* Up to three attempts per request, retrying on 429, any 5xx, and transport
  errors (connect/read/timeout), with exponential backoff between attempts.
* Non-success responses are mapped to the typed exceptions in
  :mod:`iso_obs.exceptions` so callers never handle raw status codes.

The class is internal: :class:`iso_obs.ReliabilityClient` owns one instance and
the resource namespaces route every request through it. The constructor accepts
an ``http_transport`` (e.g. :class:`httpx.MockTransport`) and a ``backoff_base``
so tests can exercise the retry policy hermetically and without sleeping,
without widening the public client signature.
"""

from __future__ import annotations

import platform
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

from .exceptions import (
    ApiError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
)

# Total attempts per request (one initial try plus two retries).
_MAX_ATTEMPTS = 3


def _user_agent() -> str:
    """Identify SDK traffic already sent to Reliability Studio."""
    try:
        sdk_version = version("iso-obs")
    except PackageNotFoundError:
        sdk_version = "development"
    return f"iso-obs/{sdk_version} Python/{platform.python_version()}"


def _exception_for(response: httpx.Response) -> ApiError:
    """Map a non-success HTTP response to its typed exception.

    Args:
        response: The failed response.

    Returns:
        The :class:`~iso_obs.exceptions.ApiError` subclass instance matching
        the response status code.
    """
    status = response.status_code
    detail: str | None = None
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("detail")
    except ValueError:
        detail = None
    message = detail or f"HTTP {status}: {response.text[:200]}"
    if status in (401, 403):
        return AuthenticationError(message, status_code=status)
    if status == 404:
        return NotFoundError(message, status_code=status)
    if status == 429:
        return RateLimitError(message, status_code=status)
    if status >= 500:
        return ServerError(message, status_code=status)
    return ApiError(message, status_code=status)


def _is_retryable(status: int) -> bool:
    """Report whether a status code warrants another attempt.

    Args:
        status: HTTP status code of the response.

    Returns:
        True for 429 (throttled) and any 5xx (transient server fault).
    """
    return status == 429 or status >= 500


class Transport:
    """Retrying, error-mapping HTTP transport shared by all resources."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        http_transport: httpx.BaseTransport | None = None,
        backoff_base: float = 0.5,
    ) -> None:
        """Initialize the transport.

        Args:
            api_key: Bearer token sent on every request.
            base_url: API root every request path is resolved against.
            timeout: Per-request timeout in seconds.
            http_transport: Optional httpx transport override; tests pass an
                :class:`httpx.MockTransport` here.
            backoff_base: First retry delay in seconds; each further retry
                doubles it. Tests pass ``0.0`` to avoid sleeping.
        """
        self._backoff_base = backoff_base
        self._http = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": _user_agent(),
            },
            timeout=timeout,
            transport=http_transport,
        )

    def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> Any:
        """Send a request, retrying transient failures, and parse the body.

        Args:
            method: HTTP method, e.g. ``"POST"``.
            path: Path relative to the base URL, e.g. ``"/runs"``.
            json: Optional JSON body.

        Returns:
            The parsed JSON response body, or ``None`` for empty responses.

        Raises:
            ApiError: Or a subclass, when the request ultimately fails. A bare
                :class:`ApiError` (``status_code=None``) means retries were
                exhausted without ever receiving an HTTP response.
        """
        response = self._send_with_retry(method, path, json)
        if not response.is_success:
            raise _exception_for(response)
        if not response.content:
            return None
        return response.json()

    def _send_with_retry(
        self, method: str, path: str, json: dict[str, Any] | None
    ) -> httpx.Response:
        """Send the request with up to ``_MAX_ATTEMPTS`` attempts.

        Args:
            method: HTTP method.
            path: Request path relative to the base URL.
            json: Optional JSON body.

        Returns:
            The final response — successful, non-retryable, or the last
            retryable failure once attempts are exhausted.

        Raises:
            ApiError: If every attempt failed at the transport level, so no
                HTTP response exists to return.
        """
        last_error: httpx.TransportError | None = None
        response: httpx.Response | None = None
        for attempt in range(_MAX_ATTEMPTS):
            if attempt:
                # Exponential backoff: base, 2*base, ... between attempts.
                time.sleep(self._backoff_base * 2 ** (attempt - 1))
            response = None
            try:
                response = self._http.request(method, path, json=json)
            except httpx.TransportError as exc:
                last_error = exc
                continue
            if not _is_retryable(response.status_code):
                return response
        if response is not None:
            return response
        raise ApiError(
            f"request failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error
