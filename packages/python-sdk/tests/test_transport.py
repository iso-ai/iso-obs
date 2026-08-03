"""Tests for the retry and error-mapping behavior of the HTTP transport."""

from __future__ import annotations

import httpx
import pytest

from iso_obs._transport import Transport
from iso_obs.exceptions import ApiError, NotFoundError, RateLimitError, ServerError


def make_transport(handler: httpx.MockTransport) -> Transport:
    """Build a Transport backed by `handler` with zero backoff delay."""
    return Transport(
        api_key="key_test",
        base_url="https://api.test/api/v1",
        timeout=5.0,
        http_transport=handler,
        backoff_base=0.0,
    )


def test_retry_then_succeed_on_server_error() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(500, json={"detail": "flaky"})
        return httpx.Response(200, json={"ok": True})

    result = make_transport(httpx.MockTransport(handler)).request("GET", "/ping")
    assert result == {"ok": True}
    assert len(calls) == 3


def test_retries_exhausted_raise_server_error() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"detail": "down"})

    with pytest.raises(ServerError) as excinfo:
        make_transport(httpx.MockTransport(handler)).request("GET", "/ping")
    assert excinfo.value.status_code == 503
    assert len(calls) == 3


def test_rate_limit_retried_then_raises_rate_limit_error() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"detail": "slow down"})

    with pytest.raises(RateLimitError):
        make_transport(httpx.MockTransport(handler)).request("GET", "/ping")
    assert len(calls) == 3


def test_transport_error_then_succeed() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"ok": True})

    result = make_transport(httpx.MockTransport(handler)).request("GET", "/ping")
    assert result == {"ok": True}
    assert len(calls) == 3


def test_transport_error_exhausted_raises_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ApiError) as excinfo:
        make_transport(httpx.MockTransport(handler)).request("GET", "/ping")
    assert excinfo.value.status_code is None
    assert not isinstance(excinfo.value, ServerError)


def test_no_retry_on_plain_4xx() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"detail": "missing"})

    with pytest.raises(NotFoundError):
        make_transport(httpx.MockTransport(handler)).request("GET", "/ping")
    assert len(calls) == 1


def test_empty_body_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    assert make_transport(httpx.MockTransport(handler)).request("POST", "/x") is None


def test_request_identifies_sdk_without_background_telemetry() -> None:
    """Attach SDK identity only to an explicit API request."""
    observed_user_agent = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_user_agent
        observed_user_agent = request.headers["User-Agent"]
        return httpx.Response(200, json={"ok": True})

    make_transport(httpx.MockTransport(handler)).request("GET", "/ping")

    assert observed_user_agent.startswith("iso-obs/")
    assert "Python/" in observed_user_agent
