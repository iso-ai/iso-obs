"""Minimal direct HTTP access for read endpoints the SDK does not expose.

The frozen ``iso_obs`` SDK surface only covers the write side of runs
(create, log events, complete, fail). ``iso run inspect`` needs the read
side, so it issues plain GETs against the same REST API using only the
standard library rather than inventing SDK methods.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from iso_obs_schemas import MetricValue, Run

_TIMEOUT_SECONDS = 30.0


class ApiError(Exception):
    """A request to the iso-obs API failed with an error response."""


class ApiUnreachableError(ApiError):
    """The iso-obs API host could not be reached at all."""


def _get_json(url: str, api_key: str) -> Any:
    """GET ``url`` with bearer auth and decode the JSON body.

    Args:
        url: Fully qualified URL to fetch.
        api_key: API key sent as a bearer token.

    Returns:
        The decoded JSON document.

    Raises:
        ApiUnreachableError: If the host cannot be reached.
        ApiError: If the server answers with an HTTP error status or a
            non-JSON body.
    """
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ApiError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ApiUnreachableError(f"cannot reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ApiError(f"{url} returned a non-JSON body") from exc


def fetch_run(run_id: str, *, api_key: str, base_url: str) -> Run:
    """Fetch a run by id.

    Args:
        run_id: Typed run id, e.g. ``run_...``.
        api_key: API key sent as a bearer token.
        base_url: API base URL, e.g. ``https://api.iso-obs.com/api/v1``.

    Returns:
        The validated :class:`~iso_obs_schemas.Run`.

    Raises:
        ApiUnreachableError: If the host cannot be reached.
        ApiError: If the server rejects the request.
    """
    data = _get_json(f"{base_url}/runs/{run_id}", api_key)
    return Run.model_validate(data)


def fetch_run_metrics(run_id: str, *, api_key: str, base_url: str) -> list[MetricValue]:
    """Fetch the metric values recorded for a run.

    Args:
        run_id: Typed run id, e.g. ``run_...``.
        api_key: API key sent as a bearer token.
        base_url: API base URL.

    Returns:
        The run's metric values; empty when the server has none (or does not
        expose the metrics resource).

    Raises:
        ApiUnreachableError: If the host cannot be reached.
    """
    try:
        data = _get_json(f"{base_url}/runs/{run_id}/metrics", api_key)
    except ApiUnreachableError:
        raise
    except ApiError:
        # Metrics are supplementary: a server without the metrics resource
        # should not make ``iso run inspect`` fail after the run fetched fine.
        return []
    return [MetricValue.model_validate(item) for item in data]
