"""Tests for ReliabilityClient construction, auth fallback, and resources."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from sdk_doubles import BASE_URL, make_client

from iso_obs import ReliabilityClient
from iso_obs.client import DEFAULT_BASE_URL
from iso_obs.exceptions import ApiError, AuthenticationError, NotFoundError
from iso_obs_schemas import IdPrefix, Project, SystemVersion, generate_id


def _project_payload(name: str) -> dict[str, Any]:
    """Build a valid Project JSON payload named `name`."""
    return {
        "id": generate_id(IdPrefix.PROJECT),
        "workspace_id": generate_id(IdPrefix.WORKSPACE),
        "name": name,
        "slug": name,
    }


class TestAuthFallback:
    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ISO_OBS_API_KEY", raising=False)
        with pytest.raises(AuthenticationError):
            ReliabilityClient()

    def test_key_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ISO_OBS_API_KEY", "key_from_env")
        client = ReliabilityClient()
        assert client._transport._http.headers["Authorization"] == (
            "Bearer key_from_env"
        )

    def test_explicit_key_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ISO_OBS_API_KEY", "key_from_env")
        client = ReliabilityClient(api_key="key_explicit")
        assert client._transport._http.headers["Authorization"] == (
            "Bearer key_explicit"
        )

    def test_base_url_falls_back_to_env_then_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # httpx normalizes base_url with a trailing slash; strip it before
        # comparing against the configured roots.
        monkeypatch.delenv("ISO_OBS_BASE_URL", raising=False)
        client = ReliabilityClient(api_key="k")
        assert str(client._transport._http.base_url).rstrip("/") == DEFAULT_BASE_URL
        monkeypatch.setenv("ISO_OBS_BASE_URL", "https://env.example/api/v1")
        client = ReliabilityClient(api_key="k")
        assert str(client._transport._http.base_url).rstrip("/") == (
            "https://env.example/api/v1"
        )


class TestProjects:
    def test_create(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_project_payload("robot-arm"))

        project = make_client(handler).projects.create("robot-arm")
        assert isinstance(project, Project)
        assert project.name == "robot-arm"
        assert seen[0].method == "POST"
        assert seen[0].url == f"{BASE_URL}/projects"
        assert json.loads(seen[0].content) == {"name": "robot-arm"}

    def test_list(self) -> None:
        payloads = [_project_payload("a"), _project_payload("b")]

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET"
            return httpx.Response(200, json=payloads)

        projects = make_client(handler).projects.list()
        assert [p.name for p in projects] == ["a", "b"]


class TestSystems:
    def test_register(self) -> None:
        seen: list[httpx.Request] = []
        payload = {
            "id": generate_id(IdPrefix.SYSTEM_VERSION),
            "system_id": generate_id(IdPrefix.SYSTEM),
            "version": "v17",
            "commit_sha": "abc123",
            "artifact_uri": "s3://models/policy-v17",
            "metadata": {"framework": "jax"},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=payload)

        version = make_client(handler).systems.register(
            "robot-arm",
            "grasp-policy",
            "v17",
            artifact_uri="s3://models/policy-v17",
            source_commit="abc123",
            framework="jax",
        )
        assert isinstance(version, SystemVersion)
        assert version.version == "v17"
        assert seen[0].url == f"{BASE_URL}/systems:register"
        body = json.loads(seen[0].content)
        assert body == {
            "project": "robot-arm",
            "name": "grasp-policy",
            "version": "v17",
            "artifact_uri": "s3://models/policy-v17",
            "source_commit": "abc123",
            "framework": "jax",
            "metadata": {},
        }


class TestErrorMapping:
    def test_404_maps_to_not_found(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "no such project"})

        with pytest.raises(NotFoundError) as excinfo:
            make_client(handler).projects.list()
        assert excinfo.value.status_code == 404
        assert "no such project" in str(excinfo.value)

    def test_other_4xx_maps_to_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"detail": "bad request"})

        with pytest.raises(ApiError) as excinfo:
            make_client(handler).projects.create("x")
        assert excinfo.value.status_code == 400

    def test_401_maps_to_authentication_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "bad key"})

        with pytest.raises(AuthenticationError):
            make_client(handler).projects.list()
