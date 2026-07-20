"""Contract tests for declared geometry and recorded playback."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iso_obs_schemas import EnvironmentRenderManifest, RunPlayback


def test_render_contracts_are_versioned_and_strict() -> None:
    """Render artifacts carry stable schema versions and reject guessed polygons."""
    manifest = EnvironmentRenderManifest(
        coordinate_frame="map",
        units="m",
        source_digest="sha256:abc",
        boundaries=[
            {
                "id": "world",
                "kind": "world",
                "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}],
            }
        ],
    )
    assert manifest.schema_version == "iso-obs.environment-render.v1"
    assert (
        RunPlayback(
            run_id="run_test",
            environment_version_id="envv_test",
        ).schema_version
        == "iso-obs.run-playback.v1"
    )

    with pytest.raises(ValidationError):
        EnvironmentRenderManifest(
            coordinate_frame="map",
            units="m",
            source_digest="sha256:abc",
            boundaries=[
                {
                    "id": "not-a-polygon",
                    "kind": "world",
                    "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
                }
            ],
        )
