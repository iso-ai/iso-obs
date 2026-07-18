"""Tests for strict local dataset artifact I/O and adapters."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from iso_obs.dataset_io import (
    DATASET_BUNDLE_SCHEMA_VERSION,
    ArtifactFormatError,
    DatasetBundle,
    ManifestDatasetAdapter,
    inspect_dataset_bundle,
    load_json_artifact,
    parse_json_artifact,
)
from iso_obs.dataset_reliability import (
    DatasetEvidenceRole,
    DatasetLabelAssertion,
    DatasetManifest,
    DatasetScope,
    EpisodeManifest,
    LabelStatus,
    ModeFamily,
)
from iso_obs.dataset_synchronization import ChannelTimingTrace
from iso_obs.evidence import NamedDigest, sha256_digest


def bundle() -> DatasetBundle:
    """Build a compact local dataset bundle."""
    manifest = DatasetManifest(
        dataset_id="robot-failures",
        dataset_version="1",
        content_digest=sha256_digest("raw"),
        source_uri="file:///datasets/robot-failures",
        license_id="Apache-2.0",
        scope=DatasetScope(
            domain_namespace="robotics",
            target_population_digest=sha256_digest("population"),
            collection_protocol_digest=sha256_digest("protocol"),
            modality_ids=("camera",),
            evidence_roles=(DatasetEvidenceRole.FAILURE_LABEL_LEARNING,),
        ),
    )
    episode = EpisodeManifest(
        dataset_manifest_digest=manifest.manifest_digest(),
        episode_id="episode-1",
        independence_unit_id="run-1",
        content_digest=sha256_digest("episode"),
        start_seconds=0.0,
        end_seconds=1.0,
        modality_digests=(NamedDigest("camera", sha256_digest("camera")),),
    )
    assertion = DatasetLabelAssertion(
        assertion_id="failure-state",
        episode_id=episode.episode_id,
        label_namespace="reliability/failure-state/v1",
        mode_family=ModeFamily.FAILURE_STATE,
        start_seconds=0.0,
        end_seconds=1.0,
        candidate_values=("contact_loss",),
        status=LabelStatus.ADJUDICATED,
        evidence=(),
    )
    trace = ChannelTimingTrace(
        episode_id=episode.episode_id,
        channel_id="camera-front",
        modality_id="camera",
        clock_id="controller",
        content_digest=sha256_digest("timing"),
        timestamps_seconds=(0.0, 0.5, 1.0),
        timestamp_uncertainty_seconds=0.001,
    )
    return DatasetBundle(
        schema_version=DATASET_BUNDLE_SCHEMA_VERSION,
        manifest=manifest,
        episodes=(episode,),
        label_assertions=(assertion,),
        timing_traces=(trace,),
    )


def test_bundle_round_trip_is_strict_and_content_addressed(
    tmp_path: Path,
) -> None:
    """Canonical JSON round-trips without changing bundle identity."""
    original = bundle()
    path = tmp_path / "bundle.json"
    path.write_text(original.to_json(), encoding="utf-8")

    loaded = load_json_artifact(path, DatasetBundle)

    assert loaded == original
    assert loaded.content_digest() == original.content_digest()


@pytest.mark.parametrize(
    "payload, message",
    [
        ('{"schema_version":"a","schema_version":"b"}', "duplicate JSON object key"),
        (
            bundle()
            .to_json()
            .replace(
                '"dataset_id":"robot-failures"',
                '"dataset_id":"robot-failures","typo":"unsafe"',
            ),
            "unknown field",
        ),
        (
            bundle()
            .to_json()
            .replace(
                '"timestamp_uncertainty_seconds":0.001',
                '"timestamp_uncertainty_seconds":NaN',
            ),
            "non-finite JSON number",
        ),
    ],
)
def test_parser_rejects_ambiguous_json(payload: str, message: str) -> None:
    """Duplicate, unknown, and non-finite values cannot be silently accepted."""
    with pytest.raises(ArtifactFormatError, match=message):
        parse_json_artifact(payload, DatasetBundle)


def test_bundle_adapter_and_inspection_preserve_missingness() -> None:
    """Inspection inventories absent evidence without converting it to zero."""
    source = replace(bundle(), label_assertions=(), timing_traces=())
    adapter = ManifestDatasetAdapter(source)

    inspection = inspect_dataset_bundle(source)

    assert adapter.dataset_manifest() == source.manifest
    assert adapter.timing_traces("episode-1") == ()
    assert inspection.assertion_count == 0
    assert inspection.timing_trace_count == 0
    assert inspection.episodes_without_assertions == ("episode-1",)
    assert inspection.episodes_without_timing_traces == ("episode-1",)


def test_bundle_rejects_cross_manifest_episode() -> None:
    """An episode cannot be smuggled into a different dataset version."""
    source = bundle()
    invalid_episode = replace(
        source.episodes[0],
        dataset_manifest_digest=sha256_digest("other-manifest"),
    )

    with pytest.raises(ValueError, match="different manifest"):
        replace(source, episodes=(invalid_episode,))
