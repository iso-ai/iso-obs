"""Tests for provenance-preserving trajectory dataset ingestion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from iso_obs.dataset_ingestion import (
    DATASET_INGESTION_PLAN_SCHEMA_VERSION,
    ChannelIngestionRule,
    DatasetIngestionDisposition,
    DatasetIngestionError,
    DatasetIngestionPlan,
    DatasetReaderUnavailableError,
    DatasetRowReader,
    DatasetSourceFormat,
    JsonlDatasetReader,
    ParquetDatasetReader,
    SourceField,
    TimestampUnit,
    ingest_dataset_rows,
    ingest_dataset_source,
)
from iso_obs.dataset_reliability import DatasetEvidenceRole, DatasetScope
from iso_obs.evidence import sha256_digest


def plan(
    *,
    source_format: DatasetSourceFormat = DatasetSourceFormat.JSONL,
    include_required_camera: bool = True,
) -> DatasetIngestionPlan:
    """Build an explicit two-channel trajectory mapping."""
    rules = [
        ChannelIngestionRule(
            channel_id="joint-state",
            modality_id="state",
            clock_id="controller",
            payload_field=SourceField(("observation", "joint_state")),
            timestamp_uncertainty_seconds=0.0001,
        )
    ]
    if include_required_camera:
        rules.append(
            ChannelIngestionRule(
                channel_id="camera-front",
                modality_id="camera",
                clock_id="camera-clock",
                payload_field=SourceField(("observation", "camera_uri")),
                timestamp_field=SourceField(("camera_timestamp_ms",)),
                timestamp_uncertainty_seconds=0.5e-3,
            )
        )
    return DatasetIngestionPlan(
        schema_version=DATASET_INGESTION_PLAN_SCHEMA_VERSION,
        plan_id="robot-jsonl-ingestion",
        plan_version="1",
        source_format=source_format,
        dataset_id="robot-trajectories",
        dataset_version="2026-07",
        source_uri="s3://research/robot-trajectories.jsonl",
        license_id="LicenseRef-research",
        scope=DatasetScope(
            domain_namespace="robot-manipulation",
            target_population_digest=sha256_digest("warehouse-picks"),
            collection_protocol_digest=sha256_digest("capture-protocol"),
            modality_ids=("camera", "state"),
            evidence_roles=(DatasetEvidenceRole.REPRESENTATION_LEARNING,),
        ),
        episode_id_field=SourceField(("episode_id",)),
        independence_unit_id_field=SourceField(("run_id",)),
        timestamp_field=SourceField(("timestamp_ms",)),
        timestamp_unit=TimestampUnit.MILLISECONDS,
        channel_rules=tuple(rules),
        source_split_field=SourceField(("source_split",)),
    )


def write_jsonl(path: Path, rows: tuple[str, ...]) -> Path:
    """Write exact JSONL source rows."""
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def test_jsonl_ingestion_is_content_addressed_and_preserves_timestamp_order(
    tmp_path: Path,
) -> None:
    """Ingestion hashes exact source bytes without repairing acquisition order."""
    source = write_jsonl(
        tmp_path / "trajectories.jsonl",
        (
            '{"episode_id":"ep-1","run_id":"run-1","timestamp_ms":1000,'
            '"camera_timestamp_ms":1001,"source_split":"train",'
            '"observation":{"joint_state":[0.1,0.2],'
            '"camera_uri":"frames/2.png"}}',
            '{"episode_id":"ep-1","run_id":"run-1","timestamp_ms":500,'
            '"camera_timestamp_ms":501,"source_split":"train",'
            '"observation":{"joint_state":[0.0,0.1],'
            '"camera_uri":"frames/1.png"}}',
        ),
    )

    first = ingest_dataset_source(plan(), source)
    second = ingest_dataset_source(plan(), source)

    assert first == second
    assert first.bundle.content_digest() == second.bundle.content_digest()
    assert first.report.disposition is (
        DatasetIngestionDisposition.INGESTED_WITHIN_DECLARED_MAPPING
    )
    assert first.report.source_content_digest == sha256_digest(source.read_bytes())
    assert first.bundle.label_assertions == ()
    assert first.bundle.episodes[0].start_seconds == 0.5
    assert first.bundle.episodes[0].end_seconds == 1.0
    traces = {item.channel_id: item for item in first.bundle.timing_traces}
    assert traces["joint-state"].timestamps_seconds == (1.0, 0.5)
    assert traces["camera-front"].timestamps_seconds == (1.001, 0.501)


def test_missing_required_channel_is_review_not_silent_success(
    tmp_path: Path,
) -> None:
    """An unused required mapping remains visible as a review obligation."""
    source = write_jsonl(
        tmp_path / "state-only.jsonl",
        (
            '{"episode_id":"ep-1","run_id":"run-1","timestamp_ms":0,'
            '"camera_timestamp_ms":0,"source_split":null,'
            '"observation":{"joint_state":[0.0],"camera_uri":null}}',
        ),
    )

    result = ingest_dataset_source(plan(), source)

    assert result.report.disposition is DatasetIngestionDisposition.REVIEW_REQUIRED
    summaries = {item.channel_id: item for item in result.report.channel_summaries}
    assert summaries["camera-front"].sample_count == 0
    assert summaries["camera-front"].missing_payload_row_count == 1
    assert result.report.issues[0].channel_id == "camera-front"


def test_empty_source_abstains_without_inventing_episodes(tmp_path: Path) -> None:
    """An empty source produces an explicit insufficient-evidence report."""
    source = write_jsonl(tmp_path / "empty.jsonl", ())

    result = ingest_dataset_source(plan(), source)

    assert result.report.disposition is (
        DatasetIngestionDisposition.INSUFFICIENT_EVIDENCE
    )
    assert result.report.row_count == 0
    assert result.bundle.episodes == ()


def test_episode_identity_cannot_change_within_source(tmp_path: Path) -> None:
    """Rows sharing an episode cannot silently cross independence units."""
    source = write_jsonl(
        tmp_path / "conflict.jsonl",
        (
            '{"episode_id":"ep-1","run_id":"run-1","timestamp_ms":0,'
            '"source_split":"train","observation":{"joint_state":[0]}}',
            '{"episode_id":"ep-1","run_id":"run-2","timestamp_ms":1,'
            '"source_split":"train","observation":{"joint_state":[1]}}',
        ),
    )

    with pytest.raises(DatasetIngestionError, match="changes independence unit"):
        ingest_dataset_source(
            plan(include_required_camera=False),
            source,
        )


def test_jsonl_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Ambiguous source objects cannot enter the normalized evidence layer."""
    source = write_jsonl(
        tmp_path / "duplicate.jsonl",
        ('{"episode_id":"ep-1","episode_id":"ep-2"}',),
    )

    with pytest.raises(DatasetIngestionError, match="duplicate JSON object key"):
        tuple(JsonlDatasetReader(source).rows())


def test_parquet_reader_has_actionable_optional_dependency_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing PyArrow reports the exact install extra instead of import noise."""
    source = tmp_path / "source.parquet"
    source.write_bytes(b"PAR1")

    def unavailable(name: str) -> None:
        """Simulate an environment without the optional Parquet dependency."""
        raise ImportError(name)

    monkeypatch.setattr(
        "iso_obs.dataset_ingestion.importlib.import_module",
        unavailable,
    )

    with pytest.raises(DatasetReaderUnavailableError, match="iso-obs\\[parquet\\]"):
        tuple(ParquetDatasetReader(source).rows())


def test_real_parquet_round_trip_when_extra_is_installed(tmp_path: Path) -> None:
    """PyArrow nested rows and binary payloads produce stable bundle evidence."""
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "trajectories.parquet"
    table = pyarrow.Table.from_pylist(
        [
            {
                "episode_id": "ep-1",
                "run_id": "run-1",
                "timestamp_ms": 0,
                "camera_timestamp_ms": 1,
                "source_split": "train",
                "observation": {
                    "joint_state": [0.0, 0.1],
                    "camera_uri": b"\x89PNG",
                },
            },
            {
                "episode_id": "ep-1",
                "run_id": "run-1",
                "timestamp_ms": 10,
                "camera_timestamp_ms": 11,
                "source_split": "train",
                "observation": {
                    "joint_state": [0.1, 0.2],
                    "camera_uri": b"\x89PNG-next",
                },
            },
        ]
    )
    parquet.write_table(table, source)

    result = ingest_dataset_source(
        plan(source_format=DatasetSourceFormat.PARQUET),
        source,
    )

    assert result.report.row_count == 2
    assert result.report.disposition is (
        DatasetIngestionDisposition.INGESTED_WITHIN_DECLARED_MAPPING
    )
    assert result.bundle.timing_traces[0].timestamps_seconds == (0.001, 0.011)


class _MemoryReader(DatasetRowReader):
    """Minimal reader used to exercise binary Parquet-style payloads."""

    @property
    def source_format(self) -> DatasetSourceFormat:
        """Return the declared in-memory format."""
        return DatasetSourceFormat.PARQUET

    def content_digest(self) -> str:
        """Return a stable synthetic source digest."""
        return sha256_digest(b"memory-parquet")

    def rows(self) -> Iterable[Mapping[str, Any]]:
        """Yield one row containing a binary payload."""
        yield {
            "episode_id": "ep-1",
            "run_id": "run-1",
            "timestamp_ms": 0,
            "source_split": None,
            "observation": {
                "joint_state": [0.0],
                "camera_uri": b"\x89PNG",
            },
            "camera_timestamp_ms": 0,
        }


class _ChangingReader(DatasetRowReader):
    """Reader whose digest changes between preflight and completion."""

    def __init__(self) -> None:
        """Initialize the digest-call counter."""
        self._digest_calls = 0

    @property
    def source_format(self) -> DatasetSourceFormat:
        """Return the declared source format."""
        return DatasetSourceFormat.JSONL

    def content_digest(self) -> str:
        """Return a different digest on each call."""
        self._digest_calls += 1
        return sha256_digest(f"source-version-{self._digest_calls}")

    def rows(self) -> Iterable[Mapping[str, Any]]:
        """Yield one otherwise valid source row."""
        yield {
            "episode_id": "ep-1",
            "run_id": "run-1",
            "timestamp_ms": 0,
            "source_split": None,
            "observation": {"joint_state": [0.0]},
        }


def test_binary_payloads_are_hashed_not_embedded() -> None:
    """Binary payload identity is preserved without copying bytes into bundles."""
    result = ingest_dataset_rows(
        plan(source_format=DatasetSourceFormat.PARQUET),
        _MemoryReader(),
    )

    assert result.report.row_count == 1
    assert b"\x89PNG" not in result.bundle.to_json().encode()


def test_source_change_during_ingestion_is_rejected() -> None:
    """A report cannot bind output rows to stale source bytes."""
    with pytest.raises(DatasetIngestionError, match="changed during ingestion"):
        ingest_dataset_rows(
            plan(include_required_camera=False),
            _ChangingReader(),
        )
