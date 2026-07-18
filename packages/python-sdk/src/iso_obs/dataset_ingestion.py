"""Provenance-preserving ingestion of trajectory datasets.

The ingestion boundary converts explicitly mapped, row-oriented source data
into :class:`iso_obs.dataset_io.DatasetBundle` artifacts. It records the exact
source digest and transformation plan, preserves observed channel timestamp
order, and reports missingness without inferring labels, clocks, or split roles.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Any

from .dataset_io import DATASET_BUNDLE_SCHEMA_VERSION, DatasetBundle
from .dataset_reliability import DatasetManifest, DatasetScope, EpisodeManifest
from .dataset_synchronization import ChannelTimingTrace
from .evidence import NamedDigest, _canonical_json, sha256_digest

DATASET_INGESTION_PLAN_SCHEMA_VERSION = "iso-obs.dataset-ingestion-plan.v1"
DATASET_INGESTION_REPORT_SCHEMA_VERSION = "iso-obs.dataset-ingestion-report.v1"
DEFAULT_MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
DEFAULT_PARQUET_BATCH_SIZE = 65_536
_FILE_DIGEST_CHUNK_BYTES = 1024 * 1024

_MISSING = object()


class DatasetSourceFormat(StrEnum):
    """Supported row-oriented source container."""

    JSONL = "jsonl"
    PARQUET = "parquet"
    MCAP = "mcap"


class TimestampUnit(StrEnum):
    """Declared unit of numeric source timestamps."""

    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    NANOSECONDS = "nanoseconds"

    @property
    def units_per_second(self) -> int:
        """Return the integer divisor used for canonical conversion."""
        return {
            TimestampUnit.SECONDS: 1,
            TimestampUnit.MILLISECONDS: 1_000,
            TimestampUnit.MICROSECONDS: 1_000_000,
            TimestampUnit.NANOSECONDS: 1_000_000_000,
        }[self]


class DatasetIngestionDisposition(StrEnum):
    """Strongest structural ingestion claim supported by the source."""

    INGESTED_WITHIN_DECLARED_MAPPING = "ingested_within_declared_mapping"
    REVIEW_REQUIRED = "review_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DatasetIngestionIssueKind(StrEnum):
    """Kind of structural ingestion limitation."""

    EMPTY_SOURCE = "empty_source"
    REQUIRED_CHANNEL_EMPTY = "required_channel_empty"
    SOURCE_MANIFEST_MISMATCH = "source_manifest_mismatch"


@dataclass(frozen=True, slots=True)
class SourceField:
    """Explicit path to a value in a nested source row."""

    path: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate a nonempty path of stable mapping keys."""
        if not self.path:
            raise ValueError("source field path must not be empty")
        normalized: list[str] = []
        for component in self.path:
            if not isinstance(component, str) or not component.strip():
                raise ValueError("source field path components must be nonempty")
            normalized.append(component.strip())
        object.__setattr__(self, "path", tuple(normalized))


@dataclass(frozen=True, slots=True)
class ChannelIngestionRule:
    """Explicit mapping from source payloads to one modality channel."""

    channel_id: str
    modality_id: str
    clock_id: str
    payload_field: SourceField
    timestamp_field: SourceField | None = None
    timestamp_uncertainty_seconds: float = 0.0
    required: bool = True
    null_payload_is_missing: bool = True

    def __post_init__(self) -> None:
        """Validate channel identity, timing uncertainty, and policies."""
        _require_text(self.channel_id, "ingestion channel ID")
        _require_text(self.modality_id, "ingestion modality ID")
        _require_text(self.clock_id, "ingestion clock ID")
        uncertainty = _nonnegative_finite(
            self.timestamp_uncertainty_seconds,
            "ingestion timestamp uncertainty",
        )
        if not isinstance(self.required, bool):
            raise ValueError("ingestion channel required must be boolean")
        if not isinstance(self.null_payload_is_missing, bool):
            raise ValueError("null payload policy must be boolean")
        object.__setattr__(
            self,
            "timestamp_uncertainty_seconds",
            uncertainty,
        )


@dataclass(frozen=True, slots=True)
class DatasetIngestionPlan:
    """Versioned mapping from one source file into a dataset bundle."""

    schema_version: str
    plan_id: str
    plan_version: str
    source_format: DatasetSourceFormat
    dataset_id: str
    dataset_version: str
    source_uri: str
    license_id: str
    scope: DatasetScope
    episode_id_field: SourceField
    independence_unit_id_field: SourceField
    timestamp_field: SourceField
    timestamp_unit: TimestampUnit
    channel_rules: tuple[ChannelIngestionRule, ...]
    source_split_field: SourceField | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate source identity, mapping completeness, and scope."""
        if self.schema_version != DATASET_INGESTION_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported dataset ingestion plan schema version")
        _require_text(self.plan_id, "dataset ingestion plan ID")
        _require_text(self.plan_version, "dataset ingestion plan version")
        _require_text(self.dataset_id, "ingested dataset ID")
        _require_text(self.dataset_version, "ingested dataset version")
        _require_text(self.source_uri, "ingested dataset source URI")
        _require_text(self.license_id, "ingested dataset license ID")
        rules = tuple(sorted(self.channel_rules, key=lambda item: item.channel_id))
        if not rules:
            raise ValueError("dataset ingestion plan requires channel rules")
        _require_unique(
            (item.channel_id for item in rules),
            "dataset ingestion channel IDs",
        )
        declared_modalities = set(self.scope.modality_ids)
        for rule in rules:
            if rule.modality_id not in declared_modalities:
                raise ValueError("ingestion rule uses an undeclared modality")
        limitations = _unique_text(self.limitations, "ingestion plan limitations")
        object.__setattr__(
            self,
            "source_format",
            DatasetSourceFormat(self.source_format),
        )
        object.__setattr__(self, "timestamp_unit", TimestampUnit(self.timestamp_unit))
        object.__setattr__(self, "channel_rules", rules)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact ingestion-plan digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class ChannelIngestionSummary:
    """Observed source coverage for one explicitly mapped channel."""

    channel_id: str
    modality_id: str
    sample_count: int
    missing_payload_row_count: int
    required: bool

    def __post_init__(self) -> None:
        """Validate channel coverage counts and identity."""
        _require_text(self.channel_id, "ingestion summary channel ID")
        _require_text(self.modality_id, "ingestion summary modality ID")
        for value in (self.sample_count, self.missing_payload_row_count):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("ingestion summary counts must be nonnegative")
        if not isinstance(self.required, bool):
            raise ValueError("ingestion summary required must be boolean")


@dataclass(frozen=True, slots=True)
class DatasetIngestionIssue:
    """One source-level limitation found during deterministic ingestion."""

    kind: DatasetIngestionIssueKind
    channel_id: str | None
    description: str

    def __post_init__(self) -> None:
        """Validate issue kind, optional channel, and description."""
        object.__setattr__(self, "kind", DatasetIngestionIssueKind(self.kind))
        if self.channel_id is not None:
            _require_text(self.channel_id, "ingestion issue channel ID")
        _require_text(self.description, "ingestion issue description")


@dataclass(frozen=True, slots=True)
class DatasetIngestionReport:
    """Versioned link between source, transformation, and output bundle."""

    schema_version: str
    plan_content_digest: str
    source_content_digest: str
    source_format: DatasetSourceFormat
    dataset_manifest_digest: str
    bundle_content_digest: str
    disposition: DatasetIngestionDisposition
    row_count: int
    episode_count: int
    channel_summaries: tuple[ChannelIngestionSummary, ...]
    issues: tuple[DatasetIngestionIssue, ...]
    scope: DatasetScope
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate report identity, counts, ordering, and disposition."""
        if self.schema_version != DATASET_INGESTION_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported dataset ingestion report schema version")
        NamedDigest("dataset ingestion plan", self.plan_content_digest)
        NamedDigest("dataset source", self.source_content_digest)
        NamedDigest("dataset manifest", self.dataset_manifest_digest)
        NamedDigest("dataset bundle", self.bundle_content_digest)
        for value in (self.row_count, self.episode_count):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("dataset ingestion counts must be nonnegative")
        summaries = tuple(
            sorted(self.channel_summaries, key=lambda item: item.channel_id)
        )
        _require_unique(
            (item.channel_id for item in summaries),
            "dataset ingestion summary channel IDs",
        )
        issues = tuple(
            sorted(
                self.issues,
                key=lambda item: (
                    item.kind.value,
                    item.channel_id or "",
                ),
            )
        )
        disposition = DatasetIngestionDisposition(self.disposition)
        if disposition is DatasetIngestionDisposition.INGESTED_WITHIN_DECLARED_MAPPING:
            if issues or self.row_count == 0:
                raise ValueError("supported ingestion report cannot contain issues")
        elif disposition is DatasetIngestionDisposition.REVIEW_REQUIRED:
            if not issues or self.row_count == 0:
                raise ValueError("review-required ingestion needs source issues")
        elif self.row_count != 0 or not any(
            item.kind is DatasetIngestionIssueKind.EMPTY_SOURCE for item in issues
        ):
            raise ValueError("insufficient ingestion requires an empty source")
        limitations = _unique_text(
            self.limitations,
            "dataset ingestion report limitations",
        )
        object.__setattr__(
            self,
            "source_format",
            DatasetSourceFormat(self.source_format),
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "channel_summaries", summaries)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact ingestion-report digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class DatasetIngestionResult:
    """Bundle and provenance report produced by one ingestion execution."""

    bundle: DatasetBundle
    report: DatasetIngestionReport

    def __post_init__(self) -> None:
        """Ensure the report identifies the paired bundle."""
        if self.report.bundle_content_digest != self.bundle.content_digest():
            raise ValueError("ingestion report identifies a different bundle")
        if (
            self.report.dataset_manifest_digest
            != self.bundle.manifest.manifest_digest()
        ):
            raise ValueError("ingestion report identifies a different manifest")


class DatasetRowReader(ABC):
    """Reader boundary for row-oriented dataset containers."""

    @property
    @abstractmethod
    def source_format(self) -> DatasetSourceFormat:
        """Return the reader's declared source format."""

    @abstractmethod
    def content_digest(self) -> str:
        """Return the exact source-file content digest."""

    @abstractmethod
    def rows(self) -> Iterable[Mapping[str, Any]]:
        """Yield source rows in their original reader order."""


class JsonlDatasetReader(DatasetRowReader):
    """Strict streaming reader for newline-delimited JSON objects."""

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES,
    ) -> None:
        """Initialize a JSONL reader.

        Args:
            path: Local JSONL source path.
            max_line_bytes: Maximum encoded size of any single source row.
        """
        if (
            isinstance(max_line_bytes, bool)
            or not isinstance(max_line_bytes, int)
            or max_line_bytes < 1
        ):
            raise ValueError("max_line_bytes must be a positive integer")
        self._path = Path(path)
        self._max_line_bytes = max_line_bytes

    @property
    def source_format(self) -> DatasetSourceFormat:
        """Return the JSONL source format."""
        return DatasetSourceFormat.JSONL

    def content_digest(self) -> str:
        """Hash the exact source bytes without normalizing line endings."""
        return file_content_digest(self._path)

    def rows(self) -> Iterator[Mapping[str, Any]]:
        """Yield strict JSON objects in physical line order."""
        with self._path.open("rb") as stream:
            for line_number, encoded in enumerate(stream, start=1):
                if len(encoded) > self._max_line_bytes:
                    raise DatasetIngestionError(
                        f"JSONL line {line_number} exceeds the "
                        f"{self._max_line_bytes}-byte limit"
                    )
                if not encoded.strip():
                    raise DatasetIngestionError(f"JSONL line {line_number} is blank")
                try:
                    text = encoded.decode("utf-8")
                    value = json.loads(
                        text,
                        object_pairs_hook=_object_without_duplicate_keys,
                        parse_constant=_reject_nonfinite_constant,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise DatasetIngestionError(
                        f"invalid JSONL line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise DatasetIngestionError(
                        f"JSONL line {line_number} must be an object"
                    )
                yield value


class ParquetDatasetReader(DatasetRowReader):
    """Lazy PyArrow reader for one local Parquet file."""

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
    ) -> None:
        """Initialize a Parquet reader.

        Args:
            path: Local Parquet source path.
            batch_size: Maximum rows materialized per PyArrow batch.
        """
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        self._path = Path(path)
        self._batch_size = batch_size

    @property
    def source_format(self) -> DatasetSourceFormat:
        """Return the Parquet source format."""
        return DatasetSourceFormat.PARQUET

    def content_digest(self) -> str:
        """Hash the exact Parquet file bytes."""
        return file_content_digest(self._path)

    def rows(self) -> Iterator[Mapping[str, Any]]:
        """Yield PyArrow-converted rows in physical scan order."""
        try:
            parquet = importlib.import_module("pyarrow.parquet")
        except ImportError as exc:
            raise DatasetReaderUnavailableError(
                "Parquet ingestion requires the 'parquet' SDK extra: "
                "pip install 'iso-obs[parquet]'"
            ) from exc
        parquet_file = parquet.ParquetFile(self._path)
        for batch in parquet_file.iter_batches(batch_size=self._batch_size):
            for row in batch.to_pylist():
                if not isinstance(row, Mapping):
                    raise DatasetIngestionError(
                        "Parquet reader produced a non-mapping row"
                    )
                yield row


class DatasetIngestionError(ValueError):
    """Raised when source rows cannot satisfy an explicit ingestion plan."""


class DatasetReaderUnavailableError(DatasetIngestionError):
    """Raised when an optional source-reader dependency is unavailable."""


@dataclass(slots=True)
class _ChannelAccumulator:
    """Mutable ingestion state for one episode channel."""

    timestamps: list[float] = field(default_factory=list)
    payloads: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class _EpisodeAccumulator:
    """Mutable ingestion state for one source episode."""

    independence_unit_id: str
    source_split: str | None
    row_timestamps: list[float] = field(default_factory=list)
    row_projections: list[dict[str, Any]] = field(default_factory=list)
    channels: dict[str, _ChannelAccumulator] = field(default_factory=dict)


def ingest_dataset_source(
    plan: DatasetIngestionPlan,
    path: str | PathLike[str],
) -> DatasetIngestionResult:
    """Ingest one local JSONL or Parquet source using its declared plan.

    Args:
        plan: Versioned field, channel, timestamp, and provenance mapping.
        path: Local source file.

    Returns:
        Content-addressed dataset bundle and ingestion report.
    """
    reader: DatasetRowReader
    if plan.source_format is DatasetSourceFormat.JSONL:
        reader = JsonlDatasetReader(path)
    elif plan.source_format is DatasetSourceFormat.PARQUET:
        reader = ParquetDatasetReader(path)
    else:
        raise DatasetReaderUnavailableError(
            "MCAP sources require an McapDatasetIngestionPlan and "
            "ingest_mcap_source()"
        )
    return ingest_dataset_rows(plan, reader)


def ingest_dataset_rows(
    plan: DatasetIngestionPlan,
    reader: DatasetRowReader,
) -> DatasetIngestionResult:
    """Convert source rows into a bundle without epistemic inference.

    Args:
        plan: Explicit ingestion mapping and dataset scope.
        reader: Format-specific reader preserving source row order.

    Returns:
        Content-addressed dataset bundle and transformation report.

    Raises:
        DatasetIngestionError: If required fields or episode identities conflict.
    """
    if reader.source_format is not plan.source_format:
        raise DatasetIngestionError("reader format does not match ingestion plan")
    source_digest = reader.content_digest()
    NamedDigest("dataset source", source_digest)
    episodes: dict[str, _EpisodeAccumulator] = {}
    sample_counts = {rule.channel_id: 0 for rule in plan.channel_rules}
    missing_counts = {rule.channel_id: 0 for rule in plan.channel_rules}
    row_count = 0

    for row_index, row in enumerate(reader.rows()):
        row_count += 1
        path = f"row {row_index + 1}"
        episode_id = _required_text_field(row, plan.episode_id_field, path)
        independence_id = _required_text_field(
            row,
            plan.independence_unit_id_field,
            path,
        )
        timestamp = _timestamp_seconds(
            _required_field(row, plan.timestamp_field, path),
            plan.timestamp_unit,
            f"{path} timestamp",
        )
        source_split = _optional_text_field(row, plan.source_split_field, path)
        accumulator = episodes.get(episode_id)
        if accumulator is None:
            accumulator = _EpisodeAccumulator(independence_id, source_split)
            episodes[episode_id] = accumulator
        elif accumulator.independence_unit_id != independence_id:
            raise DatasetIngestionError(
                f"{path} changes independence unit within episode {episode_id!r}"
            )
        elif accumulator.source_split != source_split:
            raise DatasetIngestionError(
                f"{path} changes source split within episode {episode_id!r}"
            )
        accumulator.row_timestamps.append(timestamp)
        row_projection: dict[str, Any] = {
            "channels": [],
            "timestamp_seconds": timestamp,
        }
        projected_channels: list[dict[str, Any]] = row_projection["channels"]

        for rule in plan.channel_rules:
            payload = _resolve_field(row, rule.payload_field)
            if payload is _MISSING or (
                payload is None and rule.null_payload_is_missing
            ):
                missing_counts[rule.channel_id] += 1
                projected_channels.append(
                    {
                        "channel_id": rule.channel_id,
                        "present": False,
                    }
                )
                continue
            channel_timestamp = timestamp
            if rule.timestamp_field is not None:
                channel_timestamp = _timestamp_seconds(
                    _required_field(row, rule.timestamp_field, path),
                    plan.timestamp_unit,
                    f"{path} channel {rule.channel_id!r} timestamp",
                )
            normalized_payload = _canonical_payload(payload)
            channel = accumulator.channels.setdefault(
                rule.channel_id,
                _ChannelAccumulator(),
            )
            channel.timestamps.append(channel_timestamp)
            channel.payloads.append(normalized_payload)
            sample_counts[rule.channel_id] += 1
            projected_channels.append(
                {
                    "channel_id": rule.channel_id,
                    "payload": normalized_payload,
                    "present": True,
                    "timestamp_seconds": channel_timestamp,
                }
            )
        accumulator.row_projections.append(row_projection)

    if reader.content_digest() != source_digest:
        raise DatasetIngestionError("source content changed during ingestion")

    manifest = DatasetManifest(
        dataset_id=plan.dataset_id,
        dataset_version=plan.dataset_version,
        content_digest=source_digest,
        source_uri=plan.source_uri,
        license_id=plan.license_id,
        scope=plan.scope,
    )
    episode_manifests, timing_traces = _build_episode_artifacts(
        plan,
        manifest,
        episodes,
    )
    bundle = DatasetBundle(
        schema_version=DATASET_BUNDLE_SCHEMA_VERSION,
        manifest=manifest,
        episodes=episode_manifests,
        timing_traces=timing_traces,
    )
    summaries = tuple(
        ChannelIngestionSummary(
            channel_id=rule.channel_id,
            modality_id=rule.modality_id,
            sample_count=sample_counts[rule.channel_id],
            missing_payload_row_count=missing_counts[rule.channel_id],
            required=rule.required,
        )
        for rule in plan.channel_rules
    )
    issues = _ingestion_issues(row_count, plan.channel_rules, sample_counts)
    if row_count == 0:
        disposition = DatasetIngestionDisposition.INSUFFICIENT_EVIDENCE
    elif issues:
        disposition = DatasetIngestionDisposition.REVIEW_REQUIRED
    else:
        disposition = DatasetIngestionDisposition.INGESTED_WITHIN_DECLARED_MAPPING
    report = DatasetIngestionReport(
        schema_version=DATASET_INGESTION_REPORT_SCHEMA_VERSION,
        plan_content_digest=plan.content_digest(),
        source_content_digest=source_digest,
        source_format=plan.source_format,
        dataset_manifest_digest=manifest.manifest_digest(),
        bundle_content_digest=bundle.content_digest(),
        disposition=disposition,
        row_count=row_count,
        episode_count=len(episode_manifests),
        channel_summaries=summaries,
        issues=issues,
        scope=plan.scope,
        limitations=(
            *plan.limitations,
            "Ingestion establishes structural mapping only; it does not establish "
            "label validity, temporal alignment, representativeness, or causality.",
            "No labels, clock relationships, or split roles are inferred.",
            "Payload values are content-addressed but are not copied into the "
            "dataset bundle.",
        ),
    )
    return DatasetIngestionResult(bundle=bundle, report=report)


def _build_episode_artifacts(
    plan: DatasetIngestionPlan,
    manifest: DatasetManifest,
    accumulators: Mapping[str, _EpisodeAccumulator],
) -> tuple[tuple[EpisodeManifest, ...], tuple[ChannelTimingTrace, ...]]:
    """Materialize episode and timing artifacts from mapped source rows."""
    episodes: list[EpisodeManifest] = []
    traces: list[ChannelTimingTrace] = []
    rule_by_channel = {rule.channel_id: rule for rule in plan.channel_rules}
    manifest_digest = manifest.manifest_digest()
    for episode_id, accumulator in sorted(accumulators.items()):
        modality_payloads: dict[str, list[dict[str, Any]]] = {}
        for channel_id, channel in sorted(accumulator.channels.items()):
            rule = rule_by_channel[channel_id]
            samples = [
                {"payload": payload, "timestamp_seconds": timestamp}
                for timestamp, payload in zip(
                    channel.timestamps,
                    channel.payloads,
                    strict=True,
                )
            ]
            modality_payloads.setdefault(rule.modality_id, []).append(
                {"channel_id": channel_id, "samples": samples}
            )
            traces.append(
                ChannelTimingTrace(
                    episode_id=episode_id,
                    channel_id=channel_id,
                    modality_id=rule.modality_id,
                    clock_id=rule.clock_id,
                    content_digest=sha256_digest(_canonical_json(samples)),
                    timestamps_seconds=tuple(channel.timestamps),
                    timestamp_uncertainty_seconds=(rule.timestamp_uncertainty_seconds),
                )
            )
        modality_digests = tuple(
            NamedDigest(
                modality_id,
                sha256_digest(_canonical_json(payloads)),
            )
            for modality_id, payloads in sorted(modality_payloads.items())
        )
        if not modality_digests:
            raise DatasetIngestionError(
                f"episode {episode_id!r} contains no mapped channel payloads"
            )
        episodes.append(
            EpisodeManifest(
                dataset_manifest_digest=manifest_digest,
                episode_id=episode_id,
                independence_unit_id=accumulator.independence_unit_id,
                content_digest=sha256_digest(
                    _canonical_json(accumulator.row_projections)
                ),
                start_seconds=min(accumulator.row_timestamps),
                end_seconds=max(accumulator.row_timestamps),
                modality_digests=modality_digests,
                source_split=accumulator.source_split,
            )
        )
    return tuple(episodes), tuple(traces)


def _ingestion_issues(
    row_count: int,
    rules: Sequence[ChannelIngestionRule],
    sample_counts: Mapping[str, int],
) -> tuple[DatasetIngestionIssue, ...]:
    """Identify empty sources and required channels without samples."""
    if row_count == 0:
        return (
            DatasetIngestionIssue(
                kind=DatasetIngestionIssueKind.EMPTY_SOURCE,
                channel_id=None,
                description="Source contains no dataset rows.",
            ),
        )
    return tuple(
        DatasetIngestionIssue(
            kind=DatasetIngestionIssueKind.REQUIRED_CHANNEL_EMPTY,
            channel_id=rule.channel_id,
            description="Required channel contains no mapped payload samples.",
        )
        for rule in rules
        if rule.required and sample_counts[rule.channel_id] == 0
    )


def _required_field(
    row: Mapping[str, Any],
    selector: SourceField,
    row_label: str,
) -> Any:
    """Resolve one required field or raise a path-specific source error."""
    value = _resolve_field(row, selector)
    if value is _MISSING:
        dotted = ".".join(selector.path)
        raise DatasetIngestionError(f"{row_label} is missing field {dotted!r}")
    return value


def _required_text_field(
    row: Mapping[str, Any],
    selector: SourceField,
    row_label: str,
) -> str:
    """Resolve one required nonempty string source field."""
    value = _required_field(row, selector, row_label)
    if not isinstance(value, str) or not value.strip():
        dotted = ".".join(selector.path)
        raise DatasetIngestionError(
            f"{row_label} field {dotted!r} must be a nonempty string"
        )
    return value.strip()


def _optional_text_field(
    row: Mapping[str, Any],
    selector: SourceField | None,
    row_label: str,
) -> str | None:
    """Resolve one optional string field without inventing missing values."""
    if selector is None:
        return None
    value = _resolve_field(row, selector)
    if value is _MISSING or value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        dotted = ".".join(selector.path)
        raise DatasetIngestionError(
            f"{row_label} field {dotted!r} must be a nonempty string or null"
        )
    return value.strip()


def _resolve_field(row: Mapping[str, Any], selector: SourceField) -> Any:
    """Resolve a nested mapping path, preserving absent versus null."""
    current: Any = row
    for component in selector.path:
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


def _timestamp_seconds(value: Any, unit: TimestampUnit, label: str) -> float:
    """Convert a finite numeric timestamp into seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetIngestionError(f"{label} must be numeric")
    timestamp = float(value) / unit.units_per_second
    if not math.isfinite(timestamp):
        raise DatasetIngestionError(f"{label} must be finite")
    return timestamp


def _canonical_payload(value: Any) -> Any:
    """Normalize common JSON and Parquet payload values for stable hashing."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetIngestionError("payload contains a non-finite number")
        return value
    if isinstance(value, bytes):
        return {
            "byte_count": len(value),
            "content_digest": sha256_digest(value),
            "type": "bytes",
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DatasetIngestionError("payload mapping keys must be strings")
            result[key] = _canonical_payload(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_payload(item) for item in value]
    if hasattr(value, "isoformat"):
        return {
            "type": type(value).__name__,
            "value": str(value.isoformat()),
        }
    raise DatasetIngestionError(
        f"unsupported payload value type: {type(value).__name__}"
    )


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Construct a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetIngestionError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def file_content_digest(path: str | PathLike[str]) -> str:
    """Hash a local source file incrementally to bound memory use.

    Args:
        path: Local file whose exact bytes should be identified.

    Returns:
        SHA-256 digest in the SDK's normalized representation.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_FILE_DIGEST_CHUNK_BYTES):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _reject_nonfinite_constant(value: str) -> Any:
    """Reject non-standard JSON NaN and infinity tokens."""
    raise DatasetIngestionError(f"non-finite JSON number is forbidden: {value}")


def _require_text(value: str, label: str) -> None:
    """Require a nonempty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _nonnegative_finite(value: float, label: str) -> float:
    """Return one finite, nonnegative float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _require_unique(values: Iterable[Any], label: str) -> None:
    """Require hashable values to contain no duplicates."""
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be unique")


def _unique_text(values: Iterable[str], label: str) -> tuple[str, ...]:
    """Validate, deduplicate, and sort an optional text collection."""
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain nonempty strings")
        normalized.add(value.strip())
    return tuple(sorted(normalized))
