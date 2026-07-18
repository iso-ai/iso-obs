"""Evidence-preserving rosbag2 ingestion across split MCAP recordings.

The adapter treats ``metadata.yaml`` as a declared recording manifest and
independently verifies its file statistics against CRC-validated MCAP streams.
It preserves metadata file order without inventing a global message ordering.
"""

from __future__ import annotations

import importlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any

from .dataset_ingestion import (
    DatasetIngestionError,
    DatasetIngestionIssue,
    DatasetIngestionIssueKind,
    DatasetIngestionReport,
    DatasetIngestionResult,
    DatasetReaderUnavailableError,
    DatasetRowReader,
    DatasetSourceFormat,
    file_content_digest,
    ingest_dataset_rows,
)
from .dataset_io import DatasetBundle
from .dataset_mcap import (
    McapDatasetIngestionPlan,
    McapDatasetRowReader,
    finalize_mcap_ingestion,
)
from .evidence import NamedDigest, _canonical_json, sha256_digest

ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION = "iso-obs.rosbag2-mcap-ingestion-plan.v1"
ROSBAG2_RECORDING_EVIDENCE_SCHEMA_VERSION = "iso-obs.rosbag2-recording-evidence.v1"
DEFAULT_MAX_ROSBAG2_METADATA_BYTES = 4 * 1024 * 1024
SUPPORTED_ROSBAG2_METADATA_VERSIONS = frozenset(range(5, 10))


class Rosbag2RecordingDisposition(StrEnum):
    """Strength of the rosbag2 manifest verification result."""

    VERIFIED = "verified"
    REVIEW_REQUIRED = "review_required"


class Rosbag2RecordingIssueKind(StrEnum):
    """Typed disagreement between metadata and observed recording bytes."""

    COUNT_MISMATCH = "count_mismatch"
    FILE_TIME_OVERLAP = "file_time_overlap"
    TIME_MISMATCH = "time_mismatch"
    TOPIC_MISMATCH = "topic_mismatch"


@dataclass(frozen=True, slots=True)
class Rosbag2McapIngestionPlan:
    """Versioned rosbag2 recording plan enclosing an MCAP mapping."""

    schema_version: str
    mcap_plan: McapDatasetIngestionPlan
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the wrapper contract and interpretation limits."""
        if self.schema_version != ROSBAG2_MCAP_INGESTION_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported rosbag2 MCAP ingestion plan version")
        limitations = _unique_text(self.limitations, "rosbag2 plan limitations")
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact enclosing plan digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class Rosbag2RecordingIssue:
    """One recording-level manifest verification finding."""

    kind: Rosbag2RecordingIssueKind
    relative_file_path: str | None
    description: str

    def __post_init__(self) -> None:
        """Validate issue type, optional file identity, and description."""
        object.__setattr__(self, "kind", Rosbag2RecordingIssueKind(self.kind))
        if self.relative_file_path is not None:
            _require_text(self.relative_file_path, "rosbag2 issue file path")
        _require_text(self.description, "rosbag2 issue description")


@dataclass(frozen=True, slots=True)
class Rosbag2FileEvidence:
    """Declared and independently observed evidence for one split file."""

    relative_path: str
    content_digest: str
    byte_count: int
    declared_start_time_ns: int
    declared_duration_ns: int
    declared_message_count: int
    observed_start_time_ns: int | None
    observed_end_time_ns: int | None
    observed_message_count: int

    def __post_init__(self) -> None:
        """Validate file identity, counts, times, and digest."""
        _require_text(self.relative_path, "rosbag2 relative file path")
        NamedDigest("rosbag2 file", self.content_digest)
        _nonnegative_integer(self.byte_count, "rosbag2 file byte count")
        _integer(self.declared_start_time_ns, "rosbag2 declared file start")
        _nonnegative_integer(
            self.declared_duration_ns,
            "rosbag2 declared file duration",
        )
        _nonnegative_integer(
            self.declared_message_count,
            "rosbag2 declared file message count",
        )
        _nonnegative_integer(
            self.observed_message_count,
            "rosbag2 observed file message count",
        )
        if (self.observed_start_time_ns is None) != (self.observed_end_time_ns is None):
            raise ValueError("observed rosbag2 time bounds must be supplied together")
        if self.observed_start_time_ns is not None:
            start = _integer(
                self.observed_start_time_ns,
                "rosbag2 observed file start",
            )
            end = _integer(
                self.observed_end_time_ns,
                "rosbag2 observed file end",
            )
            if end < start:
                raise ValueError("observed rosbag2 file end precedes its start")


@dataclass(frozen=True, slots=True)
class Rosbag2TopicEvidence:
    """Declared and observed identity for one recording-level topic."""

    topic_name: str
    declared_type: str | None
    declared_serialization_format: str | None
    declared_message_count: int | None
    observed_schema_names: tuple[str | None, ...]
    observed_message_encodings: tuple[str, ...]
    observed_message_count: int

    def __post_init__(self) -> None:
        """Validate topic identity, contracts, and counts."""
        _require_text(self.topic_name, "rosbag2 topic name")
        if self.declared_type is not None:
            _require_text(self.declared_type, "rosbag2 declared topic type")
        if self.declared_serialization_format is not None:
            _require_text(
                self.declared_serialization_format,
                "rosbag2 declared serialization format",
            )
        if self.declared_message_count is not None:
            _nonnegative_integer(
                self.declared_message_count,
                "rosbag2 declared topic message count",
            )
        schemas = tuple(
            sorted(
                set(self.observed_schema_names),
                key=lambda value: value or "",
            )
        )
        encodings = tuple(sorted(set(self.observed_message_encodings)))
        for value in schemas:
            if value is not None:
                _require_text(value, "observed MCAP schema name")
        for value in encodings:
            _require_text(value, "observed MCAP message encoding")
        _nonnegative_integer(
            self.observed_message_count,
            "rosbag2 observed topic message count",
        )
        object.__setattr__(self, "observed_schema_names", schemas)
        object.__setattr__(self, "observed_message_encodings", encodings)


@dataclass(frozen=True, slots=True)
class Rosbag2RecordingEvidence:
    """Content-addressed verification of one rosbag2 recording manifest."""

    schema_version: str
    plan_content_digest: str
    metadata_content_digest: str
    recording_content_digest: str
    metadata_version: int
    storage_identifier: str
    declared_start_time_ns: int
    declared_duration_ns: int
    declared_message_count: int
    files: tuple[Rosbag2FileEvidence, ...]
    topics: tuple[Rosbag2TopicEvidence, ...]
    disposition: Rosbag2RecordingDisposition
    issues: tuple[Rosbag2RecordingIssue, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate recording evidence identity, ordering, and disposition."""
        if self.schema_version != ROSBAG2_RECORDING_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported rosbag2 recording evidence version")
        NamedDigest("rosbag2 ingestion plan", self.plan_content_digest)
        NamedDigest("rosbag2 metadata", self.metadata_content_digest)
        NamedDigest("rosbag2 recording", self.recording_content_digest)
        _positive_integer(self.metadata_version, "rosbag2 metadata version")
        _require_text(self.storage_identifier, "rosbag2 storage identifier")
        _integer(self.declared_start_time_ns, "rosbag2 declared recording start")
        _nonnegative_integer(
            self.declared_duration_ns,
            "rosbag2 declared recording duration",
        )
        _nonnegative_integer(
            self.declared_message_count,
            "rosbag2 declared recording message count",
        )
        files = tuple(self.files)
        if not files:
            raise ValueError("rosbag2 recording evidence requires source files")
        _require_unique(
            (item.relative_path for item in files),
            "rosbag2 evidence file paths",
        )
        topics = tuple(sorted(self.topics, key=lambda item: item.topic_name))
        _require_unique(
            (item.topic_name for item in topics),
            "rosbag2 evidence topic names",
        )
        issues = tuple(
            sorted(
                self.issues,
                key=lambda item: (
                    item.kind.value,
                    item.relative_file_path or "",
                    item.description,
                ),
            )
        )
        disposition = Rosbag2RecordingDisposition(self.disposition)
        if disposition is Rosbag2RecordingDisposition.VERIFIED and issues:
            raise ValueError("verified rosbag2 evidence cannot contain issues")
        if disposition is Rosbag2RecordingDisposition.REVIEW_REQUIRED and not issues:
            raise ValueError("review-required rosbag2 evidence needs issues")
        limitations = _unique_text(
            self.limitations,
            "rosbag2 recording evidence limitations",
        )
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "topics", topics)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize recording evidence to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the recording-evidence artifact digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class Rosbag2DatasetIngestionResult:
    """Dataset artifacts paired with recording-level provenance evidence."""

    dataset_result: DatasetIngestionResult
    recording_evidence: Rosbag2RecordingEvidence

    def __post_init__(self) -> None:
        """Require both outputs to identify the same source and plan."""
        if (
            self.dataset_result.report.source_content_digest
            != self.recording_evidence.recording_content_digest
        ):
            raise ValueError("dataset report identifies different rosbag2 bytes")
        if (
            self.dataset_result.report.plan_content_digest
            != self.recording_evidence.plan_content_digest
        ):
            raise ValueError("dataset report identifies a different rosbag2 plan")

    @property
    def bundle(self) -> DatasetBundle:
        """Return the generated dataset bundle."""
        return self.dataset_result.bundle

    @property
    def report(self) -> DatasetIngestionReport:
        """Return the generated dataset ingestion report."""
        return self.dataset_result.report


@dataclass(frozen=True, slots=True)
class _DeclaredFile:
    """Strictly parsed rosbag2 file metadata."""

    relative_path: str
    start_time_ns: int
    duration_ns: int
    message_count: int


@dataclass(frozen=True, slots=True)
class _DeclaredRecording:
    """Strictly parsed subset of rosbag2 recording metadata."""

    metadata_version: int
    storage_identifier: str
    start_time_ns: int
    duration_ns: int
    message_count: int
    files: tuple[_DeclaredFile, ...]
    topics: tuple[_DeclaredTopic, ...]


@dataclass(frozen=True, slots=True)
class _DeclaredTopic:
    """Strictly parsed rosbag2 topic metadata."""

    topic_name: str
    topic_type: str
    serialization_format: str
    message_count: int


@dataclass(frozen=True, slots=True)
class _FileInspection:
    """File evidence plus observations used for recording-level checks."""

    evidence: Rosbag2FileEvidence
    topic_counts: Mapping[str, int]
    topic_schema_names: Mapping[str, tuple[str | None, ...]]
    topic_message_encodings: Mapping[str, tuple[str, ...]]


class Rosbag2McapDatasetRowReader(DatasetRowReader):
    """Metadata-order reader across independently identified MCAP files."""

    def __init__(
        self,
        root: Path,
        metadata_path: Path,
        plan: McapDatasetIngestionPlan,
        evidence: Rosbag2RecordingEvidence,
    ) -> None:
        """Initialize a split-file reader from verified paths and evidence."""
        self._root = root
        self._metadata_path = metadata_path
        self._plan = plan
        self._evidence = evidence

    @property
    def source_format(self) -> DatasetSourceFormat:
        """Return the underlying MCAP source format."""
        return DatasetSourceFormat.MCAP

    def content_digest(self) -> str:
        """Re-hash metadata and every declared file as one recording."""
        metadata_digest = file_content_digest(self._metadata_path)
        files = tuple(
            (item.relative_path, file_content_digest(self._file_path(item)))
            for item in self._evidence.files
        )
        digest = _recording_content_digest(metadata_digest, files)
        if digest != self._evidence.recording_content_digest:
            raise DatasetIngestionError(
                "rosbag2 recording changed after manifest verification"
            )
        return digest

    def rows(self) -> Iterator[Mapping[str, Any]]:
        """Yield mapped messages in metadata file order and file physical order."""
        for index, item in enumerate(self._evidence.files):
            reader = McapDatasetRowReader(
                self._file_path(item),
                self._plan,
                source_file_path=item.relative_path,
                source_file_content_digest=item.content_digest,
                source_file_index=index,
            )
            yield from reader.rows()

    def _file_path(self, item: Rosbag2FileEvidence) -> Path:
        """Resolve an already validated recording-relative file path."""
        return _resolve_recording_file(self._root, item.relative_path)


def inspect_rosbag2_mcap_recording(
    plan: Rosbag2McapIngestionPlan,
    source: str | PathLike[str],
) -> tuple[Path, Path, Rosbag2RecordingEvidence]:
    """Parse and independently verify one rosbag2 MCAP recording.

    Args:
        plan: Enclosing rosbag2 and MCAP mapping contract.
        source: Recording directory or its ``metadata.yaml`` path.

    Returns:
        Recording root, metadata path, and content-addressed evidence.
    """
    root, metadata_path = _recording_paths(source)
    metadata_bytes = metadata_path.read_bytes()
    if len(metadata_bytes) > DEFAULT_MAX_ROSBAG2_METADATA_BYTES:
        raise DatasetIngestionError("rosbag2 metadata exceeds the 4 MiB parsing limit")
    metadata_digest = sha256_digest(metadata_bytes)
    declared = _parse_metadata(metadata_bytes)
    paths = tuple(
        _resolve_recording_file(root, item.relative_path) for item in declared.files
    )
    inspections = tuple(
        _inspect_file(path, item, plan.mcap_plan)
        for path, item in zip(paths, declared.files, strict=True)
    )
    evidence_files = tuple(item.evidence for item in inspections)
    topic_evidence = _topic_evidence(declared.topics, inspections)
    issues = _recording_issues(declared, evidence_files, topic_evidence)
    recording_digest = _recording_content_digest(
        metadata_digest,
        tuple((item.relative_path, item.content_digest) for item in evidence_files),
    )
    disposition = (
        Rosbag2RecordingDisposition.REVIEW_REQUIRED
        if issues
        else Rosbag2RecordingDisposition.VERIFIED
    )
    evidence = Rosbag2RecordingEvidence(
        schema_version=ROSBAG2_RECORDING_EVIDENCE_SCHEMA_VERSION,
        plan_content_digest=plan.content_digest(),
        metadata_content_digest=metadata_digest,
        recording_content_digest=recording_digest,
        metadata_version=declared.metadata_version,
        storage_identifier=declared.storage_identifier,
        declared_start_time_ns=declared.start_time_ns,
        declared_duration_ns=declared.duration_ns,
        declared_message_count=declared.message_count,
        files=evidence_files,
        topics=topic_evidence,
        disposition=disposition,
        issues=issues,
        limitations=(
            *plan.limitations,
            "Metadata file order is preserved; no global message ordering is "
            "inferred across split files.",
            "Observed timing extents use MCAP log timestamps and do not establish "
            "clock alignment or causal order.",
            "Repeated transient-local messages across split boundaries are "
            "preserved and are not deduplicated.",
        ),
    )
    return root, metadata_path, evidence


def ingest_rosbag2_mcap_source(
    plan: Rosbag2McapIngestionPlan,
    source: str | PathLike[str],
) -> Rosbag2DatasetIngestionResult:
    """Ingest every MCAP declared by one rosbag2 recording manifest.

    Args:
        plan: Enclosing recording and topic-mapping contract.
        source: Recording directory or its ``metadata.yaml`` path.

    Returns:
        Dataset artifacts and recording-level provenance evidence.
    """
    root, metadata_path, evidence = inspect_rosbag2_mcap_recording(plan, source)
    reader = Rosbag2McapDatasetRowReader(
        root,
        metadata_path,
        plan.mcap_plan,
        evidence,
    )
    generic_result = ingest_dataset_rows(
        plan.mcap_plan.normalized_plan(),
        reader,
    )
    recording_issues = tuple(
        DatasetIngestionIssue(
            kind=DatasetIngestionIssueKind.SOURCE_MANIFEST_MISMATCH,
            channel_id=None,
            description=issue.description,
        )
        for issue in evidence.issues
    )
    result = finalize_mcap_ingestion(
        plan.mcap_plan,
        generic_result,
        plan_content_digest=plan.content_digest(),
        additional_issues=recording_issues,
        additional_limitations=(
            *evidence.limitations,
            "The rosbag2 recording source digest commits to metadata.yaml and "
            "each declared file path and digest.",
        ),
    )
    return Rosbag2DatasetIngestionResult(
        dataset_result=result,
        recording_evidence=evidence,
    )


def _recording_paths(
    source: str | PathLike[str],
) -> tuple[Path, Path]:
    """Resolve a recording directory and its exact metadata path."""
    path = Path(source)
    if path.is_dir():
        root = path.resolve()
        metadata_candidate = root / "metadata.yaml"
        if not metadata_candidate.is_file():
            raise DatasetIngestionError("rosbag2 metadata.yaml does not exist")
        metadata_path = metadata_candidate.resolve()
        if not metadata_path.is_relative_to(root):
            raise DatasetIngestionError(
                "rosbag2 metadata.yaml escapes its recording root"
            )
    else:
        metadata_path = path.resolve()
        root = metadata_path.parent
        if metadata_path.name != "metadata.yaml":
            raise DatasetIngestionError(
                "rosbag2 source must be a directory or metadata.yaml"
            )
    if not metadata_path.is_file():
        raise DatasetIngestionError("rosbag2 metadata.yaml does not exist")
    return root, metadata_path


def _parse_metadata(encoded: bytes) -> _DeclaredRecording:
    """Parse a strict, size-bounded rosbag2 metadata document."""
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetIngestionError("rosbag2 metadata must be valid UTF-8") from exc
    payload = _strict_yaml_load(text)
    root = _mapping(payload, "rosbag2 metadata")
    if set(root) != {"rosbag2_bagfile_information"}:
        raise DatasetIngestionError(
            "rosbag2 metadata must contain only rosbag2_bagfile_information"
        )
    info = _mapping(
        root["rosbag2_bagfile_information"],
        "rosbag2_bagfile_information",
    )
    version = _positive_integer(
        info.get("version"),
        "rosbag2 metadata version",
    )
    if version not in SUPPORTED_ROSBAG2_METADATA_VERSIONS:
        raise DatasetIngestionError(
            f"rosbag2 metadata version {version} is unsupported; expected 5 through 9"
        )
    storage = _text(info.get("storage_identifier"), "rosbag2 storage identifier")
    if storage != "mcap":
        raise DatasetIngestionError(
            f"rosbag2 storage {storage!r} is unsupported; expected 'mcap'"
        )
    compression_mode = info.get("compression_mode", "")
    compression_format = info.get("compression_format", "")
    if compression_mode not in {"", None} or compression_format not in {"", None}:
        raise DatasetIngestionError(
            "externally compressed rosbag2 files are unsupported; decompress the "
            "recording before ingestion"
        )
    relative_paths = tuple(
        _text(item, "rosbag2 relative file path")
        for item in _sequence(
            info.get("relative_file_paths"),
            "rosbag2 relative_file_paths",
        )
    )
    _require_unique(relative_paths, "rosbag2 relative file paths")
    file_nodes = _sequence(info.get("files"), "rosbag2 files")
    files = tuple(_parse_declared_file(node) for node in file_nodes)
    if tuple(item.relative_path for item in files) != relative_paths:
        raise DatasetIngestionError(
            "rosbag2 files must exactly match relative_file_paths in the same order"
        )
    if not files:
        raise DatasetIngestionError("rosbag2 recording declares no files")
    topics = tuple(
        _parse_declared_topic(node)
        for node in _sequence(
            info.get("topics_with_message_count"),
            "rosbag2 topics_with_message_count",
        )
    )
    _require_unique(
        (item.topic_name for item in topics),
        "rosbag2 declared topic names",
    )
    return _DeclaredRecording(
        metadata_version=version,
        storage_identifier=storage,
        start_time_ns=_nested_integer(
            info.get("starting_time"),
            "nanoseconds_since_epoch",
            "rosbag2 recording starting time",
        ),
        duration_ns=_nested_nonnegative_integer(
            info.get("duration"),
            "nanoseconds",
            "rosbag2 recording duration",
        ),
        message_count=_nonnegative_integer(
            info.get("message_count"),
            "rosbag2 recording message count",
        ),
        files=files,
        topics=topics,
    )


def _parse_declared_file(value: Any) -> _DeclaredFile:
    """Parse one file entry from rosbag2 metadata."""
    item = _mapping(value, "rosbag2 file information")
    return _DeclaredFile(
        relative_path=_text(item.get("path"), "rosbag2 file path"),
        start_time_ns=_nested_integer(
            item.get("starting_time"),
            "nanoseconds_since_epoch",
            "rosbag2 file starting time",
        ),
        duration_ns=_nested_nonnegative_integer(
            item.get("duration"),
            "nanoseconds",
            "rosbag2 file duration",
        ),
        message_count=_nonnegative_integer(
            item.get("message_count"),
            "rosbag2 file message count",
        ),
    )


def _parse_declared_topic(value: Any) -> _DeclaredTopic:
    """Parse one recording-level topic declaration."""
    item = _mapping(value, "rosbag2 topic information")
    metadata = _mapping(
        item.get("topic_metadata"),
        "rosbag2 topic metadata",
    )
    return _DeclaredTopic(
        topic_name=_text(metadata.get("name"), "rosbag2 topic name"),
        topic_type=_text(metadata.get("type"), "rosbag2 topic type"),
        serialization_format=_text(
            metadata.get("serialization_format"),
            "rosbag2 topic serialization format",
        ),
        message_count=_nonnegative_integer(
            item.get("message_count"),
            "rosbag2 topic message count",
        ),
    )


def _strict_yaml_load(text: str) -> Any:
    """Load safe YAML while rejecting duplicate keys and aliases."""
    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise DatasetReaderUnavailableError(
            "rosbag2 ingestion requires the 'rosbag2' SDK extra: "
            "pip install 'iso-obs[rosbag2]'"
        ) from exc

    class StrictSafeLoader(yaml.SafeLoader):  # type: ignore[name-defined, misc]
        """Safe loader that forbids ambiguous mappings and aliases."""

        def compose_node(self, parent: Any, index: Any) -> Any:
            """Reject aliases before composing a YAML node."""
            if self.check_event(yaml.AliasEvent):
                raise DatasetIngestionError(
                    "rosbag2 metadata YAML aliases are forbidden"
                )
            return super().compose_node(parent, index)

        def construct_mapping(
            self,
            node: Any,
            deep: bool = False,
        ) -> dict[Any, Any]:
            """Reject duplicate YAML mapping keys."""
            mapping: dict[Any, Any] = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                try:
                    duplicate = key in mapping
                except TypeError as exc:
                    raise DatasetIngestionError(
                        "rosbag2 metadata contains an unhashable YAML key"
                    ) from exc
                if duplicate:
                    raise DatasetIngestionError(
                        f"duplicate rosbag2 metadata key: {key!r}"
                    )
                mapping[key] = self.construct_object(value_node, deep=deep)
            return mapping

    try:
        return yaml.load(text, Loader=StrictSafeLoader)
    except DatasetIngestionError:
        raise
    except Exception as exc:
        raise DatasetIngestionError(f"failed to parse rosbag2 metadata: {exc}") from exc


def _resolve_recording_file(root: Path, relative_path: str) -> Path:
    """Resolve one unambiguous metadata path within its recording root."""
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or pure.as_posix() != relative_path
    ):
        raise DatasetIngestionError(
            f"unsafe or noncanonical rosbag2 file path: {relative_path!r}"
        )
    if pure.suffix != ".mcap":
        raise DatasetIngestionError(f"rosbag2 file is not an MCAP: {relative_path!r}")
    try:
        path = root.joinpath(*pure.parts).resolve(strict=True)
    except OSError as exc:
        raise DatasetIngestionError(
            f"rosbag2 file does not exist: {relative_path!r}"
        ) from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise DatasetIngestionError(
            f"rosbag2 file escapes its recording root: {relative_path!r}"
        )
    return path


def _inspect_file(
    path: Path,
    declared: _DeclaredFile,
    plan: McapDatasetIngestionPlan,
) -> _FileInspection:
    """Scan all physical messages in one MCAP for independent statistics."""
    try:
        reader_module = importlib.import_module("mcap.reader")
    except ImportError as exc:
        raise DatasetReaderUnavailableError(
            "rosbag2 ingestion requires the 'rosbag2' SDK extra: "
            "pip install 'iso-obs[rosbag2]'"
        ) from exc
    count = 0
    start: int | None = None
    end: int | None = None
    topic_counts: Counter[str] = Counter()
    topic_schemas: defaultdict[str, set[str | None]] = defaultdict(set)
    topic_encodings: defaultdict[str, set[str]] = defaultdict(set)
    content_digest = file_content_digest(path)
    try:
        with path.open("rb") as stream:
            reader = reader_module.NonSeekingReader(
                stream,
                validate_crcs=plan.validate_crcs,
                record_size_limit=plan.record_size_limit_bytes,
            )
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                timestamp = _integer(message.log_time, "MCAP log timestamp")
                count += 1
                start = timestamp if start is None else min(start, timestamp)
                end = timestamp if end is None else max(end, timestamp)
                topic_counts[channel.topic] += 1
                topic_schemas[channel.topic].add(
                    schema.name if schema is not None else None
                )
                topic_encodings[channel.topic].add(channel.message_encoding)
    except DatasetIngestionError:
        raise
    except Exception as exc:
        raise DatasetIngestionError(
            f"failed to inspect rosbag2 file {declared.relative_path!r}: {exc}"
        ) from exc
    if file_content_digest(path) != content_digest:
        raise DatasetIngestionError(
            f"rosbag2 file {declared.relative_path!r} changed during inspection"
        )
    return _FileInspection(
        evidence=Rosbag2FileEvidence(
            relative_path=declared.relative_path,
            content_digest=content_digest,
            byte_count=path.stat().st_size,
            declared_start_time_ns=declared.start_time_ns,
            declared_duration_ns=declared.duration_ns,
            declared_message_count=declared.message_count,
            observed_start_time_ns=start,
            observed_end_time_ns=end,
            observed_message_count=count,
        ),
        topic_counts=dict(topic_counts),
        topic_schema_names={
            topic: tuple(values) for topic, values in topic_schemas.items()
        },
        topic_message_encodings={
            topic: tuple(values) for topic, values in topic_encodings.items()
        },
    )


def _topic_evidence(
    declared_topics: Sequence[_DeclaredTopic],
    inspections: Sequence[_FileInspection],
) -> tuple[Rosbag2TopicEvidence, ...]:
    """Aggregate declared and observed identities for every topic."""
    declared_by_name = {item.topic_name: item for item in declared_topics}
    observed_counts: Counter[str] = Counter()
    observed_schemas: defaultdict[str, set[str | None]] = defaultdict(set)
    observed_encodings: defaultdict[str, set[str]] = defaultdict(set)
    for inspection in inspections:
        observed_counts.update(inspection.topic_counts)
        for topic, values in inspection.topic_schema_names.items():
            observed_schemas[topic].update(values)
        for topic, values in inspection.topic_message_encodings.items():
            observed_encodings[topic].update(values)
    topic_names = sorted(set(declared_by_name) | set(observed_counts))
    return tuple(
        Rosbag2TopicEvidence(
            topic_name=topic,
            declared_type=(
                declared_by_name[topic].topic_type
                if topic in declared_by_name
                else None
            ),
            declared_serialization_format=(
                declared_by_name[topic].serialization_format
                if topic in declared_by_name
                else None
            ),
            declared_message_count=(
                declared_by_name[topic].message_count
                if topic in declared_by_name
                else None
            ),
            observed_schema_names=tuple(observed_schemas[topic]),
            observed_message_encodings=tuple(observed_encodings[topic]),
            observed_message_count=observed_counts[topic],
        )
        for topic in topic_names
    )


def _recording_issues(
    declared: _DeclaredRecording,
    files: Sequence[Rosbag2FileEvidence],
    topics: Sequence[Rosbag2TopicEvidence],
) -> tuple[Rosbag2RecordingIssue, ...]:
    """Compare declared recording statistics with independently observed bytes."""
    issues: list[Rosbag2RecordingIssue] = []
    for item in files:
        if item.declared_message_count != item.observed_message_count:
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.COUNT_MISMATCH,
                    item.relative_path,
                    f"File {item.relative_path!r} declares "
                    f"{item.declared_message_count} messages but contains "
                    f"{item.observed_message_count}.",
                )
            )
        if item.observed_message_count == 0:
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.TIME_MISMATCH,
                    item.relative_path,
                    f"File {item.relative_path!r} contains no messages, so its "
                    "declared starting time and duration cannot be independently "
                    "verified.",
                )
            )
        if item.observed_start_time_ns is not None:
            observed_duration = (
                item.observed_end_time_ns - item.observed_start_time_ns
                if item.observed_end_time_ns is not None
                else 0
            )
            if item.declared_start_time_ns != item.observed_start_time_ns:
                issues.append(
                    _issue(
                        Rosbag2RecordingIssueKind.TIME_MISMATCH,
                        item.relative_path,
                        f"File {item.relative_path!r} declared start time does "
                        "not match its observed MCAP log-time minimum.",
                    )
                )
            if item.declared_duration_ns != observed_duration:
                issues.append(
                    _issue(
                        Rosbag2RecordingIssueKind.TIME_MISMATCH,
                        item.relative_path,
                        f"File {item.relative_path!r} declared duration does "
                        "not match its observed MCAP log-time extent.",
                    )
                )
    declared_file_count = sum(item.declared_message_count for item in files)
    observed_count = sum(item.observed_message_count for item in files)
    if declared.message_count != declared_file_count:
        issues.append(
            _issue(
                Rosbag2RecordingIssueKind.COUNT_MISMATCH,
                None,
                "Recording message count does not equal the sum of declared "
                "file message counts.",
            )
        )
    if declared.message_count != observed_count:
        issues.append(
            _issue(
                Rosbag2RecordingIssueKind.COUNT_MISMATCH,
                None,
                "Recording message count does not equal the independently "
                "observed MCAP message count.",
            )
        )
    declared_topic_count = sum(item.message_count for item in declared.topics)
    if declared.message_count != declared_topic_count:
        issues.append(
            _issue(
                Rosbag2RecordingIssueKind.COUNT_MISMATCH,
                None,
                "Recording message count does not equal the sum of declared "
                "topic message counts.",
            )
        )
    for topic in topics:
        if topic.declared_message_count is None:
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.TOPIC_MISMATCH,
                    None,
                    f"Observed topic {topic.topic_name!r} is absent from rosbag2 "
                    "topic metadata.",
                )
            )
            continue
        if topic.declared_message_count != topic.observed_message_count:
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.COUNT_MISMATCH,
                    None,
                    f"Topic {topic.topic_name!r} declares "
                    f"{topic.declared_message_count} messages but contains "
                    f"{topic.observed_message_count}.",
                )
            )
        if topic.observed_schema_names and topic.observed_schema_names != (
            topic.declared_type,
        ):
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.TOPIC_MISMATCH,
                    None,
                    f"Topic {topic.topic_name!r} declared type "
                    f"{topic.declared_type!r} but observed MCAP schemas were "
                    f"{topic.observed_schema_names!r}.",
                )
            )
        if topic.observed_message_encodings and (
            topic.observed_message_encodings != (topic.declared_serialization_format,)
        ):
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.TOPIC_MISMATCH,
                    None,
                    f"Topic {topic.topic_name!r} declared serialization "
                    f"{topic.declared_serialization_format!r} but observed MCAP "
                    f"encodings were {topic.observed_message_encodings!r}.",
                )
            )
    observed_nonempty = tuple(
        item for item in files if item.observed_start_time_ns is not None
    )
    if observed_nonempty:
        observed_start = min(
            item.observed_start_time_ns
            for item in observed_nonempty
            if item.observed_start_time_ns is not None
        )
        observed_end = max(
            item.observed_end_time_ns
            for item in observed_nonempty
            if item.observed_end_time_ns is not None
        )
        if declared.start_time_ns != observed_start:
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.TIME_MISMATCH,
                    None,
                    "Recording declared start time does not match the earliest "
                    "observed MCAP log timestamp.",
                )
            )
        if declared.duration_ns != observed_end - observed_start:
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.TIME_MISMATCH,
                    None,
                    "Recording declared duration does not match the observed "
                    "MCAP log-time extent.",
                )
            )
    for previous, current in zip(files, files[1:], strict=False):
        if (
            previous.observed_end_time_ns is not None
            and current.observed_start_time_ns is not None
            and current.observed_start_time_ns <= previous.observed_end_time_ns
        ):
            issues.append(
                _issue(
                    Rosbag2RecordingIssueKind.FILE_TIME_OVERLAP,
                    current.relative_path,
                    f"File {current.relative_path!r} begins at or before the "
                    f"observed end of {previous.relative_path!r}; source order "
                    "is preserved without deduplication.",
                )
            )
    return tuple(issues)


def _issue(
    kind: Rosbag2RecordingIssueKind,
    relative_file_path: str | None,
    description: str,
) -> Rosbag2RecordingIssue:
    """Construct one recording issue without obscuring its source file."""
    return Rosbag2RecordingIssue(
        kind=kind,
        relative_file_path=relative_file_path,
        description=description,
    )


def _recording_content_digest(
    metadata_digest: str,
    files: Sequence[tuple[str, str]],
) -> str:
    """Commit to exact metadata and ordered file identities and digests."""
    return sha256_digest(
        _canonical_json(
            {
                "metadata_content_digest": metadata_digest,
                "ordered_files": [
                    {
                        "content_digest": content_digest,
                        "relative_path": relative_path,
                    }
                    for relative_path, content_digest in files
                ],
            }
        )
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Require a string-keyed mapping."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DatasetIngestionError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    """Require a non-string sequence."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DatasetIngestionError(f"{label} must be a sequence")
    return value


def _text(value: Any, label: str) -> str:
    """Require and return nonempty text."""
    if not isinstance(value, str) or not value.strip():
        raise DatasetIngestionError(f"{label} must be nonempty text")
    return value


def _require_text(value: Any, label: str) -> None:
    """Require nonempty text."""
    _text(value, label)


def _integer(value: Any, label: str) -> int:
    """Require and return a non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetIngestionError(f"{label} must be an integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    """Require and return a positive integer."""
    result = _integer(value, label)
    if result < 1:
        raise DatasetIngestionError(f"{label} must be positive")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    """Require and return a nonnegative integer."""
    result = _integer(value, label)
    if result < 0:
        raise DatasetIngestionError(f"{label} must be nonnegative")
    return result


def _nested_integer(value: Any, key: str, label: str) -> int:
    """Read one required integer from a nested metadata mapping."""
    return _integer(_mapping(value, label).get(key), label)


def _nested_nonnegative_integer(value: Any, key: str, label: str) -> int:
    """Read one required nonnegative integer from a nested mapping."""
    return _nonnegative_integer(_mapping(value, label).get(key), label)


def _require_unique(values: Iterable[Any], label: str) -> None:
    """Require a sequence to contain no duplicate values."""
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise DatasetIngestionError(f"{label} must be unique")


def _unique_text(values: Sequence[str], label: str) -> tuple[str, ...]:
    """Validate, normalize, and sort a text collection."""
    normalized: set[str] = set()
    for value in values:
        normalized.add(_text(value, label).strip())
    return tuple(sorted(normalized))
