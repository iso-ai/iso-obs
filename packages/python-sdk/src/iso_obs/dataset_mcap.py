"""Explicit MCAP and ROS 2 ingestion into dataset reliability artifacts.

The adapter reads MCAP records in physical file order with full CRC validation,
maps only declared topics, and keeps log, publish, and ROS header timestamps
epistemically distinct. ROS 2 decoding is optional and requires an embedded
``ros2msg`` schema supported by the official MCAP decoder.
"""

from __future__ import annotations

import array
import importlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Any

from .dataset_ingestion import (
    DATASET_INGESTION_PLAN_SCHEMA_VERSION,
    ChannelIngestionRule,
    DatasetIngestionDisposition,
    DatasetIngestionError,
    DatasetIngestionIssue,
    DatasetIngestionIssueKind,
    DatasetIngestionPlan,
    DatasetIngestionResult,
    DatasetReaderUnavailableError,
    DatasetRowReader,
    DatasetSourceFormat,
    SourceField,
    TimestampUnit,
    file_content_digest,
    ingest_dataset_rows,
)
from .dataset_reliability import DatasetScope
from .evidence import NamedDigest, _canonical_json, sha256_digest

MCAP_INGESTION_PLAN_SCHEMA_VERSION = "iso-obs.mcap-ingestion-plan.v1"
DEFAULT_MCAP_RECORD_SIZE_LIMIT_BYTES = 256 * 1024 * 1024


class McapPayloadMode(StrEnum):
    """How a declared MCAP topic payload enters content addressing."""

    RAW_BYTES = "raw_bytes"
    JSON = "json"
    ROS2_DECODED = "ros2_decoded"


class McapTimestampSource(StrEnum):
    """MCAP or ROS 2 timestamp selected for one scientific role."""

    LOG_TIME = "log_time"
    PUBLISH_TIME = "publish_time"
    ROS_HEADER = "ros_header"


@dataclass(frozen=True, slots=True)
class McapEpisodeWindow:
    """Explicit episode membership interval on a declared MCAP clock."""

    episode_id: str
    independence_unit_id: str
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    source_split: str | None = None

    def __post_init__(self) -> None:
        """Validate identity and a half-open or unbounded episode interval."""
        _require_text(self.episode_id, "MCAP episode ID")
        _require_text(
            self.independence_unit_id,
            "MCAP episode independence unit ID",
        )
        if (self.start_time_ns is None) != (self.end_time_ns is None):
            raise ValueError("MCAP episode bounds must be supplied together")
        if self.start_time_ns is not None and self.end_time_ns is not None:
            start = _integer(self.start_time_ns, "MCAP episode start time")
            end = _integer(self.end_time_ns, "MCAP episode end time")
            if start >= end:
                raise ValueError("MCAP episode window must be nonempty")
            object.__setattr__(self, "start_time_ns", start)
            object.__setattr__(self, "end_time_ns", end)
        if self.source_split is not None:
            _require_text(self.source_split, "MCAP episode source split")

    def contains(self, timestamp_ns: int) -> bool:
        """Return whether a timestamp belongs to this half-open window.

        Args:
            timestamp_ns: Timestamp on the plan's episode-window clock.

        Returns:
            ``True`` for an unbounded window or ``start <= time < end``.
        """
        if self.start_time_ns is None or self.end_time_ns is None:
            return True
        return self.start_time_ns <= timestamp_ns < self.end_time_ns


@dataclass(frozen=True, slots=True)
class McapTopicRule:
    """Explicit mapping and validation policy for one MCAP topic."""

    topic: str
    channel_id: str
    modality_id: str
    clock_id: str
    payload_mode: McapPayloadMode
    timestamp_source: McapTimestampSource
    timestamp_uncertainty_seconds: float
    expected_message_encoding: str
    expected_schema_name: str | None = None
    expected_schema_encoding: str | None = None
    ros_header_stamp_path: tuple[str, ...] = ("header", "stamp")
    required: bool = True

    def __post_init__(self) -> None:
        """Validate topic identity, encoding, clock, and decoding policy."""
        _require_text(self.topic, "MCAP topic")
        _require_text(self.channel_id, "MCAP channel ID")
        _require_text(self.modality_id, "MCAP modality ID")
        _require_text(self.clock_id, "MCAP clock ID")
        _require_text(
            self.expected_message_encoding,
            "MCAP expected message encoding",
        )
        if self.expected_schema_name is not None:
            _require_text(self.expected_schema_name, "MCAP expected schema name")
        if self.expected_schema_encoding is not None:
            _require_text(
                self.expected_schema_encoding,
                "MCAP expected schema encoding",
            )
        path = _unique_path(self.ros_header_stamp_path, "ROS header stamp path")
        uncertainty = _nonnegative_finite(
            self.timestamp_uncertainty_seconds,
            "MCAP timestamp uncertainty",
        )
        if not isinstance(self.required, bool):
            raise ValueError("MCAP topic required must be boolean")
        payload_mode = McapPayloadMode(self.payload_mode)
        timestamp_source = McapTimestampSource(self.timestamp_source)
        if (
            timestamp_source is McapTimestampSource.ROS_HEADER
            and payload_mode is not McapPayloadMode.ROS2_DECODED
        ):
            raise ValueError("ROS header timestamps require decoded ROS 2 payload mode")
        object.__setattr__(self, "payload_mode", payload_mode)
        object.__setattr__(self, "timestamp_source", timestamp_source)
        object.__setattr__(self, "ros_header_stamp_path", path)
        object.__setattr__(
            self,
            "timestamp_uncertainty_seconds",
            uncertainty,
        )


@dataclass(frozen=True, slots=True)
class McapDatasetIngestionPlan:
    """Versioned topic, episode, timestamp, and provenance mapping for MCAP."""

    schema_version: str
    plan_id: str
    plan_version: str
    dataset_id: str
    dataset_version: str
    source_uri: str
    license_id: str
    scope: DatasetScope
    topic_rules: tuple[McapTopicRule, ...]
    episode_windows: tuple[McapEpisodeWindow, ...]
    episode_window_timestamp_source: McapTimestampSource
    validate_crcs: bool = True
    record_size_limit_bytes: int = DEFAULT_MCAP_RECORD_SIZE_LIMIT_BYTES
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate mappings, windows, CRC policy, and source identity."""
        if self.schema_version != MCAP_INGESTION_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported MCAP ingestion plan schema version")
        _require_text(self.plan_id, "MCAP ingestion plan ID")
        _require_text(self.plan_version, "MCAP ingestion plan version")
        _require_text(self.dataset_id, "MCAP dataset ID")
        _require_text(self.dataset_version, "MCAP dataset version")
        _require_text(self.source_uri, "MCAP source URI")
        _require_text(self.license_id, "MCAP dataset license ID")
        rules = tuple(sorted(self.topic_rules, key=lambda item: item.topic))
        if not rules:
            raise ValueError("MCAP ingestion plan requires topic rules")
        _require_unique((item.topic for item in rules), "MCAP topic rules")
        _require_unique((item.channel_id for item in rules), "MCAP channel IDs")
        declared_modalities = set(self.scope.modality_ids)
        for rule in rules:
            if rule.modality_id not in declared_modalities:
                raise ValueError("MCAP topic rule uses an undeclared modality")
        windows = tuple(
            sorted(
                self.episode_windows,
                key=lambda item: (
                    item.start_time_ns if item.start_time_ns is not None else -math.inf,
                    item.episode_id,
                ),
            )
        )
        if not windows:
            raise ValueError("MCAP ingestion plan requires episode windows")
        _require_unique((item.episode_id for item in windows), "MCAP episode IDs")
        _validate_episode_windows(windows)
        window_source = McapTimestampSource(self.episode_window_timestamp_source)
        if window_source is McapTimestampSource.ROS_HEADER:
            raise ValueError("MCAP episode windows must use log or publish timestamps")
        if not isinstance(self.validate_crcs, bool):
            raise ValueError("MCAP CRC validation policy must be boolean")
        record_limit = _positive_integer(
            self.record_size_limit_bytes,
            "MCAP record size limit",
        )
        limitations = _unique_text(self.limitations, "MCAP plan limitations")
        object.__setattr__(self, "topic_rules", rules)
        object.__setattr__(self, "episode_windows", windows)
        object.__setattr__(
            self,
            "episode_window_timestamp_source",
            window_source,
        )
        object.__setattr__(self, "record_size_limit_bytes", record_limit)
        object.__setattr__(self, "limitations", limitations)

    def to_json(self) -> str:
        """Serialize the MCAP ingestion plan to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact MCAP ingestion-plan digest."""
        return sha256_digest(self.to_json())

    def normalized_plan(self) -> DatasetIngestionPlan:
        """Build the row mapping consumed by the generic ingestion engine.

        Returns:
            Equivalent narrow-row plan whose provenance is replaced by this
            MCAP plan's digest in the final ingestion report.
        """
        return DatasetIngestionPlan(
            schema_version=DATASET_INGESTION_PLAN_SCHEMA_VERSION,
            plan_id=self.plan_id,
            plan_version=self.plan_version,
            source_format=DatasetSourceFormat.MCAP,
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            source_uri=self.source_uri,
            license_id=self.license_id,
            scope=self.scope,
            episode_id_field=SourceField(("episode_id",)),
            independence_unit_id_field=SourceField(("independence_unit_id",)),
            timestamp_field=SourceField(("window_timestamp_ns",)),
            timestamp_unit=TimestampUnit.NANOSECONDS,
            channel_rules=tuple(
                ChannelIngestionRule(
                    channel_id=rule.channel_id,
                    modality_id=rule.modality_id,
                    clock_id=rule.clock_id,
                    payload_field=SourceField(("topics", rule.topic)),
                    timestamp_field=SourceField(("topic_timestamps_ns", rule.topic)),
                    timestamp_uncertainty_seconds=(rule.timestamp_uncertainty_seconds),
                    required=rule.required,
                )
                for rule in self.topic_rules
            ),
            source_split_field=SourceField(("source_split",)),
            limitations=(
                *self.limitations,
                "MCAP topics absent from the plan are outside ingestion scope.",
                "Messages outside declared episode windows are outside ingestion "
                "scope.",
                "MCAP log, publish, and ROS header clocks remain distinct.",
            ),
        )


class McapDatasetRowReader(DatasetRowReader):
    """Physical-order, CRC-validating reader for mapped MCAP topics."""

    def __init__(
        self,
        path: str | PathLike[str],
        plan: McapDatasetIngestionPlan,
        *,
        source_file_path: str | None = None,
        source_file_content_digest: str | None = None,
        source_file_index: int | None = None,
    ) -> None:
        """Initialize a reader from a validated MCAP plan.

        Args:
            path: Local MCAP file.
            plan: Explicit topic, clock, decoding, and episode mapping.
            source_file_path: Optional recording-relative file identity.
            source_file_content_digest: Digest of the identified source file.
            source_file_index: Zero-based file position in a recording manifest.
        """
        self._path = Path(path)
        self._plan = plan
        provenance = (
            source_file_path,
            source_file_content_digest,
            source_file_index,
        )
        if any(value is not None for value in provenance) and any(
            value is None for value in provenance
        ):
            raise ValueError("MCAP source-file provenance must be supplied together")
        if source_file_path is not None:
            _require_text(source_file_path, "MCAP source file path")
        if source_file_content_digest is not None:
            NamedDigest("MCAP source file", source_file_content_digest)
        if source_file_index is not None and (
            isinstance(source_file_index, bool)
            or not isinstance(source_file_index, int)
            or source_file_index < 0
        ):
            raise ValueError("MCAP source file index must be nonnegative")
        self._source_file_path = source_file_path
        self._source_file_content_digest = source_file_content_digest
        self._source_file_index = source_file_index

    @property
    def source_format(self) -> DatasetSourceFormat:
        """Return the MCAP source format."""
        return DatasetSourceFormat.MCAP

    def content_digest(self) -> str:
        """Hash the exact MCAP bytes incrementally."""
        return file_content_digest(self._path)

    def rows(self) -> Iterator[Mapping[str, Any]]:
        """Yield one normalized sparse row per mapped MCAP message."""
        try:
            reader_module = importlib.import_module("mcap.reader")
        except ImportError as exc:
            raise DatasetReaderUnavailableError(
                "MCAP ingestion requires the 'mcap' SDK extra: "
                "pip install 'iso-obs[mcap]'"
            ) from exc
        rule_by_topic = {rule.topic: rule for rule in self._plan.topic_rules}
        topics = tuple(rule_by_topic)
        decoder_factory = _ros2_decoder_factory(self._plan.topic_rules)
        decoder_cache: dict[int, Any] = {}
        try:
            with self._path.open("rb") as stream:
                reader = reader_module.NonSeekingReader(
                    stream,
                    validate_crcs=self._plan.validate_crcs,
                    record_size_limit=self._plan.record_size_limit_bytes,
                )
                for schema, channel, message in reader.iter_messages(
                    topics=topics,
                    log_time_order=False,
                ):
                    rule = rule_by_topic[channel.topic]
                    _validate_message_contract(rule, schema, channel)
                    boundary_timestamp = _message_timestamp(
                        self._plan.episode_window_timestamp_source,
                        message,
                        None,
                        (),
                    )
                    window = _episode_for_timestamp(
                        self._plan.episode_windows,
                        boundary_timestamp,
                    )
                    if window is None:
                        continue
                    decoded = None
                    if (
                        rule.payload_mode is McapPayloadMode.ROS2_DECODED
                        or rule.timestamp_source is McapTimestampSource.ROS_HEADER
                    ):
                        decoded = _decode_ros2_message(
                            decoder_factory,
                            decoder_cache,
                            schema,
                            channel,
                            message,
                        )
                    topic_timestamp = _message_timestamp(
                        rule.timestamp_source,
                        message,
                        decoded,
                        rule.ros_header_stamp_path,
                    )
                    payload = _message_payload(rule, message.data, decoded)
                    source_file = None
                    if self._source_file_path is not None:
                        source_file = {
                            "content_digest": self._source_file_content_digest,
                            "index": self._source_file_index,
                            "relative_path": self._source_file_path,
                        }
                    mcap_metadata = {
                        "channel_metadata": dict(channel.metadata),
                        "log_time_ns": message.log_time,
                        "message_encoding": (channel.message_encoding),
                        "publish_time_ns": message.publish_time,
                        "schema_encoding": (
                            schema.encoding if schema is not None else None
                        ),
                        "schema_content_digest": (
                            sha256_digest(schema.data) if schema is not None else None
                        ),
                        "schema_name": schema.name if schema is not None else None,
                        "sequence": message.sequence,
                        "topic": channel.topic,
                    }
                    if source_file is not None:
                        mcap_metadata["source_file"] = source_file
                    yield {
                        "episode_id": window.episode_id,
                        "independence_unit_id": window.independence_unit_id,
                        "source_split": window.source_split,
                        "topic_timestamps_ns": {
                            rule.topic: topic_timestamp,
                        },
                        "topics": {
                            rule.topic: {
                                "mcap": mcap_metadata,
                                "payload": payload,
                            }
                        },
                        "window_timestamp_ns": boundary_timestamp,
                    }
        except DatasetIngestionError:
            raise
        except Exception as exc:
            raise DatasetIngestionError(f"failed to read MCAP source: {exc}") from exc


def ingest_mcap_source(
    plan: McapDatasetIngestionPlan,
    path: str | PathLike[str],
) -> DatasetIngestionResult:
    """Ingest a mapped MCAP or ROS 2 bag into reliability artifacts.

    Args:
        plan: Versioned topic, timestamp, episode, and provenance mapping.
        path: Local MCAP source file.

    Returns:
        Dataset bundle and ingestion report bound to the MCAP plan digest.
    """
    result = ingest_dataset_rows(
        plan.normalized_plan(),
        McapDatasetRowReader(path, plan),
    )
    return finalize_mcap_ingestion(plan, result)


def finalize_mcap_ingestion(
    plan: McapDatasetIngestionPlan,
    result: DatasetIngestionResult,
    *,
    plan_content_digest: str | None = None,
    additional_issues: Sequence[DatasetIngestionIssue] = (),
    additional_limitations: Sequence[str] = (),
) -> DatasetIngestionResult:
    """Apply MCAP-specific review findings to a generic ingestion result.

    Args:
        plan: MCAP mapping used to construct the generic result.
        result: Result returned by the generic row ingestion engine.
        plan_content_digest: Optional enclosing adapter-plan digest.
        additional_issues: Container-specific integrity findings.
        additional_limitations: Container-specific interpretation limits.

    Returns:
        Dataset result with MCAP provenance and disposition finalized.
    """
    issues = (
        *result.report.issues,
        *_missing_required_episode_topics(plan, result),
        *additional_issues,
    )
    disposition = result.report.disposition
    if (
        issues
        and disposition is DatasetIngestionDisposition.INGESTED_WITHIN_DECLARED_MAPPING
    ):
        disposition = DatasetIngestionDisposition.REVIEW_REQUIRED
    report = replace(
        result.report,
        plan_content_digest=plan_content_digest or plan.content_digest(),
        disposition=disposition,
        issues=issues,
        limitations=(
            *result.report.limitations,
            (
                "MCAP CRC validation was disabled by plan."
                if not plan.validate_crcs
                else "MCAP chunk and data-section CRCs were validated when " "present."
            ),
            "Physical MCAP message order was preserved.",
            *additional_limitations,
        ),
    )
    return DatasetIngestionResult(bundle=result.bundle, report=report)


def _missing_required_episode_topics(
    plan: McapDatasetIngestionPlan,
    result: DatasetIngestionResult,
) -> tuple[DatasetIngestionIssue, ...]:
    """Find required topics absent from individual declared episodes."""
    globally_observed = {
        summary.channel_id
        for summary in result.report.channel_summaries
        if summary.sample_count > 0
    }
    observed_pairs = {
        (trace.episode_id, trace.channel_id) for trace in result.bundle.timing_traces
    }
    return tuple(
        DatasetIngestionIssue(
            kind=DatasetIngestionIssueKind.REQUIRED_CHANNEL_EMPTY,
            channel_id=rule.channel_id,
            description=(
                f"Required MCAP topic {rule.topic!r} contains no mapped "
                f"samples in episode {episode.episode_id!r}."
            ),
        )
        for episode in result.bundle.episodes
        for rule in plan.topic_rules
        if rule.required
        and rule.channel_id in globally_observed
        and (episode.episode_id, rule.channel_id) not in observed_pairs
    )


def _ros2_decoder_factory(rules: Sequence[McapTopicRule]) -> Any:
    """Load the optional official ROS 2 decoder when a rule requires it."""
    if not any(
        rule.payload_mode is McapPayloadMode.ROS2_DECODED
        or rule.timestamp_source is McapTimestampSource.ROS_HEADER
        for rule in rules
    ):
        return None
    try:
        decoder_module = importlib.import_module("mcap_ros2.decoder")
    except ImportError as exc:
        raise DatasetReaderUnavailableError(
            "Decoded ROS 2 ingestion requires the 'mcap-ros2' SDK extra: "
            "pip install 'iso-obs[mcap-ros2]'"
        ) from exc
    return decoder_module.DecoderFactory()


def _decode_ros2_message(
    decoder_factory: Any,
    decoder_cache: dict[int, Any],
    schema: Any,
    channel: Any,
    message: Any,
) -> Any:
    """Decode one CDR message through the official schema-aware factory."""
    if decoder_factory is None:
        raise DatasetReaderUnavailableError("ROS 2 decoder is unavailable")
    decoder = decoder_cache.get(channel.id)
    if decoder is None:
        decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
        if decoder is None:
            raise DatasetIngestionError(
                f"topic {channel.topic!r} cannot be decoded as ROS 2 CDR; "
                "an embedded ros2msg schema is required"
            )
        decoder_cache[channel.id] = decoder
    try:
        return decoder(message.data)
    except Exception as exc:
        raise DatasetIngestionError(
            f"failed to decode ROS 2 topic {channel.topic!r}: {exc}"
        ) from exc


def _validate_message_contract(rule: McapTopicRule, schema: Any, channel: Any) -> None:
    """Require observed channel and schema identity to match the plan."""
    if channel.message_encoding != rule.expected_message_encoding:
        raise DatasetIngestionError(
            f"topic {rule.topic!r} message encoding changed from "
            f"{rule.expected_message_encoding!r} to "
            f"{channel.message_encoding!r}"
        )
    if rule.expected_schema_name is not None:
        actual = schema.name if schema is not None else None
        if actual != rule.expected_schema_name:
            raise DatasetIngestionError(
                f"topic {rule.topic!r} schema name changed from "
                f"{rule.expected_schema_name!r} to {actual!r}"
            )
    if rule.expected_schema_encoding is not None:
        actual = schema.encoding if schema is not None else None
        if actual != rule.expected_schema_encoding:
            raise DatasetIngestionError(
                f"topic {rule.topic!r} schema encoding changed from "
                f"{rule.expected_schema_encoding!r} to {actual!r}"
            )


def _message_payload(
    rule: McapTopicRule,
    raw_data: bytes,
    decoded: Any,
) -> Any:
    """Produce the explicitly selected raw, JSON, or decoded payload."""
    if rule.payload_mode is McapPayloadMode.RAW_BYTES:
        return raw_data
    if rule.payload_mode is McapPayloadMode.ROS2_DECODED:
        return _decoded_to_plain(decoded)
    try:
        return json.loads(
            raw_data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetIngestionError(
            f"topic {rule.topic!r} contains invalid JSON: {exc}"
        ) from exc


def _message_timestamp(
    source: McapTimestampSource,
    message: Any,
    decoded: Any,
    header_path: Sequence[str],
) -> int:
    """Select one timestamp without conflating MCAP and ROS clocks."""
    if source is McapTimestampSource.LOG_TIME:
        return _integer(message.log_time, "MCAP log timestamp")
    if source is McapTimestampSource.PUBLISH_TIME:
        return _integer(message.publish_time, "MCAP publish timestamp")
    stamp = _resolve_decoded_path(decoded, header_path)
    if stamp is _MISSING:
        raise DatasetIngestionError(
            f"decoded ROS message lacks header stamp path " f"{'.'.join(header_path)!r}"
        )
    sec = _decoded_member(stamp, "sec")
    nanosec = _decoded_member(stamp, "nanosec")
    seconds = _integer(sec, "ROS header stamp seconds")
    nanoseconds = _integer(nanosec, "ROS header stamp nanoseconds")
    if not 0 <= nanoseconds < 1_000_000_000:
        raise DatasetIngestionError("ROS header stamp nanoseconds must be in [0, 1e9)")
    return seconds * 1_000_000_000 + nanoseconds


def _episode_for_timestamp(
    windows: Sequence[McapEpisodeWindow],
    timestamp_ns: int,
) -> McapEpisodeWindow | None:
    """Return the sole declared episode containing a timestamp."""
    matches = tuple(window for window in windows if window.contains(timestamp_ns))
    if len(matches) > 1:
        raise DatasetIngestionError("MCAP episode windows overlap")
    return matches[0] if matches else None


def _decoded_to_plain(value: Any) -> Any:
    """Convert official ROS 2 dynamic messages into canonical Python values."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, array.array):
        if value.typecode == "B":
            return value.tobytes()
        return [_decoded_to_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _decoded_to_plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_decoded_to_plain(item) for item in value]
    slots = getattr(value, "__slots__", None)
    if slots is not None:
        return {str(name): _decoded_to_plain(getattr(value, name)) for name in slots}
    raise DatasetIngestionError(
        f"unsupported decoded ROS 2 value type: {type(value).__name__}"
    )


_MISSING = object()


def _resolve_decoded_path(value: Any, path: Sequence[str]) -> Any:
    """Resolve a path across decoded mappings and message attributes."""
    current = value
    for component in path:
        current = _decoded_member(current, component)
        if current is _MISSING:
            return _MISSING
    return current


def _decoded_member(value: Any, name: str) -> Any:
    """Read one member without treating absent and null as equivalent."""
    if isinstance(value, Mapping):
        return value.get(name, _MISSING)
    return getattr(value, name, _MISSING)


def _validate_episode_windows(windows: Sequence[McapEpisodeWindow]) -> None:
    """Reject ambiguous unbounded or overlapping episode declarations."""
    unbounded = tuple(window for window in windows if window.start_time_ns is None)
    if unbounded:
        if len(windows) != 1:
            raise ValueError("an unbounded MCAP episode must be the only window")
        return
    for left, right in zip(windows, windows[1:], strict=False):
        if left.end_time_ns is None or right.start_time_ns is None:
            raise ValueError("MCAP episode window bounds are incomplete")
        if left.end_time_ns > right.start_time_ns:
            raise ValueError("MCAP episode windows must not overlap")


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


def _reject_nonfinite_constant(value: str) -> Any:
    """Reject non-standard JSON NaN and infinity tokens."""
    raise DatasetIngestionError(f"non-finite JSON number is forbidden: {value}")


def _require_text(value: str, label: str) -> None:
    """Require a nonempty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _integer(value: Any, label: str) -> int:
    """Require a non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetIngestionError(f"{label} must be an integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    """Require a positive integer configuration value."""
    result = _integer(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    """Require a finite, nonnegative numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _unique_path(values: Iterable[str], label: str) -> tuple[str, ...]:
    """Validate a nonempty ordered field path."""
    result: list[str] = []
    for value in values:
        _require_text(value, label)
        result.append(value.strip())
    if not result:
        raise ValueError(f"{label} must not be empty")
    return tuple(result)


def _require_unique(values: Iterable[Any], label: str) -> None:
    """Require hashable values to contain no duplicates."""
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be unique")


def _unique_text(values: Iterable[str], label: str) -> tuple[str, ...]:
    """Validate, deduplicate, and sort a text collection."""
    normalized: set[str] = set()
    for value in values:
        _require_text(value, label)
        normalized.add(value.strip())
    return tuple(sorted(normalized))
