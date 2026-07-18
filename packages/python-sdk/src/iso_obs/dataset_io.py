"""Strict, content-addressed I/O for local dataset reliability workflows.

The module defines a portable JSON bundle over the existing dataset contracts
and a manifest-backed adapter for synchronization audits. Parsing is strict:
unknown fields, duplicate object keys, invalid UTF-8, and non-finite JSON
numbers are rejected rather than silently normalized.
"""

from __future__ import annotations

import json
import types
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, cast, get_args, get_origin, get_type_hints

from .dataset_reliability import (
    DatasetLabelAssertion,
    DatasetManifest,
    DatasetScope,
    EpisodeManifest,
)
from .dataset_splitting import (
    DatasetSplitAssignment,
    DatasetSplitPlan,
    DatasetSplitUnit,
)
from .dataset_synchronization import ChannelTimingTrace, DatasetAdapter
from .evidence import NamedDigest, _canonical_json, sha256_digest

DATASET_BUNDLE_SCHEMA_VERSION = "iso-obs.dataset-bundle.v1"
DATASET_BUNDLE_INSPECTION_SCHEMA_VERSION = "iso-obs.dataset-bundle-inspection.v1"
DATASET_SPLIT_REQUEST_SCHEMA_VERSION = "iso-obs.dataset-split-request.v1"
DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


class ArtifactFormatError(ValueError):
    """Raised when a local JSON artifact violates its declared contract."""


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Portable source material for local dataset reliability audits."""

    schema_version: str
    manifest: DatasetManifest
    episodes: tuple[EpisodeManifest, ...]
    label_assertions: tuple[DatasetLabelAssertion, ...] = ()
    timing_traces: tuple[ChannelTimingTrace, ...] = ()

    def __post_init__(self) -> None:
        """Validate bundle identity, references, and canonical ordering."""
        if self.schema_version != DATASET_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported dataset bundle schema version")
        manifest_digest = self.manifest.manifest_digest()
        episodes = tuple(sorted(self.episodes, key=lambda item: item.episode_id))
        _require_unique(
            (item.episode_id for item in episodes),
            "dataset bundle episode IDs",
        )
        episode_by_id = {item.episode_id: item for item in episodes}
        for episode in episodes:
            if episode.dataset_manifest_digest != manifest_digest:
                raise ValueError("bundle episode references a different manifest")

        assertions = tuple(
            sorted(self.label_assertions, key=lambda item: item.assertion_id)
        )
        _require_unique(
            (item.assertion_id for item in assertions),
            "dataset bundle assertion IDs",
        )
        for assertion in assertions:
            referenced_episode = episode_by_id.get(assertion.episode_id)
            if referenced_episode is None:
                raise ValueError("bundle assertion references an unknown episode")
            if (
                assertion.start_seconds < referenced_episode.start_seconds
                or assertion.end_seconds > referenced_episode.end_seconds
            ):
                raise ValueError("bundle assertion falls outside its episode")

        traces = tuple(
            sorted(
                self.timing_traces,
                key=lambda item: (item.episode_id, item.channel_id),
            )
        )
        _require_unique(
            ((item.episode_id, item.channel_id) for item in traces),
            "dataset bundle episode/channel timing trace pairs",
        )
        declared_modalities = set(self.manifest.scope.modality_ids)
        for trace in traces:
            if trace.episode_id not in episode_by_id:
                raise ValueError("bundle timing trace references an unknown episode")
            if trace.modality_id not in declared_modalities:
                raise ValueError("bundle timing trace uses an undeclared modality")

        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "label_assertions", assertions)
        object.__setattr__(self, "timing_traces", traces)

    def to_json(self) -> str:
        """Serialize the bundle to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact bundle content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class DatasetBundleInspection:
    """Versioned structural inventory of one local dataset bundle."""

    schema_version: str
    bundle_content_digest: str
    dataset_manifest_digest: str
    episode_count: int
    assertion_count: int
    timing_trace_count: int
    modality_episode_counts: tuple[tuple[str, int], ...]
    episodes_without_assertions: tuple[str, ...]
    episodes_without_timing_traces: tuple[str, ...]
    scope: DatasetScope
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate inspection identity, counts, and canonical ordering."""
        if self.schema_version != DATASET_BUNDLE_INSPECTION_SCHEMA_VERSION:
            raise ValueError("unsupported dataset bundle inspection schema version")
        NamedDigest("dataset bundle", self.bundle_content_digest)
        NamedDigest("dataset manifest", self.dataset_manifest_digest)
        for value in (
            self.episode_count,
            self.assertion_count,
            self.timing_trace_count,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("dataset bundle inspection counts must be nonnegative")
        modality_counts = tuple(sorted(self.modality_episode_counts))
        _require_unique(
            (item[0] for item in modality_counts),
            "dataset inspection modality count labels",
        )
        for modality_id, count in modality_counts:
            if not modality_id or count < 0:
                raise ValueError("invalid dataset inspection modality count")
        object.__setattr__(self, "modality_episode_counts", modality_counts)
        object.__setattr__(
            self,
            "episodes_without_assertions",
            tuple(sorted(set(self.episodes_without_assertions))),
        )
        object.__setattr__(
            self,
            "episodes_without_timing_traces",
            tuple(sorted(set(self.episodes_without_timing_traces))),
        )
        object.__setattr__(self, "limitations", tuple(sorted(set(self.limitations))))

    def to_json(self) -> str:
        """Serialize the inspection to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact inspection content digest."""
        return sha256_digest(self.to_json())


@dataclass(frozen=True, slots=True)
class DatasetSplitRequest:
    """Portable proposal consumed by the deterministic split guardian."""

    schema_version: str
    plan: DatasetSplitPlan
    units: tuple[DatasetSplitUnit, ...]
    assignments: tuple[DatasetSplitAssignment, ...]

    def __post_init__(self) -> None:
        """Validate request schema and canonicalize its members."""
        if self.schema_version != DATASET_SPLIT_REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported dataset split request schema version")
        object.__setattr__(
            self,
            "units",
            tuple(sorted(self.units, key=lambda item: item.unit_id)),
        )
        object.__setattr__(
            self,
            "assignments",
            tuple(sorted(self.assignments, key=lambda item: item.unit_id)),
        )

    def to_json(self) -> str:
        """Serialize the split request to canonical JSON."""
        return _canonical_json(self)

    def content_digest(self) -> str:
        """Calculate the exact split request content digest."""
        return sha256_digest(self.to_json())


class ManifestDatasetAdapter(DatasetAdapter):
    """Expose a validated dataset bundle through the SDK adapter boundary."""

    def __init__(self, bundle: DatasetBundle) -> None:
        """Initialize the adapter from an immutable bundle.

        Args:
            bundle: Validated local dataset source material.
        """
        self._bundle = bundle

    def dataset_manifest(self) -> DatasetManifest:
        """Return the immutable source dataset manifest."""
        return self._bundle.manifest

    def episode_manifests(self) -> Sequence[EpisodeManifest]:
        """Return canonical independently identified source episodes."""
        return self._bundle.episodes

    def timing_traces(self, episode_id: str) -> Sequence[ChannelTimingTrace]:
        """Return source-order timing traces for one episode.

        Args:
            episode_id: Episode whose acquisition timing should be returned.

        Returns:
            Canonically ordered channels whose timestamp arrays remain unchanged.
        """
        return tuple(
            trace
            for trace in self._bundle.timing_traces
            if trace.episode_id == episode_id
        )

    def label_assertions(self) -> Sequence[DatasetLabelAssertion]:
        """Return evidence-qualified source label assertions."""
        return self._bundle.label_assertions


def inspect_dataset_bundle(bundle: DatasetBundle) -> DatasetBundleInspection:
    """Build a structural inventory without claiming dataset adequacy.

    Args:
        bundle: Validated local dataset source material.

    Returns:
        Content-addressed counts, coverage indicators, scope, and limitations.
    """
    assertions_by_episode = {item.episode_id for item in bundle.label_assertions}
    traces_by_episode = {item.episode_id for item in bundle.timing_traces}
    modality_counts: Counter[str] = Counter()
    for episode in bundle.episodes:
        modality_counts.update(item.name for item in episode.modality_digests)
    return DatasetBundleInspection(
        schema_version=DATASET_BUNDLE_INSPECTION_SCHEMA_VERSION,
        bundle_content_digest=bundle.content_digest(),
        dataset_manifest_digest=bundle.manifest.manifest_digest(),
        episode_count=len(bundle.episodes),
        assertion_count=len(bundle.label_assertions),
        timing_trace_count=len(bundle.timing_traces),
        modality_episode_counts=tuple(modality_counts.items()),
        episodes_without_assertions=tuple(
            item.episode_id
            for item in bundle.episodes
            if item.episode_id not in assertions_by_episode
        ),
        episodes_without_timing_traces=tuple(
            item.episode_id
            for item in bundle.episodes
            if item.episode_id not in traces_by_episode
        ),
        scope=bundle.manifest.scope,
        limitations=(
            "Structural validity does not establish label validity, temporal "
            "alignment, representativeness, or split independence.",
            "Missing labels or timing traces are inventories, not zero-valued "
            "measurements or inferred defects.",
        ),
    )


def parse_json_artifact[T](text: str, artifact_type: type[T]) -> T:
    """Parse strict JSON into a declared SDK artifact type.

    Args:
        text: UTF-8 JSON text.
        artifact_type: Dataclass contract to construct recursively.

    Returns:
        Validated artifact instance.

    Raises:
        ArtifactFormatError: If JSON syntax or the artifact contract is invalid.
    """
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
        return cast(T, _coerce_value(payload, artifact_type, "$"))
    except ArtifactFormatError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactFormatError(str(exc)) from exc


def load_json_artifact[T](
    path: str | PathLike[str],
    artifact_type: type[T],
    *,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> T:
    """Load a size-bounded local JSON artifact with strict validation.

    Args:
        path: Local artifact path.
        artifact_type: Dataclass contract to construct recursively.
        max_bytes: Maximum accepted encoded artifact size.

    Returns:
        Validated artifact instance.

    Raises:
        ArtifactFormatError: If size, encoding, JSON, or contract is invalid.
        OSError: If the path cannot be read.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    artifact_path = Path(path)
    encoded = artifact_path.read_bytes()
    if len(encoded) > max_bytes:
        raise ArtifactFormatError(
            f"artifact exceeds the {max_bytes}-byte local parsing limit"
        )
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactFormatError("artifact must contain valid UTF-8 JSON") from exc
    return parse_json_artifact(text, artifact_type)


def _coerce_value(value: Any, expected_type: Any, path: str) -> Any:
    """Recursively construct one strictly typed artifact value."""
    origin = get_origin(expected_type)
    if origin in {types.UnionType, getattr(types, "UnionType", object)}:
        return _coerce_union(value, get_args(expected_type), path)
    if origin is tuple:
        return _coerce_tuple(value, get_args(expected_type), path)
    if expected_type is Any:
        return value
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        if not isinstance(value, str):
            raise ArtifactFormatError(f"{path} must be a string enum value")
        try:
            return expected_type(value)
        except ValueError as exc:
            raise ArtifactFormatError(f"{path}: {exc}") from exc
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return _coerce_dataclass(value, expected_type, path)
    if expected_type is str:
        if not isinstance(value, str):
            raise ArtifactFormatError(f"{path} must be a string")
        return value
    if expected_type is bool:
        if not isinstance(value, bool):
            raise ArtifactFormatError(f"{path} must be a boolean")
        return value
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArtifactFormatError(f"{path} must be an integer")
        return value
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArtifactFormatError(f"{path} must be a number")
        return float(value)
    if expected_type is type(None):
        if value is not None:
            raise ArtifactFormatError(f"{path} must be null")
        return None
    raise ArtifactFormatError(f"{path} uses unsupported type {expected_type!r}")


def _coerce_union(value: Any, options: tuple[Any, ...], path: str) -> Any:
    """Construct a value accepted by exactly one declared union option."""
    if value is None and type(None) in options:
        return None
    errors: list[str] = []
    for option in options:
        if option is type(None):
            continue
        try:
            return _coerce_value(value, option, path)
        except ArtifactFormatError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors)
    raise ArtifactFormatError(f"{path} does not match its declared union: {detail}")


def _coerce_tuple(value: Any, arguments: tuple[Any, ...], path: str) -> tuple[Any, ...]:
    """Construct one fixed or variadic tuple from a JSON array."""
    if not isinstance(value, list):
        raise ArtifactFormatError(f"{path} must be an array")
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return tuple(
            _coerce_value(item, arguments[0], f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if len(value) != len(arguments):
        raise ArtifactFormatError(f"{path} must contain exactly {len(arguments)} items")
    return tuple(
        _coerce_value(item, item_type, f"{path}[{index}]")
        for index, (item, item_type) in enumerate(zip(value, arguments, strict=True))
    )


def _coerce_dataclass[T](value: Any, artifact_type: type[T], path: str) -> T:
    """Construct one dataclass while rejecting missing and unknown fields."""
    if not isinstance(value, Mapping):
        raise ArtifactFormatError(f"{path} must be an object")
    declared_fields = {item.name: item for item in fields(cast(Any, artifact_type))}
    unknown = sorted(set(value) - set(declared_fields))
    if unknown:
        raise ArtifactFormatError(
            f"{path} contains unknown field(s): {', '.join(unknown)}"
        )
    type_hints = get_type_hints(artifact_type)
    keyword_arguments: dict[str, Any] = {}
    for name, field in declared_fields.items():
        if name not in value:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ArtifactFormatError(f"{path}.{name} is required")
            continue
        keyword_arguments[name] = _coerce_value(
            value[name],
            type_hints[name],
            f"{path}.{name}",
        )
    try:
        return artifact_type(**keyword_arguments)
    except (TypeError, ValueError) as exc:
        raise ArtifactFormatError(f"{path}: {exc}") from exc


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Construct a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactFormatError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    """Reject non-standard JSON NaN and infinity tokens."""
    raise ArtifactFormatError(f"non-finite JSON number is forbidden: {value}")


def _require_unique(values: Iterable[Any], label: str) -> None:
    """Require a sequence of hashable values to contain no duplicates."""
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{label} must be unique")
