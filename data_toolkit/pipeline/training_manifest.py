"""Strict multi-source training manifest creation and launch-time resolution."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

from data_toolkit.pipeline.training_source_profiles import (
    SOURCE_ACCEPTANCE_CONTRACTS,
)

KNOWN_SOURCES = ("ABO", "3D-FUTURE", "HSSD")
CANONICAL_SOURCES = ("ABO", "3D-FUTURE")
TWO_SOURCE_BUNDLE = CANONICAL_SOURCES
THREE_SOURCE_BUNDLE = ("ABO", "3D-FUTURE", "HSSD")
SUPPORTED_BUNDLES = (TWO_SOURCE_BUNDLE, THREE_SOURCE_BUNDLE)
STAGES = ("ss64", "shape512", "shape1024", "pbr1024")
SAMPLING = "proportional-unweighted-concatenation"
AUTHORIZATION = "training-input use only"
_SOURCE_SCHEMAS = {"ABO": 1, "3D-FUTURE": 2, "HSSD": 2}
_REPORT_FIELDS = {
    1: (
        "schema_version",
        "created_at",
        "source",
        "shard_id",
        "source_index",
        "acceptance_mode",
        "original_90_percent_gate_passed",
        "authorization",
        "counts",
        "eligibility_policy",
        "stages",
        "materialization_evidence",
        "observed_tool_commits",
    ),
    2: (
        "schema_version",
        "created_at",
        "source",
        "source_indexes",
        "acceptance_mode",
        "original_90_percent_gate_passed",
        "authorization",
        "counts",
        "eligibility_policy",
        "stages",
        "materialization_evidence",
        "observed_tool_commits",
    ),
}
_STAGE_COMPONENTS = {
    "ss64": {
        "base": Path(),
        "render_cond": Path("renders_cond"),
        "ss_latent": Path(
            "ss_latents/ss_enc_conv3d_16l8_fp16_64_view"
        ),
    },
    "shape512": {
        "base": Path(),
        "render_cond": Path("renders_cond"),
        "shape_latent": Path(
            "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"
        ),
    },
    "shape1024": {
        "base": Path(),
        "render_cond": Path("renders_cond"),
        "shape_latent": Path(
            "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"
        ),
    },
    "pbr1024": {
        "base": Path(),
        "render_cond": Path("renders_cond"),
        "shape_latent": Path(
            "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"
        ),
        "pbr_latent": Path(
            "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix"
        ),
    },
}


@dataclass(frozen=True)
class CombinedStage:
    """One verified source union used by a multi-view training stage."""

    source_counts: dict[str, int]
    total_count: int
    union_scope_sha256: str
    data_dir: dict[str, dict[str, str]]
    source_scopes: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class SourceTrainingData:
    """Validated source handoff and materialized stage evidence."""

    source: str
    path: Path
    sha256: str
    report_path: Path
    report_sha256: str
    handoff_path: Path
    handoff_sha256: str
    stages: dict[str, CombinedStage]


@dataclass(frozen=True)
class ResolvedTrainingData:
    """Launch-ready stage selected from a verified combined manifest."""

    path: Path
    manifest_sha256: str
    stage: str
    data_dir: dict[str, dict[str, str]]
    source_counts: dict[str, int]
    total_count: int
    source_scopes: dict[str, tuple[str, ...]]
    union_scope_sha256: str
    sampling: str


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def _regular_bytes(path: Path, label: str) -> bytes:
    path = Path(path)
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ValueError(
            f"{label} is not a regular non-symlink file: {path}"
        ) from error
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"{label} is not a regular non-symlink file: {path}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"{label} is not a regular non-symlink file: {path}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(
                f"{label} is not a regular non-symlink file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _load_json(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    raw = _regular_bytes(path, label)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value, raw


def _canonical_path(path: Path, label: str) -> Path:
    path = Path(path)
    resolved = path.resolve()
    if str(path) != str(resolved):
        raise ValueError(f"{label} path must be canonical: {path}")
    return resolved


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_source_indexes(
    source: str, report: Mapping[str, object]
) -> None:
    if source == "ABO":
        references = [
            (
                report.get("source_index"),
                f"source={source} source index",
            )
        ]
    else:
        value = report.get("source_indexes")
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"source={source} source indexes must be a non-empty list"
            )
        shard_ids = [
            reference.get("shard_id")
            for reference in value
            if isinstance(reference, Mapping)
        ]
        if (
            len(shard_ids) != len(value)
            or not all(
                isinstance(shard_id, str) and shard_id
                for shard_id in shard_ids
            )
            or len(set(shard_ids)) != len(shard_ids)
        ):
            raise ValueError(
                f"source={source} source index shard IDs are invalid"
            )
        references = [
            (
                reference,
                f"source={source} source index shard={shard_id}",
            )
            for reference, shard_id in zip(value, shard_ids, strict=True)
        ]

    canonical_paths = []
    for reference, label in references:
        expected_keys = (
            {"path", "sha256"}
            if source == "ABO"
            else {"shard_id", "path", "sha256"}
        )
        record = _exact_keys(reference, expected_keys, label)
        path_value = record["path"]
        pinned_digest = record["sha256"]
        if (
            not isinstance(path_value, str)
            or not path_value
            or not _valid_digest(pinned_digest)
        ):
            raise ValueError(f"{label} reference is invalid")
        path = _canonical_path(Path(path_value), label)
        raw = _regular_bytes(path, label)
        if _digest(raw) != pinned_digest:
            raise ValueError(
                f"source={source} source index digest changed: {label}"
            )
        canonical_paths.append(path)
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError(
            f"source={source} source index paths must be unique"
        )


def _exact_keys(
    value: object, expected: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(
            f"{label} must have exact keys {sorted(expected)}"
        )
    return value


def _stage_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(STAGES):
        raise ValueError(f"{label} must contain all four stages")
    return value


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _scope_digest(scope: list[str] | tuple[str, ...]) -> str:
    return sha256("\n".join(scope).encode()).hexdigest()


def _expected_data_dir(
    source: str, stage: str, root: Path
) -> dict[str, dict[str, str]]:
    return {
        source: {
            component: str(root / relative)
            for component, relative in _STAGE_COMPONENTS[stage].items()
        }
    }


def _validate_report_chain(
    source: str, handoff: Mapping[str, object]
) -> tuple[Path, str]:
    reference = _exact_keys(
        handoff.get("report"),
        {"path", "sha256"},
        f"source={source} report reference",
    )
    path_value = reference["path"]
    pinned_digest = reference["sha256"]
    if (
        not isinstance(path_value, str)
        or not path_value
        or not _valid_digest(pinned_digest)
    ):
        raise ValueError(f"source={source} report reference is invalid")
    path = Path(path_value)
    report, raw = _load_json(path, f"source={source} report")
    canonical_path = _canonical_path(path, f"source={source} report")
    if _digest(raw) != pinned_digest:
        raise ValueError(f"source={source} report digest changed")
    schema = _SOURCE_SCHEMAS[source]
    fields = _REPORT_FIELDS[schema]
    _exact_keys(report, set(fields), f"source={source} report")
    _validate_source_indexes(source, report)
    expected_handoff = {
        key: report[key] for key in fields
    } | {"report": dict(reference)}
    if dict(handoff) != expected_handoff:
        raise ValueError(
            f"source={source} handoff does not match report projection"
        )
    return canonical_path, pinned_digest


def _validate_materialization(
    source: str,
    stage: str,
    root: Path,
    expected_count: int,
    expected_scope_digest: str,
    pinned_digest: str,
) -> tuple[str, ...]:
    path = root / "materialization.json"
    evidence, raw = _load_json(
        path, f"source={source} stage={stage} materialization evidence"
    )
    if _digest(raw) != pinned_digest:
        raise ValueError(
            f"source={source} stage={stage} materialization digest changed"
        )
    if (
        evidence.get("source") != source
        or evidence.get("stage") != stage
        or evidence.get("stage_root") != str(root)
        or evidence.get("asset_count") != expected_count
    ):
        raise ValueError(
            f"source={source} stage={stage} materialization identity changed"
        )
    scope = evidence.get("stage_scope")
    if (
        not isinstance(scope, list)
        or not all(isinstance(asset, str) and asset for asset in scope)
        or scope != sorted(scope)
        or len(scope) != len(set(scope))
        or len(scope) != expected_count
    ):
        raise ValueError(
            f"source={source} stage={stage} materialized scope is invalid"
        )
    digest = _scope_digest(scope)
    if (
        evidence.get("stage_scope_sha256") != digest
        or expected_scope_digest != digest
    ):
        raise ValueError(
            f"source={source} stage={stage} materialized scope digest changed"
        )
    return tuple(scope)


def _validate_source_stage(
    source: str,
    stage: str,
    record: object,
    count_from_contract: int,
    materialization_summary: object,
) -> CombinedStage:
    stage_record = _exact_keys(
        record,
        {
            "root",
            "asset_count",
            "asset_scope_sha256",
            "anchors_checked",
            "validation_counts",
            "data_dir",
        },
        f"source={source} stage={stage} record",
    )
    root_value = stage_record["root"]
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(f"source={source} stage={stage} root is invalid")
    root = _canonical_path(
        Path(root_value), f"source={source} stage={stage} root"
    )
    asset_count = _count(
        stage_record["asset_count"],
        f"source={source} stage={stage} asset_count",
    )
    validation_counts = stage_record["validation_counts"]
    if (
        asset_count != count_from_contract
        or stage_record["anchors_checked"] != asset_count * 2
        or not isinstance(validation_counts, Mapping)
        or validation_counts.get("assets") != asset_count
    ):
        raise ValueError(
            f"source={source} stage={stage} count evidence is inconsistent"
        )
    scope_sha256 = stage_record["asset_scope_sha256"]
    if not _valid_digest(scope_sha256):
        raise ValueError(
            f"source={source} stage={stage} scope digest is invalid"
        )
    data_dir = stage_record["data_dir"]
    expected_data_dir = _expected_data_dir(source, stage, root)
    if not isinstance(data_dir, Mapping) or set(data_dir) != {source}:
        raise ValueError(
            f"source={source} stage={stage} data_dir source key is invalid"
        )
    components = data_dir[source]
    expected_components = expected_data_dir[source]
    if (
        not isinstance(components, Mapping)
        or set(components) != set(expected_components)
    ):
        raise ValueError(
            f"source={source} stage={stage} component keys are invalid"
        )
    if dict(components) != expected_components:
        raise ValueError(
            f"source={source} stage={stage} component path is noncanonical"
        )
    summary = _exact_keys(
        materialization_summary,
        {"sha256", "tool_commits"},
        f"source={source} stage={stage} materialization summary",
    )
    pinned_digest = summary["sha256"]
    tool_commits = summary["tool_commits"]
    if (
        not _valid_digest(pinned_digest)
        or not isinstance(tool_commits, list)
        or not all(
            isinstance(commit, str) and commit for commit in tool_commits
        )
    ):
        raise ValueError(
            f"source={source} stage={stage} materialization summary is invalid"
        )
    scope = _validate_materialization(
        source,
        stage,
        root,
        asset_count,
        scope_sha256,
        pinned_digest,
    )
    return CombinedStage(
        source_counts={source: asset_count},
        total_count=asset_count,
        union_scope_sha256=scope_sha256,
        data_dir={source: dict(components)},
        source_scopes={source: scope},
    )


def _validate_source_training_data_value(
    source: str,
    path: Path,
    value: Mapping[str, object],
    raw: bytes,
) -> SourceTrainingData:
    canonical_path = _canonical_path(
        Path(path), f"source={source} training data"
    )
    if value.get("source") != source:
        raise ValueError(f"source={source} training data source is invalid")
    if value.get("schema_version") != _SOURCE_SCHEMAS[source]:
        raise ValueError(
            f"source={source} schema_version must be "
            f"{_SOURCE_SCHEMAS[source]}"
        )
    if value.get("authorization") != AUTHORIZATION:
        raise ValueError(
            f"source={source} training data authorization is invalid"
        )
    handoff_reference = _exact_keys(
        value.get("handoff"),
        {"path", "sha256"},
        f"source={source} handoff reference",
    )
    handoff_path_value = handoff_reference["path"]
    handoff_digest = handoff_reference["sha256"]
    if (
        not isinstance(handoff_path_value, str)
        or not handoff_path_value
        or not _valid_digest(handoff_digest)
    ):
        raise ValueError(f"source={source} handoff reference is invalid")
    handoff_path = Path(handoff_path_value)
    handoff, handoff_raw = _load_json(
        handoff_path, f"source={source} handoff"
    )
    canonical_handoff_path = _canonical_path(
        handoff_path, f"source={source} handoff"
    )
    if _digest(handoff_raw) != handoff_digest:
        raise ValueError(f"source={source} handoff digest changed")
    report_path, report_digest = _validate_report_chain(source, handoff)
    expected_training_data = {
        **handoff,
        "handoff": dict(handoff_reference),
    }
    if value != expected_training_data:
        raise ValueError(
            f"source={source} training data does not match handoff bytes"
        )
    if (
        handoff.get("source") != source
        or handoff.get("schema_version") != _SOURCE_SCHEMAS[source]
        or handoff.get("authorization") != AUTHORIZATION
    ):
        raise ValueError(f"source={source} handoff identity is invalid")
    acceptance_contract = (
        value.get("acceptance_mode"),
        value.get("original_90_percent_gate_passed"),
    )
    if (
        type(acceptance_contract[1]) is not bool
        or acceptance_contract != SOURCE_ACCEPTANCE_CONTRACTS[source]
    ):
        raise ValueError(
            f"source={source} acceptance contract is invalid"
        )
    stage_records = _stage_mapping(
        value.get("stages"), f"source={source} stage records"
    )
    summaries = _stage_mapping(
        value.get("materialization_evidence"),
        f"source={source} materialization evidence",
    )
    counts = value.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError(f"source={source} counts are invalid")
    stage_counts = _stage_mapping(
        counts.get("stages"), f"source={source} count contract"
    )
    stages = {}
    for stage in STAGES:
        stages[stage] = _validate_source_stage(
            source,
            stage,
            stage_records[stage],
            _count(
                stage_counts[stage],
                f"source={source} stage={stage} count contract",
            ),
            summaries[stage],
        )
    return SourceTrainingData(
        source=source,
        path=canonical_path,
        sha256=_digest(raw),
        report_path=report_path,
        report_sha256=report_digest,
        handoff_path=canonical_handoff_path,
        handoff_sha256=handoff_digest,
        stages=stages,
    )


def _validate_source_training_data(
    source: str, path: Path
) -> SourceTrainingData:
    value, raw = _load_json(path, f"source={source} training data")
    return _validate_source_training_data_value(
        source, path, value, raw
    )


def validate_source_training_data(
    source: str, path: Path
) -> SourceTrainingData:
    """Validate one complete immutable source publication chain."""
    if source not in KNOWN_SOURCES:
        raise ValueError(
            f"source must be one of {KNOWN_SOURCES}: {source}"
        )
    return _validate_source_training_data(source, Path(path))


def _combined_stage(
    source_data: Mapping[str, SourceTrainingData],
    source_order: tuple[str, ...],
    stage: str,
) -> CombinedStage:
    scopes = {
        source: source_data[source].stages[stage].source_scopes[source]
        for source in source_order
    }
    owners: dict[str, str] = {}
    for source in source_order:
        for asset in scopes[source]:
            previous = owners.setdefault(asset, source)
            if previous != source:
                raise ValueError(
                    "cross-source asset overlap: "
                    f"{previous}/{source}: {asset}"
                )
    union = sorted(
        asset
        for source in source_order
        for asset in scopes[source]
    )
    source_counts = {
        source: len(scopes[source]) for source in source_order
    }
    data_dir = {
        source: source_data[source].stages[stage].data_dir[source]
        for source in source_order
    }
    return CombinedStage(
        source_counts=source_counts,
        total_count=sum(source_counts.values()),
        union_scope_sha256=_scope_digest(union),
        data_dir=data_dir,
        source_scopes=scopes,
    )


def _document_from_sources(
    source_data: Mapping[str, SourceTrainingData],
    source_order: tuple[str, ...],
) -> dict[str, object]:
    combined_stages = {
        stage: _combined_stage(source_data, source_order, stage)
        for stage in STAGES
    }
    return {
        "schema_version": 1,
        "authorization": AUTHORIZATION,
        "sampling": SAMPLING,
        "sources": {
            source: {
                "training_data": {
                    "path": str(source_data[source].path),
                    "sha256": source_data[source].sha256,
                },
                "handoff": {
                    "path": str(source_data[source].handoff_path),
                    "sha256": source_data[source].handoff_sha256,
                },
            }
            for source in source_order
        },
        "stages": {
            stage: {
                "source_counts": combined_stages[stage].source_counts,
                "total_count": combined_stages[stage].total_count,
                "union_scope_sha256":
                    combined_stages[stage].union_scope_sha256,
                "data_dir": combined_stages[stage].data_dir,
            }
            for stage in STAGES
        },
    }


def _source_order(
    source_paths: Mapping[str, Path],
) -> tuple[str, ...]:
    order = tuple(source_paths)
    if order not in SUPPORTED_BUNDLES:
        raise ValueError(
            f"source order must be one of {SUPPORTED_BUNDLES}: {order}"
        )
    return order


def _validate_source_paths(
    source_paths: Mapping[str, Path],
) -> tuple[dict[str, SourceTrainingData], tuple[str, ...]]:
    if not isinstance(source_paths, Mapping):
        raise ValueError("source paths must be a mapping")
    source_order = _source_order(source_paths)
    return {
        source: _validate_source_training_data(
            source, Path(source_paths[source])
        )
        for source in source_order
    }, source_order


def build_combined_training_data(
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Build a supported source bundle from pinned source handoffs."""
    source_data, source_order = _validate_source_paths(source_paths)
    return _document_from_sources(source_data, source_order)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_create_only(path: Path, payload: bytes) -> None:
    """Publish bytes once without replacing an existing identical inode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if _regular_bytes(path, "existing combined training manifest") != payload:
            raise ValueError(
                f"existing combined training manifest has different content: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                _regular_bytes(
                    path, "existing combined training manifest"
                )
                != payload
            ):
                raise ValueError(
                    "existing combined training manifest has different "
                    f"content: {path}"
                )
        else:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish_combined_training_data(
    source_paths: Mapping[str, Path], output_path: Path
) -> Path:
    """Create a validated local combined training manifest without replacement."""
    output_path = Path(output_path)
    payload = _canonical_json_bytes(
        build_combined_training_data(source_paths)
    )
    _publish_create_only(output_path, payload)
    return output_path


def _validate_combined_shape(
    value: Mapping[str, object],
) -> tuple[str, ...]:
    _exact_keys(
        value,
        {
            "schema_version",
            "authorization",
            "sampling",
            "sources",
            "stages",
        },
        "combined training manifest",
    )
    if (
        value.get("schema_version") != 1
        or value.get("authorization") != AUTHORIZATION
        or value.get("sampling") != SAMPLING
    ):
        raise ValueError("combined training manifest identity is invalid")
    sources = value.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError(
            "combined training manifest sources must be a mapping"
        )
    try:
        source_order = _source_order(sources)
    except ValueError as error:
        raise ValueError(
            f"combined training manifest sources {error}"
        ) from error
    stages = _stage_mapping(
        value.get("stages"), "combined training manifest stages"
    )
    for stage in STAGES:
        record = _exact_keys(
            stages[stage],
            {
                "source_counts",
                "total_count",
                "union_scope_sha256",
                "data_dir",
            },
            f"combined stage={stage}",
        )
        for field in ("source_counts", "data_dir"):
            source_mapping = record[field]
            if (
                not isinstance(source_mapping, Mapping)
                or list(source_mapping) != list(source_order)
            ):
                raise ValueError(
                    f"combined stage={stage} {field} sources must be "
                    f"{source_order}"
                )
    return source_order


def _resolve_combined_value(
    path: Path,
    value: Mapping[str, object],
    raw: bytes,
    stage: str,
) -> ResolvedTrainingData:
    canonical_path = _canonical_path(path, "combined training manifest")
    source_order = _validate_combined_shape(value)
    sources = value["sources"]
    source_data = {}
    for source in source_order:
        source_record = _exact_keys(
            sources[source],
            {"training_data", "handoff"},
            f"combined source={source}",
        )
        training_reference = _exact_keys(
            source_record["training_data"],
            {"path", "sha256"},
            f"combined source={source} training data reference",
        )
        handoff_reference = _exact_keys(
            source_record["handoff"],
            {"path", "sha256"},
            f"combined source={source} handoff reference",
        )
        training_path_value = training_reference["path"]
        if (
            not isinstance(training_path_value, str)
            or not training_path_value
            or not _valid_digest(training_reference["sha256"])
        ):
            raise ValueError(
                f"combined source={source} training data reference is invalid"
            )
        validated = _validate_source_training_data(
            source, Path(training_path_value)
        )
        if validated.sha256 != training_reference["sha256"]:
            raise ValueError(f"source={source} training data digest changed")
        if (
            handoff_reference["path"] != str(validated.handoff_path)
            or handoff_reference["sha256"] != validated.handoff_sha256
        ):
            raise ValueError(
                f"source={source} combined handoff reference changed"
            )
        source_data[source] = validated
    expected = _document_from_sources(source_data, source_order)
    if value != expected:
        raise ValueError(
            "combined training manifest does not match current source evidence"
        )
    combined_stage = _combined_stage(source_data, source_order, stage)
    return ResolvedTrainingData(
        path=canonical_path,
        manifest_sha256=_digest(raw),
        stage=stage,
        data_dir=combined_stage.data_dir,
        source_counts=combined_stage.source_counts,
        total_count=combined_stage.total_count,
        source_scopes=combined_stage.source_scopes,
        union_scope_sha256=combined_stage.union_scope_sha256,
        sampling=SAMPLING,
    )


def _resolve_source_value(
    path: Path,
    value: Mapping[str, object],
    raw: bytes,
    stage: str,
) -> ResolvedTrainingData:
    source = value.get("source")
    if source not in KNOWN_SOURCES:
        raise ValueError(f"unknown source training data: {source}")
    validated = _validate_source_training_data_value(
        source, path, value, raw
    )
    selected = validated.stages[stage]
    scope = selected.source_scopes[source]
    return ResolvedTrainingData(
        path=validated.path,
        manifest_sha256=validated.sha256,
        stage=stage,
        data_dir=selected.data_dir,
        source_counts={source: len(scope)},
        total_count=len(scope),
        source_scopes={source: scope},
        union_scope_sha256=selected.union_scope_sha256,
        sampling=SAMPLING,
    )


def resolve_source_training_data(
    path: Path, stage: str
) -> ResolvedTrainingData:
    value, raw = _load_json(path, "source training data")
    return _resolve_source_value(path, value, raw, stage)


def resolve_training_data(
    path: Path, stage: str
) -> ResolvedTrainingData:
    """Revalidate a supported manifest trust chain and select one stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    path = Path(path)
    value, raw = _load_json(path, "training data")
    combined_keys = {
        "schema_version",
        "authorization",
        "sampling",
        "sources",
        "stages",
    }
    source = value.get("source")
    source_keys = (
        set(_REPORT_FIELDS[_SOURCE_SCHEMAS[source]])
        | {"report", "handoff"}
        if source in KNOWN_SOURCES
        else set()
    )
    if set(value) == combined_keys:
        return _resolve_combined_value(path, value, raw, stage)
    if source_keys and set(value) == source_keys:
        return _resolve_source_value(path, value, raw, stage)
    raise ValueError("unrecognized training_data schema")


def resolve_training_input(
    config: Mapping[str, object],
    cli_data_dir: str | None,
    cli_training_data: str | Path | None,
) -> tuple[str, dict[str, object] | None]:
    """Resolve legacy data_dir or a verified manifest without touching CUDA."""
    if cli_data_dir is not None and cli_training_data is not None:
        raise ValueError(
            "--data_dir and --training_data are mutually exclusive"
        )
    if cli_training_data is None:
        configured = config.get("data_dir") if isinstance(config, Mapping) else None
        data_dir = (
            cli_data_dir
            if cli_data_dir is not None
            else configured if configured is not None else "./data/"
        )
        if not isinstance(data_dir, str):
            raise ValueError("data_dir must be a string")
        return data_dir, None
    trainer = config.get("trainer") if isinstance(config, Mapping) else None
    trainer_args = (
        trainer.get("args") if isinstance(trainer, Mapping) else None
    )
    stage = (
        trainer_args.get("multiview_stage")
        if isinstance(trainer_args, Mapping)
        else None
    )
    if stage not in STAGES:
        raise ValueError(f"unknown multiview_stage: {stage}")
    resolved = resolve_training_data(Path(cli_training_data), stage)
    evidence = {
        "training_data": {
            "path": str(resolved.path),
            "sha256": resolved.manifest_sha256,
        },
        "stage": resolved.stage,
        "sampling": resolved.sampling,
        "source_counts": resolved.source_counts,
        "total_count": resolved.total_count,
        "union_scope_sha256": resolved.union_scope_sha256,
    }
    return json.dumps(resolved.data_dir), evidence
