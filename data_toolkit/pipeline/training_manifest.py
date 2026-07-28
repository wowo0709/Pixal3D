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


CANONICAL_SOURCES = ("ABO", "3D-FUTURE")
STAGES = ("ss64", "shape512", "shape1024", "pbr1024")
SAMPLING = "proportional-unweighted-concatenation"
AUTHORIZATION = "training-input use only"
_SOURCE_SCHEMAS = {"ABO": 1, "3D-FUTURE": 2}
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


def _validate_source_training_data(
    source: str, path: Path
) -> SourceTrainingData:
    value, raw = _load_json(path, f"source={source} training data")
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
        handoff_path=canonical_handoff_path,
        handoff_sha256=handoff_digest,
        stages=stages,
    )


def _combined_stage(
    source_data: Mapping[str, SourceTrainingData], stage: str
) -> CombinedStage:
    scopes = {
        source: source_data[source].stages[stage].source_scopes[source]
        for source in CANONICAL_SOURCES
    }
    overlap = set(scopes["ABO"]) & set(scopes["3D-FUTURE"])
    if overlap:
        raise ValueError(
            f"stage={stage}: cross-source asset overlap: {sorted(overlap)[0]}"
        )
    union = sorted((*scopes["ABO"], *scopes["3D-FUTURE"]))
    source_counts = {
        source: len(scopes[source]) for source in CANONICAL_SOURCES
    }
    data_dir = {
        source: source_data[source].stages[stage].data_dir[source]
        for source in CANONICAL_SOURCES
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
) -> dict[str, object]:
    combined_stages = {
        stage: _combined_stage(source_data, stage) for stage in STAGES
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
            for source in CANONICAL_SOURCES
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


def _validate_source_paths(
    source_paths: Mapping[str, Path],
) -> dict[str, SourceTrainingData]:
    if (
        not isinstance(source_paths, Mapping)
        or set(source_paths) != set(CANONICAL_SOURCES)
    ):
        raise ValueError(
            "source paths must contain exactly ABO and 3D-FUTURE"
        )
    return {
        source: _validate_source_training_data(
            source, Path(source_paths[source])
        )
        for source in CANONICAL_SOURCES
    }


def build_combined_training_data(
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Build a strict two-source manifest from pinned source handoffs."""
    return _document_from_sources(_validate_source_paths(source_paths))


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_combined_training_data(
    source_paths: Mapping[str, Path], output_path: Path
) -> Path:
    """Atomically publish a validated local combined training manifest."""
    output_path = Path(output_path)
    payload = _canonical_json_bytes(
        build_combined_training_data(source_paths)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
        _fsync_directory(output_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _validate_combined_shape(value: Mapping[str, object]) -> None:
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
    if (
        not isinstance(sources, Mapping)
        or list(sources) != list(CANONICAL_SOURCES)
    ):
        raise ValueError(
            "combined training manifest sources must be exactly "
            "ABO then 3D-FUTURE"
        )
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
                or list(source_mapping) != list(CANONICAL_SOURCES)
            ):
                raise ValueError(
                    f"combined stage={stage} {field} sources must be "
                    "ABO then 3D-FUTURE"
                )


def resolve_training_data(
    path: Path, stage: str
) -> ResolvedTrainingData:
    """Revalidate a combined manifest trust chain and select one stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    path = Path(path)
    value, raw = _load_json(path, "combined training manifest")
    canonical_path = _canonical_path(path, "combined training manifest")
    _validate_combined_shape(value)
    sources = value["sources"]
    source_data = {}
    for source in CANONICAL_SOURCES:
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
    expected = _document_from_sources(source_data)
    if value != expected:
        raise ValueError(
            "combined training manifest does not match current source evidence"
        )
    combined_stage = _combined_stage(source_data, stage)
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
