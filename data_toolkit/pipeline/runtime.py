from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time
from typing import Callable, Mapping
from urllib.parse import urlsplit
from uuid import NAMESPACE_DNS, uuid5

import pandas as pd

from .config import PipelineConfig
from .orchestrator import (
    InfrastructureError,
    IntegrationProviderRequired,
    PipelineServices,
    _atomic_write_bytes_nofollow,
    _open_directory_nofollow,
    _read_regular_bytes_nofollow,
    _unlink_regular_nofollow,
)
from .packing import PACK_FAMILIES
from .registry import assign_shards, canonicalize_sources
from .reporting import (
    ReportValidationError,
    build_training_handoff,
    fp16_gate_summary,
    gate_measurement_summary,
    resource_peaks,
    split_overlap,
    write_report,
)
from .resources import (
    ProjectStorageAccounting,
    ResourceGuard,
    ResourcePolicy,
    ResourceSampler,
    _directory_size,
)


ARTIFACT_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 2
REFERENCE_INDEX_SCHEMA_VERSION = 2
REGISTRY_ARTIFACT_TYPES = {
    "training": "canonical_training_registry",
    "evaluation": "canonical_evaluation_registry",
}
REFERENCE_INDEX_ARTIFACT_TYPE = "canonical_raw_reference_index"
ACCOUNTING_ARTIFACT_TYPE = "project_accounting"
HARDWARE_ARTIFACT_TYPE = "hardware_preflight"


class ArtifactValidationError(IntegrationProviderRequired):
    pass


def _safe_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(_read_regular_bytes_nofollow(Path(path)))
    except (InfrastructureError, OSError) as error:
        raise ArtifactValidationError(
            f"missing or unsafe {description}: {path}: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ArtifactValidationError(
            f"corrupt {description}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"invalid {description}: {path}")
    return value


def _optional_json(path: Path, description: str) -> dict | None:
    try:
        payload = _read_regular_bytes_nofollow(Path(path), missing_ok=True)
    except (InfrastructureError, OSError) as error:
        raise ArtifactValidationError(
            f"unsafe {description}: {path}: {error}"
        ) from error
    if payload is None:
        return None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ArtifactValidationError(
            f"corrupt {description}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"invalid {description}: {path}")
    return value


def _write_json(path: Path, value: Mapping, description: str) -> None:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        _atomic_write_bytes_nofollow(path, payload)
    except (InfrastructureError, OSError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"cannot write {description}: {path}: {error}"
        ) from error


def _sha(value, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactValidationError(f"invalid {description}: {value!r}")
    return value


def _component(value: str, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "\\" in value
        or "\0" in value
    ):
        raise ArtifactValidationError(f"unsafe {description}: {value!r}")
    return value


def _nonnegative_count(value, description: str, *, positive: bool = False) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < (1 if positive else 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ArtifactValidationError(f"{description} must be {qualifier}")
    return value


def _write_csv(path: Path, frame: pd.DataFrame, description: str) -> None:
    try:
        payload = frame.to_csv(index=False).encode("utf-8")
        pd.read_csv(io.BytesIO(payload), dtype={"sha256": str})
        _atomic_write_bytes_nofollow(path, payload)
    except (InfrastructureError, OSError) as error:
        raise ArtifactValidationError(
            f"cannot write {description}: {path}: {error}"
        ) from error


class SafeRegistryStore:
    """Parquet registry bound to a strict checksum/config manifest."""

    def __init__(
        self,
        path: Path,
        config: PipelineConfig,
        *,
        partition: str = "training",
    ):
        if partition not in REGISTRY_ARTIFACT_TYPES:
            raise ValueError(f"invalid registry partition: {partition}")
        self.path = Path(path)
        self.config = config
        self.partition = partition
        self.manifest_path = self.path.with_suffix(
            self.path.suffix + ".manifest.json"
        )

    def _manifest(
        self, payload: bytes, rows: int, source_inputs: Mapping
    ) -> dict:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "artifact_type": REGISTRY_ARTIFACT_TYPES[self.partition],
            "partition": self.partition,
            "config_hash": self.config.config_hash(),
            "rows": rows,
            "sha256": sha256(payload).hexdigest(),
            "source_inputs": dict(source_inputs),
        }

    def _validate_source_inputs(self, value: Mapping) -> dict:
        if not isinstance(value, Mapping):
            raise ArtifactValidationError("invalid registry source input manifest")
        allowed = (
            set(self.config.sources)
            if self.partition == "training"
            else set(self.config.evaluation_sources)
        )
        if set(value) != allowed:
            raise ArtifactValidationError(
                "registry source input manifest must cover its partition"
            )
        result = {}
        for source, entry in value.items():
            if source not in allowed or not isinstance(entry, Mapping) or set(entry) != {
                "path",
                "sha256",
                "rows",
            }:
                raise ArtifactValidationError("invalid registry source input manifest")
            path = entry["path"]
            if not isinstance(path, str) or not path:
                raise ArtifactValidationError("invalid registry source input path")
            result[source] = {
                "path": path,
                "sha256": _sha(entry["sha256"], "registry source input checksum"),
                "rows": _nonnegative_count(
                    entry["rows"], "registry source input rows", positive=True
                ),
            }
        return result

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ArtifactValidationError("registry must not be empty")
        required = {"sha256", "owner_source", "shard_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ArtifactValidationError(
                f"registry is missing columns: {sorted(missing)}"
            )
        if frame["sha256"].duplicated().any():
            raise ArtifactValidationError("registry contains duplicate SHA-256")
        configured = (
            set(self.config.sources)
            if self.partition == "training"
            else set(self.config.evaluation_sources)
        )
        for asset_sha, source, shard in frame[
            ["sha256", "owner_source", "shard_id"]
        ].itertuples(index=False, name=None):
            _sha(asset_sha, "registry SHA-256")
            _component(source, "registry owner source")
            _component(shard, "registry shard")
            suffix = shard.removeprefix(f"{source}-")
            if (
                source not in configured
                or not shard.startswith(f"{source}-")
                or len(suffix) != 5
                or not suffix.isdigit()
            ):
                raise ArtifactValidationError(
                    f"invalid registry source/shard identity: {source}/{shard}"
                )

    def load(self) -> pd.DataFrame:
        manifest = _safe_json(self.manifest_path, "registry manifest")
        if set(manifest) != {
            "schema_version",
            "artifact_type",
            "partition",
            "config_hash",
            "rows",
            "sha256",
            "source_inputs",
        }:
            raise ArtifactValidationError("invalid registry manifest schema")
        if manifest["schema_version"] != REGISTRY_SCHEMA_VERSION:
            raise ArtifactValidationError("unsupported registry schema")
        if manifest["artifact_type"] != REGISTRY_ARTIFACT_TYPES[self.partition]:
            raise ArtifactValidationError("invalid registry artifact type")
        if manifest["partition"] != self.partition:
            raise ArtifactValidationError("registry partition mismatch")
        if manifest["config_hash"] != self.config.config_hash():
            raise ArtifactValidationError("registry config hash mismatch")
        rows = _nonnegative_count(manifest["rows"], "registry rows", positive=True)
        digest = _sha(manifest["sha256"], "registry checksum")
        self._validate_source_inputs(manifest["source_inputs"])
        try:
            payload = _read_regular_bytes_nofollow(self.path)
        except (InfrastructureError, OSError) as error:
            raise ArtifactValidationError(
                f"missing or unsafe registry: {self.path}: {error}"
            ) from error
        if sha256(payload).hexdigest() != digest:
            raise ArtifactValidationError("registry checksum mismatch")
        try:
            frame = pd.read_parquet(io.BytesIO(payload))
        except (ImportError, TypeError, ValueError) as error:
            raise ArtifactValidationError(f"corrupt registry: {error}") from error
        if len(frame) != rows or frame.empty:
            raise ArtifactValidationError("registry row count mismatch")
        self._validate_frame(frame)
        return frame

    def save(
        self, frame: pd.DataFrame, *, source_inputs: Mapping | None = None
    ) -> None:
        self._validate_frame(frame)
        source_inputs = self._validate_source_inputs(source_inputs or {})
        stream = io.BytesIO()
        try:
            frame.to_parquet(stream, index=False)
            payload = stream.getvalue()
            pd.read_parquet(io.BytesIO(payload))
            _atomic_write_bytes_nofollow(self.path, payload)
        except (InfrastructureError, OSError) as error:
            raise ArtifactValidationError(
                f"cannot write registry: {self.path}: {error}"
            ) from error
        except (ImportError, NotImplementedError, TypeError, ValueError) as error:
            raise ArtifactValidationError(
                f"cannot serialize registry: {error}"
            ) from error
        _write_json(
            self.manifest_path,
            self._manifest(payload, len(frame), source_inputs),
            "registry manifest",
        )

    def update_state(self, asset_sha256, field, state, error="") -> None:
        source_inputs = _safe_json(
            self.manifest_path, "registry manifest"
        )["source_inputs"]
        frame = self.load()
        selected = frame["sha256"] == asset_sha256
        if selected.sum() != 1:
            raise KeyError(asset_sha256)
        frame.loc[selected, field] = state.value
        if error:
            frame.loc[selected, "last_error"] = error
        self.save(frame, source_inputs=source_inputs)


class CanonicalRegistryBuilder:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        source_loader: Callable[[str], pd.DataFrame] | None = None,
        store: SafeRegistryStore | None = None,
        evaluation_store: SafeRegistryStore | None = None,
    ):
        self.config = config
        self.source_order = (*config.sources, *config.evaluation_sources)
        if not self.source_order or len(self.source_order) != len(
            set(self.source_order)
        ):
            raise ArtifactValidationError("configured source order is invalid")
        self.source_loader = source_loader or self._load_local_source
        self._input_digests: dict[str, str] = {}
        self.store = store or SafeRegistryStore(
            config.paths.data2_root / "control/assets.parquet", config
        )
        self.evaluation_store = evaluation_store or SafeRegistryStore(
            config.paths.data2_root / "control/evaluation_assets.parquet",
            config,
            partition="evaluation",
        )

    def _metadata_path(self, source: str) -> Path:
        return (
            self.config.paths.data2_root
            / "control/metadata"
            / source
            / "metadata.csv"
        )

    def _load_local_source(self, source: str) -> pd.DataFrame:
        path = self._metadata_path(source)
        try:
            payload = _read_regular_bytes_nofollow(path)
            frame = pd.read_csv(io.BytesIO(payload), dtype={"sha256": str})
        except (InfrastructureError, OSError) as error:
            raise ArtifactValidationError(
                f"missing or unsafe canonical {source} metadata: {path}: {error}"
            ) from error
        except (UnicodeError, ValueError, pd.errors.ParserError) as error:
            raise ArtifactValidationError(
                f"corrupt canonical {source} metadata: {path}: {error}"
            ) from error
        self._input_digests[source] = sha256(payload).hexdigest()
        return frame

    @staticmethod
    def _validate_source(source: str, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ArtifactValidationError(f"{source} metadata must not be empty")
        missing = {"sha256", "file_identifier"} - set(frame.columns)
        if missing:
            raise ArtifactValidationError(
                f"{source} metadata missing columns: {sorted(missing)}"
            )
        result = frame.copy()
        for value in result["sha256"]:
            _sha(value, f"{source} asset SHA-256")
        if result["sha256"].duplicated().any():
            raise ArtifactValidationError(f"duplicate {source} asset SHA-256")
        if not result["file_identifier"].map(
            lambda item: isinstance(item, str) and bool(item)
        ).all():
            raise ArtifactValidationError(f"invalid {source} file identifier")
        return result

    def __call__(self) -> pd.DataFrame:
        self._input_digests = {}
        frames = {}
        for source in self.source_order:
            _component(source, "source")
            try:
                loaded = self.source_loader(source)
            except ArtifactValidationError:
                raise
            except (OSError, TypeError, ValueError, pd.errors.ParserError) as error:
                raise ArtifactValidationError(
                    f"cannot load canonical {source} metadata: {error}"
                ) from error
            frames[source] = self._validate_source(source, loaded)
            if source not in self._input_digests:
                payload = frames[source].to_csv(index=False).encode("utf-8")
                self._input_digests[source] = sha256(payload).hexdigest()
        try:
            training = canonicalize_sources(
                {source: frames[source] for source in self.config.sources},
                self.config.render.camera_policy,
                self.config.sources,
            )
            evaluation = canonicalize_sources(
                {
                    source: frames[source]
                    for source in self.config.evaluation_sources
                },
                self.config.render.camera_policy,
                self.config.evaluation_sources,
            )
            evaluation["split"] = "evaluation"
            training = training.loc[
                ~training["sha256"].isin(set(evaluation["sha256"]))
            ].copy()
            training = assign_shards(training, self.config.shard_size)
            evaluation = assign_shards(evaluation, self.config.shard_size)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ArtifactValidationError(
                f"cannot canonicalize registry: {error}"
            ) from error
        if training.empty or evaluation.empty:
            raise ArtifactValidationError("canonical registry partitions must not be empty")

        def inputs(sources):
            return {
                source: {
                    "path": self._metadata_path(source).as_posix(),
                    "sha256": self._input_digests[source],
                    "rows": len(frames[source]),
                }
                for source in sources
            }

        training_inputs = inputs(self.config.sources)
        evaluation_inputs = inputs(self.config.evaluation_sources)
        self.store.save(training, source_inputs=training_inputs)
        self.evaluation_store.save(
            evaluation, source_inputs=evaluation_inputs
        )
        for partition, canonical, sources in (
            ("training", training, self.config.sources),
            ("evaluation", evaluation, self.config.evaluation_sources),
        ):
            for source in sources:
                selected = canonical.loc[
                    canonical["owner_source"] == source
                ].sort_values("sha256")
                _write_csv(
                    self.config.paths.data2_root
                    / "control/compatibility_metadata"
                    / partition
                    / source
                    / "metadata.csv",
                    selected,
                    f"{source} compatibility metadata",
                )

        registry_manifest = _safe_json(
            self.store.manifest_path, "training registry manifest"
        )
        references: dict[str, dict[str, list[dict[str, str]]]] = {}
        for record in training.to_dict("records"):
            source = record["owner_source"]
            raw_value = record.get("local_path") or record.get("file_identifier")
            raw_path = _canonical_raw_reference_path(source, raw_value)
            references.setdefault(source, {}).setdefault(raw_path, []).append(
                {"sha256": record["sha256"], "shard_id": record["shard_id"]}
            )
        ordered_references = {
            source: {
                path: sorted(entries, key=lambda entry: entry["sha256"])
                for path, entries in sorted(references.get(source, {}).items())
            }
            for source in self.config.sources
        }
        _write_json(
            self.config.paths.data2_root / "control/raw_references.json",
            {
                "schema_version": REFERENCE_INDEX_SCHEMA_VERSION,
                "artifact_type": REFERENCE_INDEX_ARTIFACT_TYPE,
                "config_hash": self.config.config_hash(),
                "training_registry_sha256": registry_manifest["sha256"],
                "sources": ordered_references,
            },
            "canonical raw reference index",
        )
        return training


def read_gate_report(
    config: PipelineConfig, gate: str, *, require_passed: bool = True
) -> dict:
    _component(gate, "gate")
    if gate not in {"smoke", "pilot", "production"}:
        raise ArtifactValidationError(f"unknown gate: {gate}")
    path = config.paths.data2_root / "control/reports/gates" / f"{gate}.json"
    published = _safe_json(path, f"{gate} gate report")
    derived, handoff, handoff_payload = RuntimeReportBuilder(config)._derive_gate(gate)
    if published != derived:
        raise ArtifactValidationError(
            f"{gate} gate report does not match held evidence"
        )
    if handoff is not None:
        handoff_path = (
            config.paths.data2_root / "control/splits/training_handoff.json"
        )
        try:
            held_handoff = _read_regular_bytes_nofollow(handoff_path)
        except (InfrastructureError, OSError) as error:
            raise ArtifactValidationError(
                f"missing or unsafe training handoff: {error}"
            ) from error
        if held_handoff != handoff_payload:
            raise ArtifactValidationError(
                "training handoff does not match held evidence"
            )
    if require_passed and derived["decision"] != "passed":
        raise ArtifactValidationError(f"{gate} gate has not passed")
    return derived


def _finite(value, description: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
        or (not positive and float(value) < 0)
    ):
        qualifier = "positive and finite" if positive else "finite and non-negative"
        raise ArtifactValidationError(f"{description} must be {qualifier}")
    return float(value)


def _major_minor_at_least(value: str, minimum: tuple[int, int]) -> bool:
    try:
        major, minor = value.split("+", 1)[0].split(".", 2)[:2]
        return (int(major), int(minor)) >= minimum
    except (AttributeError, TypeError, ValueError):
        return False


def _fresh_timestamp(value, description: str, *, now=None) -> str:
    if not isinstance(value, str):
        raise ArtifactValidationError(f"{description} timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ArtifactValidationError(f"invalid {description} timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactValidationError(f"{description} timestamp must be timezone-aware")
    current = now or datetime.now(timezone.utc)
    age = (current - parsed.astimezone(timezone.utc)).total_seconds()
    if age < -300 or age > 24 * 60 * 60:
        raise ArtifactValidationError(f"{description} evidence is not fresh")
    return parsed.astimezone(timezone.utc).isoformat()


def _held_timestamp(value, description: str, *, now=None) -> str:
    if not isinstance(value, str):
        raise ArtifactValidationError(
            f"{description} timestamp must be a string"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ArtifactValidationError(
            f"invalid {description} timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactValidationError(
            f"{description} timestamp must be timezone-aware"
        )
    current = now or datetime.now(timezone.utc)
    if (current - parsed.astimezone(timezone.utc)).total_seconds() < -300:
        raise ArtifactValidationError(
            f"{description} timestamp is in the future"
        )
    return parsed.astimezone(timezone.utc).isoformat()


def derive_hardware_report(
    value: Mapping,
    config: PipelineConfig,
    evidence_sha256: str,
    *,
    now=None,
) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "artifact_type",
        "config_hash",
        "created_at",
        "software",
        "gpus",
        "storage",
        "source_measurements",
    }:
        raise ArtifactValidationError("invalid hardware preflight evidence schema")
    if value["schema_version"] != 2 or value["artifact_type"] != "hardware_preflight_evidence":
        raise ArtifactValidationError("unsupported hardware preflight evidence")
    if value["config_hash"] != config.config_hash():
        raise ArtifactValidationError("hardware preflight config hash mismatch")
    created_at = _fresh_timestamp(value["created_at"], "hardware", now=now)
    evidence_digest = _sha(evidence_sha256, "hardware evidence checksum")

    software = value["software"]
    if not isinstance(software, Mapping) or set(software) != {
        "cuda_version",
        "torch_version",
        "blender_version",
        "optix_enabled",
    }:
        raise ArtifactValidationError("invalid hardware software evidence")
    for name in ("cuda_version", "torch_version", "blender_version"):
        if not isinstance(software[name], str) or not software[name]:
            raise ArtifactValidationError(f"invalid hardware {name}")
    if not isinstance(software["optix_enabled"], bool):
        raise ArtifactValidationError("invalid hardware OptiX flag")

    gpu_values = value["gpus"]
    if not isinstance(gpu_values, list) or len(gpu_values) != 7:
        raise ArtifactValidationError("hardware requires exactly seven GPUs")
    gpus = []
    for position, item in enumerate(gpu_values):
        if not isinstance(item, Mapping) or set(item) != {
            "index",
            "name",
            "cuda_visible_device",
            "cycles_device",
            "cpu_fallback_detected",
            "cube_render_sha256",
        }:
            raise ArtifactValidationError("invalid hardware GPU evidence")
        if item["index"] != position or isinstance(item["index"], bool):
            raise ArtifactValidationError("hardware GPU inventory is not contiguous")
        if item["cuda_visible_device"] != str(position):
            raise ArtifactValidationError("hardware GPU visibility is not isolated")
        if not isinstance(item["name"], str) or not item["name"]:
            raise ArtifactValidationError("hardware GPU names must not be empty")
        if not isinstance(item["cpu_fallback_detected"], bool):
            raise ArtifactValidationError("invalid CPU fallback evidence")
        _sha(item["cube_render_sha256"], "cube render checksum")
        gpus.append(dict(item))

    storage_values = value["storage"]
    if not isinstance(storage_values, Mapping) or set(storage_values) != {"local", "data2", "data3"}:
        raise ArtifactValidationError("invalid storage preflight evidence")
    storage = {}
    fixture_bytes = 10 * 1024**3
    floors = {}
    for root_name, item in storage_values.items():
        if not isinstance(item, Mapping) or set(item) != {
            "fixture_bytes",
            "write_elapsed_seconds",
            "read_elapsed_seconds",
            "write_sha256",
            "read_sha256",
            "total_bytes",
            "free_bytes_before",
            "free_bytes_after",
            "fixture_removed",
        }:
            raise ArtifactValidationError(f"invalid {root_name} storage evidence")
        if _nonnegative_count(item["fixture_bytes"], "fixture bytes", positive=True) != fixture_bytes:
            raise ArtifactValidationError("storage fixture must be exactly 10 GiB")
        write_elapsed = _finite(item["write_elapsed_seconds"], "write elapsed", positive=True)
        read_elapsed = _finite(item["read_elapsed_seconds"], "read elapsed", positive=True)
        write_sha = _sha(item["write_sha256"], "storage write checksum")
        read_sha = _sha(item["read_sha256"], "storage read checksum")
        total = _nonnegative_count(item["total_bytes"], "storage total bytes", positive=True)
        before = _nonnegative_count(item["free_bytes_before"], "storage free before")
        after = _nonnegative_count(item["free_bytes_after"], "storage free after")
        if before > total or after > total or not isinstance(item["fixture_removed"], bool):
            raise ArtifactValidationError("invalid storage capacity evidence")
        if root_name == "local":
            floor = max(
                math.ceil(total * config.limits.local_free_percent / 100),
                config.limits.local_free_gib * 1024**3,
            )
        elif root_name == "data2":
            floor = config.limits.data2_fs_free_tib * 1024**4
        else:
            floor = config.limits.data3_fs_free_tib * 1024**4
        floors[root_name] = floor
        storage[root_name] = {
            **dict(item),
            "write_mib_per_second": fixture_bytes / write_elapsed / 1024**2,
            "read_mib_per_second": fixture_bytes / read_elapsed / 1024**2,
            "free_floor_bytes": floor,
            "passed": all((write_sha == read_sha, before >= floor, after >= floor, item["fixture_removed"])),
        }

    samples = value["source_measurements"]
    if not isinstance(samples, Mapping) or set(samples) != set(config.sources):
        raise ArtifactValidationError("hardware sizing must cover every training source")
    sizing = {}
    for source in config.sources:
        values = samples[source]
        if not isinstance(values, list) or not values:
            raise ArtifactValidationError("hardware sizing measurements must not be empty")
        measured = [
            _nonnegative_count(item, f"{source} local byte sample", positive=True)
            for item in values
        ]
        p95 = math.ceil(float(pd.Series(measured, dtype=float).quantile(0.95)))
        sizing[source] = {"samples": len(measured), "p95_peak_local_bytes": p95}

    passed = all(
        (
            software["cuda_version"] == "12.8",
            _major_minor_at_least(software["torch_version"], (2, 8)),
            software["blender_version"] == config.render.blender_version,
            software["optix_enabled"],
            all(gpu["cycles_device"] == "OPTIX" for gpu in gpus),
            not any(gpu["cpu_fallback_detected"] for gpu in gpus),
            all(item["passed"] for item in storage.values()),
        )
    )
    return {
        "schema_version": 2,
        "artifact_type": HARDWARE_ARTIFACT_TYPE,
        "config_hash": config.config_hash(),
        "created_at": created_at,
        "evidence_sha256": evidence_digest,
        "decision": "passed" if passed else "failed",
        "software": dict(software),
        "gpu": {"gpu_count": 7, "devices": gpus},
        "storage": storage,
        "pilot_sizing": {"sources": sizing},
        "thresholds": {
            "fixture_bytes": fixture_bytes,
            "free_floor_bytes": floors,
            "required_gpus": 7,
            "cycles_device": config.render.cycles_device,
        },
    }


def validate_hardware_report(value: Mapping, config: PipelineConfig) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "artifact_type", "config_hash", "created_at",
        "evidence_sha256", "decision", "software", "gpu", "storage",
        "pilot_sizing", "thresholds",
    }:
        raise ArtifactValidationError("invalid hardware preflight report schema")
    if value["schema_version"] != 2 or value["artifact_type"] != HARDWARE_ARTIFACT_TYPE:
        raise ArtifactValidationError("unsupported hardware preflight report")
    if value["config_hash"] != config.config_hash():
        raise ArtifactValidationError("hardware preflight config hash mismatch")
    _fresh_timestamp(value["created_at"], "hardware report")
    _sha(value["evidence_sha256"], "hardware evidence checksum")
    if value["decision"] not in {"passed", "failed"}:
        raise ArtifactValidationError("invalid hardware preflight decision")
    return dict(value)


def read_hardware_report(config: PipelineConfig, *, require_passed=True) -> dict:
    report_path = config.paths.data2_root / "control/reports/hardware.json"
    report = validate_hardware_report(
        _safe_json(report_path, "hardware preflight report"), config
    )
    evidence_path = config.paths.data2_root / "control/report_inputs/hardware.json"
    try:
        evidence_payload = _read_regular_bytes_nofollow(evidence_path)
        evidence = json.loads(evidence_payload)
    except (InfrastructureError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"invalid hardware evidence: {error}") from error
    derived = derive_hardware_report(
        evidence,
        config,
        sha256(evidence_payload).hexdigest(),
    )
    if report != derived:
        raise ArtifactValidationError("hardware report does not match held evidence")
    if require_passed and report["decision"] != "passed":
        raise ArtifactValidationError("hardware preflight has not passed")
    return report


class PilotArtifactReader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def p95_peak_local_bytes(self, source: str) -> int:
        _component(source, "pilot source")
        pilot_path = (
            self.config.paths.data2_root
            / "control/reports/gates/pilot.json"
        )
        pilot_value = _optional_json(pilot_path, "pilot gate report")
        if pilot_value is not None:
            report = read_gate_report(self.config, "pilot")
            sources = report["capacity"]["sources"]
        else:
            sources = read_hardware_report(self.config)["pilot_sizing"][
                "sources"
            ]
        try:
            value = sources[source]["p95_peak_local_bytes"]
        except (KeyError, TypeError) as error:
            raise ArtifactValidationError(
                f"pilot report has no validated source: {source}"
            ) from error
        return _nonnegative_count(value, "pilot p95", positive=True)


def _raw_reference_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactValidationError(f"unsafe raw path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or ".." in pure.parts:
        raise ArtifactValidationError(f"unsafe raw path: {value!r}")
    for index, component in enumerate(pure.parts):
        if component.lower().endswith(".zip"):
            return PurePosixPath(*pure.parts[: index + 1]).as_posix()
    return pure.as_posix()


def _canonical_raw_reference_path(source: str, value: str) -> str:
    if source == "ObjaverseXL_github" and isinstance(value, str):
        if value.startswith("https://github.com/"):
            parts = value.split("/")
            if (
                len(parts) < 8
                or parts[:3] != ["https:", "", "github.com"]
                or parts[5] != "blob"
            ):
                raise ArtifactValidationError(
                    f"unsafe raw path: {value!r}"
                )
            organization = _component(parts[3], "GitHub organization")
            repository = _component(parts[4], "GitHub repository")
            _component(parts[6], "GitHub commit")
            return _raw_reference_path(
                f"raw/github/repos/{organization}/{repository}.zip"
            )
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc:
            parts = parsed.path.split("/")
            if parsed.scheme != "https" or parsed.query or parsed.fragment:
                raise ArtifactValidationError(
                    f"unsafe raw path: {value!r}"
                )
            if parsed.netloc.lower() == "3d-api.si.edu":
                if (
                    len(parts) != 5
                    or parts[:3] != ["", "content", "document"]
                    or not parts[3].startswith("3d_package:")
                    or not parts[4].endswith(".glb")
                ):
                    raise ArtifactValidationError(
                        f"unsafe raw path: {value!r}"
                    )
                uid = uuid5(NAMESPACE_DNS, value)
                return f"raw/smithsonian/objects/{uid}.glb"
            raise ArtifactValidationError(f"unsafe raw path: {value!r}")

    if source == "ObjaverseXL_sketchfab" and isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc:
            parts = parsed.path.split("/")
            if (
                parsed.scheme != "https"
                or parsed.netloc.lower() != "sketchfab.com"
                or len(parts) != 3
                or parts[0]
                or parts[1] != "3d-models"
            ):
                raise ArtifactValidationError(
                    f"unsafe raw path: {value!r}"
                )
            uid = parts[2]
        else:
            safe = _raw_reference_path(value)
            candidate = PurePosixPath(safe).name
            uid = candidate[:-4] if candidate.endswith(".glb") else ""
            if not uid:
                return safe
        if not (
            uid
            and all(
                character.isascii()
                and (character.isalnum() or character in "-_")
                for character in uid
            )
        ):
            raise ArtifactValidationError(f"unsafe raw path: {value!r}")
        return f"raw/hf-objaverse-v1/by-uid/{uid}.glb"

    if not isinstance(value, str) or value.startswith("raw/"):
        return _raw_reference_path(value)
    if source == "ABO":
        value = f"raw/3dmodels/original/{value}"
    elif source == "HSSD":
        value = f"raw/{value}"
    elif source == "3D-FUTURE":
        value = f"raw/{value}/raw_model.obj"
    return _raw_reference_path(value)


class FrozenReferenceCounter:
    """Caches the complete canonical reference index for one runtime."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._references: dict[str, dict[str, tuple[dict[str, str], ...]]] | None = None
        self._frozen_cache: dict[
            str, dict[str, tuple[str, str, tuple[str, ...]]]
        ] = {}

    def _reference_index(self):
        if self._references is not None:
            return self._references
        store = SafeRegistryStore(
            self.config.paths.data2_root / "control/assets.parquet", self.config
        )
        registry = store.load()
        registry_manifest = _safe_json(
            store.manifest_path, "training registry manifest"
        )
        value = _safe_json(
            self.config.paths.data2_root / "control/raw_references.json",
            "canonical raw reference index",
        )
        if (
            set(value)
            != {
                "schema_version",
                "artifact_type",
                "config_hash",
                "training_registry_sha256",
                "sources",
            }
            or value["schema_version"] != REFERENCE_INDEX_SCHEMA_VERSION
            or value["artifact_type"] != REFERENCE_INDEX_ARTIFACT_TYPE
            or value["config_hash"] != self.config.config_hash()
            or value["training_registry_sha256"] != registry_manifest["sha256"]
            or not isinstance(value["sources"], Mapping)
            or set(value["sources"]) != set(self.config.sources)
        ):
            raise ArtifactValidationError("invalid canonical raw reference index")

        expected = {}
        for record in registry.to_dict("records"):
            raw_value = record.get("local_path") or record.get("file_identifier")
            if not isinstance(raw_value, str) or not raw_value:
                raise ArtifactValidationError(
                    "training registry is missing a canonical raw path"
                )
            expected[record["sha256"]] = (
                record["owner_source"],
                record["shard_id"],
                _canonical_raw_reference_path(record["owner_source"], raw_value),
            )
        seen = set()
        result = {}
        for source in self.config.sources:
            source_value = value["sources"][source]
            if not isinstance(source_value, Mapping):
                raise ArtifactValidationError("invalid raw reference source index")
            source_result = {}
            for raw_path, entries in source_value.items():
                if _canonical_raw_reference_path(source, raw_path) != raw_path:
                    raise ArtifactValidationError("non-canonical raw reference path")
                if not isinstance(entries, list) or not entries:
                    raise ArtifactValidationError("empty raw reference entry")
                validated = []
                for entry in entries:
                    if not isinstance(entry, Mapping) or set(entry) != {
                        "sha256",
                        "shard_id",
                    }:
                        raise ArtifactValidationError("invalid raw reference entry")
                    asset = _sha(entry["sha256"], "raw reference asset")
                    shard = _component(entry["shard_id"], "raw reference shard")
                    if asset in seen or expected.get(asset) != (
                        source,
                        shard,
                        raw_path,
                    ):
                        raise ArtifactValidationError(
                            "raw reference index does not match training registry"
                        )
                    seen.add(asset)
                    validated.append({"sha256": asset, "shard_id": shard})
                source_result[raw_path] = tuple(validated)
            result[source] = source_result
        if seen != set(expected):
            raise ArtifactValidationError("raw reference index coverage is incomplete")
        self._references = result
        return result

    def _shard_directories(self, source: str) -> tuple[Path, ...]:
        root = self.config.paths.data2_root / "control/shards" / source
        try:
            root_fd = _open_directory_nofollow(root)
        except (OSError, InfrastructureError) as error:
            raise ArtifactValidationError(
                f"missing or unsafe frozen shard root: {root}: {error}"
            ) from error
        try:
            names = sorted(os.listdir(root_fd))
            result = []
            for name in names:
                _component(name, "frozen shard")
                try:
                    value = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                except OSError as error:
                    raise ArtifactValidationError(
                        f"cannot inspect frozen shard: {name}: {error}"
                    ) from error
                if not stat.S_ISDIR(value.st_mode):
                    raise ArtifactValidationError(
                        f"non-directory in frozen shard root: {name}"
                    )
                result.append(root / name)
            return tuple(result)
        finally:
            os.close(root_fd)

    def _frozen_batches(self, source: str, root: Path):
        shard = root.name
        marker = _safe_json(root / "batches.json", "frozen batch manifest")
        if (
            set(marker)
            != {
                "schema_version",
                "gate",
                "source",
                "shard_id",
                "config_hash",
                "canonical_shard_sha256",
                "scope_sha256",
                "batches",
            }
            or marker["schema_version"] != 2
            or marker["gate"] != "production"
            or marker["source"] != source
            or marker["shard_id"] != shard
            or marker["config_hash"] != self.config.config_hash()
            or not isinstance(marker["batches"], list)
            or not marker["batches"]
        ):
            raise ArtifactValidationError(
                f"invalid frozen batch manifest: {root}"
            )
        actual = []
        expected_names = []
        seen = set()
        for index, entry in enumerate(marker["batches"]):
            name = f"batch{index:03d}.txt"
            batch_id = name.removesuffix(".txt")
            expected_names.append(name)
            if (
                not isinstance(entry, dict)
                or set(entry) != {"name", "count", "sha256"}
                or entry["name"] != name
            ):
                raise ArtifactValidationError(f"invalid frozen batch entry: {name}")
            count = _nonnegative_count(
                entry["count"], "frozen batch count", positive=True
            )
            expected_sha = _sha(entry["sha256"], "frozen batch checksum")
            try:
                payload = _read_regular_bytes_nofollow(root / name)
            except (InfrastructureError, OSError) as error:
                raise ArtifactValidationError(
                    f"missing or unsafe frozen batch: {root / name}: {error}"
                ) from error
            if sha256(payload).hexdigest() != expected_sha:
                raise ArtifactValidationError("frozen batch checksum mismatch")
            try:
                text = payload.decode("ascii")
                if not text.endswith("\n"):
                    raise ValueError("missing final newline")
                assets = tuple(_sha(item, "frozen asset") for item in text.splitlines())
            except (UnicodeError, ValueError, ArtifactValidationError) as error:
                raise ArtifactValidationError(
                    f"invalid frozen batch payload: {root / name}: {error}"
                ) from error
            if (
                len(assets) != count
                or len(assets) > self.config.shard_size
                or tuple(sorted(assets)) != assets
                or len(assets) != len(set(assets))
                or seen.intersection(assets)
            ):
                raise ArtifactValidationError(f"invalid frozen batch: {root / name}")
            seen.update(assets)
            actual.append((shard, batch_id, assets))
        try:
            root_fd = _open_directory_nofollow(root)
            try:
                actual_names = sorted(
                    name
                    for name in os.listdir(root_fd)
                    if name.startswith("batch") and name.endswith(".txt")
                )
            finally:
                os.close(root_fd)
        except OSError as error:
            raise ArtifactValidationError(f"unsafe frozen shard: {root}") from error
        if actual_names != expected_names:
            raise ArtifactValidationError(f"frozen batch file set mismatch: {root}")
        flattened = tuple(asset for _, _, assets in actual for asset in assets)
        payload = "".join(f"{asset}\n" for asset in flattened).encode("ascii")
        if (
            marker["canonical_shard_sha256"] != sha256(payload).hexdigest()
            or marker["scope_sha256"] != sha256(payload).hexdigest()
        ):
            raise ArtifactValidationError("frozen production scope identity mismatch")
        return tuple(actual)

    def _frozen_assets(self, source: str):
        if source in self._frozen_cache:
            return self._frozen_cache[source]
        references = self._reference_index()[source]
        expected_shards: dict[str, set[str]] = {}
        for entries in references.values():
            for entry in entries:
                expected_shards.setdefault(entry["shard_id"], set()).add(
                    entry["sha256"]
                )
        result = {}
        root = self.config.paths.data2_root / "control/shards" / source
        try:
            directories = self._shard_directories(source)
        except ArtifactValidationError as error:
            if not root.exists() and not root.is_symlink():
                directories = ()
            else:
                raise error
        for directory in directories:
            shard_assets = []
            for shard, batch, assets in self._frozen_batches(source, directory):
                shard_assets.extend(assets)
                for asset in assets:
                    if asset in result:
                        raise ArtifactValidationError("asset frozen more than once")
                    result[asset] = (shard, batch, assets)
            if set(shard_assets) != expected_shards.get(directory.name, set()):
                raise ArtifactValidationError(
                    "frozen production shard is not the full canonical shard"
                )
        self._frozen_cache[source] = result
        return result

    def _archive_verified(
        self, source: str, shard: str, batch: str, assets: tuple[str, ...]
    ) -> bool:
        archive = (
            self.config.paths.data3_root
            / "archive/raw"
            / source
            / shard
            / f"{batch}.tar"
        )
        manifest_path = archive.with_suffix(".tar.manifest.json")
        try:
            archive_payload = _read_regular_bytes_nofollow(
                archive, missing_ok=True
            )
            manifest_payload = _read_regular_bytes_nofollow(
                manifest_path, missing_ok=True
            )
        except (InfrastructureError, OSError) as error:
            raise ArtifactValidationError(f"unsafe raw archive: {archive}: {error}") from error
        if archive_payload is None and manifest_payload is None:
            return False
        if (archive_payload is None) != (manifest_payload is None):
            raise ArtifactValidationError(
                f"incomplete raw archive publication: {archive}"
            )
        try:
            manifest = json.loads(manifest_payload)
            if not isinstance(manifest, dict):
                raise ValueError("manifest is not an object")
            if manifest.get("pack_sha256") != sha256(archive_payload).hexdigest():
                raise ValueError("archive checksum mismatch")
            expected_members = {
                item["path"]: (item["size"], item["sha256"])
                for item in manifest["members"]
            }
            actual_members = {}
            with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as bundle:
                for member in bundle:
                    name = member.name
                    pure = PurePosixPath(name)
                    if (
                        not name
                        or pure.is_absolute()
                        or pure.as_posix() != name
                        or ".." in pure.parts
                        or not member.isfile()
                        or name in actual_members
                    ):
                        raise ValueError("unsafe raw archive member")
                    stream = bundle.extractfile(member)
                    if stream is None:
                        raise ValueError("missing raw archive member")
                    payload = stream.read()
                    actual_members[name] = (len(payload), sha256(payload).hexdigest())
            if actual_members != expected_members:
                raise ValueError("raw archive member mismatch")
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError, tarfile.TarError) as error:
            raise ArtifactValidationError(
                f"corrupt raw archive: {archive}: {error}"
            ) from error
        if (
            manifest.get("family") != "raw"
            or manifest.get("gate") != "production"
            or manifest.get("shard_id") != shard
            or manifest.get("batch_id") != batch
            or manifest.get("config_hash") != self.config.config_hash()
            or tuple(manifest.get("asset_sha256s", ())) != assets
            or not isinstance(manifest.get("validated_at"), str)
            or not manifest["validated_at"]
            or manifest.get("completed_count", -1)
            + manifest.get("quarantined_count", -1)
            != len(assets)
        ):
            raise ArtifactValidationError(
                f"raw archive identity mismatch: {manifest_path}"
            )
        return True

    def pending_references(
        self,
        source: str,
        raw_relative_path: str,
        *,
        excluding_shard_id: str,
        excluding_batch_id: str,
        gate: str = "production",
    ) -> int:
        _component(source, "source")
        _component(excluding_shard_id, "excluded shard")
        _component(excluding_batch_id, "excluded batch")
        requested = _canonical_raw_reference_path(source, raw_relative_path)
        if gate not in {"smoke", "pilot", "production"}:
            raise ArtifactValidationError(f"invalid reference gate: {gate}")
        references = self._reference_index().get(source, {}).get(requested)
        if not references:
            raise ArtifactValidationError(
                f"raw path is absent from canonical reference index: {requested}"
            )
        if gate != "production":
            return len(references)
        frozen = self._frozen_assets(source)
        pending = 0
        for entry in references:
            identity = frozen.get(entry["sha256"])
            if identity is None:
                pending += 1
                continue
            shard, batch, assets = identity
            if shard == excluding_shard_id and batch == excluding_batch_id:
                continue
            if not self._archive_verified(source, shard, batch, assets):
                pending += 1
        return pending


class PersistentProjectAccounting:
    def __init__(
        self,
        config: PipelineConfig,
        accounting: ProjectStorageAccounting,
        path: Path,
    ):
        self.config = config
        self.accounting = accounting
        self.path = Path(path)

    def _persist(self) -> None:
        data2_bytes, data3_bytes = self.accounting.current_bytes()
        _write_json(
            self.path,
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_type": ACCOUNTING_ARTIFACT_TYPE,
                "config_hash": self.config.config_hash(),
                "data2_bytes": data2_bytes,
                "data3_bytes": data3_bytes,
            },
            "project accounting",
        )

    def current_bytes(self) -> tuple[int, int]:
        return self.accounting.current_bytes()

    def record_registry_delta(self, path: Path, delta_bytes: int) -> None:
        self.accounting.record_registry_delta(path, delta_bytes)
        self._persist()

    def reconcile_at_shard_boundary(self) -> tuple[int, int]:
        result = self.accounting.reconcile_at_shard_boundary()
        self._persist()
        return result


def _accounting_values(config: PipelineConfig, path: Path) -> tuple[int, int]:
    try:
        value = _safe_json(path, "project accounting")
    except ArtifactValidationError as error:
        cause = error.__cause__
        if not isinstance(cause, FileNotFoundError):
            raise
        return (0, 0)
    if set(value) != {
        "schema_version",
        "artifact_type",
        "config_hash",
        "data2_bytes",
        "data3_bytes",
    }:
        raise ArtifactValidationError("invalid project accounting schema")
    if (
        value["schema_version"] != ARTIFACT_SCHEMA_VERSION
        or value["artifact_type"] != ACCOUNTING_ARTIFACT_TYPE
    ):
        raise ArtifactValidationError("unsupported project accounting artifact")
    if value["config_hash"] != config.config_hash():
        raise ArtifactValidationError("project accounting config hash mismatch")
    return (
        _nonnegative_count(value["data2_bytes"], "data2 accounting"),
        _nonnegative_count(value["data3_bytes"], "data3 accounting"),
    )


def initialize_project_accounting(
    config: PipelineConfig,
    *,
    directory_size: Callable[[Path], int] = _directory_size,
) -> PersistentProjectAccounting:
    path = config.paths.data2_root / "control/accounting.json"
    initial_data2, initial_data3 = _accounting_values(config, path)
    accounting = ProjectStorageAccounting(
        config.paths.data2_root,
        config.paths.data3_root,
        data2_bytes=initial_data2,
        data3_bytes=initial_data3,
        directory_size=directory_size,
    )
    persistent = PersistentProjectAccounting(config, accounting, path)
    persistent.reconcile_at_shard_boundary()
    return persistent


class NoFollowTelemetryWriter:
    """Task 8 telemetry contract with descriptor-safe append creation."""

    def __init__(self, path: Path, *, clock=time.monotonic, sync_interval=30.0):
        self.path = Path(path)
        try:
            parent_fd = _open_directory_nofollow(
                self.path.parent, create=True
            )
            try:
                file_fd = os.open(
                    self.path.name,
                    os.O_WRONLY
                    | os.O_APPEND
                    | os.O_CREAT
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
            finally:
                os.close(parent_fd)
        except OSError as error:
            raise ArtifactValidationError(
                f"unsafe telemetry artifact: {self.path}: {error}"
            ) from error
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            os.close(file_fd)
            raise ArtifactValidationError(f"telemetry is not a file: {self.path}")
        self._stream = os.fdopen(file_fd, "a", encoding="utf-8")
        self._clock = clock
        self._sync_interval = float(sync_interval)
        if not math.isfinite(self._sync_interval) or self._sync_interval <= 0:
            self._stream.close()
            raise ValueError("telemetry sync interval must be positive")
        self._last_sync = clock()
        self._closed = False

    def _sync(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._last_sync = self._clock()

    def write(self, snapshot, decision, shard_id, command) -> None:
        if self._closed:
            raise ValueError("telemetry writer is closed")
        payload = asdict(snapshot)
        timestamp = snapshot.timestamp
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("resource timestamp must be timezone-aware")
        payload["timestamp"] = timestamp.isoformat()
        payload.pop("monotonic_seconds", None)
        payload.update(
            shard_id=shard_id,
            command=command,
            action=decision.action.value,
            reasons=decision.reasons,
        )
        self._stream.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._clock() - self._last_sync >= self._sync_interval:
            self._sync()

    def close(self) -> None:
        if self._closed:
            return
        sync_error = None
        close_error = None
        try:
            self._sync()
        except BaseException as error:
            sync_error = error
        try:
            self._stream.close()
        except BaseException as error:
            close_error = error
        finally:
            self._closed = True
        if sync_error is not None:
            if close_error is not None:
                raise sync_error from close_error
            raise sync_error
        if close_error is not None:
            raise close_error


def _read_bound_artifact(path: Path, expected_sha: str, description: str) -> bytes:
    digest = _sha(expected_sha, f"{description} checksum")
    try:
        payload = _read_regular_bytes_nofollow(path)
    except (InfrastructureError, OSError) as error:
        raise ArtifactValidationError(
            f"missing or unsafe {description}: {path}: {error}"
        ) from error
    if sha256(payload).hexdigest() != digest:
        raise ArtifactValidationError(f"{description} checksum mismatch")
    return payload


def _directory_names(path: Path, description: str) -> tuple[str, ...]:
    try:
        descriptor = _open_directory_nofollow(path)
    except (InfrastructureError, OSError) as error:
        raise ArtifactValidationError(
            f"missing or unsafe {description}: {path}: {error}"
        ) from error
    try:
        result = []
        for name in sorted(os.listdir(descriptor)):
            _component(name, description)
            details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(details.st_mode):
                raise ArtifactValidationError(
                    f"non-directory in {description}: {name}"
                )
            result.append(name)
        return tuple(result)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot inspect {description}: {path}: {error}"
        ) from error
    finally:
        os.close(descriptor)


def _held_pack_manifest(pack_path: Path, manifest_path: Path) -> tuple[dict, str, str]:
    try:
        pack_payload = _read_regular_bytes_nofollow(pack_path)
        manifest_payload = _read_regular_bytes_nofollow(manifest_path)
    except (InfrastructureError, OSError) as error:
        raise ArtifactValidationError(
            f"missing or unsafe pack publication: {pack_path}: {error}"
        ) from error
    try:
        manifest = json.loads(manifest_payload)
        legacy_fields = {
            "shard_id",
            "batch_id",
            "family",
            "config_hash",
            "tool_commit",
            "asset_sha256s",
            "completed_count",
            "quarantined_count",
            "created_at",
            "validated_at",
            "pack_sha256",
            "members",
            "gate",
        }
        schema_version = (
            manifest.get("schema_version", 1)
            if isinstance(manifest, dict)
            else None
        )
        expected_fields = (
            legacy_fields
            if schema_version == 1
            else legacy_fields
            | {"schema_version", "included_asset_sha256s"}
        )
        if (
            not isinstance(manifest, dict)
            or schema_version not in {1, 2}
            or set(manifest) != expected_fields
        ):
            raise ValueError("manifest is not an object")
        pack_digest = sha256(pack_payload).hexdigest()
        if _sha(manifest["pack_sha256"], "pack checksum") != pack_digest:
            raise ValueError("pack checksum mismatch")
        assets = manifest["asset_sha256s"]
        if not isinstance(assets, list):
            raise ValueError("invalid pack asset identities")
        assets = tuple(_sha(item, "pack asset") for item in assets)
        if assets != tuple(sorted(set(assets))):
            raise ValueError("invalid pack asset ordering")
        completed = _nonnegative_count(
            manifest["completed_count"], "pack completed count"
        )
        quarantined = _nonnegative_count(
            manifest["quarantined_count"], "pack quarantined count"
        )
        if schema_version == 1 and completed + quarantined != len(assets):
            raise ValueError("pack terminal counts do not match assets")
        for field in (
            "shard_id",
            "batch_id",
            "family",
            "config_hash",
            "tool_commit",
            "created_at",
            "validated_at",
            "gate",
        ):
            if not isinstance(manifest[field], str) or not manifest[field]:
                raise ValueError(f"invalid pack manifest field: {field}")
        if not isinstance(manifest["members"], list):
            raise ValueError("invalid pack members")
        expected_members = {}
        for item in manifest["members"]:
            if not isinstance(item, Mapping) or set(item) != {
                "path",
                "size",
                "sha256",
            }:
                raise ValueError("invalid pack member schema")
            name = item["path"]
            pure = PurePosixPath(name) if isinstance(name, str) else None
            if (
                pure is None
                or not name
                or pure.is_absolute()
                or pure.as_posix() != name
                or ".." in pure.parts
                or name in expected_members
            ):
                raise ValueError("unsafe or duplicate pack member")
            size = _nonnegative_count(item["size"], "pack member size")
            expected_members[name] = (
                size,
                _sha(item["sha256"], "pack member checksum"),
            )
        if schema_version == 2:
            included_values = manifest["included_asset_sha256s"]
            if not isinstance(included_values, list):
                raise ValueError("invalid pack included identities")
            included = tuple(
                _sha(item, "pack included asset")
                for item in included_values
            )
            if (
                included != tuple(sorted(set(included)))
                or not set(included) <= set(assets)
                or completed != len(included)
                or quarantined != len(assets) - len(included)
            ):
                raise ValueError("pack included scope mismatch")
        else:
            included_set = {
                component
                for name in expected_members
                for component in PurePosixPath(name).parts
                if component in set(assets)
            }
            included = tuple(
                asset for asset in assets if asset in included_set
            )
            manifest["schema_version"] = 1
            manifest["included_asset_sha256s"] = list(included)
        actual_members = {}
        with tarfile.open(fileobj=io.BytesIO(pack_payload), mode="r:") as bundle:
            for member in bundle:
                pure = PurePosixPath(member.name)
                if (
                    not member.name
                    or pure.is_absolute()
                    or pure.as_posix() != member.name
                    or ".." in pure.parts
                    or not member.isfile()
                    or member.name in actual_members
                ):
                    raise ValueError("unsafe pack member")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("missing pack member")
                payload = stream.read()
                actual_members[member.name] = (
                    len(payload),
                    sha256(payload).hexdigest(),
                )
        if actual_members != expected_members:
            raise ValueError("pack member mismatch")
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
        tarfile.TarError,
    ) as error:
        raise ArtifactValidationError(
            f"corrupt pack publication: {pack_path}: {error}"
        ) from error
    return (
        manifest,
        sha256(pack_payload).hexdigest(),
        sha256(manifest_payload).hexdigest(),
    )


def _family_root(family: str) -> Path:
    if family == "common":
        return Path("common")
    if family.startswith("SS-"):
        return Path("ss", family.removeprefix("SS-"))
    if family.startswith("shape-"):
        return Path("shape", family.removeprefix("shape-"))
    if family.startswith("PBR-"):
        return Path("pbr", family.removeprefix("PBR-"))
    raise ArtifactValidationError(f"unknown pack family: {family}")


def validate_family_memberships(
    included: Mapping[str, set[str]], config: PipelineConfig
) -> None:
    if not isinstance(included, Mapping) or set(included) != set(
        PACK_FAMILIES
    ):
        raise ArtifactValidationError(
            "family membership set is incomplete"
        )
    validated = {}
    for family in PACK_FAMILIES:
        values = included[family]
        if not isinstance(values, set):
            raise ArtifactValidationError(
                f"invalid family membership: {family}"
            )
        try:
            validated[family] = {
                _sha(asset, f"{family} included asset") for asset in values
            }
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError(
                f"invalid family membership: {family}"
            ) from error
    for resolution in config.targets.resolutions:
        pbr_family = f"PBR-{resolution}"
        shape_family = f"shape-{resolution}"
        if not validated[pbr_family] <= validated[shape_family]:
            raise ArtifactValidationError(
                f"{pbr_family} membership is not a {shape_family} subset"
            )
    ss_family = f"SS-{config.targets.ss_resolution}"
    highest_shape = f"shape-{max(config.targets.resolutions)}"
    if not validated[ss_family] <= validated[highest_shape]:
        raise ArtifactValidationError(
            f"{ss_family} membership is not a {highest_shape} subset"
        )
    non_common_union = set().union(
        *(
            validated[family]
            for family in PACK_FAMILIES
            if family != "common"
        )
    )
    if validated["common"] != non_common_union:
        raise ArtifactValidationError(
            "common membership is not the family union"
        )


def _gate_candidate(config: PipelineConfig, gate: str) -> tuple[dict, str, dict[str, bytes]]:
    path = config.paths.data2_root / "control/report_inputs" / f"{gate}.json"
    try:
        payload = _read_regular_bytes_nofollow(path)
        value = json.loads(payload)
    except (InfrastructureError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"invalid {gate} evidence manifest: {error}") from error
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "artifact_type",
        "config_hash",
        "gate",
        "created_at",
        "artifacts",
    }:
        raise ArtifactValidationError(f"invalid {gate} evidence manifest schema")
    if (
        value["schema_version"] != 2
        or value["artifact_type"] != "gate_evidence_manifest"
        or value["config_hash"] != config.config_hash()
        or value["gate"] != gate
    ):
        raise ArtifactValidationError(f"invalid {gate} evidence identity")
    created_at = _fresh_timestamp(value["created_at"], f"{gate} gate")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "measurements_sha256",
        "fp16_sha256",
        "telemetry_sha256",
    }:
        raise ArtifactValidationError(f"invalid {gate} evidence artifact schema")
    root = config.paths.data2_root / "control/report_evidence" / gate
    payloads = {
        "measurements": _read_bound_artifact(
            root / "measurements.csv",
            artifacts["measurements_sha256"],
            f"{gate} measurements",
        ),
        "fp16": _read_bound_artifact(
            root / "fp16.csv", artifacts["fp16_sha256"], f"{gate} FP16 evidence"
        ),
        "telemetry": _read_bound_artifact(
            root / "telemetry.jsonl",
            artifacts["telemetry_sha256"],
            f"{gate} telemetry",
        ),
    }
    return dict(value, created_at=created_at), sha256(payload).hexdigest(), payloads


class RuntimeReportBuilder:
    """Derives gate decisions from immutable, checksum-bound artifacts."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _quality_exclusion_evidence(
        self,
        gate: str,
        source: str,
        shard: str,
        frozen_assets: set[str],
    ) -> tuple[set[str], Mapping[str, set[str]]]:
        control = self.config.paths.data2_root / "control"
        root = (
            control / "quality"
            if gate == "production"
            else control / "qualification" / gate / "quality"
        )
        path = root / source / f"{shard}.json"
        try:
            value = json.loads(_read_regular_bytes_nofollow(path))
        except (
            InfrastructureError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
        ) as error:
            raise ArtifactValidationError(
                f"family exclusion evidence is unavailable: {path}: {error}"
            ) from error
        required = {
            "schema_version",
            "source",
            "shard_id",
            "gate",
            "batches",
            "entries",
            "quarantine",
            "family_exclusions",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != required
            or value["schema_version"] != 3
            or value["source"] != source
            or value["shard_id"] != shard
            or value["gate"] != gate
            or not isinstance(value["quarantine"], Mapping)
            or not isinstance(value["family_exclusions"], Mapping)
        ):
            raise ArtifactValidationError(
                f"invalid family exclusion evidence: {path}"
            )

        def validate_record(record, description: str) -> None:
            if (
                not isinstance(record, Mapping)
                or set(record)
                != {"category", "stage", "reason", "attempts"}
                or not all(
                    isinstance(record[field], str) and record[field]
                    for field in ("category", "stage", "reason")
                )
                or not isinstance(record["attempts"], int)
                or isinstance(record["attempts"], bool)
                or record["attempts"] < 0
            ):
                raise ArtifactValidationError(
                    f"invalid {description}: {path}"
                )

        quarantine = set()
        for asset, record in value["quarantine"].items():
            _sha(asset, "quarantined asset")
            validate_record(record, "global quarantine evidence")
            quarantine.add(asset)
        exclusions = {}
        for asset, records in value["family_exclusions"].items():
            _sha(asset, "family-excluded asset")
            if not isinstance(records, Mapping) or not records:
                raise ArtifactValidationError(
                    f"invalid family exclusion evidence: {path}"
                )
            families = set()
            for family, record in records.items():
                if family not in set(PACK_FAMILIES) - {"common"}:
                    raise ArtifactValidationError(
                        f"invalid family exclusion evidence: {path}"
                    )
                validate_record(record, "family exclusion evidence")
                families.add(family)
            exclusions[asset] = families
        if not (quarantine | set(exclusions)) <= frozen_assets:
            raise ArtifactValidationError(
                f"family exclusion evidence contains an unfrozen asset: {path}"
            )
        return quarantine, exclusions

    def _frozen_scopes(self, gate: str, registry: pd.DataFrame):
        control = self.config.paths.data2_root / "control"
        if gate == "production":
            root = control / "shards"
        else:
            root = control / "qualification" / gate / "shards"
        sources = _directory_names(root, f"{gate} frozen source root")
        if set(sources) != set(self.config.sources):
            raise ArtifactValidationError(
                f"{gate} frozen scopes must cover every training source"
            )
        scopes = []
        assets = {}
        batches = {}
        for source in self.config.sources:
            for shard in _directory_names(root / source, f"{gate} frozen shards"):
                canonical = tuple(
                    sorted(
                        registry.loc[
                            (registry["owner_source"] == source)
                            & (registry["shard_id"] == shard),
                            "sha256",
                        ]
                    )
                )
                if not canonical:
                    raise ArtifactValidationError(
                        f"frozen scope is absent from training registry: {source}/{shard}"
                    )
                scope_root = root / source / shard
                marker_path = scope_root / "batches.json"
                try:
                    marker_payload = _read_regular_bytes_nofollow(marker_path)
                    marker = json.loads(marker_payload)
                except (
                    InfrastructureError,
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                ) as error:
                    raise ArtifactValidationError(
                        f"invalid frozen scope manifest: {marker_path}: {error}"
                    ) from error
                if (
                    not isinstance(marker, Mapping)
                    or set(marker)
                    != {
                        "schema_version",
                        "gate",
                        "source",
                        "shard_id",
                        "config_hash",
                        "canonical_shard_sha256",
                        "scope_sha256",
                        "batches",
                    }
                    or marker["schema_version"] != 2
                    or marker["gate"] != gate
                    or marker["source"] != source
                    or marker["shard_id"] != shard
                    or marker["config_hash"] != self.config.config_hash()
                    or marker["canonical_shard_sha256"]
                    != sha256("".join(f"{item}\n" for item in canonical).encode("ascii")).hexdigest()
                    or not isinstance(marker["batches"], list)
                    or not marker["batches"]
                ):
                    raise ArtifactValidationError(
                        f"invalid frozen scope identity: {marker_path}"
                    )
                flattened = []
                for index, entry in enumerate(marker["batches"]):
                    name = f"batch{index:03d}.txt"
                    if (
                        not isinstance(entry, Mapping)
                        or set(entry) != {"name", "count", "sha256"}
                        or entry["name"] != name
                    ):
                        raise ArtifactValidationError("invalid frozen batch entry")
                    payload = _read_bound_artifact(
                        scope_root / name,
                        entry["sha256"],
                        f"{gate} frozen batch",
                    )
                    try:
                        text = payload.decode("ascii")
                        if not text.endswith("\n"):
                            raise ValueError("missing final newline")
                        batch_assets = tuple(
                            _sha(item, "frozen asset") for item in text.splitlines()
                        )
                    except (UnicodeError, ValueError) as error:
                        raise ArtifactValidationError("invalid frozen batch payload") from error
                    if len(batch_assets) != entry["count"] or tuple(sorted(batch_assets)) != batch_assets:
                        raise ArtifactValidationError("invalid frozen batch identity")
                    batches[(source, shard, name.removesuffix(".txt"))] = batch_assets
                    flattened.extend(batch_assets)
                flattened = tuple(flattened)
                try:
                    descriptor = _open_directory_nofollow(scope_root)
                    try:
                        actual_names = set(os.listdir(descriptor))
                    finally:
                        os.close(descriptor)
                except (InfrastructureError, OSError) as error:
                    raise ArtifactValidationError(
                        f"cannot inspect frozen scope: {scope_root}: {error}"
                    ) from error
                expected_names = {
                    "batches.json",
                    *(entry["name"] for entry in marker["batches"]),
                }
                if actual_names != expected_names:
                    raise ArtifactValidationError(
                        "frozen scope contains unindexed artifacts"
                    )
                scope_digest = sha256(
                    "".join(f"{item}\n" for item in flattened).encode("ascii")
                ).hexdigest()
                if marker["scope_sha256"] != scope_digest:
                    raise ArtifactValidationError("frozen scope checksum mismatch")
                if gate == "production":
                    valid_scope = flattened == canonical
                else:
                    valid_scope = (
                        bool(flattened)
                        and set(flattened).issubset(set(canonical))
                        and flattened
                        == tuple(item for item in canonical if item in set(flattened))
                    )
                if not valid_scope or set(flattened) & set(assets):
                    raise ArtifactValidationError("frozen scope asset identity mismatch")
                for asset in flattened:
                    assets[asset] = (source, shard)
                scopes.append(
                    {
                        "gate": gate,
                        "source": source,
                        "shard_id": shard,
                        "assets": len(flattened),
                        "scope_sha256": scope_digest,
                        "manifest_sha256": sha256(marker_payload).hexdigest(),
                    }
                )
        if gate == "production" and set(assets) != set(registry["sha256"]):
            raise ArtifactValidationError(
                "production scopes do not cover the full training registry"
            )
        return scopes, assets, batches

    def _publications(self, gate: str, batches):
        prepared = self.config.paths.data2_root / "prepared"
        prefix = Path() if gate == "production" else Path("qualification", gate)
        pack_inventory = []
        archive_inventory = []
        grouped = {}
        shard_assets = {}
        for (source, shard, _batch), assets in batches.items():
            shard_assets.setdefault((source, shard), set()).update(assets)
        exclusion_evidence = {}
        expected_batches = {}
        for source, shard, batch in batches:
            expected_batches.setdefault((source, shard), set()).add(batch)
        indexes = {}
        for source, shard, batch in sorted(batches):
            batch_assets = batches[(source, shard, batch)]
            index_path = prepared / prefix / "index" / source / f"{shard}.json"
            index_key = (source, shard)
            if index_key not in indexes:
                try:
                    index_payload = _read_regular_bytes_nofollow(index_path)
                    index = json.loads(index_payload)
                except (
                    InfrastructureError,
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    TypeError,
                ) as error:
                    raise ArtifactValidationError(
                        f"invalid held shard index: {index_path}: {error}"
                    ) from error
                if (
                    not isinstance(index, Mapping)
                    or set(index) != {"gate", "source", "shard_id", "batches"}
                    or index["gate"] != gate
                    or index["source"] != source
                    or index["shard_id"] != shard
                    or not isinstance(index["batches"], Mapping)
                    or set(index["batches"]) != expected_batches[index_key]
                ):
                    raise ArtifactValidationError(
                        "published shard index identity mismatch"
                    )
                indexes[index_key] = index
            entries = indexes[index_key]["batches"][batch]
            if not isinstance(entries, Mapping) or set(entries) != set(PACK_FAMILIES):
                raise ArtifactValidationError("published pack family set mismatch")
            batch_included = {}
            for family in PACK_FAMILIES:
                relative = prefix / _family_root(family) / source / shard / f"{batch}.tar"
                manifest_relative = relative.with_suffix(".tar.manifest.json")
                entry = entries[family]
                if (
                    not isinstance(entry, Mapping)
                    or set(entry)
                    != {"pack", "pack_sha256", "manifest", "manifest_sha256"}
                    or entry["pack"] != relative.as_posix()
                    or entry["manifest"] != manifest_relative.as_posix()
                ):
                    raise ArtifactValidationError("non-canonical pack index path")
                manifest, pack_sha, manifest_sha = _held_pack_manifest(
                    prepared / relative, prepared / manifest_relative
                )
                if (
                    manifest.get("gate") != gate
                    or manifest.get("shard_id") != shard
                    or manifest.get("batch_id") != batch
                    or manifest.get("family") != family
                    or manifest.get("config_hash") != self.config.config_hash()
                    or tuple(manifest.get("asset_sha256s", ())) != batch_assets
                    or not manifest.get("validated_at")
                    or entry.get("pack_sha256") != pack_sha
                    or entry.get("manifest_sha256") != manifest_sha
                ):
                    raise ArtifactValidationError("pack publication identity mismatch")
                included_assets = tuple(
                    manifest.get("included_asset_sha256s", ())
                )
                batch_included[family] = set(included_assets)
                pack_inventory.append(
                    {
                        "source": source,
                        "shard_id": shard,
                        "batch_id": batch,
                        "family": family,
                        "path": (prepared / relative).as_posix(),
                        "pack_sha256": pack_sha,
                        "manifest_sha256": manifest_sha,
                        "included_count": len(included_assets),
                        "excluded_count": len(batch_assets)
                        - len(included_assets),
                    }
                )
                grouped.setdefault((source, shard, batch), set()).add(family)
            validate_family_memberships(batch_included, self.config)
            excluded_by_family = {
                family: set(batch_assets) - batch_included[family]
                for family in PACK_FAMILIES
            }
            if any(excluded_by_family.values()):
                evidence_key = (source, shard)
                if evidence_key not in exclusion_evidence:
                    exclusion_evidence[evidence_key] = (
                        self._quality_exclusion_evidence(
                            gate,
                            source,
                            shard,
                            shard_assets[evidence_key],
                        )
                    )
                quarantine, family_exclusions = exclusion_evidence[
                    evidence_key
                ]
                for family in PACK_FAMILIES:
                    for asset in excluded_by_family[family]:
                        if asset in quarantine:
                            continue
                        if (
                            family != "common"
                            and family
                            in family_exclusions.get(asset, set())
                        ):
                            continue
                        raise ArtifactValidationError(
                            "family exclusion evidence is missing: "
                            f"{source}/{shard}/{batch}/{family}/{asset}"
                        )
                    for asset in batch_included[family]:
                        if asset in quarantine or family in family_exclusions.get(
                            asset, set()
                        ):
                            raise ArtifactValidationError(
                                "family exclusion evidence conflicts with "
                                f"published membership: {family}/{asset}"
                            )

            archive_root = self.config.paths.data3_root / "archive"
            if gate == "production":
                archive_root = archive_root / "raw"
            else:
                archive_root = archive_root / "qualification" / gate / "raw"
            archive = archive_root / source / shard / f"{batch}.tar"
            manifest, pack_sha, manifest_sha = _held_pack_manifest(
                archive, archive.with_suffix(".tar.manifest.json")
            )
            if (
                manifest.get("gate") != gate
                or manifest.get("shard_id") != shard
                or manifest.get("batch_id") != batch
                or manifest.get("family") != "raw"
                or manifest.get("config_hash") != self.config.config_hash()
                or tuple(manifest.get("asset_sha256s", ())) != batch_assets
                or manifest.get("schema_version") != 2
                or tuple(manifest.get("included_asset_sha256s", ()))
                != tuple(
                    asset
                    for asset in batch_assets
                    if asset in batch_included["common"]
                )
                or not manifest.get("validated_at")
            ):
                raise ArtifactValidationError("raw archive publication identity mismatch")
            archive_inventory.append(
                {
                    "source": source,
                    "shard_id": shard,
                    "batch_id": batch,
                    "path": archive.as_posix(),
                    "pack_sha256": pack_sha,
                    "manifest_sha256": manifest_sha,
                }
            )
        if any(value != set(PACK_FAMILIES) for value in grouped.values()):
            raise ArtifactValidationError("pack family coverage is incomplete")
        return pack_inventory, archive_inventory

    @staticmethod
    def _telemetry(payload: bytes, gate: str, scope_pairs: set[tuple[str, str]]):
        records = []
        seen_scopes = set()
        try:
            for line in payload.decode("utf-8").splitlines():
                if not line:
                    continue
                record = json.loads(line)
                if (
                    not isinstance(record, Mapping)
                    or record.get("gate") != gate
                    or (record.get("source"), record.get("shard_id")) not in scope_pairs
                ):
                    raise ValueError("telemetry scope mismatch")
                _held_timestamp(record.get("timestamp"), f"{gate} telemetry")
                seen_scopes.add((record["source"], record["shard_id"]))
                records.append(record)
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise ArtifactValidationError(f"invalid held gate telemetry: {error}") from error
        if not records or seen_scopes != scope_pairs:
            raise ArtifactValidationError(
                "gate telemetry must cover every frozen scope"
            )
        try:
            return resource_peaks(records)
        except ReportValidationError as error:
            raise ArtifactValidationError(f"invalid gate telemetry: {error}") from error

    def _derive_gate(self, gate: str):
        candidate, candidate_sha, payloads = _gate_candidate(self.config, gate)
        training_store = SafeRegistryStore(
            self.config.paths.data2_root / "control/assets.parquet", self.config
        )
        evaluation_store = SafeRegistryStore(
            self.config.paths.data2_root / "control/evaluation_assets.parquet",
            self.config,
            partition="evaluation",
        )
        training = training_store.load()
        evaluation = evaluation_store.load()
        training_manifest = _safe_json(
            training_store.manifest_path, "training registry manifest"
        )
        evaluation_manifest = _safe_json(
            evaluation_store.manifest_path, "evaluation registry manifest"
        )
        scopes, frozen_assets, batches = self._frozen_scopes(gate, training)
        packs, archives = self._publications(gate, batches)
        family_counts = {
            family: {
                "included": sum(
                    pack["included_count"]
                    for pack in packs
                    if pack["family"] == family
                ),
                "excluded": sum(
                    pack["excluded_count"]
                    for pack in packs
                    if pack["family"] == family
                ),
            }
            for family in PACK_FAMILIES
        }

        try:
            measurements = pd.read_csv(
                io.BytesIO(payloads["measurements"]), dtype={"sha256": str}
            )
            fp16 = pd.read_csv(io.BytesIO(payloads["fp16"]), dtype={"sha256": str})
        except (UnicodeError, ValueError, pd.errors.ParserError) as error:
            raise ArtifactValidationError(f"invalid gate measurement CSV: {error}") from error
        if set(measurements.get("sha256", ())) != set(frozen_assets):
            raise ArtifactValidationError("gate measurements do not match frozen scope")
        for record in measurements.to_dict("records"):
            if frozen_assets.get(record["sha256"]) != (
                record["source"],
                record["shard_id"],
            ):
                raise ArtifactValidationError("gate measurement scope identity mismatch")
        if not set(fp16.get("sha256", ())).issubset(set(frozen_assets)):
            raise ArtifactValidationError("FP16 evidence contains an unfrozen asset")

        hardware = read_hardware_report(self.config)
        hardware_path = self.config.paths.data2_root / "control/reports/hardware.json"
        hardware_payload = _read_regular_bytes_nofollow(hardware_path)
        local_limit = (
            hardware["storage"]["local"]["free_bytes_after"]
            - hardware["storage"]["local"]["free_floor_bytes"]
        )
        if local_limit <= 0:
            raise ArtifactValidationError("hardware local capacity is exhausted")
        try:
            summary = gate_measurement_summary(
                measurements,
                total_assets=len(training),
                local_limit_bytes=local_limit,
                data2_limit_bytes=self.config.limits.data2_soft_tib * 1024**4,
                data3_limit_bytes=self.config.limits.data3_soft_tib * 1024**4,
            )
            parity = fp16_gate_summary(
                fp16, self.config.targets.latent_dtype
            )
            combined = pd.concat(
                (
                    training[["sha256", "split"]],
                    evaluation[["sha256", "split"]],
                ),
                ignore_index=True,
            )
            overlap = split_overlap(combined)
        except ReportValidationError as error:
            raise ArtifactValidationError(f"invalid derived gate evidence: {error}") from error
        if set(summary["source_counts"]) != set(self.config.sources):
            raise ArtifactValidationError("gate measurements must cover every training source")
        peaks = self._telemetry(
            payloads["telemetry"],
            gate,
            {(scope["source"], scope["shard_id"]) for scope in scopes},
        )
        capacity_passed = all(
            value["within_limit"]
            for value in summary["capacity"]["projections"].values()
        )
        if gate == "smoke":
            quality_passed = (
                summary["quality"]["failure_rate"] <= 0.10
                and summary["quality"]["schema_failures"] == 0
                and all(
                    value["failure_rate"] <= 0.10
                    and value["schema_failures"] == 0
                    for value in summary["quality"]["sources"].values()
                )
            )
        else:
            quality_passed = (
                summary["quality"]["failure_rate"] <= 0.10
                and summary["quality"]["schema_failure_rate"] <= 0.05
            )
            quality_passed = quality_passed and all(
                value["passed"]
                for value in summary["quality"]["sources"].values()
            )
        parity_passed = parity["passed"]
        passed = all(
            (
                hardware["decision"] == "passed",
                capacity_passed,
                quality_passed,
                parity_passed,
                overlap["passed"],
                bool(packs),
                bool(archives),
            )
        )

        handoff = None
        if gate == "production" and passed:
            handoff = build_training_handoff(
                config_hash=self.config.config_hash(),
                registry_checksum=training_manifest["sha256"],
                evaluation_registry_checksum=evaluation_manifest["sha256"],
                frozen_scopes=scopes,
                packs=packs,
                archives=archives,
                train_ids=sorted(training.loc[training["split"] == "train", "sha256"]),
                validation_ids=sorted(training.loc[training["split"] == "validation", "sha256"]),
                evaluation_ids=sorted(evaluation["sha256"]),
                created_at=candidate["created_at"],
                family_counts=family_counts,
            )
        handoff_payload = (
            json.dumps(handoff, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if handoff is not None
            else None
        )
        handoff_path = self.config.paths.data2_root / "control/splits/training_handoff.json"
        report = {
            "schema_version": 2,
            "report_type": "pipeline_gate_derived",
            "gate": gate,
            "decision": "passed" if passed else "failed",
            "config_hash": self.config.config_hash(),
            "created_at": candidate["created_at"],
            "evidence": {
                "manifest_sha256": candidate_sha,
                "training_registry_sha256": training_manifest["sha256"],
                "evaluation_registry_sha256": evaluation_manifest["sha256"],
                "hardware_sha256": sha256(hardware_payload).hexdigest(),
                "measurements_sha256": sha256(payloads["measurements"]).hexdigest(),
                "fp16_sha256": sha256(payloads["fp16"]).hexdigest(),
                "telemetry_sha256": sha256(payloads["telemetry"]).hexdigest(),
                "frozen_scopes": scopes,
                "packs": packs,
                "archives": archives,
            },
            **summary,
            "resource_peaks": peaks,
            "fp16_parity": parity,
            "split_overlap": overlap,
            "checksums": {
                "algorithm": "sha256",
                "verified": 3 + len(scopes) + len(packs) + len(archives),
                "failed": 0,
            },
            "hardware": {
                "report_sha256": sha256(hardware_payload).hexdigest(),
                "gpu_count": hardware["gpu"]["gpu_count"],
                "checked": hardware["decision"] == "passed",
            },
            "audits": {
                "registry": True,
                "frozen_scopes": True,
                "pack_verification": True,
                "raw_archive_verification": True,
                "split_overlap": overlap["passed"],
            },
            "handoff": {
                "ready": handoff is not None,
                "path": handoff_path.as_posix() if handoff is not None else "",
                "sha256": sha256(handoff_payload).hexdigest() if handoff_payload is not None else "",
                "pack_families": len(PACK_FAMILIES),
                "stage_extractable": bool(packs),
                "family_counts": family_counts,
            },
        }
        return report, handoff, handoff_payload

    def _revoke_production_publication(self) -> None:
        root = self.config.paths.data2_root / "control"
        for path in (
            root / "splits/training_handoff.json",
            root / "reports/gates/production.json",
            root / "reports/gates/production.md",
        ):
            _unlink_regular_nofollow(path, missing_ok=True)

    def __call__(self, gate: str | None, hardware_check: bool = False):
        if gate is None:
            if not hardware_check:
                raise ArtifactValidationError(
                    "report requires a gate or hardware check"
                )
            input_path = (
                self.config.paths.data2_root
                / "control/report_inputs/hardware.json"
            )
            try:
                evidence_payload = _read_regular_bytes_nofollow(input_path)
                evidence = json.loads(evidence_payload)
            except (
                InfrastructureError,
                OSError,
                UnicodeError,
                json.JSONDecodeError,
            ) as error:
                raise ArtifactValidationError(
                    f"invalid hardware report input: {error}"
                ) from error
            report = derive_hardware_report(
                evidence,
                self.config,
                sha256(evidence_payload).hexdigest(),
            )
            return write_report(
                self.config.paths.data2_root / "control/reports",
                "hardware",
                report,
            )
        if gate not in {"smoke", "pilot", "production"}:
            raise ArtifactValidationError("report requires a known gate")
        if gate == "production":
            self._revoke_production_publication()
        report, handoff, handoff_payload = self._derive_gate(gate)
        if hardware_check and not report["hardware"]["checked"]:
            raise ArtifactValidationError("hardware check did not pass")
        try:
            published = write_report(
                self.config.paths.data2_root / "control/reports/gates",
                gate,
                report,
            )
        except BaseException as error:
            if gate == "production":
                try:
                    self._revoke_production_publication()
                except BaseException as revoke_error:
                    raise error from revoke_error
            raise
        if handoff is not None:
            try:
                _atomic_write_bytes_nofollow(
                    self.config.paths.data2_root
                    / "control/splits/training_handoff.json",
                    handoff_payload,
                )
            except BaseException as error:
                try:
                    self._revoke_production_publication()
                except BaseException as revoke_error:
                    raise error from revoke_error
                raise
        return published


def build_read_only_services(config: PipelineConfig) -> PipelineServices:
    return PipelineServices(
        config,
        registry_store=SafeRegistryStore(
            config.paths.data2_root / "control/assets.parquet", config
        ),
        pilot_reader=PilotArtifactReader(config),
    )


def _ensure_runtime_roots(config: PipelineConfig) -> None:
    for root in (
        config.paths.data2_root,
        config.paths.data3_root,
        config.paths.local_root,
    ):
        try:
            descriptor = _open_directory_nofollow(root, create=True)
            os.close(descriptor)
        except (InfrastructureError, OSError) as error:
            raise ArtifactValidationError(
                f"cannot initialize runtime root {root}: {error}"
            ) from error


class MutatingRuntime:
    """Lazily constructs production providers at the first mutating call."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._services: PipelineServices | None = None
        self._telemetry: NoFollowTelemetryWriter | None = None

    def _initialize(self) -> PipelineServices:
        if self._services is not None:
            return self._services
        _ensure_runtime_roots(self.config)
        accounting = initialize_project_accounting(self.config)
        sampler = ResourceSampler(self.config, accounting)
        telemetry = NoFollowTelemetryWriter(
            self.config.paths.data2_root / "control/telemetry/resources.jsonl"
        )
        self._telemetry = telemetry
        try:
            guard = ResourceGuard(
                sampler,
                ResourcePolicy(self.config.limits),
                telemetry,
                time.monotonic,
                time.sleep,
            )
            registry = SafeRegistryStore(
                self.config.paths.data2_root / "control/assets.parquet", self.config
            )
            self._services = PipelineServices(
                self.config,
                resource_guard=guard,
                pilot_reader=PilotArtifactReader(self.config),
                reference_counter=FrozenReferenceCounter(self.config),
                project_accounting=accounting,
                registry_store=registry,
                registry_builder=CanonicalRegistryBuilder(
                    self.config, store=registry
                ),
                report_builder=RuntimeReportBuilder(self.config),
            )
        except BaseException as error:
            try:
                self.close()
            except BaseException as close_error:
                raise error from close_error
            raise
        return self._services

    @property
    def services(self) -> PipelineServices:
        return self._initialize()

    def invoke(self, method: str, *arguments):
        service = self._initialize()
        return getattr(service, method)(*arguments)

    def close(self) -> None:
        if self._telemetry is not None:
            telemetry = self._telemetry
            self._telemetry = None
            telemetry.close()

    def __enter__(self) -> "MutatingRuntime":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if exc_value is None:
                raise
            raise exc_value.with_traceback(traceback) from close_error


def build_mutating_services(config: PipelineConfig) -> MutatingRuntime:
    return MutatingRuntime(config)
