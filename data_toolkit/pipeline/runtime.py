from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import time
from typing import Callable, Mapping

import pandas as pd

from .config import PipelineConfig
from .orchestrator import (
    InfrastructureError,
    IntegrationProviderRequired,
    PipelineServices,
    _atomic_write_bytes_nofollow,
    _open_directory_nofollow,
    _read_regular_bytes_nofollow,
    _regular_file_stat_nofollow,
)
from .packing import verify_pack
from .registry import assign_shards, canonicalize_sources
from .reporting import (
    ReportValidationError,
    validate_gate_report,
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
REGISTRY_ARTIFACT_TYPE = "canonical_registry"
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


def _safe_csv(path: Path, description: str) -> pd.DataFrame:
    try:
        payload = _read_regular_bytes_nofollow(path)
        return pd.read_csv(io.BytesIO(payload), dtype={"sha256": str})
    except (InfrastructureError, OSError) as error:
        raise ArtifactValidationError(
            f"missing or unsafe {description}: {path}: {error}"
        ) from error
    except Exception as error:
        raise ArtifactValidationError(
            f"corrupt {description}: {path}: {error}"
        ) from error


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

    def __init__(self, path: Path, config: PipelineConfig):
        self.path = Path(path)
        self.config = config
        self.manifest_path = self.path.with_suffix(
            self.path.suffix + ".manifest.json"
        )

    def _manifest(self, payload: bytes, rows: int) -> dict:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": REGISTRY_ARTIFACT_TYPE,
            "config_hash": self.config.config_hash(),
            "rows": rows,
            "sha256": sha256(payload).hexdigest(),
        }

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
        configured = {*self.config.sources, *self.config.evaluation_sources}
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
            "config_hash",
            "rows",
            "sha256",
        }:
            raise ArtifactValidationError("invalid registry manifest schema")
        if manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError("unsupported registry schema")
        if manifest["artifact_type"] != REGISTRY_ARTIFACT_TYPE:
            raise ArtifactValidationError("invalid registry artifact type")
        if manifest["config_hash"] != self.config.config_hash():
            raise ArtifactValidationError("registry config hash mismatch")
        rows = _nonnegative_count(manifest["rows"], "registry rows", positive=True)
        digest = _sha(manifest["sha256"], "registry checksum")
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
        except Exception as error:
            raise ArtifactValidationError(f"corrupt registry: {error}") from error
        if len(frame) != rows or frame.empty:
            raise ArtifactValidationError("registry row count mismatch")
        self._validate_frame(frame)
        return frame

    def save(self, frame: pd.DataFrame) -> None:
        self._validate_frame(frame)
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
        except Exception as error:
            raise ArtifactValidationError(
                f"cannot serialize registry: {error}"
            ) from error
        _write_json(
            self.manifest_path,
            self._manifest(payload, len(frame)),
            "registry manifest",
        )

    def update_state(self, asset_sha256, field, state, error="") -> None:
        frame = self.load()
        selected = frame["sha256"] == asset_sha256
        if selected.sum() != 1:
            raise KeyError(asset_sha256)
        frame.loc[selected, field] = state.value
        if error:
            frame.loc[selected, "last_error"] = error
        self.save(frame)


class CanonicalRegistryBuilder:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        source_loader: Callable[[str], pd.DataFrame] | None = None,
        store: SafeRegistryStore | None = None,
    ):
        self.config = config
        self.source_order = (*config.sources, *config.evaluation_sources)
        if not self.source_order or len(self.source_order) != len(
            set(self.source_order)
        ):
            raise ArtifactValidationError("configured source order is invalid")
        self.source_loader = source_loader or self._load_local_source
        self.store = store or SafeRegistryStore(
            config.paths.data2_root / "control/assets.parquet", config
        )

    def _metadata_path(self, source: str) -> Path:
        return (
            self.config.paths.data2_root
            / "control/metadata"
            / source
            / "metadata.csv"
        )

    def _load_local_source(self, source: str) -> pd.DataFrame:
        return _safe_csv(
            self._metadata_path(source), f"canonical {source} metadata"
        )

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
        frames = {}
        for source in self.source_order:
            _component(source, "source")
            frames[source] = self._validate_source(
                source, self.source_loader(source)
            )
        try:
            canonical = canonicalize_sources(
                frames, self.config.render.camera_policy, self.source_order
            )
            canonical = assign_shards(canonical, self.config.shard_size)
        except Exception as error:
            raise ArtifactValidationError(
                f"cannot canonicalize registry: {error}"
            ) from error
        if canonical.empty:
            raise ArtifactValidationError("canonical registry must not be empty")
        self.store.save(canonical)
        for source in self.source_order:
            selected = canonical.loc[
                canonical["owner_source"] == source
            ].sort_values("sha256")
            _write_csv(
                self._metadata_path(source),
                selected,
                f"{source} compatibility metadata",
            )
        return canonical


def read_gate_report(
    config: PipelineConfig, gate: str, *, require_passed: bool = True
) -> dict:
    _component(gate, "gate")
    path = (
        config.paths.data2_root / "control/reports/gates" / f"{gate}.json"
    )
    value = _safe_json(path, f"{gate} gate report")
    try:
        validated = validate_gate_report(
            value, expected_config_hash=config.config_hash()
        )
    except ReportValidationError as error:
        raise ArtifactValidationError(
            f"invalid {gate} gate report: {error}"
        ) from error
    if validated["gate"] != gate:
        raise ArtifactValidationError(f"gate report identity mismatch: {gate}")
    if require_passed and validated["decision"] != "passed":
        raise ArtifactValidationError(f"{gate} gate has not passed")
    return validated


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


def validate_hardware_report(value: Mapping, config: PipelineConfig) -> dict:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "artifact_type",
        "config_hash",
        "decision",
        "gpu",
        "storage",
        "pilot_sizing",
    }:
        raise ArtifactValidationError("invalid hardware preflight schema")
    if (
        value["schema_version"] != ARTIFACT_SCHEMA_VERSION
        or value["artifact_type"] != HARDWARE_ARTIFACT_TYPE
    ):
        raise ArtifactValidationError("unsupported hardware preflight schema")
    if value["config_hash"] != config.config_hash():
        raise ArtifactValidationError("hardware preflight config hash mismatch")
    if value["decision"] not in {"passed", "failed"}:
        raise ArtifactValidationError("invalid hardware preflight decision")

    gpu = value["gpu"]
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "checked",
        "cycles_device",
        "gpu_count",
        "device_names",
        "cpu_fallback_detected",
        "cube_render_sha256",
    }:
        raise ArtifactValidationError("invalid hardware GPU schema")
    if not isinstance(gpu["checked"], bool):
        raise ArtifactValidationError("hardware checked must be boolean")
    gpu_count = _nonnegative_count(gpu["gpu_count"], "hardware GPU count")
    device_names = gpu["device_names"]
    if (
        not isinstance(device_names, list)
        or len(device_names) != gpu_count
        or len(device_names) != len(set(device_names))
        or not all(isinstance(name, str) and name for name in device_names)
    ):
        raise ArtifactValidationError("invalid hardware GPU device names")
    if not isinstance(gpu["cpu_fallback_detected"], bool):
        raise ArtifactValidationError("CPU fallback flag must be boolean")
    _sha(gpu["cube_render_sha256"], "cube render checksum")

    storage = value["storage"]
    if not isinstance(storage, Mapping) or set(storage) != {
        "local",
        "data2",
        "data3",
    }:
        raise ArtifactValidationError("invalid storage preflight schema")
    for root_name, result in storage.items():
        if not isinstance(result, Mapping) or set(result) != {
            "read_mib_per_second",
            "write_mib_per_second",
            "free_bytes_before",
            "free_bytes_after",
            "fixture_removed",
        }:
            raise ArtifactValidationError(
                f"invalid {root_name} storage result schema"
            )
        _finite(
            result["read_mib_per_second"],
            f"{root_name} read throughput",
            positive=True,
        )
        _finite(
            result["write_mib_per_second"],
            f"{root_name} write throughput",
            positive=True,
        )
        _nonnegative_count(
            result["free_bytes_before"], f"{root_name} free bytes before"
        )
        _nonnegative_count(
            result["free_bytes_after"], f"{root_name} free bytes after"
        )
        if not isinstance(result["fixture_removed"], bool):
            raise ArtifactValidationError(
                f"{root_name} fixture removal must be boolean"
            )

    sizing = value["pilot_sizing"]
    if not isinstance(sizing, Mapping) or set(sizing) != {"sources"}:
        raise ArtifactValidationError("invalid pilot sizing schema")
    sources = sizing["sources"]
    if not isinstance(sources, Mapping) or not sources:
        raise ArtifactValidationError("pilot sizing sources must not be empty")
    for source, result in sources.items():
        _component(source, "pilot sizing source")
        if not isinstance(result, Mapping) or set(result) != {
            "p95_peak_local_bytes"
        }:
            raise ArtifactValidationError("invalid pilot sizing source schema")
        _nonnegative_count(
            result["p95_peak_local_bytes"],
            f"{source} pilot sizing p95",
            positive=True,
        )

    expected_pass = all(
        (
            gpu["checked"],
            gpu["cycles_device"] == "OPTIX",
            gpu_count >= max(
                config.workers.render_workers, config.workers.encoder_ranks
            ),
            not gpu["cpu_fallback_detected"],
            all(result["fixture_removed"] for result in storage.values()),
        )
    )
    if (value["decision"] == "passed") != expected_pass:
        raise ArtifactValidationError(
            "hardware preflight decision contradicts validated checks"
        )
    return dict(value)


def read_hardware_report(config: PipelineConfig, *, require_passed=True) -> dict:
    path = config.paths.data2_root / "control/reports/hardware.json"
    report = validate_hardware_report(
        _safe_json(path, "hardware preflight report"), config
    )
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
            try:
                report = validate_gate_report(
                    pilot_value,
                    expected_config_hash=self.config.config_hash(),
                )
            except ReportValidationError as error:
                raise ArtifactValidationError(
                    f"invalid pilot gate report: {error}"
                ) from error
            if report["gate"] != "pilot" or report["decision"] != "passed":
                raise ArtifactValidationError("pilot gate has not passed")
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


class FrozenReferenceCounter:
    """Counts references from immutable batch identity and verified archives."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _metadata(self, source: str) -> dict[str, str]:
        path = (
            self.config.paths.data2_root
            / "raw"
            / source
            / "raw/metadata.csv"
        )
        frame = _safe_csv(path, f"{source} raw metadata")
        missing = {"sha256", "local_path"} - set(frame.columns)
        if missing or frame.empty:
            raise ArtifactValidationError(
                f"invalid canonical raw metadata: missing {sorted(missing)}"
            )
        result = {}
        for record in frame.to_dict("records"):
            asset = _sha(record["sha256"], "raw metadata SHA-256")
            if asset in result:
                raise ArtifactValidationError("duplicate raw metadata SHA-256")
            result[asset] = _raw_reference_path(record["local_path"])
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
            != {"schema_version", "source", "shard_id", "config_hash", "batches"}
            or marker["schema_version"] != ARTIFACT_SCHEMA_VERSION
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
        return tuple(actual)

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
            archive_stat = _regular_file_stat_nofollow(
                archive, missing_ok=True
            )
            manifest_stat = _regular_file_stat_nofollow(
                manifest_path, missing_ok=True
            )
        except OSError as error:
            raise ArtifactValidationError(
                f"unsafe raw archive: {archive}: {error}"
            ) from error
        if archive_stat is None and manifest_stat is None:
            return False
        if (archive_stat is None) != (manifest_stat is None):
            raise ArtifactValidationError(
                f"incomplete raw archive publication: {archive}"
            )
        try:
            verify_pack(archive, manifest_path)
            manifest = _safe_json(manifest_path, "raw archive manifest")
        except Exception as error:
            if isinstance(error, ArtifactValidationError):
                raise
            raise ArtifactValidationError(
                f"corrupt raw archive: {archive}: {error}"
            ) from error
        if (
            manifest.get("family") != "raw"
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
    ) -> int:
        _component(source, "source")
        _component(excluding_shard_id, "excluded shard")
        _component(excluding_batch_id, "excluded batch")
        requested = _raw_reference_path(raw_relative_path)
        metadata = self._metadata(source)
        pending = 0
        for root in self._shard_directories(source):
            for shard, batch, assets in self._frozen_batches(source, root):
                unknown = set(assets) - set(metadata)
                if unknown:
                    raise ArtifactValidationError(
                        "frozen assets are missing from canonical raw metadata"
                    )
                references = tuple(
                    asset for asset in assets if metadata[asset] == requested
                )
                if not references:
                    continue
                if shard == excluding_shard_id and batch == excluding_batch_id:
                    continue
                if not self._archive_verified(source, shard, batch, assets):
                    pending += len(references)
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


class RuntimeReportBuilder:
    """Validates a machine-produced candidate before gate publication."""

    def __init__(self, config: PipelineConfig):
        self.config = config

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
            report = validate_hardware_report(
                _safe_json(input_path, "hardware report input"), self.config
            )
            return write_report(
                self.config.paths.data2_root / "control/reports",
                "hardware",
                report,
            )
        if gate not in {"smoke", "pilot", "production"}:
            raise ArtifactValidationError("report requires a known gate")
        input_path = (
            self.config.paths.data2_root
            / "control/report_inputs"
            / f"{gate}.json"
        )
        candidate = _safe_json(input_path, f"{gate} report input")
        try:
            report = validate_gate_report(
                candidate, expected_config_hash=self.config.config_hash()
            )
        except ReportValidationError as error:
            raise ArtifactValidationError(
                f"invalid {gate} report input: {error}"
            ) from error
        if report["gate"] != gate:
            raise ArtifactValidationError("report input gate mismatch")
        if hardware_check and not report["hardware"]["checked"]:
            raise ArtifactValidationError("hardware check did not pass")
        return write_report(
            self.config.paths.data2_root / "control/reports/gates",
            gate,
            report,
        )


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
        descriptor = _open_directory_nofollow(root, create=True)
        os.close(descriptor)


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
        self._telemetry = telemetry
        return self._services

    @property
    def services(self) -> PipelineServices:
        return self._initialize()

    def invoke(self, method: str, *arguments):
        service = self._initialize()
        return getattr(service, method)(*arguments)

    def close(self) -> None:
        if self._telemetry is not None:
            self._telemetry.close()
            self._telemetry = None

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
