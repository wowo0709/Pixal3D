from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping

import yaml


@dataclass(frozen=True)
class PathConfig:
    data2_root: Path
    data3_root: Path
    local_root: Path


@dataclass(frozen=True)
class RenderConfig:
    num_views: int
    resolution: int
    fov_min_degrees: float
    fov_max_degrees: float
    camera_policy: str
    blender_version: str
    cycles_device: str


@dataclass(frozen=True)
class TargetConfig:
    views: tuple[int, ...]
    resolutions: tuple[int, ...]
    ss_resolution: int
    latent_dtype: str


@dataclass(frozen=True)
class BatchConfig:
    smoke_max_assets: int
    pilot_max_assets: int
    production_max_assets: int


@dataclass(frozen=True)
class WorkerConfig:
    cpu_threads: int
    dump_workers: int
    voxel_workers: int
    voxel_threads_per_worker: int
    render_workers: int
    encoder_ranks: int
    encoder_loader_threads: int
    encoder_saver_threads: int


@dataclass(frozen=True)
class LimitConfig:
    cpu_soft_percent: float
    cpu_hard_percent: float
    load_soft: float
    io_wait_soft_percent: float
    ram_soft_available_gib: int
    ram_hard_available_gib: int
    local_free_percent: float
    local_free_gib: int
    data2_soft_tib: int
    data2_hard_tib: int
    data2_fs_free_tib: int
    data3_soft_tib: int
    data3_fs_free_tib: int
    swap_soft_mib_per_minute: int
    swap_soft_samples: int
    cpu_temp_soft_celsius: int
    cpu_temp_hard_celsius: int
    gpu_temp_soft_celsius: int
    gpu_temp_hard_celsius: int
    temperature_soft_seconds: int
    recovery_stable_seconds: int


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_version: str
    sources: tuple[str, ...]
    evaluation_sources: tuple[str, ...]
    shard_size: int
    paths: PathConfig
    render: RenderConfig
    targets: TargetConfig
    batching: BatchConfig
    workers: WorkerConfig
    limits: LimitConfig

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return sha256(payload).hexdigest()


ROOT_KEYS = {
    "pipeline_version",
    "sources",
    "evaluation_sources",
    "shard_size",
    "paths",
    "render",
    "targets",
    "batching",
    "workers",
    "limits",
}

SECTION_KEYS = {
    "paths": {"data2_root", "data3_root", "local_root"},
    "render": {
        "num_views",
        "resolution",
        "fov_min_degrees",
        "fov_max_degrees",
        "camera_policy",
        "blender_version",
        "cycles_device",
    },
    "targets": {"views", "resolutions", "ss_resolution", "latent_dtype"},
    "batching": {"smoke_max_assets", "pilot_max_assets", "production_max_assets"},
    "workers": {
        "cpu_threads",
        "dump_workers",
        "voxel_workers",
        "voxel_threads_per_worker",
        "render_workers",
        "encoder_ranks",
        "encoder_loader_threads",
        "encoder_saver_threads",
    },
    "limits": {
        "cpu_soft_percent",
        "cpu_hard_percent",
        "load_soft",
        "io_wait_soft_percent",
        "ram_soft_available_gib",
        "ram_hard_available_gib",
        "local_free_percent",
        "local_free_gib",
        "data2_soft_tib",
        "data2_hard_tib",
        "data2_fs_free_tib",
        "data3_soft_tib",
        "data3_fs_free_tib",
        "swap_soft_mib_per_minute",
        "swap_soft_samples",
        "cpu_temp_soft_celsius",
        "cpu_temp_hard_celsius",
        "gpu_temp_soft_celsius",
        "gpu_temp_hard_celsius",
        "temperature_soft_seconds",
        "recovery_stable_seconds",
    },
}


def _read_config(path: Path) -> bytes:
    absolute = Path(os.path.abspath(path))
    directory_flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = None
    try:
        directory_fd = os.open("/", directory_flags)
        for component in absolute.parts[1:-1]:
            next_fd = os.open(
                component, directory_flags, dir_fd=directory_fd
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | os.O_CLOEXEC
            | os.O_NONBLOCK
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise ValueError(f"unsafe config: {path}: {error}") from error
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError(f"unsafe config: {path}: not a regular file")
        if details.st_size > 1024 * 1024:
            raise ValueError(f"unsafe config: {path}: file is too large")
        chunks = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65536))
            if not chunk:
                raise ValueError(f"unsafe config: {path}: truncated read")
            chunks.append(chunk)
            remaining -= len(chunk)
        final_details = os.fstat(descriptor)
        if (
            final_details.st_size != details.st_size
            or final_details.st_mtime_ns != details.st_mtime_ns
            or final_details.st_ctime_ns != details.st_ctime_ns
        ):
            raise ValueError(f"unsafe config: {path}: changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _mapping(value, keys: set[str], description: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"invalid {description} schema")
    return value


def _string(value, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a nonempty string")
    return value


def _positive_int(value, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _number(
    value, description: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{description} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{description} is above its maximum")
    return result


def _sources(value, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{description} sources must be a nonempty list")
    result = []
    for source in value:
        source = _string(source, f"{description} source")
        if (
            source in {".", ".."}
            or Path(source).name != source
            or "\\" in source
            or "\0" in source
        ):
            raise ValueError(f"unsafe {description} source: {source!r}")
        result.append(source)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate {description} source")
    return tuple(result)


def _paths(value: Mapping) -> PathConfig:
    value = _mapping(value, SECTION_KEYS["paths"], "paths")
    roots = []
    for name in ("data2_root", "data3_root", "local_root"):
        raw = _string(value[name], f"paths {name}")
        if "\0" in raw:
            raise ValueError(f"root contains a null byte: {name}")
        path = Path(os.path.normpath(raw))
        if not path.is_absolute():
            raise ValueError(f"root must be absolute: {name}")
        roots.append(path)
    if len(set(roots)) != len(roots):
        raise ValueError("roots must be distinct")
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError("roots must not be nested")
    return PathConfig(*roots)


def _render(value: Mapping) -> RenderConfig:
    value = _mapping(value, SECTION_KEYS["render"], "render")
    num_views = _positive_int(value["num_views"], "render num_views")
    if num_views != 8:
        raise ValueError("render num_views must be exactly 8")
    resolution = _positive_int(value["resolution"], "render resolution")
    if resolution != 512:
        raise ValueError("render resolution must be exactly 512")
    minimum = _number(value["fov_min_degrees"], "render fov_min_degrees", minimum=0.01, maximum=179.0)
    maximum = _number(value["fov_max_degrees"], "render fov_max_degrees", minimum=0.01, maximum=179.0)
    if minimum >= maximum:
        raise ValueError("render FOV minimum must be less than maximum")
    camera_policy = _string(value["camera_policy"], "render camera_policy")
    blender_version = _string(value["blender_version"], "render blender_version")
    cycles_device = _string(value["cycles_device"], "render cycles_device")
    if cycles_device != "OPTIX":
        raise ValueError("render cycles_device must be OPTIX")
    return RenderConfig(
        num_views,
        resolution,
        minimum,
        maximum,
        camera_policy,
        blender_version,
        cycles_device,
    )


def _targets(value: Mapping) -> TargetConfig:
    value = _mapping(value, SECTION_KEYS["targets"], "targets")
    if (
        value["views"] != [0, 1]
        or any(type(item) is not int for item in value["views"])
    ):
        raise ValueError("targets views must be exactly [0, 1]")
    if value["resolutions"] != [256, 512, 1024]:
        raise ValueError("targets resolutions must be exactly [256, 512, 1024]")
    if value["ss_resolution"] != 64 or isinstance(value["ss_resolution"], bool):
        raise ValueError("targets ss_resolution must be exactly 64")
    latent_dtype = _string(value["latent_dtype"], "targets latent_dtype")
    if latent_dtype not in {"float16", "float32"}:
        raise ValueError("targets latent_dtype must be float16 or float32")
    return TargetConfig((0, 1), (256, 512, 1024), 64, latent_dtype)


def _batching(value: Mapping, shard_size: int) -> BatchConfig:
    value = _mapping(value, SECTION_KEYS["batching"], "batching")
    values = {
        name: _positive_int(value[name], f"batching {name}")
        for name in SECTION_KEYS["batching"]
    }
    if not (
        values["smoke_max_assets"]
        <= values["pilot_max_assets"]
        <= values["production_max_assets"]
        <= shard_size
    ):
        raise ValueError("batching caps must be ordered and fit shard_size")
    return BatchConfig(**values)


def _workers(value: Mapping) -> WorkerConfig:
    value = _mapping(value, SECTION_KEYS["workers"], "workers")
    return WorkerConfig(
        **{
            name: _positive_int(value[name], f"workers {name}")
            for name in SECTION_KEYS["workers"]
        }
    )


def _limits(value: Mapping) -> LimitConfig:
    value = _mapping(value, SECTION_KEYS["limits"], "limits")
    float_values = {
        "cpu_soft_percent": _number(value["cpu_soft_percent"], "limits cpu_soft_percent", minimum=0.01, maximum=100.0),
        "cpu_hard_percent": _number(value["cpu_hard_percent"], "limits cpu_hard_percent", minimum=0.01, maximum=100.0),
        "load_soft": _number(value["load_soft"], "limits load_soft", minimum=0.01),
        "io_wait_soft_percent": _number(value["io_wait_soft_percent"], "limits io_wait_soft_percent", minimum=0.0, maximum=100.0),
        "local_free_percent": _number(value["local_free_percent"], "limits local_free_percent", minimum=0.01, maximum=100.0),
    }
    integer_names = SECTION_KEYS["limits"] - set(float_values)
    integers = {
        name: _positive_int(value[name], f"limits {name}")
        for name in integer_names
    }
    if float_values["cpu_soft_percent"] >= float_values["cpu_hard_percent"]:
        raise ValueError("limits CPU soft threshold must be below hard threshold")
    if integers["ram_soft_available_gib"] <= integers["ram_hard_available_gib"]:
        raise ValueError("limits RAM soft threshold must exceed hard threshold")
    if integers["data2_soft_tib"] >= integers["data2_hard_tib"]:
        raise ValueError("limits data2 soft threshold must be below hard threshold")
    if integers["cpu_temp_soft_celsius"] >= integers["cpu_temp_hard_celsius"]:
        raise ValueError("limits CPU temperature soft threshold must be below hard threshold")
    if integers["gpu_temp_soft_celsius"] >= integers["gpu_temp_hard_celsius"]:
        raise ValueError("limits GPU temperature soft threshold must be below hard threshold")
    return LimitConfig(**float_values, **integers)


def load_config(path: Path) -> PipelineConfig:
    try:
        raw = yaml.safe_load(_read_config(Path(path)))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid config syntax: {path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("invalid root config schema")
    unknown = set(raw) - ROOT_KEYS
    if unknown:
        raise ValueError(f"unknown_key: {sorted(unknown)}")
    missing = ROOT_KEYS - set(raw)
    if missing:
        raise ValueError(f"missing root config keys: {sorted(missing)}")
    sources = _sources(raw["sources"], "training")
    evaluation_sources = _sources(raw["evaluation_sources"], "evaluation")
    if set(sources) & set(evaluation_sources):
        raise ValueError("training and evaluation sources must be disjoint")
    return PipelineConfig(
        pipeline_version=_string(raw["pipeline_version"], "pipeline_version"),
        sources=sources,
        evaluation_sources=evaluation_sources,
        shard_size=_positive_int(raw["shard_size"], "shard_size"),
        paths=_paths(raw["paths"]),
        render=_render(raw["render"]),
        targets=_targets(raw["targets"]),
        batching=_batching(raw["batching"], _positive_int(raw["shard_size"], "shard_size")),
        workers=_workers(raw["workers"]),
        limits=_limits(raw["limits"]),
    )
