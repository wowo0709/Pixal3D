from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

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


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_version: str
    sources: tuple[str, ...]
    evaluation_sources: tuple[str, ...]
    shard_size: int
    paths: PathConfig
    render: RenderConfig
    targets: TargetConfig
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
    "workers",
    "limits",
}


def load_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text())
    unknown = set(raw) - ROOT_KEYS
    if unknown:
        raise ValueError(f"unknown_key: {sorted(unknown)}")
    return PipelineConfig(
        pipeline_version=raw["pipeline_version"],
        sources=tuple(raw["sources"]),
        evaluation_sources=tuple(raw["evaluation_sources"]),
        shard_size=int(raw["shard_size"]),
        paths=PathConfig(**{key: Path(value) for key, value in raw["paths"].items()}),
        render=RenderConfig(**raw["render"]),
        targets=TargetConfig(
            tuple(raw["targets"]["views"]),
            tuple(raw["targets"]["resolutions"]),
            raw["targets"]["ss_resolution"],
            raw["targets"]["latent_dtype"],
        ),
        workers=WorkerConfig(**raw["workers"]),
        limits=LimitConfig(**raw["limits"]),
    )
