from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from .config import PipelineConfig
from .parallelism import geometry_profile


@dataclass(frozen=True)
class WorkerProfile:
    dump_workers: int
    voxel_workers: int
    voxel_threads_per_worker: int
    render_workers: int
    encoder_ranks: int
    render_workers_per_gpu: int = 2


def select_render_workers(
    *,
    current: int,
    peak_percent: float,
    temperature_celsius: float,
    failed: bool,
    steps: tuple[int, ...] = (2, 3, 4),
) -> int:
    if (
        not steps
        or any(type(step) is not int or step <= 0 for step in steps)
        or tuple(sorted(set(steps))) != tuple(steps)
    ):
        raise ValueError(
            "render worker steps must be increasing positive integers"
        )
    if current not in steps:
        raise ValueError("current render workers must be a configured step")
    for name, value in (
        ("peak percent", peak_percent),
        ("temperature", temperature_celsius),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"render {name} must be finite")
    if type(failed) is not bool:
        raise ValueError("render failed flag must be boolean")

    index = steps.index(current)
    if failed or peak_percent > 80 or temperature_celsius >= 80:
        return steps[max(0, index - 1)]
    if peak_percent < 70 and temperature_celsius < 75:
        return steps[min(len(steps) - 1, index + 1)]
    return current


def _render_gpu_state(
    recent_snapshots: Sequence[Mapping[str, object]],
) -> tuple[float, float] | None:
    states = []
    for snapshot in recent_snapshots:
        metrics = snapshot.get("gpu_metrics", ())
        if not isinstance(metrics, (list, tuple)):
            continue
        for metric in metrics:
            if not isinstance(metric, Mapping):
                continue
            used = metric.get("memory_used_mib")
            total = metric.get("memory_total_mib")
            temperature = metric.get("temperature_celsius")
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in (used, total, temperature)
            ):
                continue
            if total <= 0 or used < 0:
                continue
            states.append((100.0 * used / total, float(temperature)))
    if not states:
        return None
    return max(value[0] for value in states), max(value[1] for value in states)


def choose_worker_profile(
    recent_snapshots: Sequence[Mapping[str, object]],
    config: PipelineConfig,
    previous: WorkerProfile | None = None,
) -> WorkerProfile:
    profiles = config.worker_tuning.voxel_profiles
    dump_steps = config.worker_tuning.dump_steps
    if previous is None:
        geometry = geometry_profile(config.parallelism)
        return WorkerProfile(
            dump_workers=dump_steps[-1],
            voxel_workers=geometry.processes,
            voxel_threads_per_worker=geometry.native_threads,
            render_workers=config.worker_tuning.render_workers,
            encoder_ranks=config.worker_tuning.encoder_ranks,
            render_workers_per_gpu=(
                config.parallelism.render_workers_per_gpu_steps[
                    min(1, len(config.parallelism.render_workers_per_gpu_steps) - 1)
                ]
            ),
        )
    if not recent_snapshots:
        return previous
    pressure = any(
        float(snapshot.get("cpu_percent", 0)) >= 80
        or float(snapshot.get("available_ram_gib", 10**9))
        < config.limits.ram_soft_available_gib
        or float(snapshot.get("io_wait_percent", 0)) >= 10
        or any(
            "temp" in str(reason).lower()
            or "thermal" in str(reason).lower()
            for reason in snapshot.get("reasons", ())
        )
        for snapshot in recent_snapshots
    )
    stable = len(recent_snapshots) >= 3 and all(
        not snapshot.get("reasons")
        and float(snapshot.get("cpu_percent", 100)) < 70
        and float(snapshot.get("io_wait_percent", 100)) < 5
        and float(snapshot.get("available_ram_gib", 0))
        >= config.limits.ram_soft_available_gib
        for snapshot in recent_snapshots[-3:]
    )
    current_dump = dump_steps.index(previous.dump_workers)
    current_geometry = (
        previous.voxel_workers,
        previous.voxel_threads_per_worker,
    )
    full_geometry = (
        config.parallelism.cpu_physical_cores,
        1,
    )
    if current_geometry == full_geometry:
        voxel_workers, native_threads = (
            profiles[-1] if pressure else current_geometry
        )
        if pressure:
            current_dump = max(0, current_dump - 1)
    else:
        current_voxel = profiles.index(current_geometry)
        if pressure:
            current_dump = max(0, current_dump - 1)
            current_voxel = max(0, current_voxel - 1)
        elif stable:
            current_dump = min(len(dump_steps) - 1, current_dump + 1)
            current_voxel = min(len(profiles) - 1, current_voxel + 1)
        voxel_workers, native_threads = profiles[current_voxel]
    render_workers_per_gpu = previous.render_workers_per_gpu
    render_state = _render_gpu_state(recent_snapshots)
    if render_state is not None:
        peak_percent, temperature_celsius = render_state
        render_workers_per_gpu = select_render_workers(
            current=render_workers_per_gpu,
            peak_percent=peak_percent,
            temperature_celsius=temperature_celsius,
            failed=False,
            steps=config.parallelism.render_workers_per_gpu_steps,
        )
    return WorkerProfile(
        dump_workers=dump_steps[current_dump],
        voxel_workers=voxel_workers,
        voxel_threads_per_worker=native_threads,
        render_workers=previous.render_workers,
        encoder_ranks=previous.encoder_ranks,
        render_workers_per_gpu=render_workers_per_gpu,
    )


@dataclass(frozen=True)
class ShardContext:
    source: str
    shard_id: str
    instances: Path
    metadata_root: Path
    source_root: Path
    download_root: Path
    work_root: Path
    output_root: Path
    batch_id: str
    gate: str = "production"
    record_prefix: str = ""

    @classmethod
    def for_test(
        cls, root: Path, source: str, shard_id: str, *, gate: str = "production"
    ) -> "ShardContext":
        _validate_gate(gate)
        return cls(
            source,
            shard_id,
            root / "instances.txt",
            root / "metadata",
            root / "source",
            root / "raw",
            root / "work",
            root / "output",
            "batch000",
            gate,
        )

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        source: str,
        shard_id: str,
        batch_id: str,
        *,
        gate: str = "production",
    ) -> "ShardContext":
        for name, value in (
            ("source", source),
            ("shard_id", shard_id),
            ("batch_id", batch_id),
        ):
            _validate_identifier(name, value)
        _validate_gate(gate)

        if gate == "production":
            local = config.paths.local_root / "preprocess" / "active"
            shard_root = config.paths.data2_root / "control" / "shards"
        else:
            local = (
                config.paths.local_root / "preprocess" / "qualification" / gate
            )
            shard_root = (
                config.paths.data2_root
                / "control"
                / "qualification"
                / gate
                / "shards"
            )
        local = local / shard_id / batch_id
        control = config.paths.data2_root / "control"
        return cls(
            source=source,
            shard_id=shard_id,
            instances=shard_root / source / shard_id / f"{batch_id}.txt",
            metadata_root=control / "metadata" / source,
            source_root=config.paths.data2_root / "raw" / source,
            download_root=local / "source",
            work_root=local / "work",
            output_root=local / "output",
            batch_id=batch_id,
            gate=gate,
        )


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    gpu_ranks: int = 0
    workers_per_gpu: int = 1


CPU_ENV = (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
)
RENDER_ENV = (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
)
ENCODE_ENV = CPU_ENV + (
    ("ATTN_BACKEND", "sdpa"),
    ("SPARSE_ATTN_BACKEND", "sdpa"),
)


def _validate_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise ValueError(f"unsafe {name}: {value!r}")


def _validate_gate(value: str) -> None:
    if value not in {"smoke", "pilot", "production"}:
        raise ValueError(f"invalid gate: {value!r}")


def python_command(script: str, *args: str) -> tuple[str, ...]:
    python = sys.executable
    override = os.environ.get("PIXAL3D_LEAF_WORKER")
    if override:
        return (python, override, "--original-script", script, *args)
    return (python, f"data_toolkit/{script}", *args)


def dataset_args(source: str) -> tuple[str, ...]:
    if source == "ObjaverseXL_sketchfab":
        return ("ObjaverseXL", "--source", "sketchfab")
    if source == "ObjaverseXL_github":
        return ("ObjaverseXL", "--source", "github")
    return (source,)


def expand_ranked(
    command: CommandSpec,
) -> tuple[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], ...]:
    if type(command.gpu_ranks) is not int or command.gpu_ranks < 0:
        raise ValueError("GPU rank count must be a nonnegative integer")
    if (
        type(command.workers_per_gpu) is not int
        or command.workers_per_gpu <= 0
    ):
        raise ValueError("workers per GPU must be a positive integer")
    if not command.gpu_ranks:
        if command.workers_per_gpu != 1:
            raise ValueError("workers per GPU requires GPU ranks")
        return ((command.argv, command.env),)
    raw_indices = os.environ.get("PIXAL3D_GPU_INDICES")
    if raw_indices is None:
        gpu_indices = tuple(range(command.gpu_ranks))
    else:
        try:
            gpu_indices = tuple(int(value) for value in raw_indices.split(","))
        except ValueError as error:
            raise ValueError("PIXAL3D_GPU_INDICES must be comma-separated integers") from error
        if (
            not gpu_indices
            or any(index < 0 for index in gpu_indices)
            or len(set(gpu_indices)) != len(gpu_indices)
        ):
            raise ValueError("PIXAL3D_GPU_INDICES must contain unique nonnegative GPU indices")
        gpu_indices = gpu_indices[:command.gpu_ranks]
    total = len(gpu_indices) * command.workers_per_gpu
    if total > 28:
        raise ValueError("ranked command exceeds the 28-process cap")
    return tuple(
        (
            (
                *command.argv,
                "--rank",
                str(rank),
                "--world_size",
                str(total),
            ),
            (
                *command.env,
                ("CUDA_VISIBLE_DEVICES", str(gpu_indices[rank % len(gpu_indices)])),
            ),
        )
        for rank in range(total)
    )


def build_preprocessing_dag(
    context: ShardContext,
    config: PipelineConfig,
    profile: WorkerProfile | None = None,
) -> tuple[CommandSpec, ...]:
    if profile is None:
        geometry = geometry_profile(config.parallelism)
        profile = WorkerProfile(
            dump_workers=config.workers.dump_workers,
            voxel_workers=geometry.processes,
            voxel_threads_per_worker=geometry.native_threads,
            render_workers=config.workers.render_workers,
            encoder_ranks=config.workers.encoder_ranks,
            render_workers_per_gpu=(
                config.parallelism.render_workers_per_gpu_steps[
                    min(1, len(config.parallelism.render_workers_per_gpu_steps) - 1)
                ]
            ),
        )
    dataset = dataset_args(context.source)
    record_args = (
        ("--record_prefix", context.record_prefix)
        if context.record_prefix
        else ()
    )
    base = (
        *dataset,
        "--root",
        str(context.metadata_root),
        "--instances",
        str(context.instances),
    )
    blender = (
        config.paths.local_root
        / "tools"
        / f"blender-{config.render.blender_version}-linux-x64"
        / "blender"
    )
    commands: list[CommandSpec] = [
        CommandSpec(
            "download",
            python_command(
                "download.py",
                *base,
                "--download_root",
                str(context.source_root),
                "--max_workers",
                "8",
            ),
            CPU_ENV,
        ),
        CommandSpec("stage_raw", ("internal:stage_raw",)),
        CommandSpec(
            "dump_mesh",
            python_command(
                "dump_mesh.py",
                *base,
                "--download_root",
                str(context.download_root),
                "--mesh_dump_root",
                str(context.work_root),
                "--blender_path",
                str(blender),
                "--max_workers",
                str(profile.dump_workers),
            ),
            CPU_ENV,
        ),
        CommandSpec(
            "dump_pbr",
            python_command(
                "dump_pbr.py",
                *base,
                "--download_root",
                str(context.download_root),
                "--pbr_dump_root",
                str(context.work_root),
                "--blender_path",
                str(blender),
                "--max_workers",
                str(profile.dump_workers),
            ),
            CPU_ENV,
        ),
        CommandSpec(
            "asset_stats",
            python_command(
                "asset_stats.py",
                "--root",
                str(context.metadata_root),
                "--instances",
                str(context.instances),
                "--mesh_dump_root",
                str(context.work_root),
                "--pbr_dump_root",
                str(context.work_root),
                "--max_workers",
                str(profile.dump_workers),
                *record_args,
            ),
            CPU_ENV,
        ),
        CommandSpec(
            "render_cond",
            python_command(
                "render_cond.py",
                *base,
                "--download_root",
                str(context.download_root),
                "--render_cond_root",
                str(context.output_root),
                "--num_cond_views",
                str(config.render.num_views),
                "--cond_resolution",
                str(config.render.resolution),
                "--blender_path",
                str(blender),
                "--cycles_device",
                config.render.cycles_device,
                "--max_workers",
                "1",
                *record_args,
            ),
            RENDER_ENV,
            gpu_ranks=profile.render_workers,
            workers_per_gpu=profile.render_workers_per_gpu,
        ),
    ]

    for resolution in config.targets.resolutions:
        common = (
            "--resolution",
            str(resolution),
            "--view_indices",
            "0-1",
        )
        commands.extend(
            [
                CommandSpec(
                    f"dual_grid_{resolution}",
                    python_command(
                        "dual_grid_view.py",
                        *base,
                        "--mesh_dump_root",
                        str(context.work_root),
                        "--transform_root",
                        str(context.output_root / "renders_cond"),
                        "--dual_grid_root",
                        str(context.work_root),
                        *common,
                        "--max_workers",
                        str(profile.voxel_workers),
                        "--native_threads",
                        str(profile.voxel_threads_per_worker),
                        *record_args,
                    ),
                    CPU_ENV,
                ),
                CommandSpec(
                    f"voxelize_pbr_{resolution}",
                    python_command(
                        "voxelize_pbr_view.py",
                        *base,
                        "--pbr_dump_root",
                        str(context.work_root),
                        "--transform_root",
                        str(context.output_root / "renders_cond"),
                        "--pbr_voxel_root",
                        str(context.work_root),
                        *common,
                        "--max_workers",
                        str(profile.voxel_workers),
                        "--native_threads",
                        str(profile.voxel_threads_per_worker),
                        *record_args,
                    ),
                    CPU_ENV,
                ),
                CommandSpec(
                    f"encode_shape_{resolution}",
                    python_command(
                        "encode_shape_latent_view.py",
                        "--root",
                        str(context.metadata_root),
                        "--instances",
                        str(context.instances),
                        "--dual_grid_root",
                        str(context.work_root),
                        "--shape_latent_root",
                        str(context.output_root),
                        *common,
                        "--loader_workers",
                        str(config.workers.encoder_loader_threads),
                        "--saver_workers",
                        str(config.workers.encoder_saver_threads),
                        "--latent_dtype",
                        config.targets.latent_dtype,
                        "--micro_batch_size",
                        str(config.parallelism.micro_batch(resolution)),
                        "--gpu_memory_target_percent",
                        str(config.parallelism.gpu_memory_target_percent),
                    ),
                    ENCODE_ENV,
                    gpu_ranks=profile.encoder_ranks,
                ),
                CommandSpec(
                    f"encode_pbr_{resolution}",
                    python_command(
                        "encode_pbr_latent_view.py",
                        "--root",
                        str(context.metadata_root),
                        "--instances",
                        str(context.instances),
                        "--pbr_voxel_root",
                        str(context.work_root),
                        "--pbr_latent_root",
                        str(context.output_root),
                        *common,
                        "--loader_workers",
                        str(config.workers.encoder_loader_threads),
                        "--saver_workers",
                        str(config.workers.encoder_saver_threads),
                        "--latent_dtype",
                        config.targets.latent_dtype,
                        "--micro_batch_size",
                        str(config.parallelism.micro_batch(resolution)),
                        "--gpu_memory_target_percent",
                        str(config.parallelism.gpu_memory_target_percent),
                    ),
                    ENCODE_ENV,
                    gpu_ranks=profile.encoder_ranks,
                ),
                CommandSpec(
                    f"cleanup_voxels_{resolution}",
                    ("internal:cleanup_voxels", str(resolution)),
                ),
            ]
        )

    ss_resolution = config.targets.ss_resolution
    shape_resolution = max(config.targets.resolutions)
    commands.extend(
        [
            CommandSpec(
                f"encode_ss_{ss_resolution}",
                python_command(
                    "encode_ss_latent_view.py",
                    "--root",
                    str(context.metadata_root),
                    "--instances",
                    str(context.instances),
                    "--shape_latent_root",
                    str(context.output_root),
                    "--ss_latent_root",
                    str(context.output_root),
                    "--shape_latent_name",
                    f"shape_enc_next_dc_f16c32_fp16_{shape_resolution}",
                    "--resolution",
                    str(ss_resolution),
                    "--view_indices",
                    "0-1",
                    "--loader_workers",
                    str(config.workers.encoder_loader_threads),
                    "--saver_workers",
                    str(config.workers.encoder_saver_threads),
                    "--micro_batch_size",
                    str(config.parallelism.micro_batch(config.targets.ss_resolution)),
                    "--gpu_memory_target_percent",
                    str(config.parallelism.gpu_memory_target_percent),
                ),
                ENCODE_ENV,
                gpu_ranks=profile.encoder_ranks,
            ),
            CommandSpec("validate_outputs", ("internal:validate_outputs",)),
            CommandSpec("build_packs", ("internal:build_packs",)),
            CommandSpec("archive_raw", ("internal:archive_raw",)),
            CommandSpec("cleanup_local", ("internal:cleanup_local",)),
        ]
    )
    return tuple(commands)
