from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig


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
    return ("python", f"data_toolkit/{script}", *args)


def dataset_args(source: str) -> tuple[str, ...]:
    if source == "ObjaverseXL_sketchfab":
        return ("ObjaverseXL", "--source", "sketchfab")
    if source == "ObjaverseXL_github":
        return ("ObjaverseXL", "--source", "github")
    return (source,)


def expand_ranked(
    command: CommandSpec,
) -> tuple[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], ...]:
    if not command.gpu_ranks:
        return ((command.argv, command.env),)
    return tuple(
        (
            (
                *command.argv,
                "--rank",
                str(rank),
                "--world_size",
                str(command.gpu_ranks),
            ),
            (*command.env, ("CUDA_VISIBLE_DEVICES", str(rank))),
        )
        for rank in range(command.gpu_ranks)
    )


def build_preprocessing_dag(
    context: ShardContext, config: PipelineConfig
) -> tuple[CommandSpec, ...]:
    dataset = dataset_args(context.source)
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
                "--max_workers",
                str(config.workers.dump_workers),
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
                "--max_workers",
                str(config.workers.dump_workers),
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
                str(config.workers.dump_workers),
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
            ),
            RENDER_ENV,
            gpu_ranks=config.workers.render_workers,
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
                        str(config.workers.voxel_workers),
                        "--native_threads",
                        str(config.workers.voxel_threads_per_worker),
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
                        str(config.workers.voxel_workers),
                        "--native_threads",
                        str(config.workers.voxel_threads_per_worker),
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
                    ),
                    CPU_ENV,
                    gpu_ranks=config.workers.encoder_ranks,
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
                    ),
                    CPU_ENV,
                    gpu_ranks=config.workers.encoder_ranks,
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
                ),
                CPU_ENV,
                gpu_ranks=config.workers.encoder_ranks,
            ),
            CommandSpec("validate_outputs", ("internal:validate_outputs",)),
            CommandSpec("build_packs", ("internal:build_packs",)),
            CommandSpec("archive_raw", ("internal:archive_raw",)),
            CommandSpec("cleanup_local", ("internal:cleanup_local",)),
        ]
    )
    return tuple(commands)
