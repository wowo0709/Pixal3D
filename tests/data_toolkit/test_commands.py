from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from data_toolkit.pipeline.commands import (
    CommandSpec,
    ShardContext,
    WorkerProfile,
    build_preprocessing_dag,
    choose_worker_profile,
    expand_ranked,
    select_render_workers,
)
from data_toolkit.pipeline.parallelism import geometry_profile


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


def _by_name(dag, name):
    return next(command for command in dag if command.name == name)


def _python_command(script, *args):
    return ("python", f"data_toolkit/{script}", *args)


def test_worker_profile_ramps_after_stable_telemetry(config):
    first = WorkerProfile(32, 8, 4, 7, 7, 2)
    stable = [
        {"cpu_percent": 60, "io_wait_percent": 2, "available_ram_gib": 200, "reasons": []}
    ] * 3
    second = choose_worker_profile(stable, config, first)
    assert second.dump_workers == 36
    assert (second.voxel_workers, second.voxel_threads_per_worker) == (10, 4)


def test_initial_worker_profile_owns_full_geometry_lane(config):
    selected = choose_worker_profile((), config)

    assert selected.dump_workers == 44
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (11, 4)


def test_worker_profile_steps_down_on_pressure(config):
    previous = WorkerProfile(40, 10, 4, 7, 7)
    pressure = [{"cpu_percent": 70, "io_wait_percent": 12, "available_ram_gib": 100, "reasons": ["I/O wait"]}]
    selected = choose_worker_profile(pressure, config, previous)
    assert selected.dump_workers == 36
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (8, 4)


def test_geometry_profile_steps_up_only_after_three_stable_boundaries(config):
    previous = WorkerProfile(32, 8, 4, 7, 7, 2)
    stable = {
        "cpu_percent": 60,
        "io_wait_percent": 2,
        "available_ram_gib": 200,
        "reasons": [],
    }

    assert choose_worker_profile([stable] * 2, config, previous) == previous
    selected = choose_worker_profile([stable] * 3, config, previous)
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (10, 4)


def test_worker_profile_applies_render_step_at_next_dag_boundary(config):
    previous = choose_worker_profile((), config)
    stable_gpu = [
        {
            "cpu_percent": 60,
            "io_wait_percent": 2,
            "available_ram_gib": 200,
            "reasons": [],
            "gpu_metrics": [
                {
                    "memory_used_mib": 60_000,
                    "memory_total_mib": 97_887,
                    "temperature_celsius": 70,
                }
            ],
        }
    ] * 3

    selected = choose_worker_profile(stable_gpu, config, previous)

    assert previous.render_workers_per_gpu == 2
    assert selected.render_workers_per_gpu == 3


def test_dag_accepts_worker_profile(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    profile = WorkerProfile(44, 11, 4, 7, 7)
    dag = build_preprocessing_dag(context, config, profile)
    assert str(profile.dump_workers) in _by_name(dag, "dump_mesh").argv
    assert str(profile.voxel_workers) in _by_name(dag, "dual_grid_256").argv


def test_dag_has_exact_order_for_all_configured_resolutions(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")

    dag = build_preprocessing_dag(context, config)

    expected_names = [
        "download",
        "stage_raw",
        "dump_mesh",
        "dump_pbr",
        "asset_stats",
        "render_cond",
    ]
    for resolution in config.targets.resolutions:
        expected_names.extend(
            [
                f"dual_grid_{resolution}",
                f"voxelize_pbr_{resolution}",
                f"encode_shape_{resolution}",
                f"encode_pbr_{resolution}",
                f"cleanup_voxels_{resolution}",
            ]
        )
    expected_names.extend(
        [
            f"encode_ss_{config.targets.ss_resolution}",
            "validate_outputs",
            "build_packs",
            "archive_raw",
            "cleanup_local",
        ]
    )
    assert [command.name for command in dag] == expected_names


def test_commands_have_exact_parser_compatible_argv(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)
    base = (
        "ABO",
        "--root",
        str(context.metadata_root),
        "--instances",
        str(context.instances),
    )

    assert _by_name(dag, "download").argv == _python_command(
        "download.py",
        *base,
        "--download_root",
        str(context.source_root),
        "--max_workers",
        "8",
    )
    assert _by_name(dag, "dump_mesh").argv == _python_command(
        "dump_mesh.py",
        *base,
        "--download_root",
        str(context.download_root),
        "--mesh_dump_root",
        str(context.work_root),
        "--max_workers",
        str(config.workers.dump_workers),
    )
    assert _by_name(dag, "dump_pbr").argv == _python_command(
        "dump_pbr.py",
        *base,
        "--download_root",
        str(context.download_root),
        "--pbr_dump_root",
        str(context.work_root),
        "--max_workers",
        str(config.workers.dump_workers),
    )
    assert _by_name(dag, "asset_stats").argv == _python_command(
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
    )
    assert _by_name(dag, "render_cond").argv == _python_command(
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
        str(
            config.paths.local_root
            / "tools"
            / f"blender-{config.render.blender_version}-linux-x64"
            / "blender"
        ),
        "--cycles_device",
        config.render.cycles_device,
        "--max_workers",
        "1",
    )

    geometry = geometry_profile(config.parallelism)
    for resolution in config.targets.resolutions:
        common = (
            "--resolution",
            str(resolution),
            "--view_indices",
            "0-1",
        )
        assert _by_name(dag, f"dual_grid_{resolution}").argv == _python_command(
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
            str(geometry.processes),
            "--native_threads",
            str(geometry.native_threads),
        )
        assert _by_name(
            dag, f"voxelize_pbr_{resolution}"
        ).argv == _python_command(
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
            str(geometry.processes),
            "--native_threads",
            str(geometry.native_threads),
        )
        assert _by_name(
            dag, f"encode_shape_{resolution}"
        ).argv == _python_command(
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
        )
        assert _by_name(dag, f"encode_pbr_{resolution}").argv == _python_command(
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
        )

    ss_resolution = config.targets.ss_resolution
    shape_resolution = max(config.targets.resolutions)
    assert _by_name(dag, f"encode_ss_{ss_resolution}").argv == _python_command(
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
        str(config.parallelism.micro_batch(ss_resolution)),
        "--gpu_memory_target_percent",
        str(config.parallelism.gpu_memory_target_percent),
    )


def test_internal_command_names_and_arguments_are_exact(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)

    assert _by_name(dag, "stage_raw").argv == ("internal:stage_raw",)
    for resolution in config.targets.resolutions:
        assert _by_name(dag, f"cleanup_voxels_{resolution}").argv == (
            "internal:cleanup_voxels",
            str(resolution),
        )
    assert _by_name(dag, "validate_outputs").argv == (
        "internal:validate_outputs",
    )
    assert _by_name(dag, "build_packs").argv == ("internal:build_packs",)
    assert _by_name(dag, "archive_raw").argv == ("internal:archive_raw",)
    assert _by_name(dag, "cleanup_local").argv == ("internal:cleanup_local",)


@pytest.mark.parametrize(
    ("source", "canonical_source"),
    [
        ("ObjaverseXL_sketchfab", "sketchfab"),
        ("ObjaverseXL_github", "github"),
    ],
)
def test_objaversexl_source_mapping(source, canonical_source, config, tmp_path):
    context = ShardContext.for_test(tmp_path, source, f"{source}-00000")
    dag = build_preprocessing_dag(context, config)

    dataset_commands = [
        command
        for command in dag
        if command.name
        in {
            "download",
            "dump_mesh",
            "dump_pbr",
            "render_cond",
            *(f"dual_grid_{value}" for value in config.targets.resolutions),
            *(f"voxelize_pbr_{value}" for value in config.targets.resolutions),
        }
    ]
    assert dataset_commands
    assert all(
        command.argv[2:5] == ("ObjaverseXL", "--source", canonical_source)
        for command in dataset_commands
    )


def test_render_and_cpu_stages_apply_thread_and_worker_caps(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)
    render = _by_name(dag, "render_cond")

    assert render.env == RENDER_ENV
    assert render.gpu_ranks == config.workers.render_workers
    assert render.workers_per_gpu == 2
    assert render.argv[render.argv.index("--max_workers") + 1] == "1"
    assert render.argv[render.argv.index("--num_cond_views") + 1] == "8"
    assert render.argv[render.argv.index("--cond_resolution") + 1] == "512"
    assert render.argv[render.argv.index("--cycles_device") + 1] == "OPTIX"
    assert render.argv[render.argv.index("--blender_path") + 1].endswith(
        "blender-4.5.1-linux-x64/blender"
    )
    assert "--camera_policy" not in render.argv
    assert "--fov_min_degrees" not in render.argv
    assert "--fov_max_degrees" not in render.argv

    external_cpu = [
        command
        for command in dag
        if command.argv[0] == "python" and command.name != "render_cond"
    ]
    assert all(command.env == CPU_ENV for command in external_cpu)
    assert all(
        command.gpu_ranks == config.workers.encoder_ranks
        for command in external_cpu
        if command.name.startswith("encode_")
    )
    assert all(
        command.gpu_ranks == 0
        for command in external_cpu
        if not command.name.startswith("encode_")
    )
    assert all(
        command.env == () and command.gpu_ranks == 0
        for command in dag
        if command.argv[0].startswith("internal:")
    )


def test_all_voxel_and_encoder_commands_use_anchor_views(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)
    view_commands = [
        command
        for command in dag
        if command.name.startswith(
            ("dual_grid_", "voxelize_pbr_", "encode_shape_", "encode_pbr_", "encode_ss_")
        )
    ]

    assert view_commands
    assert all(
        command.argv[command.argv.index("--view_indices") + 1] == "0-1"
        for command in view_commands
    )


def test_render_workers_map_round_robin_to_seven_gpus(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    render = _by_name(build_preprocessing_dag(context, config), "render_cond")

    expanded = expand_ranked(render)

    assert len(expanded) == 14
    assert [dict(env)["CUDA_VISIBLE_DEVICES"] for _, env in expanded] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    for rank, (argv, env) in enumerate(expanded):
        assert argv == (
            *render.argv,
            "--rank",
            str(rank),
            "--world_size",
            "14",
        )
        assert env == (
            *RENDER_ENV,
            ("CUDA_VISIBLE_DEVICES", str(rank % config.workers.render_workers)),
        )


def test_unranked_command_expands_once_without_mutation():
    command = CommandSpec("cpu", ("python", "script.py"), CPU_ENV)

    assert expand_ranked(command) == ((command.argv, command.env),)
    assert command == CommandSpec("cpu", ("python", "script.py"), CPU_ENV)


@pytest.mark.parametrize(
    "command",
    [
        CommandSpec("negative", ("worker",), gpu_ranks=-1),
        CommandSpec(
            "zero-workers", ("worker",), gpu_ranks=7, workers_per_gpu=0
        ),
        CommandSpec("too-many", ("worker",), gpu_ranks=7, workers_per_gpu=5),
        CommandSpec("cpu-multiplier", ("worker",), workers_per_gpu=2),
    ],
)
def test_rank_expansion_rejects_invalid_worker_counts(command):
    with pytest.raises(ValueError):
        expand_ranked(command)


def test_render_worker_selector_steps_only_at_profile_boundaries():
    steps = (2, 3, 4)

    assert select_render_workers(
        current=2,
        peak_percent=65.0,
        temperature_celsius=70.0,
        failed=False,
        steps=steps,
    ) == 3
    assert select_render_workers(
        current=3,
        peak_percent=81.0,
        temperature_celsius=70.0,
        failed=False,
        steps=steps,
    ) == 2
    assert select_render_workers(
        current=4,
        peak_percent=75.0,
        temperature_celsius=80.0,
        failed=False,
        steps=steps,
    ) == 3
    assert select_render_workers(
        current=3,
        peak_percent=75.0,
        temperature_celsius=70.0,
        failed=True,
        steps=steps,
    ) == 2


def test_context_from_config_builds_paths_under_trusted_roots(config):
    context = ShardContext.from_config(
        config, "ObjaverseXL_github", "ObjaverseXL_github-00000", "batch000"
    )
    local = (
        config.paths.local_root
        / "preprocess"
        / "active"
        / "ObjaverseXL_github-00000"
        / "batch000"
    )

    assert context == ShardContext(
        source="ObjaverseXL_github",
        shard_id="ObjaverseXL_github-00000",
        instances=config.paths.data2_root
        / "control/shards/ObjaverseXL_github/ObjaverseXL_github-00000/batch000.txt",
        metadata_root=config.paths.data2_root
        / "control/metadata/ObjaverseXL_github",
        source_root=config.paths.data2_root / "raw/ObjaverseXL_github",
        download_root=local / "source",
        work_root=local / "work",
        output_root=local / "output",
        batch_id="batch000",
    )
    assert context.instances.is_relative_to(config.paths.data2_root / "control")
    assert context.metadata_root.is_relative_to(config.paths.data2_root / "control")
    assert context.source_root.is_relative_to(config.paths.data2_root / "raw")
    assert context.download_root.is_relative_to(config.paths.local_root)
    assert context.work_root.is_relative_to(config.paths.local_root)
    assert context.output_root.is_relative_to(config.paths.local_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", ""),
        ("source", "../ABO"),
        ("source", "/absolute"),
        ("source", "ABO\\escape"),
        ("shard_id", "."),
        ("shard_id", ".."),
        ("shard_id", "ABO/00000"),
        ("batch_id", "../batch000"),
        ("batch_id", "/tmp/batch000"),
        ("batch_id", "batch000\\escape"),
    ],
)
def test_context_from_config_rejects_unsafe_identifiers(config, field, value):
    values = {
        "source": "ABO",
        "shard_id": "ABO-00000",
        "batch_id": "batch000",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        ShardContext.from_config(config, **values)


def test_context_and_dag_are_frozen_and_deterministic(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    first = build_preprocessing_dag(context, config)
    second = build_preprocessing_dag(context, config)

    assert isinstance(first, tuple)
    assert first == second
    assert hash(context)
    assert hash(first)
    with pytest.raises(FrozenInstanceError):
        context.source = "HSSD"
    with pytest.raises(FrozenInstanceError):
        first[0].name = "changed"
