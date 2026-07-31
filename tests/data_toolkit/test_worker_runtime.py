from pathlib import Path
from types import SimpleNamespace

from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.gpu_policy import GpuRuntimePolicy
from data_toolkit.pipeline.worker_registry import WorkerRegistration

import pytest

from data_toolkit.pipeline.worker_runtime import (
    WorkerAlreadyRunningError,
    WorkerRuntimeError,
    execution_config,
    validate_worker_environment,
    worker_process_lock,
)


CONFIG = Path("data_toolkit/configs/multiview_preprocess.yaml")


def test_execution_config_preserves_gate_identity_and_overrides_node_resources():
    canonical = load_config(CONFIG)
    registration = WorkerRegistration(
        node_id="node16",
        ssh_target="youngwoo@n16.unist.info:55555",
        cpu_limit=40,
        gpu_indices=(0, 1, 2, 3),
        data2_root=Path("/file2/youngwoo/pixal3d"),
        data3_root=Path("/file3/youngwoo/pixal3d"),
        local_root=Path("/home/youngwoo/data/pixal3d"),
    )

    configured = execution_config(canonical, registration)

    assert configured.config_hash() == canonical.config_hash()
    assert configured.paths.data2_root == registration.data2_root
    assert configured.paths.data3_root == registration.data3_root
    assert configured.paths.local_root == registration.local_root
    assert configured.parallelism.cpu_physical_cores == 40
    assert configured.parallelism.gpu_count == 4
    assert configured.workers.render_workers == 4
    assert configured.workers.encoder_ranks == 4
    assert configured.worker_tuning.render_workers == 4
    assert configured.worker_tuning.encoder_ranks == 4
    assert configured.worker_tuning.dump_steps == (32, 36, 40)
    assert configured.worker_tuning.voxel_profiles == ((8, 4), (10, 4))
    assert canonical.parallelism.gpu_count == 7


def test_execution_config_applies_gpu_policy_without_changing_identity():
    canonical = load_config(CONFIG)
    registration = WorkerRegistration(
        node_id="node16",
        ssh_target="node16",
        cpu_limit=40,
        gpu_indices=(0, 1, 2, 3),
        data2_root=Path("/file2/youngwoo/pixal3d"),
        data3_root=Path("/file3/youngwoo/pixal3d"),
        local_root=Path("/home/youngwoo/data/pixal3d"),
    )

    configured = execution_config(
        canonical,
        registration,
        GpuRuntimePolicy(80, 100),
    )

    assert configured.config_hash() == canonical.config_hash()
    assert canonical.parallelism.gpu_memory_hard_percent == 90
    assert configured.parallelism.gpu_memory_target_percent == 80
    assert configured.parallelism.gpu_memory_hard_percent == 100


def test_worker_process_lock_rejects_duplicate_for_same_node(tmp_path):
    with worker_process_lock(tmp_path, "node16"):
        with pytest.raises(WorkerAlreadyRunningError, match="node16"):
            with worker_process_lock(tmp_path, "node16"):
                pytest.fail("duplicate worker acquired the lock")

    with worker_process_lock(tmp_path, "node16"):
        pass


def test_worker_process_lock_is_independent_per_node(tmp_path):
    with worker_process_lock(tmp_path, "node16"):
        with worker_process_lock(tmp_path, "node17"):
            pass


def test_worker_environment_validates_native_modules_cuda_and_blender(
    tmp_path,
):
    canonical = load_config(CONFIG)
    registration = WorkerRegistration(
        node_id="node16",
        ssh_target="node16",
        cpu_limit=40,
        gpu_indices=(0, 1, 2, 3),
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
        local_root=tmp_path / "local",
    )
    configured = execution_config(canonical, registration)
    blender = (
        registration.local_root
        / "tools/blender-4.5.1-linux-x64/blender"
    )
    blender.parent.mkdir(parents=True)
    blender.write_text("fixture")
    blender.chmod(0o700)
    torch = SimpleNamespace(
        __version__="2.8.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
        ),
    )
    imported = []

    def importer(name):
        imported.append(name)
        return torch if name == "torch" else object()

    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(stdout="Blender 4.5.1\n")

    validate_worker_environment(
        configured,
        registration,
        importer=importer,
        process_runner=run,
    )

    assert {"cumesh", "flex_gemm", "nvdiffrast", "o_voxel"}.issubset(
        imported
    )
    assert commands[0][0] == [str(blender), "--version"]


def test_worker_environment_fails_before_claim_when_native_module_is_missing(
    tmp_path,
):
    canonical = load_config(CONFIG)
    registration = WorkerRegistration(
        node_id="node16",
        ssh_target="node16",
        cpu_limit=40,
        gpu_indices=(0,),
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
        local_root=tmp_path / "local",
    )

    def importer(name):
        if name == "o_voxel":
            raise ModuleNotFoundError(name)
        return object()

    with pytest.raises(WorkerRuntimeError, match="o_voxel"):
        validate_worker_environment(
            execution_config(canonical, registration),
            registration,
            importer=importer,
        )
