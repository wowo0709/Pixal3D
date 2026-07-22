from pathlib import Path

from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.worker_registry import WorkerRegistration
from data_toolkit.pipeline.worker_runtime import execution_config


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
    assert canonical.parallelism.gpu_count == 7
