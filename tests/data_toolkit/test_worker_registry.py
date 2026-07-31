from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from data_toolkit.pipeline.worker_registry import WorkerRegistry, WorkerRegistration


def test_registry_persists_registration_drain_and_heartbeat(tmp_path):
    registry = WorkerRegistry(tmp_path / "workers.json")
    registered = registry.register(
        WorkerRegistration(
            node_id="node16",
            ssh_target="youngwoo@n16.unist.info:55555",
            cpu_limit=40,
            gpu_indices=(0, 1, 2, 3),
            data2_root=Path("/file2/youngwoo/pixal3d"),
            data3_root=Path("/file3/youngwoo/pixal3d"),
            local_root=Path("/home/youngwoo/data/pixal3d"),
        ),
        now=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )

    registry.heartbeat("node16", now=datetime(2026, 7, 20, 0, 0, 15, tzinfo=timezone.utc))
    registry.drain("node16")

    restored = WorkerRegistry(tmp_path / "workers.json").read()["node16"]
    assert registered.node_id == "node16"
    assert restored.state == "draining"
    assert restored.healthy(now=datetime(2026, 7, 20, 0, 1, 0, tzinfo=timezone.utc))
    assert not restored.healthy(now=datetime(2026, 7, 20, 0, 2, 0, tzinfo=timezone.utc))
    assert restored.registration.gpu_indices == (0, 1, 2, 3)
    assert restored.registration.data2_root == Path("/file2/youngwoo/pixal3d")


def test_registry_can_remove_and_reactivate_worker(tmp_path):
    registry = WorkerRegistry(tmp_path / "workers.json")
    registration = WorkerRegistration(
        node_id="node17",
        ssh_target="local://node17",
        cpu_limit=44,
        gpu_indices=(1, 2, 3, 4, 5, 6),
        data2_root=Path("/root/data2/pixal3d"),
        data3_root=Path("/root/data3/pixal3d"),
        local_root=Path("/root/node17/data/pixal3d"),
    )
    registry.register(registration, now=datetime(2026, 7, 20, tzinfo=timezone.utc))

    assert registry.remove("node17").state == "removed"
    assert registry.activate(
        "node17", now=datetime(2026, 7, 20, 0, 1, tzinfo=timezone.utc)
    ).state == "active"


def test_registration_rejects_relative_execution_paths():
    with pytest.raises(ValueError, match="absolute"):
        WorkerRegistration(
            node_id="node16",
            ssh_target="node16",
            cpu_limit=40,
            gpu_indices=(0,),
            data2_root=Path("relative/data2"),
            data3_root=Path("/data3"),
            local_root=Path("/local"),
        )


def test_concurrent_registry_updates_do_not_lose_workers(tmp_path):
    registry = WorkerRegistry(tmp_path / "workers.json")
    barrier = threading.Barrier(12)

    def register(index):
        barrier.wait()
        registry.register(
            WorkerRegistration(
                node_id=f"node{index}",
                ssh_target=f"node{index}",
                cpu_limit=4,
                gpu_indices=(index,),
                data2_root=Path(f"/data2/node{index}"),
                data3_root=Path(f"/data3/node{index}"),
                local_root=Path(f"/local/node{index}"),
            ),
            now=datetime(2026, 7, 20, tzinfo=timezone.utc),
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        tuple(executor.map(register, range(12)))

    assert set(registry.read()) == {f"node{index}" for index in range(12)}
