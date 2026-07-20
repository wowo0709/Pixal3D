from datetime import datetime, timedelta, timezone

from data_toolkit.pipeline.worker_registry import WorkerRegistry, WorkerRegistration


def test_registry_persists_registration_drain_and_heartbeat(tmp_path):
    registry = WorkerRegistry(tmp_path / "workers.json")
    registered = registry.register(
        WorkerRegistration(
            node_id="node16",
            ssh_target="youngwoo@n16.unist.info:55555",
            cpu_limit=40,
            gpu_indices=(2, 3, 4, 5),
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
