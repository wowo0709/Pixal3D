from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from data_toolkit.pipeline.worker_supervisor import ProductionWorkerSupervisor
from data_toolkit.pipeline.work_queue import ProductionWorkQueue, WorkUnit
from data_toolkit.pipeline.worker_registry import (
    WorkerRegistration,
    WorkerRegistry,
)


NOW = datetime(2026, 7, 22, 5, 0, tzinfo=timezone.utc)


def setup_runtime(tmp_path):
    unit = WorkUnit("ABO", "ABO-00000", "batch000", 256)
    queue = ProductionWorkQueue(
        tmp_path / "queue", lease_timeout=timedelta(minutes=5)
    )
    queue.initialize("a" * 64, (unit,), now=NOW)
    registry = WorkerRegistry(tmp_path / "workers.json")
    registry.register(
        WorkerRegistration(
            "node17",
            "local",
            44,
            (1, 2, 3, 4, 5, 6),
            tmp_path / "data2",
            tmp_path / "data3",
            tmp_path / "local",
        ),
        now=NOW,
    )
    return queue, registry


def supervisor(tmp_path, *, process_runner, sleeper):
    queue, registry = setup_runtime(tmp_path)
    return (
        ProductionWorkerSupervisor(
            queue,
            registry,
            "node17",
            ("pixal3d", "worker", "--node-id", "node17"),
            process_runner=process_runner,
            sleeper=sleeper,
            poll_interval=0.25,
        ),
        queue,
        registry,
    )


def test_supervisor_restarts_worker_exit_while_node_is_active(tmp_path):
    calls = []
    sleeps = []
    subject, _, _ = supervisor(
        tmp_path,
        process_runner=lambda command, check: (
            calls.append((command, check)) or SimpleNamespace(returncode=7)
        ),
        sleeper=sleeps.append,
    )

    assert subject.run_cycle() is True
    assert subject.run_cycle() is True

    assert calls == [
        (("pixal3d", "worker", "--node-id", "node17"), False),
        (("pixal3d", "worker", "--node-id", "node17"), False),
    ]
    assert sleeps == [0.25, 0.25]


def test_supervisor_waits_while_draining_and_resumes_after_activation(tmp_path):
    calls = []
    sleeps = []
    subject, _, registry = supervisor(
        tmp_path,
        process_runner=lambda command, check: (
            calls.append(command) or SimpleNamespace(returncode=0)
        ),
        sleeper=sleeps.append,
    )
    registry.drain("node17")

    assert subject.run_cycle() is True
    assert calls == []

    registry.activate("node17", now=NOW + timedelta(seconds=1))
    assert subject.run_cycle() is True
    assert calls == [("pixal3d", "worker", "--node-id", "node17")]
    assert sleeps == [0.25, 0.25]


def test_supervisor_exits_when_queue_is_terminal(tmp_path):
    calls = []
    subject, queue, _ = supervisor(
        tmp_path,
        process_runner=lambda command, check: calls.append(command),
        sleeper=lambda seconds: None,
    )
    lease = queue.claim("node17", now=NOW, token="done")
    queue.complete(lease, now=NOW)

    assert subject.run_cycle() is False
    assert calls == []


def test_supervisor_exits_when_worker_is_removed(tmp_path):
    calls = []
    subject, _, registry = supervisor(
        tmp_path,
        process_runner=lambda command, check: calls.append(command),
        sleeper=lambda seconds: None,
    )
    registry.remove("node17")

    assert subject.run_cycle() is False
    assert calls == []
