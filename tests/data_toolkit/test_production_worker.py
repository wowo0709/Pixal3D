from datetime import datetime, timedelta, timezone
import threading
import time

import pytest

from data_toolkit.pipeline.production_worker import ProductionWorker
from data_toolkit.pipeline.work_queue import ProductionWorkQueue, WorkUnit
from data_toolkit.pipeline.worker_registry import (
    WorkerRegistration,
    WorkerRegistry,
)


NOW = datetime(2026, 7, 22, 5, 0, tzinfo=timezone.utc)


def registration(tmp_path):
    return WorkerRegistration(
        "node17",
        "local",
        44,
        (1, 2, 3, 4, 5, 6),
        tmp_path / "data2",
        tmp_path / "data3",
        tmp_path / "local",
    )


def setup_runtime(tmp_path, *, unit_count=1):
    units = tuple(
        WorkUnit("ABO", "ABO-00000", f"batch{index:03d}", 256)
        for index in range(unit_count)
    )
    queue = ProductionWorkQueue(
        tmp_path / "queue", lease_timeout=timedelta(minutes=5)
    )
    queue.initialize("a" * 64, units, now=NOW)
    registry = WorkerRegistry(tmp_path / "workers.json")
    registry.register(registration(tmp_path), now=NOW)
    return queue, registry, units


def test_active_worker_executes_and_completes_one_claimed_batch(tmp_path):
    queue, registry, units = setup_runtime(tmp_path)
    calls = []
    worker = ProductionWorker(
        queue,
        registry,
        "node17",
        calls.append,
        heartbeat_interval=0.01,
    )

    assert worker.run_once(now=NOW) is True

    assert calls == [units[0]]
    assert queue.status(now=NOW)["completed"] == 1


def test_draining_worker_does_not_claim_new_batch(tmp_path):
    queue, registry, _ = setup_runtime(tmp_path)
    registry.drain("node17")
    worker = ProductionWorker(queue, registry, "node17", lambda unit: None)

    assert worker.run_once(now=NOW) is False
    assert queue.status(now=NOW)["pending"] == 1


def test_failed_batch_is_released_and_next_worker_can_retry(tmp_path):
    queue, registry, units = setup_runtime(tmp_path)
    worker = ProductionWorker(
        queue,
        registry,
        "node17",
        lambda unit: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        worker.run_once(now=NOW)

    lease = queue.claim("node16", now=NOW + timedelta(seconds=1), token="retry")
    assert lease is not None
    assert lease.unit == units[0]
    assert lease.attempt == 2


def test_heartbeat_keeps_long_running_batch_owned(tmp_path):
    queue, registry, _ = setup_runtime(tmp_path)
    started = threading.Event()
    finish = threading.Event()

    def execute(_unit):
        started.set()
        assert finish.wait(timeout=1)

    worker = ProductionWorker(
        queue,
        registry,
        "node17",
        execute,
        heartbeat_interval=0.01,
    )
    thread = threading.Thread(target=worker.run_once, kwargs={"now": NOW})
    thread.start()
    assert started.wait(timeout=1)
    time.sleep(0.04)

    assert queue.status(now=datetime.now(timezone.utc))["running"] == 1
    finish.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
