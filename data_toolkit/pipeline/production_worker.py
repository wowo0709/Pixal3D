from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
from typing import Callable

from .work_queue import ProductionWorkQueue, WorkLease, WorkUnit
from .worker_registry import WorkerRegistry


class ProductionWorker:
    """Claim and execute independent frozen production batches for one node."""

    def __init__(
        self,
        queue: ProductionWorkQueue,
        registry: WorkerRegistry,
        node_id: str,
        execute: Callable[[WorkUnit], None],
        *,
        heartbeat_interval: float = 30.0,
        poll_interval: float = 5.0,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if heartbeat_interval <= 0 or poll_interval <= 0:
            raise ValueError("worker intervals must be positive")
        self.queue = queue
        self.registry = registry
        self.node_id = node_id
        self.execute = execute
        self.heartbeat_interval = heartbeat_interval
        self.poll_interval = poll_interval
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleeper = sleeper

    def run_once(self, *, now: datetime | None = None) -> bool:
        claimed_at = now or self.clock()
        statuses = self.registry.read()
        if self.node_id not in statuses:
            raise ValueError(f"production worker is not registered: {self.node_id}")
        if statuses[self.node_id].state != "active":
            return False
        self.registry.heartbeat(self.node_id, now=claimed_at)
        lease = self.queue.claim(self.node_id, now=claimed_at)
        if lease is None:
            return False

        stopped = threading.Event()
        heartbeat_errors: list[BaseException] = []
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(lease, stopped, heartbeat_errors),
            name=f"pixal3d-heartbeat-{self.node_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            self.execute(lease.unit)
            if heartbeat_errors:
                raise heartbeat_errors[0]
        except BaseException as error:
            stopped.set()
            heartbeat.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self.queue.release(
                lease,
                reason=f"{type(error).__name__}: {error}"[:1000],
                now=self.clock(),
            )
            raise
        stopped.set()
        heartbeat.join(timeout=max(1.0, self.heartbeat_interval * 2))
        if heartbeat.is_alive():
            raise RuntimeError("production worker heartbeat did not stop")
        self.queue.complete(lease, now=self.clock())
        self.registry.heartbeat(self.node_id, now=self.clock())
        return True

    def run_forever(self) -> None:
        while True:
            status = self.registry.read().get(self.node_id)
            if status is None:
                raise ValueError(
                    f"production worker is not registered: {self.node_id}"
                )
            if status.state in {"draining", "removed", "cordoned"}:
                return
            try:
                worked = self.run_once()
            except Exception:
                self.sleeper(self.poll_interval)
                continue
            if worked:
                continue
            queue_status = self.queue.status(now=self.clock())
            if (
                queue_status["completed"] + queue_status["failed"]
                == queue_status["total"]
            ):
                return
            self.sleeper(self.poll_interval)

    def _heartbeat_loop(
        self,
        lease: WorkLease,
        stopped: threading.Event,
        errors: list[BaseException],
    ) -> None:
        current = lease
        while not stopped.wait(self.heartbeat_interval):
            try:
                now = self.clock()
                current = self.queue.heartbeat(
                    current, stage="running", now=now
                )
                self.registry.heartbeat(self.node_id, now=now)
            except BaseException as error:
                errors.append(error)
                stopped.set()
                return
