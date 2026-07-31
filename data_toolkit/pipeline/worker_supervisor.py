from __future__ import annotations

from datetime import datetime, timezone
import subprocess
import time
from typing import Callable, Sequence

from .work_queue import ProductionWorkQueue
from .worker_registry import WorkerRegistry


class ProductionWorkerSupervisor:
    """Keep one node worker available while its registration is active."""

    def __init__(
        self,
        queue: ProductionWorkQueue,
        registry: WorkerRegistry,
        node_id: str,
        command: Sequence[str],
        *,
        poll_interval: float = 10.0,
        process_runner: Callable = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] | None = None,
        on_exit: Callable[[int], None] | None = None,
    ) -> None:
        values = tuple(command)
        if not node_id or not values or any(not value for value in values):
            raise ValueError("supervisor node and command must be non-empty")
        if poll_interval <= 0:
            raise ValueError("supervisor poll interval must be positive")
        self.queue = queue
        self.registry = registry
        self.node_id = node_id
        self.command = values
        self.poll_interval = poll_interval
        self.process_runner = process_runner
        self.sleeper = sleeper
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.on_exit = on_exit or (lambda returncode: None)

    def run_cycle(self) -> bool:
        statuses = self.registry.read()
        if self.node_id not in statuses:
            raise ValueError(
                f"production worker is not registered: {self.node_id}"
            )
        status = statuses[self.node_id]
        if status.state == "removed":
            return False
        counts = self.queue.status(now=self.clock())
        if counts["completed"] + counts["failed"] == counts["total"]:
            return False
        if status.state != "active":
            self.sleeper(self.poll_interval)
            return True
        result = self.process_runner(self.command, check=False)
        self.on_exit(result.returncode)
        self.sleeper(self.poll_interval)
        return True

    def run_forever(self) -> None:
        while self.run_cycle():
            pass
