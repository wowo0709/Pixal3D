from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .orchestrator import _atomic_write_bytes_nofollow, _read_regular_bytes_nofollow


HEARTBEAT_TIMEOUT = timedelta(seconds=90)


@dataclass(frozen=True)
class WorkerRegistration:
    node_id: str
    ssh_target: str
    cpu_limit: int
    gpu_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("worker node id must be non-empty")
        if not isinstance(self.ssh_target, str) or not self.ssh_target:
            raise ValueError("worker SSH target must be non-empty")
        if type(self.cpu_limit) is not int or self.cpu_limit <= 0:
            raise ValueError("worker CPU limit must be positive")
        if not self.gpu_indices or any(type(value) is not int or value < 0 for value in self.gpu_indices):
            raise ValueError("worker GPU indices must be nonnegative integers")


@dataclass(frozen=True)
class WorkerStatus:
    registration: WorkerRegistration
    state: str
    heartbeat_at: datetime

    @property
    def node_id(self) -> str:
        return self.registration.node_id

    def healthy(self, *, now: datetime) -> bool:
        return now - self.heartbeat_at <= HEARTBEAT_TIMEOUT


class WorkerRegistry:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, WorkerStatus]:
        payload = _read_regular_bytes_nofollow(self.path, missing_ok=True)
        if payload is None:
            return {}
        value = json.loads(payload)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("invalid worker registry")
        result = {}
        for item in value.get("workers", []):
            registration = WorkerRegistration(
                node_id=item["node_id"], ssh_target=item["ssh_target"],
                cpu_limit=item["cpu_limit"], gpu_indices=tuple(item["gpu_indices"]),
            )
            result[registration.node_id] = WorkerStatus(
                registration=registration, state=item["state"],
                heartbeat_at=datetime.fromisoformat(item["heartbeat_at"]),
            )
        return result

    def register(self, registration: WorkerRegistration, *, now: datetime) -> WorkerStatus:
        workers = self.read()
        status = WorkerStatus(registration, "active", now)
        workers[registration.node_id] = status
        self._write(workers)
        return status

    def heartbeat(self, node_id: str, *, now: datetime) -> WorkerStatus:
        workers = self.read()
        current = workers[node_id]
        status = WorkerStatus(current.registration, current.state, now)
        workers[node_id] = status
        self._write(workers)
        return status

    def drain(self, node_id: str) -> WorkerStatus:
        workers = self.read()
        current = workers[node_id]
        status = WorkerStatus(current.registration, "draining", current.heartbeat_at)
        workers[node_id] = status
        self._write(workers)
        return status

    def _write(self, workers: dict[str, WorkerStatus]) -> None:
        value = {"schema_version": 1, "workers": [
            {**asdict(status.registration), "state": status.state,
             "heartbeat_at": status.heartbeat_at.astimezone(timezone.utc).isoformat()}
            for _, status in sorted(workers.items())
        ]}
        _atomic_write_bytes_nofollow(
            self.path, json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        )
