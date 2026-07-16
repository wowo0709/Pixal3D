from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import os
from pathlib import Path
import subprocess
from typing import Callable

import psutil

from .config import LimitConfig, PipelineConfig


GIB = 1024**3
TIB = 1024**4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class GpuMetric:
    index: int
    utilization_percent: float
    memory_used_mib: float
    temperature_celsius: float
    power_watts: float


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: datetime
    cpu_percent: float
    load_1m: float
    io_wait_percent: float
    available_ram_gib: float
    swap_in_bytes: int
    local_free_gib: float
    local_free_percent: float
    data2_project_tib: float
    data2_fs_free_tib: float
    data3_project_tib: float
    data3_fs_free_tib: float
    gpu_metrics: tuple[GpuMetric, ...] = ()
    gpu_query_error: str | None = None


def _directory_size(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += (Path(directory) / filename).stat().st_size
            except FileNotFoundError:
                continue
    return total


class ProjectStorageAccounting:
    """Cached registry totals, reconciled against disk only at shard boundaries."""

    def __init__(
        self,
        data2_root: Path,
        data3_root: Path,
        *,
        data2_bytes: int = 0,
        data3_bytes: int = 0,
        directory_size: Callable[[Path], int] = _directory_size,
    ):
        self.data2_root = Path(data2_root)
        self.data3_root = Path(data3_root)
        self._data2_bytes = data2_bytes
        self._data3_bytes = data3_bytes
        self._directory_size = directory_size

    def current_bytes(self) -> tuple[int, int]:
        return self._data2_bytes, self._data3_bytes

    def record_registry_delta(self, path: Path, delta_bytes: int) -> None:
        candidate = Path(path).resolve()
        for attribute, root in (
            ("_data2_bytes", self.data2_root.resolve()),
            ("_data3_bytes", self.data3_root.resolve()),
        ):
            if candidate == root or root in candidate.parents:
                updated = getattr(self, attribute) + delta_bytes
                if updated < 0:
                    raise ValueError("negative project accounting")
                setattr(self, attribute, updated)
                return
        raise ValueError(f"path outside configured project roots: {path}")

    def reconcile_at_shard_boundary(self) -> tuple[int, int]:
        self._data2_bytes = self._directory_size(self.data2_root)
        self._data3_bytes = self._directory_size(self.data3_root)
        return self.current_bytes()


class ResourceSampler:
    GPU_QUERY = (
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )

    def __init__(
        self,
        config: PipelineConfig,
        project_accounting: ProjectStorageAccounting,
        *,
        psutil_api=psutil,
        gpu_runner=subprocess.run,
        clock: Callable[[], datetime] = _utc_now,
    ):
        self.config = config
        self.project_accounting = project_accounting
        self.psutil = psutil_api
        self.gpu_runner = gpu_runner
        self.clock = clock
        self._previous_swap_in: int | None = None

    def _gpu_metrics(self) -> tuple[tuple[GpuMetric, ...], str | None]:
        try:
            completed = self.gpu_runner(
                list(self.GPU_QUERY), capture_output=True, text=True, check=True
            )
            metrics = []
            for line in completed.stdout.splitlines():
                if not line.strip():
                    continue
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 5:
                    raise ValueError(f"unexpected nvidia-smi row: {line!r}")
                metrics.append(
                    GpuMetric(
                        index=int(fields[0]),
                        utilization_percent=float(fields[1]),
                        memory_used_mib=float(fields[2]),
                        temperature_celsius=float(fields[3]),
                        power_watts=float(fields[4]),
                    )
                )
            return tuple(metrics), None
        except Exception as error:
            return (), str(error) or type(error).__name__

    def reconcile_at_shard_boundary(self) -> tuple[int, int]:
        return self.project_accounting.reconcile_at_shard_boundary()

    def __call__(self) -> ResourceSnapshot:
        paths = self.config.paths
        cpu_times = self.psutil.cpu_times_percent(interval=None)
        memory = self.psutil.virtual_memory()
        swap_in = self.psutil.swap_memory().sin
        swap_delta = (
            0
            if self._previous_swap_in is None
            else max(0, swap_in - self._previous_swap_in)
        )
        self._previous_swap_in = swap_in
        local_usage = self.psutil.disk_usage(paths.local_root)
        data2_usage = self.psutil.disk_usage(paths.data2_root)
        data3_usage = self.psutil.disk_usage(paths.data3_root)
        data2_bytes, data3_bytes = self.project_accounting.current_bytes()
        gpu_metrics, gpu_error = self._gpu_metrics()
        local_free_percent = (
            100.0 * local_usage.free / local_usage.total if local_usage.total else 0.0
        )
        return ResourceSnapshot(
            timestamp=self.clock(),
            cpu_percent=self.psutil.cpu_percent(interval=None),
            load_1m=self.psutil.getloadavg()[0],
            io_wait_percent=getattr(cpu_times, "iowait", 0.0),
            available_ram_gib=memory.available / GIB,
            swap_in_bytes=swap_delta,
            local_free_gib=local_usage.free / GIB,
            local_free_percent=local_free_percent,
            data2_project_tib=data2_bytes / TIB,
            data2_fs_free_tib=data2_usage.free / TIB,
            data3_project_tib=data3_bytes / TIB,
            data3_fs_free_tib=data3_usage.free / TIB,
            gpu_metrics=gpu_metrics,
            gpu_query_error=gpu_error,
        )


class ResourceAction(str, Enum):
    RUN = "run"
    PAUSE = "pause"
    STOP = "stop"


@dataclass(frozen=True)
class ResourceDecision:
    action: ResourceAction
    reasons: tuple[str, ...]


class ResourceLimitExceeded(RuntimeError):
    def __init__(self, reasons: tuple[str, ...]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def _snapshot_payload(snapshot: ResourceSnapshot) -> dict:
    payload = asdict(snapshot)
    payload["timestamp"] = snapshot.timestamp.isoformat()
    return payload


class TelemetryWriter:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
        sync_interval: timedelta = timedelta(seconds=30),
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        self._clock = clock
        self._sync_interval = sync_interval
        self._last_sync = clock()
        self._closed = False

    def _sync(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._last_sync = self._clock()

    def write(
        self,
        snapshot: ResourceSnapshot,
        decision: ResourceDecision,
        shard_id: str,
        command: str,
    ) -> None:
        if self._closed:
            raise ValueError("telemetry writer is closed")
        payload = _snapshot_payload(snapshot)
        payload.update(
            shard_id=shard_id,
            command=command,
            action=decision.action.value,
            reasons=decision.reasons,
        )
        self._stream.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._clock() - self._last_sync >= self._sync_interval:
            self._sync()

    def close(self) -> None:
        if not self._closed:
            self._sync()
            self._stream.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class ResourceGuard:
    def __init__(
        self,
        sample: Callable[[], ResourceSnapshot],
        policy: "ResourcePolicy",
        telemetry_writer: TelemetryWriter,
        clock: Callable[[], datetime],
        sleeper: Callable[[float], None],
    ):
        self.sample = sample
        self.policy = policy
        self.telemetry_writer = telemetry_writer
        self.clock = clock
        self.sleeper = sleeper
        self._snapshots = deque(maxlen=60)
        self._recovery_required = False
        self._stable_since: datetime | None = None

    def check(self, shard_id: str, command: str) -> ResourceDecision:
        snapshot = self.sample()
        decision = self.policy.evaluate(snapshot)
        now = self.clock()
        if decision.action == ResourceAction.RUN:
            if self._recovery_required:
                if self._stable_since is None:
                    self._stable_since = now
                if now - self._stable_since < timedelta(minutes=5):
                    decision = ResourceDecision(
                        ResourceAction.PAUSE, ("resource recovery period",)
                    )
                else:
                    self._recovery_required = False
                    self._stable_since = None
        else:
            self._recovery_required = True
            self._stable_since = None
        self._snapshots.append(snapshot)
        self.telemetry_writer.write(snapshot, decision, shard_id, command)
        return decision

    def wait_for_admission(
        self, shard_id: str, command: str
    ) -> ResourceDecision:
        while True:
            decision = self.check(shard_id, command)
            if decision.action == ResourceAction.STOP:
                raise ResourceLimitExceeded(decision.reasons)
            if decision.action == ResourceAction.RUN:
                return decision
            self.sleeper(5)

    def last_five_minutes(self) -> tuple[dict, ...]:
        return tuple(_snapshot_payload(snapshot) for snapshot in self._snapshots)


class ResourcePolicy:
    def __init__(self, limits: LimitConfig):
        self.limits = limits
        self.first_seen: dict[str, datetime] = {}

    def duration(self, key: str, active: bool, now: datetime) -> timedelta:
        if not active:
            self.first_seen.pop(key, None)
            return timedelta()
        self.first_seen.setdefault(key, now)
        return now - self.first_seen[key]

    def evaluate(self, value: ResourceSnapshot) -> ResourceDecision:
        now = value.timestamp
        hard = []
        soft = []
        if (
            value.local_free_gib < self.limits.local_free_gib
            or value.local_free_percent < self.limits.local_free_percent
        ):
            hard.append("local free-space floor")
        if value.data2_project_tib >= self.limits.data2_hard_tib:
            hard.append("data2 hard project limit")
        if value.data2_fs_free_tib < self.limits.data2_fs_free_tib:
            hard.append("data2 filesystem free-space floor")
        if value.data3_fs_free_tib < self.limits.data3_fs_free_tib:
            hard.append("data3 filesystem free-space floor")
        if value.available_ram_gib < self.limits.ram_hard_available_gib:
            hard.append("RAM hard floor")
        if self.duration(
            "cpu_hard", value.cpu_percent > self.limits.cpu_hard_percent, now
        ) >= timedelta(minutes=5):
            hard.append("CPU hard duration")
        if self.duration(
            "cpu_soft", value.cpu_percent > self.limits.cpu_soft_percent, now
        ) >= timedelta(minutes=2):
            soft.append("CPU soft duration")
        if self.duration("load", value.load_1m > self.limits.load_soft, now) >= timedelta(
            minutes=2
        ):
            soft.append("load soft duration")
        if self.duration(
            "iowait", value.io_wait_percent > self.limits.io_wait_soft_percent, now
        ) >= timedelta(minutes=2):
            soft.append("I/O wait")
        if value.available_ram_gib < self.limits.ram_soft_available_gib:
            soft.append("RAM soft floor")
        if value.swap_in_bytes > 0:
            soft.append("swap-in activity")
        if (
            value.data2_project_tib >= self.limits.data2_soft_tib
            or value.data3_project_tib >= self.limits.data3_soft_tib
        ):
            soft.append("project storage soft limit")
        action = ResourceAction.STOP if hard else ResourceAction.PAUSE if soft else ResourceAction.RUN
        return ResourceDecision(action, tuple(hard or soft))
