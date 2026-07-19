from __future__ import annotations

from dataclasses import dataclass
import math
import threading

from .config import ParallelismConfig


@dataclass(frozen=True)
class GeometryProfile:
    processes: int
    native_threads: int

    def __post_init__(self) -> None:
        if type(self.processes) is not int or self.processes <= 0:
            raise ValueError("geometry processes must be a positive integer")
        if type(self.native_threads) is not int or self.native_threads <= 0:
            raise ValueError(
                "geometry native threads must be a positive integer"
            )


def configure_geometry_threads(native_threads: int, *, torch_module=None) -> None:
    if type(native_threads) is not int or native_threads <= 0:
        raise ValueError("geometry native threads must be a positive integer")
    if torch_module is None:
        import torch as torch_module
    torch_module.set_num_threads(native_threads)
    try:
        torch_module.set_num_interop_threads(1)
    except RuntimeError:
        # A reused interpreter may already have initialized inter-op workers.
        # The intra-op cap above still bounds the compute kernels in this child.
        pass


def geometry_profile(config: ParallelismConfig) -> GeometryProfile:
    return GeometryProfile(
        processes=config.cpu_physical_cores,
        native_threads=1,
    )


def geometry_affinity_sets(
    profile: GeometryProfile,
) -> tuple[tuple[int, ...], ...]:
    if profile.processes * profile.native_threads > 44:
        raise ValueError("geometry affinity profile exceeds 44 physical cores")
    physical_cores = (*range(0, 20), *range(24, 48))
    groups = tuple(
        tuple(physical_cores[index : index + profile.native_threads])
        for index in range(0, len(physical_cores), profile.native_threads)
    )
    return groups[: profile.processes]


@dataclass(frozen=True)
class GpuMemoryState:
    index: int
    used_mib: float
    total_mib: float

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("GPU index must be a nonnegative integer")
        if not math.isfinite(self.used_mib) or self.used_mib < 0:
            raise ValueError("GPU memory used must be finite and nonnegative")
        if not math.isfinite(self.total_mib) or self.total_mib <= 0:
            raise ValueError("GPU memory total must be finite and positive")
        if self.used_mib > self.total_mib:
            raise ValueError("GPU memory used cannot exceed total")

    @property
    def percent(self) -> float:
        return self.used_mib * 100.0 / self.total_mib


def select_micro_batch(
    *,
    resolution: int,
    configured: int,
    peak_percent: float,
    oom: bool,
    config: ParallelismConfig,
) -> int:
    if type(resolution) is not int or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    if type(configured) is not int or configured <= 0:
        raise ValueError("configured micro-batch must be a positive integer")
    if not isinstance(peak_percent, (int, float)) or isinstance(
        peak_percent, bool
    ):
        raise ValueError("peak memory percent must be numeric")
    peak_percent = float(peak_percent)
    if not math.isfinite(peak_percent) or peak_percent < 0:
        raise ValueError("peak memory percent must be finite and nonnegative")
    if type(oom) is not bool:
        raise ValueError("oom must be boolean")

    if oom or peak_percent > config.gpu_memory_target_percent:
        return max(1, configured // 2)
    if peak_percent < 70.0:
        return min(config.micro_batch(resolution), configured * 2)
    return configured


class _NodeResourceLease:
    def __init__(
        self,
        broker: "NodeResourceBroker",
        cpu_cores: int,
        gpu_indices: tuple[int, ...],
        gpu_memory_percent: float,
    ) -> None:
        self._broker = broker
        self.cpu_cores = cpu_cores
        self.gpu_indices = gpu_indices
        self.gpu_memory_percent = gpu_memory_percent
        self._released = False
        self._lock = threading.Lock()

    def __enter__(self) -> "_NodeResourceLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._broker._release(self)


class NodeResourceBroker:
    def __init__(
        self,
        *,
        cpu_limit: int,
        gpu_count: int,
        gpu_hard_percent: float = 90.0,
    ) -> None:
        if type(cpu_limit) is not int or cpu_limit <= 0:
            raise ValueError("CPU limit must be a positive integer")
        if type(gpu_count) is not int or gpu_count <= 0:
            raise ValueError("GPU count must be a positive integer")
        if not isinstance(gpu_hard_percent, (int, float)) or isinstance(
            gpu_hard_percent, bool
        ):
            raise ValueError("GPU hard percent must be numeric")
        gpu_hard_percent = float(gpu_hard_percent)
        if not math.isfinite(gpu_hard_percent) or not (
            0 < gpu_hard_percent <= 100
        ):
            raise ValueError("GPU hard percent must be in (0, 100]")

        self.cpu_limit = cpu_limit
        self.gpu_count = gpu_count
        self.gpu_hard_percent = gpu_hard_percent
        self._cpu_allocated = 0
        self._gpu_allocated_percent = [0.0] * gpu_count
        self._lock = threading.Lock()

    def try_acquire(
        self,
        cpu_cores: int,
        gpu_indices: tuple[int, ...],
        gpu_memory_percent: float = 0.0,
    ) -> _NodeResourceLease | None:
        if type(cpu_cores) is not int or cpu_cores < 0:
            raise ValueError("CPU cores must be a nonnegative integer")
        try:
            gpu_indices = tuple(gpu_indices)
        except TypeError as error:
            raise ValueError("GPU indices must be iterable") from error
        if any(type(index) is not int for index in gpu_indices):
            raise ValueError("GPU indices must be integers")
        if len(gpu_indices) != len(set(gpu_indices)):
            raise ValueError("GPU indices must be unique")
        if any(index < 0 or index >= self.gpu_count for index in gpu_indices):
            raise ValueError("GPU index is outside the configured node")
        if not isinstance(gpu_memory_percent, (int, float)) or isinstance(
            gpu_memory_percent, bool
        ):
            raise ValueError("GPU memory percent must be numeric")
        gpu_memory_percent = float(gpu_memory_percent)
        if not math.isfinite(gpu_memory_percent) or gpu_memory_percent < 0:
            raise ValueError(
                "GPU memory percent must be finite and nonnegative"
            )
        if not gpu_indices and gpu_memory_percent:
            raise ValueError("GPU memory cannot be reserved without a GPU")

        with self._lock:
            if self._cpu_allocated + cpu_cores > self.cpu_limit:
                return None
            if any(
                self._gpu_allocated_percent[index] + gpu_memory_percent
                >= self.gpu_hard_percent
                for index in gpu_indices
            ):
                return None
            self._cpu_allocated += cpu_cores
            for index in gpu_indices:
                self._gpu_allocated_percent[index] += gpu_memory_percent
            return _NodeResourceLease(
                self,
                cpu_cores,
                gpu_indices,
                gpu_memory_percent,
            )

    def _release(self, lease: _NodeResourceLease) -> None:
        with self._lock:
            self._cpu_allocated -= lease.cpu_cores
            for index in lease.gpu_indices:
                self._gpu_allocated_percent[index] -= lease.gpu_memory_percent
