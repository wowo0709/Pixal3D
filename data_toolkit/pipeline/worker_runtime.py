from __future__ import annotations

from dataclasses import replace

from .config import PipelineConfig
from .worker_registry import WorkerRegistration


class WorkerExecutionConfig:
    """Node-local execution overrides bound to a canonical pipeline identity."""

    def __init__(
        self, canonical: PipelineConfig, registration: WorkerRegistration
    ) -> None:
        gpu_count = len(registration.gpu_indices)
        self._canonical = canonical
        self.paths = replace(
            canonical.paths,
            data2_root=registration.data2_root,
            data3_root=registration.data3_root,
            local_root=registration.local_root,
        )
        self.parallelism = replace(
            canonical.parallelism,
            gpu_count=gpu_count,
            cpu_physical_cores=registration.cpu_limit,
        )
        self.workers = replace(
            canonical.workers,
            cpu_threads=registration.cpu_limit,
            render_workers=gpu_count,
            encoder_ranks=gpu_count,
        )
        self.worker_tuning = replace(
            canonical.worker_tuning,
            render_workers=gpu_count,
            encoder_ranks=gpu_count,
        )

    def config_hash(self) -> str:
        return self._canonical.config_hash()

    def __getattr__(self, name: str):
        return getattr(self._canonical, name)


def execution_config(
    canonical: PipelineConfig, registration: WorkerRegistration
) -> WorkerExecutionConfig:
    return WorkerExecutionConfig(canonical, registration)
