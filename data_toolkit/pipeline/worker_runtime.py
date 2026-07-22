from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Callable, Iterator

from .config import PipelineConfig
from .worker_registry import WorkerRegistration


class WorkerAlreadyRunningError(RuntimeError):
    pass


class WorkerRuntimeError(RuntimeError):
    pass


@contextmanager
def worker_process_lock(local_root: Path, node_id: str) -> Iterator[Path]:
    """Hold one crash-safe, non-blocking worker lock for a node."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node_id) is None:
        raise ValueError(f"invalid worker node id: {node_id!r}")
    lock_root = Path(local_root) / "control/runtime/worker-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{node_id}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkerAlreadyRunningError(
                f"production worker is already running for {node_id}"
            ) from error
        payload = json.dumps(
            {"node_id": node_id, "pid": os.getpid()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _version_pair(value: str, name: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise WorkerRuntimeError(f"cannot determine {name} version: {value!r}")
    return int(match.group(1)), int(match.group(2))


def validate_worker_environment(
    config,
    registration: WorkerRegistration,
    *,
    importer: Callable[[str], object] = importlib.import_module,
    process_runner: Callable = subprocess.run,
) -> None:
    """Fail before claiming work when a node cannot execute every stage."""
    modules = (
        "torch",
        "cv2",
        "easydict",
        "objaverse",
        "cumesh",
        "flex_gemm",
        "nvdiffrast",
        "o_voxel",
        "data_toolkit.datasets.ABO",
        "data_toolkit.datasets.HSSD",
        "data_toolkit.datasets.3D-FUTURE",
        "data_toolkit.datasets.ObjaverseXL",
    )
    loaded = {}
    for name in modules:
        try:
            loaded[name] = importer(name)
        except BaseException as error:
            raise WorkerRuntimeError(
                f"worker dependency is unavailable: {name}: {error}"
            ) from error

    torch = loaded["torch"]
    torch_version = str(getattr(torch, "__version__", ""))
    if _version_pair(torch_version, "PyTorch") < (2, 8):
        raise WorkerRuntimeError(
            f"PyTorch 2.8 or newer is required, found {torch_version!r}"
        )
    cuda_version = str(getattr(getattr(torch, "version", None), "cuda", ""))
    if _version_pair(cuda_version, "CUDA") != (12, 8):
        raise WorkerRuntimeError(
            f"CUDA 12.8 is required, found {cuda_version!r}"
        )
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise WorkerRuntimeError("CUDA is unavailable to the production worker")
    device_count = cuda.device_count()
    if (
        type(device_count) is not int
        or device_count <= 0
        or max(registration.gpu_indices) >= device_count
    ):
        raise WorkerRuntimeError(
            "registered GPU indices are unavailable to the production worker"
        )

    blender_path = (
        config.paths.local_root
        / "tools"
        / f"blender-{config.render.blender_version}-linux-x64"
        / "blender"
    )
    if not blender_path.is_file() or not os.access(blender_path, os.X_OK):
        raise WorkerRuntimeError(
            f"configured Blender binary is unavailable: {blender_path}"
        )
    try:
        result = process_runner(
            [str(blender_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise WorkerRuntimeError(
            f"configured Blender binary is unusable: {blender_path}: {error}"
        ) from error
    first_line = str(result.stdout).splitlines()[0] if result.stdout else ""
    if not first_line.startswith(f"Blender {config.render.blender_version}"):
        raise WorkerRuntimeError(
            f"Blender {config.render.blender_version} is required, "
            f"found {first_line!r}"
        )


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
