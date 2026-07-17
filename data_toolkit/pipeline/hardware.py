from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Callable

import torch

from .atomic_io import atomic_write_json
from .blender import ensure_blender
from .config import PipelineConfig


STORAGE_FIXTURE_NAME = ".pixal3d-storage-preflight.bin"


class HardwarePreflightError(RuntimeError):
    pass


def build_hardware_evidence(
    config: PipelineConfig,
    *,
    software: dict,
    gpus: list[dict],
    storage: dict,
    bootstrap_peak_local_bytes: int,
) -> dict:
    if bootstrap_peak_local_bytes <= 0:
        raise ValueError("bootstrap peak local bytes must be positive")
    return {
        "schema_version": 2,
        "artifact_type": "hardware_preflight_evidence",
        "config_hash": config.config_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": software,
        "gpus": gpus,
        "storage": storage,
        "source_measurements": {
            source: [bootstrap_peak_local_bytes] for source in config.sources
        },
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_storage(
    root: Path,
    *,
    fixture_bytes: int = 10 * 1024**3,
    chunk_bytes: int = 8 * 1024**2,
) -> dict:
    if fixture_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("storage fixture sizes must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fixture = root / STORAGE_FIXTURE_NAME
    if fixture.exists() or fixture.is_symlink():
        raise FileExistsError(f"storage preflight fixture already exists: {fixture}")

    usage_before = shutil.disk_usage(root)
    seed = sha256(b"Pixal3D storage preflight fixture").digest()
    block = (seed * ((chunk_bytes + len(seed) - 1) // len(seed)))[:chunk_bytes]
    write_digest = sha256()
    read_digest = sha256()
    write_elapsed = 0.0
    read_elapsed = 0.0
    try:
        started = time.monotonic()
        with fixture.open("xb") as stream:
            remaining = fixture_bytes
            while remaining:
                chunk = block[: min(remaining, len(block))]
                stream.write(chunk)
                write_digest.update(chunk)
                remaining -= len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        write_elapsed = time.monotonic() - started

        started = time.monotonic()
        with fixture.open("rb") as stream:
            for chunk in iter(lambda: stream.read(chunk_bytes), b""):
                read_digest.update(chunk)
        read_elapsed = time.monotonic() - started
    finally:
        fixture.unlink(missing_ok=True)

    usage_after = shutil.disk_usage(root)
    return {
        "fixture_bytes": fixture_bytes,
        "write_elapsed_seconds": write_elapsed,
        "read_elapsed_seconds": read_elapsed,
        "write_sha256": write_digest.hexdigest(),
        "read_sha256": read_digest.hexdigest(),
        "total_bytes": usage_before.total,
        "free_bytes_before": usage_before.free,
        "free_bytes_after": usage_after.free,
        "fixture_removed": not fixture.exists(),
    }


def render_gpu_cube(
    *,
    blender_path: Path,
    script_path: Path,
    gpu_index: int,
    output_root: Path,
    gpu_name: str,
    runner: Callable | None = None,
) -> dict:
    if gpu_index < 0:
        raise ValueError("GPU index must be non-negative")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run = runner or subprocess.run
    with tempfile.TemporaryDirectory(
        dir=output_root, prefix=f"gpu-{gpu_index}."
    ) as temporary:
        temporary_root = Path(temporary)
        output = temporary_root / "cube.png"
        metadata = temporary_root / "metadata.json"
        argv = [
            str(blender_path),
            "--background",
            "--factory-startup",
            "--python",
            str(script_path),
            "--",
            "--output",
            str(output),
            "--metadata",
            str(metadata),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_index),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        run(
            argv,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"GPU {gpu_index} cube render is empty")
        try:
            details = json.loads(metadata.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"GPU {gpu_index} metadata is invalid") from error
        selected = details.get("selected_devices")
        cpu_fallback = details.get("cpu_fallback_detected")
        if cpu_fallback is not False or not isinstance(selected, list):
            raise RuntimeError(f"GPU {gpu_index} CPU fallback detected")
        if not selected or any("CPU" in str(device).upper() for device in selected):
            raise RuntimeError(f"GPU {gpu_index} CPU fallback detected")
        if len(selected) != 1:
            raise RuntimeError(
                f"GPU {gpu_index} must expose exactly one OptiX device"
            )
        return {
            "index": gpu_index,
            "name": gpu_name,
            "cuda_visible_device": str(gpu_index),
            "cycles_device": "OPTIX",
            "cpu_fallback_detected": False,
            "cube_render_sha256": _file_sha256(output),
        }


def _gpu_inventory() -> list[tuple[int, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    inventory = []
    for line in result.stdout.splitlines():
        index, name = line.split(",", 1)
        inventory.append((int(index.strip()), name.strip()))
    if [index for index, _ in inventory] != list(range(7)):
        raise RuntimeError("hardware preflight requires contiguous GPUs 0 through 6")
    return inventory


def _software_inventory(blender_path: Path) -> dict:
    result = subprocess.run(
        [str(blender_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    first_line = result.stdout.splitlines()[0].strip()
    if not first_line.startswith("Blender "):
        raise RuntimeError("unable to determine Blender version")
    blender_version = first_line.split()[1]
    return {
        "cuda_version": str(torch.version.cuda or ""),
        "torch_version": str(torch.__version__),
        "blender_version": blender_version,
        "optix_enabled": True,
    }


def _collect_hardware_preflight(
    config: PipelineConfig,
    *,
    bootstrap_peak_local_bytes: int,
    blender_installer: Callable | None = None,
    gpu_inventory: Callable | None = None,
    software_inventory: Callable | None = None,
    gpu_renderer: Callable | None = None,
    storage_benchmark: Callable | None = None,
) -> Path:
    install = blender_installer or ensure_blender
    inventory = gpu_inventory or _gpu_inventory
    inspect_software = software_inventory or _software_inventory
    render = gpu_renderer or render_gpu_cube
    benchmark = storage_benchmark or benchmark_storage

    blender_path = install(config.paths.local_root / "tools")
    devices = inventory()
    if len(devices) != 7:
        raise RuntimeError("hardware preflight requires exactly seven GPUs")
    script_path = (
        Path(__file__).resolve().parents[1]
        / "blender_script"
        / "hardware_cube.py"
    )
    gpu_output = config.paths.local_root / "preflight" / "gpu"
    gpus = [
        render(
            blender_path=blender_path,
            script_path=script_path,
            gpu_index=index,
            output_root=gpu_output,
            gpu_name=name,
        )
        for index, name in devices
    ]
    roots = {
        "local": config.paths.local_root,
        "data2": config.paths.data2_root,
        "data3": config.paths.data3_root,
    }
    storage = {name: benchmark(root) for name, root in roots.items()}
    evidence = build_hardware_evidence(
        config,
        software=inspect_software(blender_path),
        gpus=gpus,
        storage=storage,
        bootstrap_peak_local_bytes=bootstrap_peak_local_bytes,
    )
    input_root = config.paths.data2_root / "control" / "report_inputs"
    provenance_path = input_root / "hardware_sizing_provenance.json"
    atomic_write_json(
        provenance_path,
        {
            "config_hash": config.config_hash(),
            "created_at": evidence["created_at"],
            "mode": "bootstrap_reservation",
            "peak_local_bytes_per_asset": bootstrap_peak_local_bytes,
            "sources": list(config.sources),
        },
    )
    evidence_path = input_root / "hardware.json"
    atomic_write_json(evidence_path, evidence)
    return evidence_path


def collect_hardware_preflight(
    config: PipelineConfig,
    *,
    bootstrap_peak_local_bytes: int,
    blender_installer: Callable | None = None,
    gpu_inventory: Callable | None = None,
    software_inventory: Callable | None = None,
    gpu_renderer: Callable | None = None,
    storage_benchmark: Callable | None = None,
) -> Path:
    try:
        return _collect_hardware_preflight(
            config,
            bootstrap_peak_local_bytes=bootstrap_peak_local_bytes,
            blender_installer=blender_installer,
            gpu_inventory=gpu_inventory,
            software_inventory=software_inventory,
            gpu_renderer=gpu_renderer,
            storage_benchmark=storage_benchmark,
        )
    except HardwarePreflightError:
        raise
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise HardwarePreflightError(str(error)) from error
