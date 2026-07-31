import json
from pathlib import Path
import subprocess

import pytest

from data_toolkit.pipeline.hardware import (
    benchmark_storage,
    build_hardware_evidence,
    collect_hardware_preflight,
    render_gpu_cube,
)
from data_toolkit.pipeline.runtime import derive_hardware_report


def test_storage_benchmark_round_trips_and_removes_fixture(tmp_path):
    result = benchmark_storage(
        tmp_path,
        fixture_bytes=12 * 1024,
        chunk_bytes=4 * 1024,
    )

    assert result["fixture_bytes"] == 12 * 1024
    assert result["write_sha256"] == result["read_sha256"]
    assert result["write_elapsed_seconds"] > 0
    assert result["read_elapsed_seconds"] > 0
    assert result["fixture_removed"] is True
    assert not (tmp_path / ".pixal3d-storage-preflight.bin").exists()


def test_gpu_cube_render_is_isolated_and_checksummed(tmp_path):
    calls = []

    def fake_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        arguments = argv[argv.index("--") + 1 :]
        output = Path(arguments[arguments.index("--output") + 1])
        metadata = Path(arguments[arguments.index("--metadata") + 1])
        output.write_bytes(b"non-empty cube render")
        metadata.write_text(
            json.dumps(
                {
                    "selected_devices": ["RTX fixture"],
                    "cpu_fallback_detected": False,
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = render_gpu_cube(
        blender_path=tmp_path / "blender",
        script_path=tmp_path / "cube.py",
        gpu_index=3,
        output_root=tmp_path / "output",
        gpu_name="RTX fixture",
        runner=fake_runner,
    )

    argv, kwargs = calls[0]
    assert argv[:3] == [str(tmp_path / "blender"), "--background", "--factory-startup"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert result["index"] == 3
    assert result["cuda_visible_device"] == "3"
    assert result["cycles_device"] == "OPTIX"
    assert result["cpu_fallback_detected"] is False
    assert len(result["cube_render_sha256"]) == 64


def test_gpu_cube_render_rejects_cpu_fallback(tmp_path):
    def fake_runner(argv, **kwargs):
        arguments = argv[argv.index("--") + 1 :]
        Path(arguments[arguments.index("--output") + 1]).write_bytes(b"render")
        Path(arguments[arguments.index("--metadata") + 1]).write_text(
            json.dumps(
                {
                    "selected_devices": ["CPU"],
                    "cpu_fallback_detected": True,
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(RuntimeError, match="CPU fallback"):
        render_gpu_cube(
            blender_path=tmp_path / "blender",
            script_path=tmp_path / "cube.py",
            gpu_index=0,
            output_root=tmp_path / "output",
            gpu_name="RTX fixture",
            runner=fake_runner,
        )


def test_gpu_cube_render_rejects_multiple_visible_optix_devices(tmp_path):
    def fake_runner(argv, **kwargs):
        arguments = argv[argv.index("--") + 1 :]
        Path(arguments[arguments.index("--output") + 1]).write_bytes(b"render")
        Path(arguments[arguments.index("--metadata") + 1]).write_text(
            json.dumps(
                {
                    "selected_devices": ["GPU fixture 0", "GPU fixture 1"],
                    "cpu_fallback_detected": False,
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(RuntimeError, match="exactly one OptiX device"):
        render_gpu_cube(
            blender_path=tmp_path / "blender",
            script_path=tmp_path / "cube.py",
            gpu_index=0,
            output_root=tmp_path / "output",
            gpu_name="RTX fixture",
            runner=fake_runner,
        )


def test_hardware_evidence_uses_explicit_bootstrap_reservation(config):
    fixture_bytes = 10 * 1024**3
    storage = {
        name: {
            "fixture_bytes": fixture_bytes,
            "write_elapsed_seconds": 10.0,
            "read_elapsed_seconds": 8.0,
            "write_sha256": "a" * 64,
            "read_sha256": "a" * 64,
            "total_bytes": 100 * 1024**4,
            "free_bytes_before": 50 * 1024**4,
            "free_bytes_after": 50 * 1024**4,
            "fixture_removed": True,
        }
        for name in ("local", "data2", "data3")
    }
    gpus = [
        {
            "index": index,
            "name": f"GPU {index}",
            "cuda_visible_device": str(index),
            "cycles_device": "OPTIX",
            "cpu_fallback_detected": False,
            "cube_render_sha256": f"{index + 1:064x}",
        }
        for index in range(7)
    ]
    reservation = 350 * 1024**3

    evidence = build_hardware_evidence(
        config,
        software={
            "cuda_version": "12.8",
            "torch_version": "2.8.0+cu128",
            "blender_version": "4.5.1",
            "optix_enabled": True,
        },
        gpus=gpus,
        storage=storage,
        bootstrap_peak_local_bytes=reservation,
    )

    assert evidence["source_measurements"] == {
        source: [reservation] for source in config.sources
    }
    report = derive_hardware_report(evidence, config, "b" * 64)
    assert report["decision"] == "passed"
    assert all(
        item["p95_peak_local_bytes"] == reservation
        for item in report["pilot_sizing"]["sources"].values()
    )


@pytest.mark.parametrize(
    ("cuda_version", "torch_version"),
    (("12.7", "2.8.0+cu128"), ("12.8", "2.7.1+cu128")),
)
def test_hardware_report_requires_cuda_12_8_and_torch_2_8_or_newer(
    config, cuda_version, torch_version
):
    fixture_bytes = 10 * 1024**3
    storage = {
        name: {
            "fixture_bytes": fixture_bytes,
            "write_elapsed_seconds": 10.0,
            "read_elapsed_seconds": 8.0,
            "write_sha256": "a" * 64,
            "read_sha256": "a" * 64,
            "total_bytes": 100 * 1024**4,
            "free_bytes_before": 50 * 1024**4,
            "free_bytes_after": 50 * 1024**4,
            "fixture_removed": True,
        }
        for name in ("local", "data2", "data3")
    }
    gpus = [
        {
            "index": index,
            "name": f"GPU {index}",
            "cuda_visible_device": str(index),
            "cycles_device": "OPTIX",
            "cpu_fallback_detected": False,
            "cube_render_sha256": f"{index + 1:064x}",
        }
        for index in range(7)
    ]
    evidence = build_hardware_evidence(
        config,
        software={
            "cuda_version": cuda_version,
            "torch_version": torch_version,
            "blender_version": "4.5.1",
            "optix_enabled": True,
        },
        gpus=gpus,
        storage=storage,
        bootstrap_peak_local_bytes=350 * 1024**3,
    )

    report = derive_hardware_report(evidence, config, "e" * 64)

    assert report["decision"] == "failed"


def test_collector_runs_gpu_and_storage_checks_sequentially(
    config, monkeypatch, tmp_path
):
    from dataclasses import replace

    roots = replace(
        config.paths,
        local_root=tmp_path / "local",
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
    )
    config = replace(config, paths=roots)
    events = []

    def fake_installer(root):
        events.append(("install", root))
        return root / "blender-4.5.1-linux-x64/blender"

    def fake_inventory():
        return [(index, f"GPU {index}") for index in range(7)]

    def fake_software(blender_path):
        return {
            "cuda_version": "12.8",
            "torch_version": "2.8.0+cu128",
            "blender_version": "4.5.1",
            "optix_enabled": True,
        }

    def fake_gpu_renderer(**kwargs):
        index = kwargs["gpu_index"]
        events.append(("gpu", index))
        return {
            "index": index,
            "name": kwargs["gpu_name"],
            "cuda_visible_device": str(index),
            "cycles_device": "OPTIX",
            "cpu_fallback_detected": False,
            "cube_render_sha256": f"{index + 1:064x}",
        }

    def fake_storage(root):
        name = {roots.local_root: "local", roots.data2_root: "data2", roots.data3_root: "data3"}[root]
        events.append(("storage", name))
        return {
            "fixture_bytes": 10 * 1024**3,
            "write_elapsed_seconds": 10.0,
            "read_elapsed_seconds": 8.0,
            "write_sha256": "c" * 64,
            "read_sha256": "c" * 64,
            "total_bytes": 100 * 1024**4,
            "free_bytes_before": 50 * 1024**4,
            "free_bytes_after": 50 * 1024**4,
            "fixture_removed": True,
        }

    evidence_path = collect_hardware_preflight(
        config,
        bootstrap_peak_local_bytes=350 * 1024**3,
        blender_installer=fake_installer,
        gpu_inventory=fake_inventory,
        software_inventory=fake_software,
        gpu_renderer=fake_gpu_renderer,
        storage_benchmark=fake_storage,
    )

    assert events == [
        ("install", roots.local_root / "tools"),
        *[("gpu", index) for index in range(7)],
        ("storage", "local"),
        ("storage", "data2"),
        ("storage", "data3"),
    ]
    evidence = json.loads(evidence_path.read_text())
    assert derive_hardware_report(evidence, config, "d" * 64)["decision"] == "passed"
    provenance = json.loads(
        evidence_path.with_name("hardware_sizing_provenance.json").read_text()
    )
    assert provenance["mode"] == "bootstrap_reservation"
    assert provenance["peak_local_bytes_per_asset"] == 350 * 1024**3
