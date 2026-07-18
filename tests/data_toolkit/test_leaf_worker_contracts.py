import importlib
import json
import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data_toolkit.pipeline.validation import ValidationError, validate_sparse_latent


CPU_SCRIPTS = (
    "dump_mesh.py",
    "dump_pbr.py",
    "dual_grid_view.py",
    "voxelize_pbr_view.py",
)
ENCODERS = (
    "encode_shape_latent_view.py",
    "encode_pbr_latent_view.py",
    "encode_ss_latent_view.py",
)


def _help(script):
    return subprocess.run(
        [sys.executable, f"data_toolkit/{script}", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("script", CPU_SCRIPTS + ENCODERS)
def test_timeout_flag(script):
    result = _help(script)

    assert result.returncode == 0, result.stderr
    assert "--timeout_seconds" in result.stdout


@pytest.mark.parametrize("script", ENCODERS)
def test_encoder_bounds_and_dtype(script):
    result = _help(script)

    assert result.returncode == 0, result.stderr
    assert "--loader_workers" in result.stdout
    assert "--saver_workers" in result.stdout
    assert "--latent_dtype" in result.stdout


@pytest.mark.parametrize("script", ("dual_grid_view.py", "voxelize_pbr_view.py"))
def test_voxel_native_thread_bound(script):
    result = _help(script)

    assert result.returncode == 0, result.stderr
    assert "--native_threads" in result.stdout


def test_build_metadata_help_does_not_import_help_as_a_dataset():
    result = _help("build_metadata.py")

    assert result.returncode == 0, result.stderr
    assert "--from_merged_records" in result.stdout


@pytest.mark.parametrize(
    "module_name",
    (
        "data_toolkit.encode_shape_latent_view",
        "data_toolkit.encode_pbr_latent_view",
    ),
)
@pytest.mark.parametrize("bad_coordinate", (-1, 256))
def test_sparse_coordinates_are_rejected_before_uint8_narrowing(
    tmp_path, module_name, bad_coordinate
):
    worker = importlib.import_module(module_name)
    source_scale = tmp_path / "source_scale.json"
    source_scale.write_text('{"total_scale": 1.0}')
    destination_scale = tmp_path / "view00_scale.json"
    output = tmp_path / "view00.npz"

    def encode():
        return SimpleNamespace(
            feats=torch.tensor([[1.0]], dtype=torch.float32),
            coords=torch.tensor(
                [[0, bad_coordinate, 0, 0]], dtype=torch.int64
            ),
        )

    with pytest.raises(ValueError, match="coordinate"):
        worker._encode_sparse_output(
            output,
            encode,
            grid_resolution=1024,
            latent_dtype="float32",
            scale_source=source_scale,
            scale_destination=destination_scale,
        )

    assert not output.exists()
    assert not destination_scale.exists()
    assert not list(tmp_path.glob(".*.npz"))


def test_corrupt_sparse_output_is_replaced_by_one_stubbed_asset(tmp_path):
    worker = importlib.import_module("data_toolkit.encode_shape_latent_view")
    output = tmp_path / "view00.npz"
    output.write_bytes(b"not an npz")
    source_scale = tmp_path / "source_scale.json"
    source_scale.write_text('{"total_scale": 1.0}')
    destination_scale = tmp_path / "view00_scale.json"
    input_marker = object()

    class StubEncoder:
        calls = 0

        def __call__(self, value):
            assert value is input_marker
            self.calls += 1
            return SimpleNamespace(
                feats=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                coords=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
            )

    encoder = StubEncoder()
    token_count = worker._encode_sparse_output(
        output,
        lambda: encoder(input_marker),
        grid_resolution=16,
        latent_dtype="float16",
        scale_source=source_scale,
        scale_destination=destination_scale,
    )

    assert encoder.calls == 1
    assert token_count == 1
    validate_sparse_latent(output, grid_resolution=16, max_tokens=16**3)
    with np.load(output, allow_pickle=False) as data:
        assert data["feats"].dtype == np.float16
        assert data["coords"].dtype == np.uint8
    assert json.loads(destination_scale.read_text()) == {"total_scale": 1.0}


def test_invalid_sparse_encoder_result_is_not_published(tmp_path):
    worker = importlib.import_module("data_toolkit.encode_shape_latent_view")
    output = tmp_path / "view00.npz"
    source_scale = tmp_path / "source_scale.json"
    source_scale.write_text('{"total_scale": 1.0}')
    destination_scale = tmp_path / "view00_scale.json"

    def encode():
        return SimpleNamespace(
            feats=torch.tensor([[1.0]], dtype=torch.float32),
            coords=torch.tensor([[0, 20, 0, 0]], dtype=torch.int64),
        )

    with pytest.raises(Exception, match="outside grid"):
        worker._encode_sparse_output(
            output,
            encode,
            grid_resolution=16,
            latent_dtype="float32",
            scale_source=source_scale,
            scale_destination=destination_scale,
        )

    assert not output.exists()
    assert not destination_scale.exists()
    assert not list(tmp_path.glob(".*.npz"))


@pytest.mark.parametrize(
    "module_name",
    (
        "data_toolkit.encode_shape_latent_view",
        "data_toolkit.encode_pbr_latent_view",
    ),
)
@pytest.mark.parametrize("source_contents", (None, "not-json"))
def test_sparse_encoder_requires_valid_source_scale(
    tmp_path, module_name, source_contents
):
    worker = importlib.import_module(module_name)
    output = tmp_path / "view00.npz"
    destination_scale = tmp_path / "view00_scale.json"
    source_scale = tmp_path / "source_scale.json"
    if source_contents is not None:
        source_scale.write_text(source_contents)
    encoder_called = False

    def encode():
        nonlocal encoder_called
        encoder_called = True
        raise AssertionError("scale must be checked before encoding")

    with pytest.raises(Exception, match="scale metadata"):
        worker._encode_sparse_output(
            output,
            encode,
            grid_resolution=16,
            latent_dtype="float32",
            scale_source=source_scale,
            scale_destination=destination_scale,
        )

    assert not encoder_called
    assert not output.exists()
    assert not destination_scale.exists()


@pytest.mark.parametrize("source_contents", (None, "not-json"))
def test_ss_encoder_requires_valid_source_scale(tmp_path, source_contents):
    worker = importlib.import_module("data_toolkit.encode_ss_latent_view")
    encoder_called = False
    source_scale = tmp_path / "source_scale.json"
    if source_contents is not None:
        source_scale.write_text(source_contents)

    def encode():
        nonlocal encoder_called
        encoder_called = True
        raise AssertionError("scale must be checked before encoding")

    with pytest.raises(Exception, match="scale metadata"):
        worker._encode_ss_output(
            tmp_path / "view00.npz",
            encode,
            latent_dtype="float32",
            scale_source=source_scale,
            scale_destination=tmp_path / "view00_scale.json",
        )

    assert not encoder_called
    assert not (tmp_path / "view00.npz").exists()
    assert not (tmp_path / "view00_scale.json").exists()


@pytest.mark.parametrize(
    "module_name",
    (
        "data_toolkit.encode_shape_latent_view",
        "data_toolkit.encode_pbr_latent_view",
    ),
)
def test_sparse_pair_cleanup_when_final_reopen_fails(
    monkeypatch, tmp_path, module_name
):
    worker = importlib.import_module(module_name)
    output = tmp_path / "view00.npz"
    source_scale = tmp_path / "source_scale.json"
    source_scale.write_text('{"total_scale": 1.0}')
    destination_scale = tmp_path / "view00_scale.json"
    real_validate = worker.validate_sparse_latent

    def fail_final(path, grid_resolution, max_tokens):
        if Path(path) == output:
            raise ValidationError("injected final reopen failure")
        return real_validate(path, grid_resolution, max_tokens)

    monkeypatch.setattr(worker, "validate_sparse_latent", fail_final)
    z = SimpleNamespace(
        feats=torch.tensor([[1.0]], dtype=torch.float32),
        coords=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
    )

    with pytest.raises(ValidationError, match="injected"):
        worker._encode_sparse_output(
            output,
            lambda: z,
            grid_resolution=16,
            latent_dtype="float32",
            scale_source=source_scale,
            scale_destination=destination_scale,
        )

    assert not output.exists()
    assert not destination_scale.exists()
    assert not list(tmp_path.glob(".*.npz"))


def test_ss_pair_cleanup_when_final_reopen_fails(monkeypatch, tmp_path):
    worker = importlib.import_module("data_toolkit.encode_ss_latent_view")
    output = tmp_path / "view00.npz"
    source_scale = tmp_path / "source_scale.json"
    source_scale.write_text('{"total_scale": 1.0}')
    destination_scale = tmp_path / "view00_scale.json"
    real_validate = worker.validate_ss_latent

    def fail_final(path):
        if Path(path) == output:
            raise ValidationError("injected final reopen failure")
        return real_validate(path)

    monkeypatch.setattr(worker, "validate_ss_latent", fail_final)
    z = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(ValidationError, match="injected"):
        worker._encode_ss_output(
            output,
            lambda: z,
            latent_dtype="float32",
            scale_source=source_scale,
            scale_destination=destination_scale,
        )

    assert not output.exists()
    assert not destination_scale.exists()
    assert not list(tmp_path.glob(".*.npz"))


@pytest.mark.parametrize(
    ("module_name", "function_name", "directory", "record_key"),
    (
        ("data_toolkit.dump_mesh", "_dump_mesh", "mesh_dumps", "mesh_dumped"),
        ("data_toolkit.dump_pbr", "_dump_pbr", "pbr_dumps", "pbr_dumped"),
    ),
)
def test_dump_worker_propagates_timeout_and_reopens_pickle(
    monkeypatch, tmp_path, module_name, function_name, directory, record_key
):
    worker = importlib.import_module(module_name)
    observed = {}

    def fake_run(args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        temporary = Path(args[args.index("--output_path") + 1])
        with temporary.open("wb") as stream:
            pickle.dump({"objects": []}, stream)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    result = getattr(worker, function_name)(
        "fixture.glb", "abc123", str(tmp_path), timeout_seconds=17
    )

    output = tmp_path / directory / "abc123.pickle"
    assert observed["timeout"] == 17
    assert result == {"sha256": "abc123", record_key: True}
    with output.open("rb") as stream:
        assert pickle.load(stream) == {"objects": []}
    assert not list(output.parent.glob(".abc123.pickle.*"))


def test_pbr_failure_record_classifies_official_unsupported_marker():
    worker = importlib.import_module("data_toolkit.dump_pbr")
    asset_sha = "a" * 64

    record = worker._pbr_failure_record(
        asset_sha, "Material is not supported"
    )

    assert record == {
        "sha256": asset_sha,
        "pbr_dumped": False,
        "error_category": "unsupported_shader",
        "error_reason": "Material is not supported",
    }


def test_pbr_dump_returns_parser_evidence_without_publishing_output(
    monkeypatch, tmp_path
):
    worker = importlib.import_module("data_toolkit.dump_pbr")
    asset_sha = "b" * 64

    def reject_material(args, **kwargs):
        temporary = Path(args[args.index("--output_path") + 1])
        Path(f"{temporary}_error.txt").write_text(
            "[['Principled BSDF'], ['Material Output'], "
            "['Normal Map'], ['Image Texture']]"
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(worker.subprocess, "run", reject_material)

    record = worker._dump_pbr(
        "fixture.glb", asset_sha, tmp_path, timeout_seconds=17
    )

    assert record["error_category"] == "unsupported_shader"
    assert record["error_reason"].startswith("Material is not supported: ")
    assert "Normal Map" in record["error_reason"]
    output_dir = tmp_path / "pbr_dumps"
    assert not (output_dir / f"{asset_sha}.pickle").exists()
    assert not list(output_dir.glob(f".{asset_sha}.pickle.*"))


def test_pbr_dump_returns_timeout_evidence(monkeypatch, tmp_path):
    worker = importlib.import_module("data_toolkit.dump_pbr")
    asset_sha = "c" * 64

    def time_out(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(worker.subprocess, "run", time_out)

    record = worker._dump_pbr(
        "fixture.glb", asset_sha, tmp_path, timeout_seconds=17
    )

    assert record == {
        "sha256": asset_sha,
        "pbr_dumped": False,
        "error_category": "timeout",
        "error_reason": "PBR dump timed out after 17 seconds",
    }


@pytest.mark.parametrize(
    ("module_name", "function_name", "directory"),
    (
        ("data_toolkit.dump_mesh", "_dump_mesh", "mesh_dumps"),
        ("data_toolkit.dump_pbr", "_dump_pbr", "pbr_dumps"),
    ),
)
def test_dump_removes_final_when_post_replace_reopen_fails(
    monkeypatch, tmp_path, module_name, function_name, directory
):
    worker = importlib.import_module(module_name)

    def fake_run(args, **kwargs):
        temporary = Path(args[args.index("--output_path") + 1])
        with temporary.open("wb") as stream:
            pickle.dump({"objects": []}, stream)
        return SimpleNamespace(returncode=0)

    real_read = worker._read_pickle

    def fail_final(path):
        path = Path(path)
        if path.name == "abc123.pickle":
            raise ValueError("injected final reopen failure")
        return real_read(path)

    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    monkeypatch.setattr(worker, "_read_pickle", fail_final)

    with pytest.raises(ValueError, match="injected"):
        getattr(worker, function_name)(
            "fixture.glb", "abc123", str(tmp_path), timeout_seconds=17
        )

    output_dir = tmp_path / directory
    assert not (output_dir / "abc123.pickle").exists()
    assert not list(output_dir.glob(".abc123.pickle.*"))


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_voxel_worker_does_not_require_unused_local_path(module_name):
    worker = importlib.import_module(module_name)

    class PathRequiringAdapter:
        @staticmethod
        def _process_instance(args):
            metadatum, output_dir, func = args
            return func(
                str(Path(output_dir) / metadatum["local_path"]),
                metadatum["sha256"],
            )

    result = worker._run_foreach_bounded(
        PathRequiringAdapter,
        pd.DataFrame([{"sha256": "asset"}]),
        "/unused",
        lambda file_path, sha256: {
            "sha256": sha256,
            "source_file_unused": file_path is None,
        },
        max_workers=1,
        desc="fixture",
        timeout_seconds=5,
        requires_local_path=False,
    )

    assert result.to_dict("records") == [
        {"sha256": "asset", "source_file_unused": True}
    ]


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_vxz_writer_uses_native_thread_bound_and_native_temp_suffix(
    monkeypatch, tmp_path, module_name
):
    worker = importlib.import_module(module_name)
    observed = {}

    def fake_write(path, coord, attr, num_threads):
        observed["path"] = Path(path)
        observed["threads"] = num_threads
        Path(path).write_bytes(b"valid-vxz")

    def fake_read(path):
        assert Path(path).read_bytes() == b"valid-vxz"
        return {"num_voxel": 3}

    monkeypatch.setattr(worker.o_voxel.io, "write_vxz", fake_write)
    monkeypatch.setattr(worker.o_voxel.io, "read_vxz_info", fake_read)
    output = tmp_path / "asset" / "view00.vxz"

    info = worker._atomic_write_vxz(output, object(), {}, native_threads=5)

    assert info == {"num_voxel": 3}
    assert observed["threads"] == 5
    assert observed["path"].suffix == ".vxz"
    assert observed["path"].parent == output.parent
    assert output.read_bytes() == b"valid-vxz"
    assert not list(output.parent.glob(".*.vxz"))


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_vxz_pair_cleanup_when_final_reopen_fails(
    monkeypatch, tmp_path, module_name
):
    worker = importlib.import_module(module_name)
    output = tmp_path / "asset" / "view00.vxz"
    scale = output.with_name("view00_scale.json")

    def fake_write(path, coord, attr, num_threads):
        Path(path).write_bytes(b"valid-vxz")

    def fake_read(path):
        if Path(path) == output:
            raise ValueError("injected final reopen failure")
        return {"num_voxel": 3}

    monkeypatch.setattr(worker.o_voxel.io, "write_vxz", fake_write)
    monkeypatch.setattr(worker.o_voxel.io, "read_vxz_info", fake_read)

    with pytest.raises(ValueError, match="injected"):
        worker._publish_vxz_pair(
            output,
            scale,
            {"total_scale": 1.0},
            object(),
            {},
            native_threads=5,
        )

    assert not output.exists()
    assert not scale.exists()
    assert not list(output.parent.glob(".*.vxz"))


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_vxz_pair_rejects_invalid_scale_before_final_marker(
    monkeypatch, tmp_path, module_name
):
    worker = importlib.import_module(module_name)
    output = tmp_path / "asset" / "view00.vxz"
    scale = output.with_name("view00_scale.json")
    writer_called = False

    def fake_write(path, coord, attr, num_threads):
        nonlocal writer_called
        writer_called = True

    monkeypatch.setattr(worker.o_voxel.io, "write_vxz", fake_write)

    with pytest.raises(Exception, match="scale metadata"):
        worker._publish_vxz_pair(
            output,
            scale,
            {"total_scale": float("nan")},
            object(),
            {},
            native_threads=5,
        )

    assert not writer_called
    assert not output.exists()
    assert not scale.exists()


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_voxel_adapter_child_stays_in_inherited_supervisor_group(
    tmp_path, monkeypatch, module_name
):
    worker = importlib.import_module(module_name)
    result_path = tmp_path / "result.pickle"
    error_path = tmp_path / "error.txt"

    class Adapter:
        @staticmethod
        def foreach_instance(*args, **kwargs):
            raise AssertionError("nested adapter executor must not be used")

        @staticmethod
        def _process_instance(args):
            metadatum, output_dir, func = args
            return func(
                os.path.join(output_dir, metadatum["local_path"]),
                metadatum["sha256"],
            )

    monkeypatch.setattr(
        worker.os,
        "setsid",
        lambda: (_ for _ in ()).throw(
            AssertionError("nested session escaped supervisor group")
        ),
    )

    worker._foreach_child(
        result_path,
        error_path,
        Adapter,
        pd.DataFrame([{"sha256": "asset", "local_path": "raw/asset.glb"}]),
        tmp_path,
        lambda path, sha256: {
            "sha256": sha256,
            "path": os.fspath(path),
            "processed": True,
        },
        "fixture",
    )

    assert result_path.is_file()
    assert not error_path.exists()
    with result_path.open("rb") as stream:
        result = pickle.load(stream)
    assert result.to_dict("records") == [
        {
            "sha256": "asset",
            "path": os.fspath(tmp_path / "raw/asset.glb"),
            "processed": True,
        }
    ]


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_voxel_adapter_timeout_uses_bounded_direct_pidfd_cleanup(
    monkeypatch, module_name
):
    worker = importlib.import_module(module_name)
    sent = []
    closed = []

    class StubbornProcess:
        pid = 700

        def __init__(self):
            self.alive = True
            self.joins = []

        def join(self, timeout):
            self.joins.append(timeout)

        def is_alive(self):
            return self.alive

    process = StubbornProcess()
    monkeypatch.setattr(worker, "_pidfd_open", lambda pid: 81)

    def send_signal(pidfd, sent_signal, siginfo=None, flags=0):
        sent.append((pidfd, sent_signal))
        if sent_signal == signal.SIGKILL:
            process.alive = False

    monkeypatch.setattr(worker, "_pidfd_send_signal", send_signal)
    monkeypatch.setattr(worker.os, "close", closed.append)

    worker._terminate_process(process)

    assert sent == [(81, signal.SIGTERM), (81, signal.SIGKILL)]
    assert process.joins == [0.5, 0.5]
    assert closed == [81]


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_voxel_adapter_timeout_kills_process_group(
    tmp_path, module_name
):
    worker = importlib.import_module(module_name)
    pid_path = tmp_path / f"{module_name.rsplit('.', 1)[-1]}.pid"

    class HangingAdapter:
        @staticmethod
        def _process_instance(args):
            pid_path.write_text(str(os.getpid()))
            while True:
                time.sleep(1)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        worker._run_foreach_bounded(
            HangingAdapter,
            pd.DataFrame([{"sha256": "asset"}]),
            str(tmp_path),
            lambda *args: None,
            max_workers=1,
            desc="fixture",
            timeout_seconds=0.25,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    child_pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_voxel_adapter_timeout_is_per_asset(module_name):
    worker = importlib.import_module(module_name)

    class ProgressiveAdapter:
        @staticmethod
        def _process_instance(args):
            metadatum, output_dir, func = args
            time.sleep(0.12)
            return {"sha256": metadatum["sha256"], "processed": True}

    started = time.monotonic()
    result = worker._run_foreach_bounded(
        ProgressiveAdapter,
        pd.DataFrame([{"sha256": "first"}, {"sha256": "second"}]),
        None,
        lambda *args: None,
        max_workers=1,
        desc="fixture",
        timeout_seconds=0.2,
    )
    elapsed = time.monotonic() - started

    assert elapsed > 0.2
    assert result.to_dict("records") == [
        {"sha256": "first", "processed": True},
        {"sha256": "second", "processed": True},
    ]


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_voxel_adapter_timeout_isolates_healthy_asset(
    tmp_path, module_name
):
    worker = importlib.import_module(module_name)
    healthy_path = tmp_path / "healthy.txt"
    hanging_pid_path = tmp_path / "hanging.pid"

    class MixedAdapter:
        @staticmethod
        def _process_instance(args):
            metadatum, output_dir, func = args
            sha256 = metadatum["sha256"]
            if sha256 == "hang":
                hanging_pid_path.write_text(str(os.getpid()))
                while True:
                    time.sleep(1)
            healthy_path.write_text(sha256)
            return {"sha256": sha256, "processed": True}

    with pytest.raises(TimeoutError, match="hang"):
        worker._run_foreach_bounded(
            MixedAdapter,
            pd.DataFrame([{"sha256": "healthy"}, {"sha256": "hang"}]),
            str(tmp_path),
            lambda *args: None,
            max_workers=2,
            desc="fixture",
            timeout_seconds=0.2,
        )

    assert healthy_path.read_text() == "healthy"
    hanging_pid = int(hanging_pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(hanging_pid, 0)


@pytest.mark.parametrize(
    "module_name",
    ("data_toolkit.dual_grid_view", "data_toolkit.voxelize_pbr_view"),
)
def test_voxel_adapter_never_exceeds_configured_workers(module_name):
    worker = importlib.import_module(module_name)
    context = multiprocessing.get_context("fork")
    active = context.Value("i", 0)
    peak = context.Value("i", 0)

    class ConcurrencyAdapter:
        @staticmethod
        def _process_instance(args):
            metadatum, output_dir, func = args
            with active.get_lock():
                active.value += 1
                peak.value = max(peak.value, active.value)
            try:
                time.sleep(0.12)
                return {"sha256": metadatum["sha256"], "processed": True}
            finally:
                with active.get_lock():
                    active.value -= 1

    result = worker._run_foreach_bounded(
        ConcurrencyAdapter,
        pd.DataFrame([{"sha256": f"asset-{index}"} for index in range(6)]),
        None,
        lambda *args: None,
        max_workers=2,
        desc="fixture",
        timeout_seconds=0.5,
    )

    assert len(result) == 6
    assert peak.value == 2


@pytest.mark.parametrize(
    "module_name",
    (
        "data_toolkit.encode_shape_latent_view",
        "data_toolkit.encode_pbr_latent_view",
        "data_toolkit.encode_ss_latent_view",
    ),
)
def test_encoder_pipeline_returns_boundedly_from_stuck_loader(module_name):
    worker = importlib.import_module(module_name)
    release = threading.Event()

    def load(task, cancel):
        release.wait()
        return None, None

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="loader"):
            worker._run_bounded_pipeline(
                [("asset", 0)],
                load=load,
                process=lambda task, payload: payload,
                save=lambda task, payload, cancel: None,
                cleanup=lambda task: None,
                loader_workers=1,
                saver_workers=1,
                timeout_seconds=0.2,
            )
    finally:
        release.set()

    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize(
    "module_name",
    (
        "data_toolkit.encode_shape_latent_view",
        "data_toolkit.encode_pbr_latent_view",
        "data_toolkit.encode_ss_latent_view",
    ),
)
def test_encoder_pipeline_cancels_stuck_saver_without_publication(
    tmp_path, module_name
):
    worker = importlib.import_module(module_name)
    release = threading.Event()
    marker = tmp_path / "published"
    cleaned = []

    def save(task, payload, cancel):
        release.wait()
        if not cancel.is_set():
            marker.write_text("published")
        return {"sha256": task[0]}

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="saver"):
            worker._run_bounded_pipeline(
                [("asset", 0)],
                load=lambda task, cancel: (object(), None),
                process=lambda task, payload: payload,
                save=save,
                cleanup=lambda task: cleaned.append(task),
                loader_workers=1,
                saver_workers=1,
                timeout_seconds=0.2,
            )
    finally:
        release.set()
        time.sleep(0.05)

    assert time.monotonic() - started < 1.0
    assert cleaned == [("asset", 0)]
    assert not marker.exists()


def test_build_metadata_reads_merged_records_from_the_selected_directory(tmp_path):
    worker = importlib.import_module("data_toolkit.build_metadata")
    path = tmp_path / "mesh_dumps"
    (path / "new_records").mkdir(parents=True)
    (path / "merged_records").mkdir()
    pd.DataFrame([{"sha256": "new", "value": 1}]).to_csv(
        path / "new_records" / "part_0.csv", index=False
    )
    merged = path / "merged_records" / "200_part_0.csv"
    pd.DataFrame([{"sha256": "merged", "value": 2}]).to_csv(
        merged, index=False
    )
    opt = SimpleNamespace(from_merged_records=True, record_start=100)

    metadata = worker.update_metadata(path, opt)

    assert list(metadata.index) == ["merged"]
    assert merged.exists()
    assert not (path / "metadata.csv.tmp").exists()


def test_merged_records_are_applied_in_sorted_order(tmp_path):
    worker = importlib.import_module("data_toolkit.build_metadata")
    path = tmp_path / "mesh_dumps"
    merged = path / "merged_records"
    merged.mkdir(parents=True)
    (path / "new_records").mkdir()
    pd.DataFrame([{"sha256": "same", "value": "newer"}]).to_csv(
        merged / "200_part_0.csv", index=False
    )
    pd.DataFrame([{"sha256": "same", "value": "older"}]).to_csv(
        merged / "100_part_0.csv", index=False
    )
    opt = SimpleNamespace(from_merged_records=True, record_start=0)

    metadata = worker.update_metadata(path, opt)

    assert metadata.loc["same", "value"] == "newer"


def test_missing_optional_directory_is_an_empty_input(tmp_path):
    worker = importlib.import_module("data_toolkit.build_metadata")

    assert worker._list_optional_directory(tmp_path / "missing") == []
