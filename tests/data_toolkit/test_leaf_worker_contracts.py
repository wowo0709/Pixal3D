import importlib
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data_toolkit.pipeline.validation import validate_sparse_latent


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


def test_corrupt_sparse_output_is_replaced_by_one_stubbed_asset(tmp_path):
    worker = importlib.import_module("data_toolkit.encode_shape_latent_view")
    output = tmp_path / "view00.npz"
    output.write_bytes(b"not an npz")
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
    )

    assert encoder.calls == 1
    assert token_count == 1
    validate_sparse_latent(output, grid_resolution=16, max_tokens=16**3)
    with np.load(output, allow_pickle=False) as data:
        assert data["feats"].dtype == np.float16
        assert data["coords"].dtype == np.uint8


def test_invalid_sparse_encoder_result_is_not_published(tmp_path):
    worker = importlib.import_module("data_toolkit.encode_shape_latent_view")
    output = tmp_path / "view00.npz"

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
        )

    assert not output.exists()
    assert not list(tmp_path.iterdir())


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


def test_missing_optional_directory_is_an_empty_input(tmp_path):
    worker = importlib.import_module("data_toolkit.build_metadata")

    assert worker._list_optional_directory(tmp_path / "missing") == []
