import os
from pathlib import Path
import stat

import numpy as np
import pytest

from data_toolkit.pipeline import atomic_io
from data_toolkit.pipeline.atomic_io import (
    atomic_copy,
    atomic_save_npz,
    atomic_write_json,
)


def _write_case(case: str, path: Path, value: int) -> None:
    if case == "npz":
        atomic_save_npz(path, value=np.asarray([value], dtype=np.int64))
    elif case == "json":
        atomic_write_json(path, {"value": value})
    else:
        source = path.parent / f"source-{value}"
        source.write_bytes(str(value).encode())
        atomic_copy(source, path)


def test_atomic_npz(tmp_path):
    path = tmp_path / "view00.npz"

    atomic_save_npz(
        path,
        feats=np.ones((2, 3), np.float32),
        coords=np.zeros((2, 3), np.uint8),
    )

    with np.load(path) as data:
        assert data["feats"].shape == (2, 3)
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob("*.tmp.npz")) == []


def test_atomic_json_and_copy_round_trip(tmp_path):
    json_path = tmp_path / "metadata" / "scale.json"
    source = tmp_path / "source.bin"
    destination = tmp_path / "copies" / "source.bin"
    source.write_bytes(b"payload")

    atomic_write_json(json_path, {"scale": 1.25, "name": "fixture"})
    atomic_copy(source, destination)

    assert json_path.read_text() == (
        '{\n  "name": "fixture",\n  "scale": 1.25\n}'
    )
    assert destination.read_bytes() == b"payload"


@pytest.mark.parametrize("case", ["npz", "json", "copy"])
def test_atomic_writers_fsync_and_use_unique_same_directory_temps(
    case, monkeypatch, tmp_path
):
    destination = tmp_path / "nested" / f"output.{case}"
    destination.parent.mkdir()
    events = []
    temporary_paths = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(file_descriptor):
        kind = (
            "directory"
            if stat.S_ISDIR(os.fstat(file_descriptor).st_mode)
            else "file"
        )
        events.append(kind)
        real_fsync(file_descriptor)

    def tracked_replace(source, target):
        events.append("replace")
        temporary_paths.append(Path(source))
        real_replace(source, target)

    monkeypatch.setattr(atomic_io.os, "fsync", tracked_fsync)
    monkeypatch.setattr(atomic_io.os, "replace", tracked_replace)

    _write_case(case, destination, 1)
    _write_case(case, destination, 2)

    assert events == [
        "file",
        "replace",
        "directory",
        "file",
        "replace",
        "directory",
    ]
    assert len(set(temporary_paths)) == 2
    assert all(path.parent == destination.parent for path in temporary_paths)
    assert all(path.suffix == ".tmp" for path in temporary_paths)


@pytest.mark.parametrize("case", ["npz", "json", "copy"])
def test_atomic_writers_keep_destination_and_cleanup_when_replace_fails(
    case, monkeypatch, tmp_path
):
    destination = tmp_path / f"output.{case}"
    destination.write_bytes(b"existing")

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _write_case(case, destination, 1)

    assert destination.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("case", ["npz", "json", "copy"])
def test_atomic_writers_cleanup_when_writing_fails(
    case, monkeypatch, tmp_path
):
    destination = tmp_path / f"output.{case}"

    if case == "npz":
        def fail_write(*args, **kwargs):
            raise OSError("write failed")

        def operation():
            atomic_save_npz(destination, value=np.asarray([1]))

        monkeypatch.setattr(atomic_io.np, "savez_compressed", fail_write)
    elif case == "json":
        def operation():
            atomic_write_json(destination, {"bad": object()})
    else:
        def operation():
            atomic_copy(tmp_path / "missing", destination)

    with pytest.raises((OSError, TypeError, FileNotFoundError)):
        operation()

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))
