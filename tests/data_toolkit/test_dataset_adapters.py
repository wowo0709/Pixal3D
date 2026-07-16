import importlib
from hashlib import sha256
import inspect
from io import BytesIO
from pathlib import Path
import stat
import tarfile
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest


def test_adapter_download_contract():
    for name in ("ABO", "HSSD", "3D-FUTURE", "Toys4k", "ObjaverseXL"):
        module = importlib.import_module(f"data_toolkit.datasets.{name}")
        parameters = inspect.signature(module.download).parameters
        assert "metadata" in parameters
        assert "output_dir" in parameters


def test_public_metadata_paths(monkeypatch):
    seen = []
    monkeypatch.setattr(
        pd, "read_csv", lambda path: seen.append(path) or pd.DataFrame()
    )
    for name in ("HSSD", "3D-FUTURE", "Toys4k"):
        importlib.import_module(f"data_toolkit.datasets.{name}").get_metadata()
    assert seen == [
        "hf://datasets/JeffreyXiang/TRELLIS-500K/HSSD.csv",
        "hf://datasets/JeffreyXiang/TRELLIS-500K/3D-FUTURE.csv",
        "hf://datasets/JeffreyXiang/TRELLIS-500K/Toys4k.csv",
    ]


def test_hssd_snapshot_is_bounded_and_only_verified_files_are_returned(
    monkeypatch, tmp_path
):
    module = importlib.import_module("data_toolkit.datasets.HSSD")
    good = b"verified"
    calls = []

    monkeypatch.setattr(module.huggingface_hub, "whoami", lambda: {"name": "test"})

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        root = Path(kwargs["local_dir"])
        (root / "models").mkdir(parents=True)
        (root / "models/good.glb").write_bytes(good)
        (root / "models/bad.glb").write_bytes(b"wrong")

    monkeypatch.setattr(
        module.huggingface_hub, "snapshot_download", snapshot_download
    )
    metadata = pd.DataFrame(
        [
            {
                "file_identifier": "models/good.glb",
                "sha256": sha256(good).hexdigest(),
            },
            {"file_identifier": "models/bad.glb", "sha256": "0" * 64},
        ]
    )

    result = module.download(metadata, str(tmp_path), max_workers=64)

    assert calls == [
        {
            "repo_id": "hssd/hssd-models",
            "repo_type": "dataset",
            "allow_patterns": ["models/good.glb", "models/bad.glb"],
            "local_dir": str(tmp_path / "raw"),
            "max_workers": 8,
        }
    ]
    assert result.to_dict("records") == [
        {
            "sha256": sha256(good).hexdigest(),
            "local_path": "raw/models/good.glb",
        }
    ]


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


def test_3d_future_extracts_only_selected_verified_directory(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.3D-FUTURE")
    selected_image = b"selected image"
    archive_path = tmp_path / "3D-FUTURE-model.zip"
    _write_zip(
        archive_path,
        {
            "3D-FUTURE-model/selected/image.jpg": selected_image,
            "3D-FUTURE-model/selected/raw_model.obj": b"mesh",
            "3D-FUTURE-model/ignored/image.jpg": b"ignored",
            "3D-FUTURE-model/ignored/raw_model.obj": b"ignored mesh",
        },
    )
    metadata = pd.DataFrame(
        [
            {
                "file_identifier": "3D-FUTURE-model/selected",
                "sha256": sha256(selected_image).hexdigest(),
            }
        ]
    )

    result = module.download(metadata, str(tmp_path), max_workers=64)

    assert result.to_dict("records") == [
        {
            "sha256": sha256(selected_image).hexdigest(),
            "local_path": "raw/3D-FUTURE-model/selected/raw_model.obj",
        }
    ]
    assert not (tmp_path / "raw/3D-FUTURE-model/ignored").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_3d_future_rejects_unsafe_zip_members_before_extraction(tmp_path, name):
    module = importlib.import_module("data_toolkit.datasets.3D-FUTURE")
    _write_zip(
        tmp_path / "3D-FUTURE-model.zip",
        {
            "3D-FUTURE-model/selected/image.jpg": b"selected",
            name: b"unsafe",
        },
    )

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        module.download(pd.DataFrame(columns=["file_identifier", "sha256"]), tmp_path)

    assert not (tmp_path / "raw/3D-FUTURE-model").exists()


def test_toys4k_rejects_symlink_member_before_extraction(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.Toys4k")
    archive_path = tmp_path / "toys4k_blend_files.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("toys4k_blend_files/selected.blend", b"blend")
        symlink = zipfile.ZipInfo("toys4k_blend_files/link.blend")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink, "selected.blend")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        module.download(pd.DataFrame(columns=["file_identifier", "sha256"]), tmp_path)

    assert not (tmp_path / "raw/toys4k_blend_files").exists()


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, contents in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, BytesIO(contents))


def test_abo_extracts_only_selected_verified_member(tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ABO")
    raw = tmp_path / "raw"
    raw.mkdir()
    selected = b"selected glb"
    _write_tar(
        raw / "abo-3dmodels.tar",
        {
            "3dmodels/original/selected.glb": selected,
            "3dmodels/original/ignored.glb": b"ignored",
        },
    )
    metadata = pd.DataFrame(
        [
            {
                "file_identifier": "selected.glb",
                "sha256": sha256(selected).hexdigest(),
            }
        ]
    )

    result = module.download(metadata, str(tmp_path), max_workers=64)

    assert result.to_dict("records") == [
        {
            "sha256": sha256(selected).hexdigest(),
            "local_path": "raw/3dmodels/original/selected.glb",
        }
    ]
    assert not (raw / "3dmodels/original/ignored.glb").exists()


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_abo_rejects_tar_links_before_extraction(tmp_path, link_type):
    module = importlib.import_module("data_toolkit.datasets.ABO")
    raw = tmp_path / "raw"
    raw.mkdir()
    with tarfile.open(raw / "abo-3dmodels.tar", "w") as archive:
        info = tarfile.TarInfo("3dmodels/original/link.glb")
        info.type = link_type
        info.linkname = "../../outside"
        archive.addfile(info)

    with pytest.raises(ValueError, match="Unsafe TAR member"):
        module.download(pd.DataFrame(columns=["file_identifier", "sha256"]), tmp_path)


def test_objaverse_download_processes_are_bounded(monkeypatch, tmp_path):
    module = importlib.import_module("data_toolkit.datasets.ObjaverseXL")
    metadata = pd.DataFrame(
        [{"file_identifier": "object.glb", "sha256": "a" * 64}]
    )
    monkeypatch.setattr(module.oxl, "get_annotations", lambda: metadata.copy())
    seen = []

    def download_objects(annotations, **kwargs):
        seen.append(kwargs)
        path = tmp_path / "raw/object.glb"
        path.write_bytes(b"object")
        return {"object.glb": str(path)}

    monkeypatch.setattr(module.oxl, "download_objects", download_objects)

    module.download(metadata, str(tmp_path), max_workers=64)

    assert seen[0]["processes"] == 8


def test_download_wrapper_maps_alias_and_merges_records(monkeypatch, tmp_path):
    module = importlib.import_module("data_toolkit.download")
    root = tmp_path / "source"
    root.mkdir()
    pd.DataFrame(
        [
            {"sha256": "a" * 64, "file_identifier": "a.glb"},
            {"sha256": "b" * 64, "file_identifier": "b.glb"},
        ]
    ).to_csv(root / "metadata.csv", index=False)
    calls = []

    def add_args(parser):
        parser.add_argument("--source", default="sketchfab")

    def download(metadata, output_dir, **kwargs):
        calls.append((metadata.copy(), output_dir, kwargs))
        return metadata[["sha256"]].assign(
            local_path=lambda frame: "raw/" + frame["sha256"] + ".glb"
        )

    adapter = SimpleNamespace(add_args=add_args, download=download)
    imported = []

    def import_module(name):
        imported.append(name)
        return adapter

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    for rank in (0, 1):
        module.main(
            [
                "ObjaverseXL_github",
                "--root",
                str(root),
                "--rank",
                str(rank),
                "--world_size",
                "2",
                "--max_workers",
                "5",
            ]
        )

    assert imported == ["datasets.ObjaverseXL", "datasets.ObjaverseXL"]
    assert [call[2]["source"] for call in calls] == ["github", "github"]
    assert [call[2]["max_workers"] for call in calls] == [5, 5]
    merged = pd.read_csv(root / "raw/metadata.csv")
    assert merged["sha256"].tolist() == ["a" * 64, "b" * 64]
    assert not (root / "raw/metadata.csv.tmp").exists()
