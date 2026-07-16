from dataclasses import asdict, replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from data_toolkit.pipeline import packing
from data_toolkit.pipeline.packing import (
    PACK_FAMILIES,
    PackManifest,
    PackMember,
    build_pack,
    publish_pack,
    verify_pack,
)
from data_toolkit.pipeline.validation import ValidationError


PACK_METADATA = {
    "batch_id": "batch000",
    "family": "common",
    "config_hash": "c" * 64,
    "tool_commit": "abc123",
    "asset_sha256s": ("a" * 64,),
    "completed_count": 1,
    "quarantined_count": 0,
}


def _write_tar(path: Path, members: list[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as bundle:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            info.mtime = 0
            if kind == "file":
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                bundle.addfile(info)
            else:
                raise AssertionError(f"unsupported test member kind: {kind}")


def _write_manifest(path: Path, pack_path: Path, names: list[str]) -> None:
    members = tuple(
        PackMember(name, 1, sha256(b"x").hexdigest()) for name in names
    )
    manifest = PackManifest(
        shard_id="ABO-00000",
        batch_id="batch000",
        family="common",
        config_hash="c" * 64,
        tool_commit="abc123",
        asset_sha256s=("a" * 64,),
        completed_count=1,
        quarantined_count=0,
        created_at="2026-07-16T00:00:00+00:00",
        validated_at="",
        pack_sha256=sha256(pack_path.read_bytes()).hexdigest(),
        members=members,
    )
    path.write_text(json.dumps(asdict(manifest)))


def _family_members(source: Path) -> dict[str, list[Path]]:
    result = {}
    for family in PACK_FAMILIES:
        relative = Path(family) / "payload.bin"
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(family.encode())
        result[family] = [relative]
    return result


def _publish(
    data2_root: Path,
    source: Path,
    family_members: dict[str, list[Path]],
    **overrides,
):
    metadata = {
        "batch_id": "batch000",
        "source": "ABO",
        "config_hash": "c" * 64,
        "tool_commit": "abc123",
        "asset_sha256s": ("a" * 64,),
        "completed_count": 1,
        "quarantined_count": 0,
    }
    metadata.update(overrides)
    return publish_pack(
        data2_root,
        source,
        family_members,
        "ABO-00000",
        **metadata,
    )


def test_pack_is_deterministic(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_text("a")
    (source / "b").write_text("b")

    first = build_pack(
        source,
        [Path("b"), Path("a")],
        tmp_path / "one.tar",
        "ABO-00000",
        **PACK_METADATA,
    )
    second = build_pack(
        source,
        [Path("a"), Path("b")],
        tmp_path / "two.tar",
        "ABO-00000",
        **PACK_METADATA,
    )

    assert first.pack_sha256 == second.pack_sha256
    verify_pack(tmp_path / "one.tar", tmp_path / "one.tar.manifest.json")


def test_symlink_is_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "target").write_text("x")
    (source / "link").symlink_to("target")

    with pytest.raises(ValueError, match="symlink"):
        build_pack(
            source,
            [Path("link")],
            tmp_path / "bad.tar",
            "ABO-00000",
            **PACK_METADATA,
        )

    assert not (tmp_path / "bad.tar").exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("member", [Path("../escape"), Path("/absolute")])
def test_unsafe_source_member_is_rejected(tmp_path: Path, member: Path):
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="unsafe member"):
        build_pack(
            source,
            [member],
            tmp_path / "bad.tar",
            "ABO-00000",
            **PACK_METADATA,
        )


def test_duplicate_source_member_is_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_text("a")

    with pytest.raises(ValueError, match="duplicate member"):
        build_pack(
            source,
            [Path("a"), Path("a")],
            tmp_path / "bad.tar",
            "ABO-00000",
            **PACK_METADATA,
        )


@pytest.mark.parametrize(
    "members, message",
    [
        ([("../escape", b"x", "file")], "unsafe tar member"),
        ([("link", b"", "symlink")], "unsafe tar member"),
        (
            [("same", b"x", "file"), ("same", b"x", "file")],
            "duplicate tar member",
        ),
    ],
)
def test_verify_rejects_unsafe_or_duplicate_tar_members(
    tmp_path: Path,
    members: list[tuple[str, bytes, str]],
    message: str,
):
    pack_path = tmp_path / "bad.tar"
    manifest_path = tmp_path / "bad.tar.manifest.json"
    _write_tar(pack_path, members)
    manifest_names = list(dict.fromkeys(name for name, _, _ in members))
    if manifest_names == ["../escape"]:
        manifest_names = ["safe"]
    _write_manifest(manifest_path, pack_path, manifest_names)

    with pytest.raises(ValidationError, match=message):
        verify_pack(pack_path, manifest_path)


def test_verify_rejects_duplicate_manifest_members(tmp_path: Path):
    pack_path = tmp_path / "bad.tar"
    manifest_path = tmp_path / "bad.tar.manifest.json"
    _write_tar(pack_path, [("same", b"x", "file")])
    _write_manifest(manifest_path, pack_path, ["same", "same"])

    with pytest.raises(ValidationError, match="duplicate manifest member"):
        verify_pack(pack_path, manifest_path)


def test_build_failure_preserves_existing_pack_and_cleans_temporary(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_text("a")
    output = tmp_path / "pack.tar"
    output.write_bytes(b"existing")

    def fail_replace(source_path, destination_path):
        raise OSError("replace failed")

    monkeypatch.setattr(packing.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        build_pack(
            source,
            [Path("a")],
            output,
            "ABO-00000",
            **PACK_METADATA,
        )

    assert output.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_requires_exactly_eight_families(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)
    family_members.pop("PBR-1024")

    with pytest.raises(ValueError, match="exactly these pack families"):
        _publish(tmp_path / "data2", source, family_members)


def test_publish_validates_all_packs_and_writes_complete_shard_index(
    tmp_path: Path,
):
    data2_root = tmp_path / "data2"
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)

    manifests = _publish(data2_root, source, family_members)

    assert tuple(manifest.family for manifest in manifests) == PACK_FAMILIES
    assert all(manifest.validated_at for manifest in manifests)
    prepared = sorted((data2_root / "prepared").rglob("*.tar"))
    assert len(prepared) == 8
    assert {
        path.relative_to(data2_root / "prepared").as_posix()
        for path in prepared
    } == {
        "common/ABO/ABO-00000/batch000.tar",
        "ss/64/ABO/ABO-00000/batch000.tar",
        "shape/256/ABO/ABO-00000/batch000.tar",
        "shape/512/ABO/ABO-00000/batch000.tar",
        "shape/1024/ABO/ABO-00000/batch000.tar",
        "pbr/256/ABO/ABO-00000/batch000.tar",
        "pbr/512/ABO/ABO-00000/batch000.tar",
        "pbr/1024/ABO/ABO-00000/batch000.tar",
    }
    assert not list((data2_root / "staging").rglob("*"))
    for pack_path in prepared:
        verify_pack(
            pack_path, pack_path.with_suffix(pack_path.suffix + ".manifest.json")
        )

    index_path = data2_root / "prepared" / "index" / "ABO" / "ABO-00000.json"
    index = json.loads(index_path.read_text())
    assert index["source"] == "ABO"
    assert index["shard_id"] == "ABO-00000"
    assert set(index["batches"]) == {"batch000"}
    assert set(index["batches"]["batch000"]) == set(PACK_FAMILIES)
    for family, entry in index["batches"]["batch000"].items():
        manifest_path = data2_root / "prepared" / entry["manifest"]
        assert entry["manifest_sha256"] == sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        assert json.loads(manifest_path.read_text())["family"] == family


def test_valid_republish_is_idempotent(tmp_path: Path, monkeypatch):
    data2_root = tmp_path / "data2"
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)
    first = _publish(data2_root, source, family_members)
    existing = {
        path: (path.stat().st_ino, path.read_bytes())
        for path in (data2_root / "prepared").rglob("*")
        if path.is_file()
    }

    monkeypatch.setattr(
        packing,
        "_utc_now",
        lambda: "2099-01-01T00:00:00+00:00",
    )
    second = _publish(data2_root, source, family_members)

    assert second == first
    assert {
        path: (path.stat().st_ino, path.read_bytes())
        for path in (data2_root / "prepared").rglob("*")
        if path.is_file()
    } == existing


def test_publish_merges_batches_without_changing_prior_entries(tmp_path: Path):
    data2_root = tmp_path / "data2"
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)
    _publish(data2_root, source, family_members)
    index_path = data2_root / "prepared" / "index" / "ABO" / "ABO-00000.json"
    first_batch = json.loads(index_path.read_text())["batches"]["batch000"]

    _publish(
        data2_root,
        source,
        family_members,
        batch_id="batch001",
        asset_sha256s=("b" * 64,),
    )

    index = json.loads(index_path.read_text())
    assert list(index["batches"]) == ["batch000", "batch001"]
    assert index["batches"]["batch000"] == first_batch
    assert set(index["batches"]["batch001"]) == set(PACK_FAMILIES)


def test_publish_refuses_to_replace_valid_different_pack(tmp_path: Path):
    data2_root = tmp_path / "data2"
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)
    _publish(data2_root, source, family_members)
    existing = {
        path: path.read_bytes()
        for path in (data2_root / "prepared").rglob("*")
        if path.is_file()
    }
    (source / "common" / "payload.bin").write_bytes(b"different")

    with pytest.raises(ValidationError, match="different valid published pack"):
        _publish(data2_root, source, family_members)

    assert {
        path: path.read_bytes()
        for path in (data2_root / "prepared").rglob("*")
        if path.is_file()
    } == existing


def test_interrupted_publication_has_no_index_entry_and_can_resume(
    tmp_path: Path, monkeypatch
):
    data2_root = tmp_path / "data2"
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)
    real_replace_and_sync = packing._replace_and_sync
    prepared_renames = 0

    def interrupt_manifest_rename(temporary, destination):
        nonlocal prepared_renames
        if (data2_root / "prepared") in destination.parents:
            prepared_renames += 1
            if prepared_renames == 2:
                raise OSError("publication interrupted")
        return real_replace_and_sync(temporary, destination)

    with monkeypatch.context() as context:
        context.setattr(packing, "_replace_and_sync", interrupt_manifest_rename)
        with pytest.raises(OSError, match="publication interrupted"):
            _publish(data2_root, source, family_members)

    index_path = data2_root / "prepared" / "index" / "ABO" / "ABO-00000.json"
    assert not index_path.exists()

    resumed = _publish(data2_root, source, family_members)

    assert len(resumed) == 8
    assert "batch000" in json.loads(index_path.read_text())["batches"]


def test_publish_failure_does_not_write_index(tmp_path: Path, monkeypatch):
    data2_root = tmp_path / "data2"
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)
    real_verify_pack = packing.verify_pack
    calls = 0

    def fail_last_pack(pack_path, manifest_path):
        nonlocal calls
        calls += 1
        if calls == len(PACK_FAMILIES):
            raise ValidationError("injected verification failure")
        return real_verify_pack(pack_path, manifest_path)

    monkeypatch.setattr(packing, "verify_pack", fail_last_pack)

    with pytest.raises(ValidationError, match="injected verification failure"):
        _publish(data2_root, source, family_members)

    assert not (data2_root / "prepared").exists()
    assert not list((data2_root / "staging").rglob("*"))
