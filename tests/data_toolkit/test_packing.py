from contextlib import contextmanager
from dataclasses import asdict, replace
import errno
from hashlib import sha256
import io
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
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


def _publish_worker(
    result_queue,
    data2_root: Path,
    source: Path,
    family_members: dict[str, list[Path]],
    overrides: dict,
) -> None:
    name = multiprocessing.current_process().name
    try:
        manifests = _publish(
            data2_root,
            source,
            family_members,
            **overrides,
        )
    except BaseException as error:
        result_queue.put((name, "error", type(error).__name__, str(error)))
    else:
        result_queue.put((name, "ok", len(manifests)))


def _run_interleaved_publishers(
    monkeypatch,
    data2_root: Path,
    first_source: Path,
    first_members: dict[str, list[Path]],
    first_overrides: dict,
    second_source: Path,
    second_members: dict[str, list[Path]],
    second_overrides: dict,
) -> dict[str, tuple]:
    context = multiprocessing.get_context("fork")
    first_loaded = context.Event()
    release_first = context.Event()
    second_attempting = context.Event()
    second_acquired = context.Event()
    result_queue = context.Queue()
    real_lock = packing._source_shard_lock
    real_load_index = packing._load_index

    @contextmanager
    def observed_lock(*args, **kwargs):
        is_second = multiprocessing.current_process().name == "publisher-second"
        if is_second:
            second_attempting.set()
        with real_lock(*args, **kwargs):
            if is_second:
                second_acquired.set()
            yield

    def controlled_load_index(*args, **kwargs):
        value = real_load_index(*args, **kwargs)
        if multiprocessing.current_process().name == "publisher-first":
            first_loaded.set()
            if not release_first.wait(10):
                raise TimeoutError("timed out waiting to release first publisher")
        return value

    monkeypatch.setattr(packing, "_source_shard_lock", observed_lock)
    monkeypatch.setattr(packing, "_load_index", controlled_load_index)
    first = context.Process(
        name="publisher-first",
        target=_publish_worker,
        args=(
            result_queue,
            data2_root,
            first_source,
            first_members,
            first_overrides,
        ),
    )
    second = context.Process(
        name="publisher-second",
        target=_publish_worker,
        args=(
            result_queue,
            data2_root,
            second_source,
            second_members,
            second_overrides,
        ),
    )
    processes = (first, second)
    try:
        first.start()
        assert first_loaded.wait(10), "first publisher did not load the index"
        second.start()
        assert second_attempting.wait(10), "second publisher did not reach the lock"
        assert not second_acquired.wait(0.2)
        release_first.set()
        for process in processes:
            process.join(10)
            assert not process.is_alive(), f"publisher hung: {process.name}"
            assert process.exitcode == 0
        results = {}
        for _ in processes:
            try:
                result = result_queue.get(timeout=5)
            except Empty as error:
                raise AssertionError("publisher returned no result") from error
            results[result[0]] = result[1:]
        return results
    finally:
        release_first.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
        result_queue.close()
        result_queue.join_thread()


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


def test_backslash_source_member_is_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    relative = Path("bad\\name")
    (source / relative).write_text("x")

    with pytest.raises(ValueError, match="unsafe member"):
        build_pack(
            source,
            [relative],
            tmp_path / "bad.tar",
            "ABO-00000",
            **PACK_METADATA,
        )


def test_component_swap_cannot_pack_content_outside_source_root(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    member_directory = source / "nested"
    member_directory.mkdir(parents=True)
    (member_directory / "payload").write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload").write_bytes(b"outside")
    real_safe_source_member = packing._safe_source_member
    swapped = False

    def validate_then_swap(*args, **kwargs):
        nonlocal swapped
        result = real_safe_source_member(*args, **kwargs)
        if not swapped:
            swapped = True
            member_directory.rename(source / "original")
            member_directory.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(packing, "_safe_source_member", validate_then_swap)

    with pytest.raises(ValueError, match="symlink|unsafe member"):
        build_pack(
            source,
            [Path("nested/payload")],
            tmp_path / "bad.tar",
            "ABO-00000",
            **PACK_METADATA,
        )

    assert not (tmp_path / "bad.tar").exists()


@pytest.mark.parametrize(
    "failure_point,error_number",
    (("root", errno.EIO), ("component", errno.ESTALE), ("final", errno.EIO)),
)
def test_build_pack_propagates_source_open_io_failures(
    tmp_path: Path, monkeypatch, failure_point: str, error_number: int
):
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "payload").write_bytes(b"payload")
    real_open = os.open

    def fail_selected_open(path, flags, *args, **kwargs):
        is_root = Path(path) == source and "dir_fd" not in kwargs
        is_component = path == "nested" and "dir_fd" in kwargs
        is_final = path == "payload" and "dir_fd" in kwargs
        if (
            (failure_point == "root" and is_root)
            or (failure_point == "component" and is_component)
            or (failure_point == "final" and is_final)
        ):
            raise OSError(error_number, f"source {failure_point} I/O")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(packing.os, "open", fail_selected_open)

    with pytest.raises(OSError, match=f"source {failure_point} I/O") as caught:
        build_pack(
            source,
            [Path("nested/payload")],
            tmp_path / "bad.tar",
            "ABO-00000",
            **PACK_METADATA,
        )

    assert caught.value.errno == error_number
    assert not (tmp_path / "bad.tar").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_rejected_source_member_closes_all_file_descriptors(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(tmp_path / "outside")
    before = len(os.listdir("/proc/self/fd"))

    for index in range(3):
        with pytest.raises(ValueError, match="symlink"):
            build_pack(
                source,
                [Path("link")],
                tmp_path / f"bad-{index}.tar",
                "ABO-00000",
                **PACK_METADATA,
            )

    assert len(os.listdir("/proc/self/fd")) == before


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


def test_verify_pack_propagates_manifest_read_eio(tmp_path, monkeypatch):
    pack_path = tmp_path / "pack.tar"
    manifest_path = tmp_path / "pack.tar.manifest.json"
    _write_tar(pack_path, [("member", b"value", "file")])
    _write_manifest(manifest_path, pack_path, ["member"])
    real_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if Path(path) == manifest_path:
            raise OSError(errno.EIO, "manifest EIO")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    with pytest.raises(OSError, match="manifest EIO"):
        verify_pack(pack_path, manifest_path)


def test_verify_pack_propagates_pack_read_estale(tmp_path, monkeypatch):
    pack_path = tmp_path / "pack.tar"
    manifest_path = tmp_path / "pack.tar.manifest.json"
    _write_tar(pack_path, [("member", b"value", "file")])
    _write_manifest(manifest_path, pack_path, ["member"])
    real_open = Path.open

    def open_path(path, *args, **kwargs):
        if Path(path) == pack_path:
            raise OSError(errno.ESTALE, "pack ESTALE")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)

    with pytest.raises(OSError, match="pack ESTALE"):
        verify_pack(pack_path, manifest_path)


def test_load_index_propagates_read_eio(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {"source": "ABO", "shard_id": "ABO-00000", "batches": {}}
        )
    )
    real_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if Path(path) == index_path:
            raise OSError(errno.EIO, "index EIO")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    with pytest.raises(OSError, match="index EIO"):
        packing._load_index(
            index_path,
            tmp_path,
            "ABO",
            "ABO-00000",
            replacing_batch="batch000",
        )


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


def test_concurrent_same_batch_cannot_replace_valid_different_pack(
    tmp_path: Path, monkeypatch
):
    data2_root = tmp_path / "data2"
    first_source = tmp_path / "source-first"
    second_source = tmp_path / "source-second"
    first_source.mkdir()
    second_source.mkdir()
    first_members = _family_members(first_source)
    second_members = _family_members(second_source)
    (second_source / "common" / "payload.bin").write_bytes(b"different")

    results = _run_interleaved_publishers(
        monkeypatch,
        data2_root,
        first_source,
        first_members,
        {},
        second_source,
        second_members,
        {},
    )

    assert results["publisher-first"] == ("ok", 8)
    assert results["publisher-second"][:2] == ("error", "ValidationError")
    assert "different valid published pack" in results["publisher-second"][2]
    common = (
        data2_root
        / "prepared/common/ABO/ABO-00000/batch000.tar"
    )
    with tarfile.open(common) as bundle:
        assert bundle.extractfile("common/payload.bin").read() == b"common"


def test_concurrent_different_batches_merge_index_entries(
    tmp_path: Path, monkeypatch
):
    data2_root = tmp_path / "data2"
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)

    results = _run_interleaved_publishers(
        monkeypatch,
        data2_root,
        source,
        family_members,
        {"batch_id": "batch000"},
        source,
        family_members,
        {"batch_id": "batch001", "asset_sha256s": ("b" * 64,)},
    )

    assert results == {
        "publisher-first": ("ok", 8),
        "publisher-second": ("ok", 8),
    }
    index_path = data2_root / "prepared/index/ABO/ABO-00000.json"
    index = json.loads(index_path.read_text())
    assert list(index["batches"]) == ["batch000", "batch001"]
    assert all(
        set(entries) == set(PACK_FAMILIES)
        for entries in index["batches"].values()
    )


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


def test_final_prepared_verification_failure_does_not_write_index(
    tmp_path: Path, monkeypatch
):
    data2_root = tmp_path / "data2"
    source = tmp_path / "source"
    source.mkdir()
    family_members = _family_members(source)
    real_verify_pack = packing.verify_pack
    prepared_root = data2_root / "prepared"
    prepared_verifications = []

    def fail_prepared_pack(pack_path, manifest_path):
        pack_path = Path(pack_path)
        if prepared_root in pack_path.parents:
            prepared_verifications.append(pack_path)
            raise ValidationError("injected verification failure")
        return real_verify_pack(pack_path, manifest_path)

    monkeypatch.setattr(packing, "verify_pack", fail_prepared_pack)

    with pytest.raises(ValidationError, match="injected verification failure"):
        _publish(data2_root, source, family_members)

    assert len(prepared_verifications) == 1
    assert len(list(prepared_root.rglob("*.tar"))) == 8
    assert not (prepared_root / "index/ABO/ABO-00000.json").exists()
    assert not list((data2_root / "staging").rglob("*"))
