from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
from typing import BinaryIO, Mapping, Sequence

from .atomic_io import atomic_write_json
from .validation import ValidationError


PACK_FAMILIES = (
    "common",
    "SS-64",
    "shape-256",
    "shape-512",
    "shape-1024",
    "PBR-256",
    "PBR-512",
    "PBR-1024",
)

_FAMILY_DIRECTORIES = {
    "common": Path("common"),
    "SS-64": Path("ss", "64"),
    "shape-256": Path("shape", "256"),
    "shape-512": Path("shape", "512"),
    "shape-1024": Path("shape", "1024"),
    "PBR-256": Path("pbr", "256"),
    "PBR-512": Path("pbr", "512"),
    "PBR-1024": Path("pbr", "1024"),
}


@dataclass(frozen=True)
class PackMember:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PackManifest:
    shard_id: str
    batch_id: str
    family: str
    config_hash: str
    tool_commit: str
    asset_sha256s: tuple[str, ...]
    completed_count: int
    quarantined_count: int
    created_at: str
    validated_at: str
    pack_sha256: str
    members: tuple[PackMember, ...]


class _HashingReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = sha256()

    def read(self, size: int = -1) -> bytes:
        value = self.stream.read(size)
        self.digest.update(value)
        return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha(path: Path) -> str:
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _replace_and_sync(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory = os.open(destination.parent, flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _manifest_path(pack_path: Path) -> Path:
    return pack_path.with_suffix(pack_path.suffix + ".manifest.json")


def _safe_source_member(source_root: Path, relative: Path) -> tuple[Path, str]:
    relative = Path(relative)
    name = relative.as_posix()
    if (
        relative.is_absolute()
        or not name
        or name == "."
        or ".." in relative.parts
    ):
        raise ValueError(f"unsafe member: {relative}")

    path = source_root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink member: {relative}")

    try:
        path.resolve(strict=True).relative_to(source_root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"unsafe member: {relative}") from error
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise ValueError(f"invalid member: {relative}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"non-file member: {relative}")
    return path, name


def build_pack(
    source_root: Path,
    members: list[Path],
    output: Path,
    shard_id: str,
    *,
    batch_id: str,
    family: str,
    config_hash: str,
    tool_commit: str,
    asset_sha256s: tuple[str, ...],
    completed_count: int,
    quarantined_count: int,
) -> PackManifest:
    source_root = Path(source_root).resolve(strict=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (_safe_source_member(source_root, item) for item in members),
        key=lambda item: item[1],
    )
    names = [name for _, name in ordered]
    if len(names) != len(set(names)):
        raise ValueError("duplicate member")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)

        recorded = []
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as bundle:
            for path, name in ordered:
                with path.open("rb") as stream:
                    file_stat = os.fstat(stream.fileno())
                    if not stat.S_ISREG(file_stat.st_mode):
                        raise ValueError(f"non-file member: {name}")
                    info = tarfile.TarInfo(name)
                    info.size = file_stat.st_size
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    hashing_stream = _HashingReader(stream)
                    bundle.addfile(info, hashing_stream)
                    recorded.append(
                        PackMember(
                            name,
                            info.size,
                            hashing_stream.digest.hexdigest(),
                        )
                    )

        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        _replace_and_sync(temporary, output)
        temporary = None

        manifest = PackManifest(
            shard_id=shard_id,
            batch_id=batch_id,
            family=family,
            config_hash=config_hash,
            tool_commit=tool_commit,
            asset_sha256s=tuple(sorted(asset_sha256s)),
            completed_count=completed_count,
            quarantined_count=quarantined_count,
            created_at=_utc_now(),
            validated_at="",
            pack_sha256=file_sha(output),
            members=tuple(recorded),
        )
        atomic_write_json(_manifest_path(output), asdict(manifest))
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_string(value, field: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"invalid pack manifest field: {field}")
    return value


def _require_count(value, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"invalid pack manifest field: {field}")
    return value


def _manifest_from_value(value, manifest_path: Path) -> PackManifest:
    if not isinstance(value, dict):
        raise ValidationError(f"invalid pack manifest: {manifest_path}")
    try:
        asset_values = value["asset_sha256s"]
        member_values = value["members"]
        if not isinstance(asset_values, (list, tuple)) or not all(
            isinstance(item, str) for item in asset_values
        ):
            raise ValidationError("invalid pack manifest field: asset_sha256s")
        if not isinstance(member_values, (list, tuple)):
            raise ValidationError("invalid pack manifest field: members")

        members = []
        for item in member_values:
            if not isinstance(item, dict):
                raise ValidationError("invalid pack manifest member")
            members.append(
                PackMember(
                    path=_require_string(item["path"], "members.path"),
                    size=_require_count(item["size"], "members.size"),
                    sha256=_require_string(item["sha256"], "members.sha256"),
                )
            )
        return PackManifest(
            shard_id=_require_string(value["shard_id"], "shard_id"),
            batch_id=_require_string(value["batch_id"], "batch_id"),
            family=_require_string(value["family"], "family"),
            config_hash=_require_string(value["config_hash"], "config_hash"),
            tool_commit=_require_string(value["tool_commit"], "tool_commit"),
            asset_sha256s=tuple(asset_values),
            completed_count=_require_count(
                value["completed_count"], "completed_count"
            ),
            quarantined_count=_require_count(
                value["quarantined_count"], "quarantined_count"
            ),
            created_at=_require_string(value["created_at"], "created_at"),
            validated_at=_require_string(
                value["validated_at"], "validated_at"
            ),
            pack_sha256=_require_string(
                value["pack_sha256"], "pack_sha256"
            ),
            members=tuple(members),
        )
    except KeyError as error:
        raise ValidationError(
            f"invalid pack manifest: {manifest_path}: missing {error.args[0]}"
        ) from error


def _load_manifest(manifest_path: Path) -> PackManifest:
    try:
        value = json.loads(Path(manifest_path).read_text())
    except Exception as error:
        raise ValidationError(
            f"invalid pack manifest: {manifest_path}: {error}"
        ) from error
    return _manifest_from_value(value, Path(manifest_path))


def _safe_archive_name(name: str) -> bool:
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == name
        and name != "."
    )


def verify_pack(pack_path: Path, manifest_path: Path) -> None:
    pack_path = Path(pack_path)
    manifest_path = Path(manifest_path)
    expected = _load_manifest(manifest_path)
    expected_members = {}
    for member in expected.members:
        if not _safe_archive_name(member.path):
            raise ValidationError(f"unsafe manifest member: {member.path}")
        if member.path in expected_members:
            raise ValidationError(f"duplicate manifest member: {member.path}")
        expected_members[member.path] = member

    try:
        actual_pack_sha = file_sha(pack_path)
        if actual_pack_sha != expected.pack_sha256:
            raise ValidationError(f"pack checksum mismatch: {pack_path}")

        actual = {}
        with tarfile.open(pack_path, mode="r:") as bundle:
            for item in bundle:
                name = item.name
                if (
                    not _safe_archive_name(name)
                    or item.issym()
                    or item.islnk()
                    or not item.isfile()
                ):
                    raise ValidationError(f"unsafe tar member: {name}")
                if name in actual:
                    raise ValidationError(f"duplicate tar member: {name}")
                stream = bundle.extractfile(item)
                if stream is None:
                    raise ValidationError(f"unreadable tar member: {name}")
                member_sha = sha256()
                with stream:
                    for block in iter(
                        lambda: stream.read(8 * 1024 * 1024), b""
                    ):
                        member_sha.update(block)
                actual[name] = (item.size, member_sha.hexdigest())
    except ValidationError:
        raise
    except (OSError, EOFError, tarfile.TarError) as error:
        raise ValidationError(f"invalid pack: {pack_path}: {error}") from error

    if set(actual) != set(expected_members):
        raise ValidationError("pack member set mismatch")
    for name, (size, digest_value) in actual.items():
        expected_member = expected_members[name]
        if size != expected_member.size:
            raise ValidationError(f"pack member size mismatch: {name}")
        if digest_value != expected_member.sha256:
            raise ValidationError(f"pack member checksum mismatch: {name}")


def _validate_component(value: str, description: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or Path(value).name != value
        or "\\" in value
    ):
        raise ValueError(f"unsafe {description}: {value}")


def _prepared_pack_relative(
    family: str, source: str, shard_id: str, batch_id: str
) -> Path:
    return _FAMILY_DIRECTORIES[family] / source / shard_id / f"{batch_id}.tar"


def _same_publication(first: PackManifest, second: PackManifest) -> bool:
    fields = (
        "shard_id",
        "batch_id",
        "family",
        "config_hash",
        "tool_commit",
        "asset_sha256s",
        "completed_count",
        "quarantined_count",
        "pack_sha256",
        "members",
    )
    return all(getattr(first, field) == getattr(second, field) for field in fields)


def _index_entry(
    prepared_root: Path, pack_path: Path, manifest: PackManifest
) -> dict[str, str]:
    manifest_path = _manifest_path(pack_path)
    return {
        "pack": pack_path.relative_to(prepared_root).as_posix(),
        "pack_sha256": manifest.pack_sha256,
        "manifest": manifest_path.relative_to(prepared_root).as_posix(),
        "manifest_sha256": file_sha(manifest_path),
    }


def _load_index(
    index_path: Path,
    prepared_root: Path,
    source: str,
    shard_id: str,
    replacing_batch: str,
) -> dict:
    if not index_path.exists():
        return {"source": source, "shard_id": shard_id, "batches": {}}
    try:
        value = json.loads(index_path.read_text())
    except Exception as error:
        raise ValidationError(f"invalid shard index: {index_path}: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("source") != source
        or value.get("shard_id") != shard_id
        or not isinstance(value.get("batches"), dict)
    ):
        raise ValidationError(f"invalid shard index: {index_path}")

    for batch_id, entries in value["batches"].items():
        try:
            _validate_component(batch_id, "indexed batch id")
        except ValueError as error:
            raise ValidationError(
                f"invalid shard index batch: {batch_id}"
            ) from error
        if not isinstance(entries, dict) or set(entries) != set(PACK_FAMILIES):
            raise ValidationError(f"incomplete shard index batch: {batch_id}")
        for family in PACK_FAMILIES:
            entry = entries[family]
            pack_relative = _prepared_pack_relative(
                family, source, shard_id, batch_id
            )
            manifest_relative = _manifest_path(pack_relative)
            if (
                not isinstance(entry, dict)
                or entry.get("pack") != pack_relative.as_posix()
                or entry.get("manifest") != manifest_relative.as_posix()
            ):
                raise ValidationError(
                    f"invalid shard index entry: {batch_id}: {family}"
                )
            if batch_id == replacing_batch:
                continue

            pack_path = prepared_root / pack_relative
            manifest_path = prepared_root / manifest_relative
            verify_pack(pack_path, manifest_path)
            manifest = _load_manifest(manifest_path)
            if (
                manifest.shard_id != shard_id
                or manifest.batch_id != batch_id
                or manifest.family != family
                or entry.get("pack_sha256") != manifest.pack_sha256
                or entry.get("manifest_sha256") != file_sha(manifest_path)
            ):
                raise ValidationError(
                    f"shard index checksum mismatch: {batch_id}: {family}"
                )
    return value


def _cleanup_staging(run_directory: Path, batch_directory: Path) -> None:
    shutil.rmtree(run_directory, ignore_errors=True)
    for path in (batch_directory, batch_directory.parent):
        try:
            path.rmdir()
        except OSError:
            pass


def publish_pack(
    data2_root: Path,
    source_root: Path,
    members_by_family: Mapping[str, Sequence[Path]],
    shard_id: str,
    *,
    source: str,
    batch_id: str,
    config_hash: str,
    tool_commit: str,
    asset_sha256s: tuple[str, ...],
    completed_count: int,
    quarantined_count: int,
) -> tuple[PackManifest, ...]:
    if set(members_by_family) != set(PACK_FAMILIES):
        expected = ", ".join(PACK_FAMILIES)
        raise ValueError(f"expected exactly these pack families: {expected}")
    _validate_component(source, "source")
    _validate_component(shard_id, "shard id")
    _validate_component(batch_id, "batch id")

    data2_root = Path(data2_root)
    source_root = Path(source_root)
    prepared_root = data2_root / "prepared"
    batch_staging = data2_root / "staging" / shard_id / batch_id
    batch_staging.mkdir(parents=True, exist_ok=True)
    run_directory = Path(
        tempfile.mkdtemp(prefix=".publish-", dir=batch_staging)
    )
    index_path = prepared_root / "index" / source / f"{shard_id}.json"

    try:
        staged = {}
        for family in PACK_FAMILIES:
            pack_path = run_directory / f"{family}.tar"
            manifest = build_pack(
                source_root,
                list(members_by_family[family]),
                pack_path,
                shard_id,
                batch_id=batch_id,
                family=family,
                config_hash=config_hash,
                tool_commit=tool_commit,
                asset_sha256s=asset_sha256s,
                completed_count=completed_count,
                quarantined_count=quarantined_count,
            )
            manifest_path = _manifest_path(pack_path)
            verify_pack(pack_path, manifest_path)
            manifest = replace(manifest, validated_at=_utc_now())
            atomic_write_json(manifest_path, asdict(manifest))
            verify_pack(pack_path, manifest_path)
            staged[family] = (pack_path, manifest)

        index = _load_index(
            index_path,
            prepared_root,
            source,
            shard_id,
            replacing_batch=batch_id,
        )

        reusable = {}
        for family in PACK_FAMILIES:
            destination = prepared_root / _prepared_pack_relative(
                family, source, shard_id, batch_id
            )
            destination_manifest = _manifest_path(destination)
            if destination.exists() and destination_manifest.exists():
                try:
                    verify_pack(destination, destination_manifest)
                except ValidationError:
                    pass
                else:
                    existing = _load_manifest(destination_manifest)
                    if not _same_publication(existing, staged[family][1]):
                        raise ValidationError(
                            f"different valid published pack: {destination}"
                        )
                    reusable[family] = existing

        for family in PACK_FAMILIES:
            if family in reusable:
                continue
            staged_pack, _ = staged[family]
            destination = prepared_root / _prepared_pack_relative(
                family, source, shard_id, batch_id
            )
            _replace_and_sync(staged_pack, destination)
            _replace_and_sync(
                _manifest_path(staged_pack), _manifest_path(destination)
            )

        published = []
        entries = {}
        for family in PACK_FAMILIES:
            destination = prepared_root / _prepared_pack_relative(
                family, source, shard_id, batch_id
            )
            destination_manifest = _manifest_path(destination)
            verify_pack(destination, destination_manifest)
            manifest = _load_manifest(destination_manifest)
            if not manifest.validated_at:
                raise ValidationError(
                    f"published pack is not validated: {destination}"
                )
            published.append(manifest)
            entries[family] = _index_entry(
                prepared_root, destination, manifest
            )

        batches = dict(index["batches"])
        batches[batch_id] = entries
        updated_index = {
            "source": source,
            "shard_id": shard_id,
            "batches": {name: batches[name] for name in sorted(batches)},
        }
        if updated_index != index:
            atomic_write_json(index_path, updated_index)
        return tuple(published)
    finally:
        _cleanup_staging(run_directory, batch_staging)
