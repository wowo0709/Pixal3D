from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import errno
import fcntl
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

PACK_MANIFEST_SCHEMA_VERSION = 2

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

_VALIDATION_READ_ERRNOS = {
    errno.ENOENT,
    errno.ENOTDIR,
    errno.ELOOP,
    errno.EISDIR,
}


@dataclass(frozen=True)
class PackMember:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PackManifest:
    schema_version: int
    shard_id: str
    batch_id: str
    family: str
    config_hash: str
    tool_commit: str
    asset_sha256s: tuple[str, ...]
    included_asset_sha256s: tuple[str, ...]
    completed_count: int
    quarantined_count: int
    created_at: str
    validated_at: str
    pack_sha256: str
    members: tuple[PackMember, ...]
    gate: str = "production"


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
        or "\\" in name
    ):
        raise ValueError(f"unsafe member: {relative}")
    return relative, name


_STRUCTURAL_SOURCE_ERRNOS = frozenset(
    (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
)


def _raise_source_open_error(
    error: OSError, directory_fd: int, component: str, name: str
) -> None:
    if error.errno not in _STRUCTURAL_SOURCE_ERRNOS:
        raise error
    try:
        component_mode = os.stat(
            component, dir_fd=directory_fd, follow_symlinks=False
        ).st_mode
    except OSError as stat_error:
        if stat_error.errno not in _STRUCTURAL_SOURCE_ERRNOS:
            raise
        component_mode = None
    if error.errno == errno.ELOOP or (
        component_mode is not None and stat.S_ISLNK(component_mode)
    ):
        raise ValueError(f"symlink member: {name}") from error
    raise ValueError(f"unsafe member: {name}") from error


def _open_source_member(root_fd: int, relative: Path, name: str) -> int:
    directory_fd = os.dup(root_fd)
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                _raise_source_open_error(error, directory_fd, component, name)
            os.close(directory_fd)
            directory_fd = next_fd

        final_component = relative.parts[-1]
        try:
            file_fd = os.open(
                final_component,
                os.O_RDONLY
                | os.O_NOFOLLOW
                | os.O_CLOEXEC
                | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except OSError as error:
            _raise_source_open_error(
                error, directory_fd, final_component, name
            )
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError(f"non-file member: {name}")
            return file_fd
        except BaseException:
            os.close(file_fd)
            raise
    finally:
        os.close(directory_fd)


def _open_source_root(source_root: Path) -> int:
    try:
        return os.open(
            source_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as error:
        if error.errno not in _STRUCTURAL_SOURCE_ERRNOS:
            raise
        raise ValueError(f"unsafe source root: {source_root}") from error


def _canonical_asset_scope(
    values: Sequence[str], field: str
) -> tuple[str, ...]:
    assets = tuple(values)
    if not all(
        isinstance(asset, str)
        and len(asset) == 64
        and all(character in "0123456789abcdef" for character in asset)
        for asset in assets
    ):
        raise ValueError(f"invalid {field}")
    if len(assets) != len(set(assets)):
        raise ValueError(f"duplicate {field}")
    return tuple(sorted(assets))


def _resolve_manifest_scopes(
    asset_sha256s: Sequence[str],
    included_asset_sha256s: Sequence[str] | None,
    completed_count: int | None,
    quarantined_count: int | None,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    frozen = _canonical_asset_scope(asset_sha256s, "asset_sha256s")
    if included_asset_sha256s is None:
        if completed_count != len(frozen) or quarantined_count != 0:
            raise ValueError(
                "included_asset_sha256s is required for a partial scope"
            )
        included = frozen
    else:
        included = _canonical_asset_scope(
            included_asset_sha256s, "included_asset_sha256s"
        )
    if not set(included) <= set(frozen):
        raise ValueError("included_asset_sha256s is not a frozen subset")
    expected_completed = len(included)
    expected_quarantined = len(frozen) - expected_completed
    if completed_count is not None and completed_count != expected_completed:
        raise ValueError("completed_count does not match included scope")
    if (
        quarantined_count is not None
        and quarantined_count != expected_quarantined
    ):
        raise ValueError("quarantined_count does not match excluded scope")
    return frozen, included, expected_completed, expected_quarantined


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
    included_asset_sha256s: tuple[str, ...] | None = None,
    completed_count: int | None = None,
    quarantined_count: int | None = None,
    gate: str = "production",
) -> PackManifest:
    (
        frozen_assets,
        included_assets,
        completed_count,
        quarantined_count,
    ) = _resolve_manifest_scopes(
        asset_sha256s,
        included_asset_sha256s,
        completed_count,
        quarantined_count,
    )
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
        root_fd = _open_source_root(source_root)
        try:
            with tarfile.open(
                temporary, "w", format=tarfile.PAX_FORMAT
            ) as bundle:
                for relative, name in ordered:
                    file_fd = _open_source_member(root_fd, relative, name)
                    try:
                        stream = os.fdopen(file_fd, "rb", closefd=True)
                    except BaseException:
                        os.close(file_fd)
                        raise
                    with stream:
                        file_stat = os.fstat(stream.fileno())
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
        finally:
            os.close(root_fd)

        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        _replace_and_sync(temporary, output)
        temporary = None

        manifest = PackManifest(
            schema_version=PACK_MANIFEST_SCHEMA_VERSION,
            shard_id=shard_id,
            batch_id=batch_id,
            family=family,
            config_hash=config_hash,
            tool_commit=tool_commit,
            asset_sha256s=frozen_assets,
            included_asset_sha256s=included_assets,
            completed_count=completed_count,
            quarantined_count=quarantined_count,
            created_at=_utc_now(),
            validated_at="",
            pack_sha256=file_sha(output),
            members=tuple(recorded),
            gate=gate,
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


def _manifest_asset_scope(value, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValidationError(f"invalid pack manifest field: {field}")
    try:
        canonical = _canonical_asset_scope(value, field)
    except ValueError as error:
        raise ValidationError(
            f"invalid pack manifest field: {field}"
        ) from error
    if tuple(value) != canonical:
        raise ValidationError(
            f"non-canonical pack manifest field: {field}"
        )
    return canonical


def _legacy_included_scope(
    frozen: tuple[str, ...], member_values: Sequence[object]
) -> tuple[str, ...]:
    frozen_set = set(frozen)
    found = set()
    for item in member_values:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str):
            continue
        found.update(
            component
            for component in PurePosixPath(path).parts
            if component in frozen_set
        )
    return tuple(asset for asset in frozen if asset in found)


def _manifest_from_value(value, manifest_path: Path) -> PackManifest:
    if not isinstance(value, dict):
        raise ValidationError(f"invalid pack manifest: {manifest_path}")
    try:
        asset_values = value["asset_sha256s"]
        member_values = value["members"]
        if not isinstance(member_values, (list, tuple)):
            raise ValidationError("invalid pack manifest field: members")

        schema_version = value.get("schema_version", 1)
        if schema_version not in {1, PACK_MANIFEST_SCHEMA_VERSION}:
            raise ValidationError(
                "invalid pack manifest field: schema_version"
            )
        frozen = _manifest_asset_scope(asset_values, "asset_sha256s")
        if schema_version == PACK_MANIFEST_SCHEMA_VERSION:
            included = _manifest_asset_scope(
                value["included_asset_sha256s"],
                "included_asset_sha256s",
            )
            if not set(included) <= set(frozen):
                raise ValidationError(
                    "pack included scope is not a frozen subset"
                )
        else:
            included = _legacy_included_scope(frozen, member_values)

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
        completed_count = _require_count(
            value["completed_count"], "completed_count"
        )
        quarantined_count = _require_count(
            value["quarantined_count"], "quarantined_count"
        )
        if schema_version == PACK_MANIFEST_SCHEMA_VERSION and (
            completed_count != len(included)
            or quarantined_count != len(frozen) - len(included)
        ):
            raise ValidationError(
                "pack manifest counts do not match included scope"
            )
        return PackManifest(
            schema_version=schema_version,
            shard_id=_require_string(value["shard_id"], "shard_id"),
            batch_id=_require_string(value["batch_id"], "batch_id"),
            family=_require_string(value["family"], "family"),
            config_hash=_require_string(value["config_hash"], "config_hash"),
            tool_commit=_require_string(value["tool_commit"], "tool_commit"),
            asset_sha256s=frozen,
            included_asset_sha256s=included,
            completed_count=completed_count,
            quarantined_count=quarantined_count,
            created_at=_require_string(value["created_at"], "created_at"),
            validated_at=_require_string(
                value["validated_at"], "validated_at"
            ),
            pack_sha256=_require_string(
                value["pack_sha256"], "pack_sha256"
            ),
            members=tuple(members),
            gate=_require_string(value["gate"], "gate"),
        )
    except KeyError as error:
        raise ValidationError(
            f"invalid pack manifest: {manifest_path}: missing {error.args[0]}"
        ) from error


def _load_manifest(manifest_path: Path) -> PackManifest:
    try:
        value = json.loads(Path(manifest_path).read_text())
    except OSError as error:
        if error.errno not in _VALIDATION_READ_ERRNOS:
            raise
        raise ValidationError(
            f"invalid pack manifest: {manifest_path}: {error}"
        ) from error
    except (UnicodeError, json.JSONDecodeError, TypeError) as error:
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
    except OSError as error:
        if error.errno not in _VALIDATION_READ_ERRNOS:
            raise
        raise ValidationError(f"invalid pack: {pack_path}: {error}") from error
    except (EOFError, tarfile.TarError) as error:
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
    family: str,
    source: str,
    shard_id: str,
    batch_id: str,
    gate: str = "production",
) -> Path:
    prefix = Path() if gate == "production" else Path("qualification", gate)
    return prefix / _FAMILY_DIRECTORIES[family] / source / shard_id / f"{batch_id}.tar"


def _same_publication(first: PackManifest, second: PackManifest) -> bool:
    fields = (
        "schema_version",
        "shard_id",
        "batch_id",
        "family",
        "config_hash",
        "tool_commit",
        "asset_sha256s",
        "included_asset_sha256s",
        "completed_count",
        "quarantined_count",
        "pack_sha256",
        "members",
        "gate",
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
    gate: str = "production",
) -> dict:
    try:
        value = json.loads(index_path.read_text())
    except FileNotFoundError:
        return {
            "gate": gate,
            "source": source,
            "shard_id": shard_id,
            "batches": {},
        }
    except OSError as error:
        if error.errno not in _VALIDATION_READ_ERRNOS:
            raise
        raise ValidationError(
            f"invalid shard index: {index_path}: {error}"
        ) from error
    except (UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise ValidationError(f"invalid shard index: {index_path}: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("gate") != gate
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
                family, source, shard_id, batch_id, gate
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
                or manifest.gate != gate
                or manifest.batch_id != batch_id
                or manifest.family != family
                or entry.get("pack_sha256") != manifest.pack_sha256
                or entry.get("manifest_sha256") != file_sha(manifest_path)
            ):
                raise ValidationError(
                    f"shard index checksum mismatch: {batch_id}: {family}"
                )
    return value


@contextmanager
def _source_shard_lock(
    data2_root: Path, source: str, shard_id: str, gate: str
):
    lock_directory = (
        Path(data2_root)
        / "control"
        / "locks"
        / "packing"
        / gate
        / source
    )
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{shard_id}.lock"
    lock_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _cleanup_staging(run_directory: Path, batch_directory: Path) -> None:
    shutil.rmtree(run_directory, ignore_errors=True)
    for path in (
        batch_directory,
        batch_directory.parent,
        batch_directory.parent.parent,
    ):
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
    included_asset_sha256s_by_family: Mapping[
        str, tuple[str, ...]
    ] | None = None,
    completed_count: int | None = None,
    quarantined_count: int | None = None,
    gate: str = "production",
) -> tuple[PackManifest, ...]:
    if set(members_by_family) != set(PACK_FAMILIES):
        expected = ", ".join(PACK_FAMILIES)
        raise ValueError(f"expected exactly these pack families: {expected}")
    frozen_assets = _canonical_asset_scope(
        asset_sha256s, "asset_sha256s"
    )
    if included_asset_sha256s_by_family is None:
        if completed_count != len(frozen_assets) or quarantined_count != 0:
            raise ValueError(
                "included_asset_sha256s_by_family is required for partial scopes"
            )
        included_by_family = {
            family: frozen_assets for family in PACK_FAMILIES
        }
    else:
        if set(included_asset_sha256s_by_family) != set(PACK_FAMILIES):
            expected = ", ".join(PACK_FAMILIES)
            raise ValueError(
                "expected included scopes for exactly these pack families: "
                f"{expected}"
            )
        included_by_family = {
            family: _canonical_asset_scope(
                included_asset_sha256s_by_family[family],
                f"{family} included_asset_sha256s",
            )
            for family in PACK_FAMILIES
        }
        for family, included in included_by_family.items():
            if not set(included) <= set(frozen_assets):
                raise ValueError(
                    f"{family} included scope is not a frozen subset"
                )
        if completed_count is not None or quarantined_count is not None:
            if any(
                completed_count != len(included)
                or quarantined_count
                != len(frozen_assets) - len(included)
                for included in included_by_family.values()
            ):
                raise ValueError(
                    "legacy counts do not match every family scope"
                )
    for resolution in (256, 512, 1024):
        pbr_family = f"PBR-{resolution}"
        shape_family = f"shape-{resolution}"
        if not set(included_by_family[pbr_family]) <= set(
            included_by_family[shape_family]
        ):
            raise ValueError(
                f"{pbr_family} included scope is not a {shape_family} subset"
            )
    if not set(included_by_family["SS-64"]) <= set(
        included_by_family["shape-1024"]
    ):
        raise ValueError(
            "SS-64 included scope is not a shape-1024 subset"
        )
    non_common_union = set().union(
        *(
            set(included_by_family[family])
            for family in PACK_FAMILIES
            if family != "common"
        )
    )
    if set(included_by_family["common"]) != non_common_union:
        raise ValueError("common included scope is not the family union")
    _validate_component(source, "source")
    _validate_component(shard_id, "shard id")
    _validate_component(batch_id, "batch id")
    if gate not in {"smoke", "pilot", "production"}:
        raise ValueError(f"invalid gate: {gate}")

    data2_root = Path(data2_root)
    source_root = Path(source_root)
    prepared_root = data2_root / "prepared"
    batch_staging = data2_root / "staging" / gate / shard_id / batch_id
    batch_staging.mkdir(parents=True, exist_ok=True)
    run_directory = Path(
        tempfile.mkdtemp(prefix=".publish-", dir=batch_staging)
    )
    prefix = Path() if gate == "production" else Path("qualification", gate)
    index_path = prepared_root / prefix / "index" / source / f"{shard_id}.json"

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
                asset_sha256s=frozen_assets,
                included_asset_sha256s=included_by_family[family],
                gate=gate,
            )
            manifest_path = _manifest_path(pack_path)
            verify_pack(pack_path, manifest_path)
            manifest = replace(manifest, validated_at=_utc_now())
            atomic_write_json(manifest_path, asdict(manifest))
            verify_pack(pack_path, manifest_path)
            staged[family] = (pack_path, manifest)

        with _source_shard_lock(data2_root, source, shard_id, gate):
            index = _load_index(
                index_path,
                prepared_root,
                source,
                shard_id,
                replacing_batch=batch_id,
                gate=gate,
            )

            reusable = {}
            for family in PACK_FAMILIES:
                destination = prepared_root / _prepared_pack_relative(
                    family, source, shard_id, batch_id, gate
                )
                destination_manifest = _manifest_path(destination)
                if destination.exists() and destination_manifest.exists():
                    try:
                        verify_pack(destination, destination_manifest)
                    except ValidationError:
                        pass
                    else:
                        existing = _load_manifest(destination_manifest)
                        if not _same_publication(
                            existing, staged[family][1]
                        ):
                            raise ValidationError(
                                "different valid published pack: "
                                f"{destination}"
                            )
                        reusable[family] = existing

            for family in PACK_FAMILIES:
                if family in reusable:
                    continue
                staged_pack, _ = staged[family]
                destination = prepared_root / _prepared_pack_relative(
                    family, source, shard_id, batch_id, gate
                )
                _replace_and_sync(staged_pack, destination)
                _replace_and_sync(
                    _manifest_path(staged_pack), _manifest_path(destination)
                )

            published = []
            entries = {}
            for family in PACK_FAMILIES:
                destination = prepared_root / _prepared_pack_relative(
                    family, source, shard_id, batch_id, gate
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
                "gate": gate,
                "source": source,
                "shard_id": shard_id,
                "batches": {
                    name: batches[name] for name in sorted(batches)
                },
            }
            if updated_index != index:
                atomic_write_json(index_path, updated_index)
            return tuple(published)
    finally:
        _cleanup_staging(run_directory, batch_staging)
