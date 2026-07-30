"""Resumable, fail-closed HSSD transfer and Node17 publication."""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Callable, Mapping, Sequence

from data_toolkit.pipeline import training_preflight
from data_toolkit.pipeline.training_config_policy import rebase_json_paths
from data_toolkit.pipeline.training_manifest import (
    STAGES,
    SourceTrainingData,
    validate_source_training_data,
)
from data_toolkit.pipeline.training_preflight import StagePreflight
from data_toolkit.pipeline.training_source_profiles import build_source_spec


GIB = 1024**3
SOURCE_HOST = "youngwoo@n16.unist.info"
SOURCE_PORT = 55555
SOURCE_ROOT = Path(
    "/home/youngwoo/data/pixal3d/train/production/hssd"
)
NODE16_DATA2_ROOT = "/file2/youngwoo/pixal3d"
NODE17_DATA2_ROOT = Path("/root/data2/pixal3d")
STAGING_NAME = ".hssd-node16-transfer"
_STAGE_TOP_LEVEL = frozenset(STAGES)
_ROLLED_BACK_TOP_LEVEL = _STAGE_TOP_LEVEL | {
    "publication",
    "training_data.json",
}


CommandRunner = Callable[
    [Sequence[str]], subprocess.CompletedProcess[str]
]


@dataclass(frozen=True)
class Node17HssdTransferPaths:
    source_host: str
    source_port: int
    source_root: Path
    data2_root: Path
    production_root: Path
    staging_root: Path

    @property
    def canonical_root(self) -> Path:
        return self.production_root / "hssd"


@dataclass(frozen=True)
class TreeInventory:
    file_count: int
    logical_bytes: int


@dataclass(frozen=True)
class HssdTransferResult:
    source_inventory: TreeInventory
    target_inventory: TreeInventory
    original_materialization_sha256: dict[str, str]
    canonical_materialization_sha256: dict[str, str]
    training_data: Path
    training_data_sha256: str
    stage_counts: dict[str, int]
    elapsed_seconds: dict[str, float]


def _canonical_absolute(path: Path, label: str) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        raise ValueError(f"{label} must be absolute: {selected}")
    resolved = selected.resolve(strict=False)
    if selected != resolved:
        raise ValueError(
            f"{label} must be canonical and non-symlinked: "
            f"{selected} resolves to {resolved}"
        )
    return selected


def _safe_directory(path: Path, label: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ValueError(f"{label} is missing or unsafe: {path}") from error
    if not stat.S_ISDIR(mode):
        raise ValueError(f"{label} is missing or unsafe: {path}")


def _validate_safe_tree(root: Path, label: str) -> None:
    _safe_directory(root, label)
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            path = current_path / name
            try:
                mode = os.lstat(path).st_mode
            except OSError as error:
                raise ValueError(
                    f"{label} contains an unsafe path: {path}"
                ) from error
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(
                    f"{label} contains an unsafe path: {path}"
                )


def _validate_staging(paths: Node17HssdTransferPaths) -> None:
    staging = paths.staging_root
    if not os.path.lexists(staging):
        return
    try:
        _validate_safe_tree(staging, "unsafe staging")
        entries = {entry.name for entry in os.scandir(staging)}
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(f"unsafe staging directory: {staging}") from error
    if not entries <= _ROLLED_BACK_TOP_LEVEL:
        raise ValueError(
            "unsafe staging topology: "
            f"unexpected={sorted(entries - _ROLLED_BACK_TOP_LEVEL)}"
        )
    publication = staging / "publication"
    if os.path.lexists(publication):
        _safe_directory(publication, "unsafe staging publication")
        names = {entry.name for entry in os.scandir(publication)}
        if not names <= {"report.json", "handoff.json"}:
            raise ValueError(
                "unsafe staging publication topology: "
                f"unexpected={sorted(names - {'report.json', 'handoff.json'})}"
            )


def _validate_canonical_chain_binding(
    paths: Node17HssdTransferPaths,
    validated: SourceTrainingData,
) -> None:
    canonical = paths.canonical_root
    expected_paths = {
        "training data": canonical / "training_data.json",
        "report": canonical / "publication/report.json",
        "handoff": canonical / "publication/handoff.json",
    }
    actual_paths = {
        "training data": Path(validated.path),
        "report": Path(validated.report_path),
        "handoff": Path(validated.handoff_path),
    }
    if (
        validated.source != "HSSD"
        or actual_paths != expected_paths
        or set(validated.stages) != set(STAGES)
    ):
        raise ValueError(
            "validated canonical HSSD chain is not bound to its "
            f"canonical root: {canonical}"
        )
    for stage in STAGES:
        expected_data_dir = training_preflight.stage_data_dir(
            "HSSD", stage, canonical / stage / "active"
        )
        if validated.stages[stage].data_dir != expected_data_dir:
            raise ValueError(
                "validated canonical HSSD stage is not bound to its "
                f"canonical root: stage={stage} root={canonical}"
            )


def _validate_existing_canonical(
    paths: Node17HssdTransferPaths,
) -> SourceTrainingData | None:
    canonical = paths.canonical_root
    if not os.path.lexists(canonical):
        return None
    try:
        _safe_directory(canonical, "canonical HSSD")
        validated = validate_source_training_data(
            "HSSD", canonical / "training_data.json"
        )
        _validate_canonical_chain_binding(paths, validated)
        _validate_published_topology(paths)
        return validated
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "canonical HSSD conflicts with transfer and its full source "
            f"chain is invalid: {canonical}"
        ) from error


def _validate_paths(paths: Node17HssdTransferPaths) -> None:
    if not isinstance(paths, Node17HssdTransferPaths):
        raise TypeError("paths must be Node17HssdTransferPaths")
    if paths.source_host != SOURCE_HOST:
        raise ValueError(
            f"source host must be exactly {SOURCE_HOST!r}"
        )
    if type(paths.source_port) is not int or paths.source_port != SOURCE_PORT:
        raise ValueError(
            f"source port must be exactly {SOURCE_PORT}"
        )
    if Path(paths.source_root) != SOURCE_ROOT:
        raise ValueError(
            f"source root must be exactly {SOURCE_ROOT}"
        )
    if Path(paths.data2_root) != NODE17_DATA2_ROOT:
        raise ValueError(
            f"data2 root must be exactly {NODE17_DATA2_ROOT}"
        )
    data2 = _canonical_absolute(paths.data2_root, "data2 root")
    production = _canonical_absolute(
        paths.production_root, "production root"
    )
    staging = _canonical_absolute(paths.staging_root, "staging root")
    canonical = _canonical_absolute(paths.canonical_root, "canonical root")
    if staging.parent != production or staging.name != STAGING_NAME:
        raise ValueError(
            "staging root must be the exact hidden transfer directory: "
            f"{production / STAGING_NAME}"
        )
    if canonical != production / "hssd":
        raise ValueError(
            "canonical root must be production_root / 'hssd'"
        )
    if data2 != NODE17_DATA2_ROOT:
        raise ValueError(
            f"data2 root must be exactly {NODE17_DATA2_ROOT}"
        )
    if os.path.lexists(production):
        _safe_directory(production, "production root")
    _validate_staging(paths)


def _rsync_base_command(
    paths: Node17HssdTransferPaths,
) -> list[str]:
    return [
        "rsync",
        "-a",
        "--partial",
        "--info=progress2",
        "-e",
        f"ssh -p {paths.source_port}",
        *(f"--include=/{stage}/***" for stage in STAGES),
        "--exclude=*",
        f"{paths.source_host}:{paths.source_root}/",
        f"{paths.staging_root}/",
    ]


def _rsync_verification_command(
    paths: Node17HssdTransferPaths,
) -> list[str]:
    command = _rsync_base_command(paths)
    command.remove("--info=progress2")
    insertion = len(command) - 2
    command[insertion:insertion] = [
        "--checksum",
        "--dry-run",
        "--itemize-changes",
        "--delete",
    ]
    return command


_REMOTE_INVENTORY_SCRIPT = r"""
import json
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
stages = ("ss64", "shape512", "shape1024", "pbr1024")
files = 0
logical_bytes = 0
for stage in stages:
    stage_root = root / stage
    mode = os.lstat(stage_root).st_mode
    if not stat.S_ISDIR(mode):
        raise ValueError(f"unsafe stage root: {stage_root}")
    for current, directories, names in os.walk(
        stage_root, followlinks=False
    ):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if not stat.S_ISDIR(os.lstat(path).st_mode):
                raise ValueError(f"unsafe directory: {path}")
        for name in names:
            path = current_path / name
            item = os.lstat(path)
            if not stat.S_ISREG(item.st_mode) or item.st_size <= 0:
                raise ValueError(f"unsafe or empty file: {path}")
            files += 1
            logical_bytes += item.st_size
print(json.dumps({
    "file_count": files,
    "logical_bytes": logical_bytes,
}, sort_keys=True))
""".strip()

_REMOTE_MATERIALIZATIONS_SCRIPT = r"""
import base64
import json
import os
from pathlib import Path
import stat
import sys

root = Path(sys.argv[1])
stages = ("ss64", "shape512", "shape1024", "pbr1024")
materializations = {}
for stage in stages:
    path = root / stage / "active/materialization.json"
    item = os.lstat(path)
    if not stat.S_ISREG(item.st_mode) or item.st_size <= 0:
        raise ValueError(f"unsafe or empty materialization: {path}")
    materializations[stage] = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")
print(json.dumps(materializations, sort_keys=True))
""".strip()


def _remote_python_command(
    paths: Node17HssdTransferPaths, script: str
) -> list[str]:
    encoded = base64.b64encode(
        script.encode("utf-8")
    ).decode("ascii")
    remote = (
        "python3 -c \"import base64;"
        f"exec(base64.b64decode('{encoded}'))\" "
        f"{paths.source_root}"
    )
    return [
        "ssh",
        "-p",
        str(paths.source_port),
        paths.source_host,
        remote,
    ]


def _remote_inventory_command(
    paths: Node17HssdTransferPaths,
) -> list[str]:
    return _remote_python_command(paths, _REMOTE_INVENTORY_SCRIPT)


def _remote_materializations_command(
    paths: Node17HssdTransferPaths,
) -> list[str]:
    return _remote_python_command(
        paths, _REMOTE_MATERIALIZATIONS_SCRIPT
    )


def plan_hssd_transfer(
    paths: Node17HssdTransferPaths,
) -> dict[str, object]:
    """Validate immutable identities and return a non-mutating command plan."""
    _validate_paths(paths)
    existing = _validate_existing_canonical(paths)
    return {
        "execute": False,
        "source": f"{paths.source_host}:{paths.source_root}",
        "source_port": paths.source_port,
        "staging_root": str(paths.staging_root),
        "canonical_root": str(paths.canonical_root),
        "existing_canonical_valid": existing is not None,
        "source_inventory_command": _remote_inventory_command(paths),
        "transfer_command": _rsync_base_command(paths),
        "verification_command": _rsync_verification_command(paths),
    }


def _run_command(
    command: Sequence[str], command_runner: CommandRunner
) -> subprocess.CompletedProcess[str]:
    selected = [str(argument) for argument in command]
    if command_runner is subprocess.run:
        return subprocess.run(
            selected,
            check=False,
            capture_output=True,
            text=True,
        )
    return command_runner(selected)


def _checked_command(
    command: Sequence[str],
    command_runner: CommandRunner,
    label: str,
) -> subprocess.CompletedProcess[str]:
    result = _run_command(command, command_runner)
    if type(result.returncode) is not int or result.returncode != 0:
        stderr = getattr(result, "stderr", None) or ""
        raise RuntimeError(
            f"{label} failed with returncode={result.returncode}: "
            f"{stderr.strip()}"
        )
    return result


def _remote_inventory(
    paths: Node17HssdTransferPaths,
    command_runner: CommandRunner,
) -> TreeInventory:
    result = _checked_command(
        _remote_inventory_command(paths),
        command_runner,
        "remote HSSD inventory",
    )
    try:
        value = json.loads(result.stdout or "")
        if not isinstance(value, dict) or set(value) != {
            "file_count",
            "logical_bytes",
        }:
            raise ValueError("inventory must have exact fields")
        file_count = value["file_count"]
        logical_bytes = value["logical_bytes"]
        if (
            type(file_count) is not int
            or file_count <= 0
            or type(logical_bytes) is not int
            or logical_bytes <= 0
        ):
            raise ValueError("inventory counts must be positive integers")
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            "remote HSSD inventory output is invalid"
        ) from error
    return TreeInventory(file_count, logical_bytes)


def _remote_materializations(
    paths: Node17HssdTransferPaths,
    command_runner: CommandRunner,
) -> dict[str, bytes]:
    result = _checked_command(
        _remote_materializations_command(paths),
        command_runner,
        "remote HSSD materialization evidence",
    )
    try:
        value = json.loads(result.stdout or "")
        if not isinstance(value, dict) or set(value) != set(STAGES):
            raise ValueError(
                "materializations must contain all four stages"
            )
        decoded = {
            stage: base64.b64decode(value[stage], validate=True)
            for stage in STAGES
        }
        if any(not raw for raw in decoded.values()):
            raise ValueError("materialization bytes must be non-empty")
    except (
        ValueError,
        TypeError,
        json.JSONDecodeError,
        base64.binascii.Error,
    ) as error:
        raise ValueError(
            "remote HSSD materialization output is invalid"
        ) from error
    return decoded


def _existing_ancestor(path: Path) -> Path:
    selected = Path(path)
    while not selected.exists() and selected.parent != selected:
        selected = selected.parent
    return selected


def _admit_free_space(
    paths: Node17HssdTransferPaths, inventory: TreeInventory
) -> dict[str, int | str]:
    margin = max(inventory.logical_bytes // 10, 10 * GIB)
    required = inventory.logical_bytes + margin
    usage = shutil.disk_usage(_existing_ancestor(paths.production_root))
    if usage.free < required:
        raise ValueError(
            "insufficient local free space for HSSD transfer: "
            f"required={required} free={usage.free} "
            f"path={paths.production_root}"
        )
    return {
        "path": str(paths.production_root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_bytes": required,
    }


def _run_transfer(
    command: Sequence[str], command_runner: CommandRunner
) -> None:
    _checked_command(command, command_runner, "resumable HSSD rsync")


def _run_verification(
    command: Sequence[str], command_runner: CommandRunner
) -> None:
    result = _checked_command(
        command, command_runner, "HSSD checksum verification"
    )
    if (result.stdout or "").strip():
        raise ValueError(
            "HSSD checksum verification reported itemized differences: "
            f"{result.stdout.strip()}"
        )


def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise ValueError(
            f"{label} is not a regular non-symlink file: {path}"
        ) from error
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"{label} is not a regular non-symlink file: {path}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(
                f"{label} is not a regular non-symlink file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _tree_inventory(root: Path) -> TreeInventory:
    root = Path(root)
    _safe_directory(root, "transferred HSSD root")
    top_level = {entry.name for entry in os.scandir(root)}
    if not _STAGE_TOP_LEVEL <= top_level:
        raise ValueError(
            "transferred HSSD is missing stage directories: "
            f"missing={sorted(_STAGE_TOP_LEVEL - top_level)}"
        )
    if not top_level <= _ROLLED_BACK_TOP_LEVEL:
        raise ValueError(
            "transferred HSSD has unexpected top-level paths: "
            f"unexpected={sorted(top_level - _ROLLED_BACK_TOP_LEVEL)}"
        )
    file_count = 0
    logical_bytes = 0
    for stage in STAGES:
        stage_root = root / stage
        _safe_directory(stage_root, f"transferred stage={stage}")
        for current, directories, files in os.walk(
            stage_root, followlinks=False
        ):
            current_path = Path(current)
            for name in directories:
                path = current_path / name
                try:
                    mode = os.lstat(path).st_mode
                except OSError as error:
                    raise ValueError(
                        f"unsafe transferred directory: {path}"
                    ) from error
                if not stat.S_ISDIR(mode):
                    raise ValueError(
                        f"unsafe transferred directory: {path}"
                    )
            for name in files:
                path = current_path / name
                try:
                    item = os.lstat(path)
                except OSError as error:
                    raise ValueError(
                        f"unsafe transferred file: {path}"
                    ) from error
                if not stat.S_ISREG(item.st_mode) or item.st_size <= 0:
                    raise ValueError(
                        f"unsafe or empty transferred file: {path}"
                    )
                file_count += 1
                logical_bytes += item.st_size
    return TreeInventory(file_count, logical_bytes)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    current_mode = stat.S_IMODE(os.lstat(path).st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, current_mode)
        os.replace(temporary, path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _rebase_json_paths(
    value: object, replacements: Sequence[tuple[str, str]]
) -> object:
    return rebase_json_paths(value, replacements)


def _contains_text(value: object, text: str) -> bool:
    if isinstance(value, dict):
        return any(
            text in str(key) or _contains_text(item, text)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_text(item, text) for item in value)
    return isinstance(value, str) and text in value


def _json_document(raw: bytes, path: Path) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"invalid materialization JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"materialization JSON must be an object: {path}"
        )
    return value


def _staged_materialization_path(
    paths: Node17HssdTransferPaths, stage: str
) -> Path:
    return paths.staging_root / stage / "active/materialization.json"


def _node16_path(value: object, label: str) -> None:
    prefix = NODE16_DATA2_ROOT
    if not isinstance(value, str) or not (
        value == prefix or value.startswith(prefix + "/")
    ):
        raise ValueError(
            f"materialization Node16 origin has invalid {label}: {value!r}"
        )


def _validate_node16_materialization(
    paths: Node17HssdTransferPaths,
    stage: str,
    document: Mapping[str, object],
) -> None:
    expected_root = paths.source_root / stage / "active"
    source_indexes = document.get("source_indexes")
    packs = document.get("packs")
    if (
        document.get("source") != "HSSD"
        or document.get("stage") != stage
        or document.get("stage_root") != str(expected_root)
        or not isinstance(source_indexes, list)
        or not source_indexes
        or not isinstance(packs, list)
        or not packs
    ):
        raise ValueError(
            "materialization Node16 origin identity is invalid: "
            f"stage={stage}"
        )
    for index, reference in enumerate(source_indexes):
        if not isinstance(reference, Mapping):
            raise ValueError(
                "materialization Node16 origin source index is invalid: "
                f"stage={stage} index={index}"
            )
        _node16_path(
            reference.get("path"),
            f"stage={stage} source_indexes[{index}].path",
        )
    for index, pack in enumerate(packs):
        if not isinstance(pack, Mapping):
            raise ValueError(
                "materialization Node16 origin pack is invalid: "
                f"stage={stage} index={index}"
            )
        path_fields = [
            name
            for name in ("pack", "manifest", "path")
            if name in pack
        ]
        if not path_fields:
            raise ValueError(
                "materialization Node16 origin pack lacks source paths: "
                f"stage={stage} index={index}"
            )
        for name in path_fields:
            _node16_path(
                pack[name],
                f"stage={stage} packs[{index}].{name}",
            )


def _prepare_original_materializations(
    paths: Node17HssdTransferPaths,
    originals: Mapping[str, bytes],
    target_root: Path,
) -> tuple[dict[str, str], dict[str, bytes]]:
    if tuple(originals) != tuple(STAGES):
        raise ValueError(
            "original materializations must use canonical stage order"
        )
    original_digests = {}
    transformed_bytes = {}
    replacements = (
        (str(paths.source_root), str(target_root)),
        (NODE16_DATA2_ROOT, str(paths.data2_root)),
    )
    for stage in STAGES:
        path = paths.source_root / stage / "active/materialization.json"
        raw = originals[stage]
        original_digests[stage] = sha256(raw).hexdigest()
        document = _json_document(raw, path)
        _validate_node16_materialization(paths, stage, document)
        rebased = _rebase_json_paths(document, replacements)
        for old_prefix in (str(paths.source_root), NODE16_DATA2_ROOT):
            if _contains_text(rebased, old_prefix):
                raise ValueError(
                    "old Node16 prefix remains in rebased "
                    f"materialization stage={stage}: {old_prefix}"
                )
        transformed_bytes[stage] = _canonical_json_bytes(rebased)
    return original_digests, transformed_bytes


def _rebase_staged_materializations(
    paths: Node17HssdTransferPaths,
) -> dict[str, str]:
    originals = {
        stage: _regular_bytes(
            _staged_materialization_path(paths, stage),
            f"stage={stage} materialization",
        )
        for stage in STAGES
    }
    original_digests, staged_bytes = _prepare_original_materializations(
        paths, originals, paths.staging_root
    )
    for stage in STAGES:
        _atomic_write(
            _staged_materialization_path(paths, stage),
            staged_bytes[stage],
        )
    return original_digests


def _validate_runtime_configs(
    runtime_configs: Mapping[str, Path],
) -> dict[str, Path]:
    if not isinstance(runtime_configs, Mapping) or tuple(
        runtime_configs
    ) != tuple(STAGES):
        raise ValueError(
            "runtime configs must contain all four stages in canonical order"
        )
    return {
        stage: Path(runtime_configs[stage]) for stage in STAGES
    }


def _preflight_staging(
    paths: Node17HssdTransferPaths,
    runtime_configs: Mapping[str, Path],
) -> tuple[object, dict[str, StagePreflight]]:
    configs = _validate_runtime_configs(runtime_configs)
    spec = build_source_spec("hssd", paths.data2_root)
    results = {}
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        for stage in STAGES:
            results[stage] = training_preflight.preflight_stage(
                spec,
                stage,
                paths.staging_root / stage / "active",
                configs[stage],
            )
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
    return spec, results


def _promote_stage_preflight(
    paths: Node17HssdTransferPaths, result: StagePreflight
) -> StagePreflight:
    if result.stage not in STAGES:
        raise ValueError(f"unknown preflight stage: {result.stage}")
    expected_root = (
        paths.staging_root / result.stage / "active"
    )
    if Path(result.root) != expected_root:
        raise ValueError(
            "strict preflight root is not the staged active root: "
            f"stage={result.stage} actual={result.root}"
        )
    document = _json_document(
        result.materialization_bytes,
        expected_root / "materialization.json",
    )
    promoted = _rebase_json_paths(
        document,
        ((str(paths.staging_root), str(paths.canonical_root)),),
    )
    if _contains_text(promoted, str(paths.staging_root)):
        raise ValueError(
            "staging prefix remains in canonical materialization: "
            f"stage={result.stage}"
        )
    return replace(
        result,
        root=paths.canonical_root / result.stage / "active",
        materialization_bytes=_canonical_json_bytes(promoted),
    )


def _write_promoted_materializations(
    paths: Node17HssdTransferPaths,
    results: Mapping[str, StagePreflight],
) -> None:
    if tuple(results) != tuple(STAGES):
        raise ValueError(
            "promoted preflight results must use canonical stage order"
        )
    for stage in STAGES:
        _atomic_write(
            _staged_materialization_path(paths, stage),
            results[stage].materialization_bytes,
        )


def _restore_staging_materializations(
    paths: Node17HssdTransferPaths,
) -> None:
    for stage in STAGES:
        path = _staged_materialization_path(paths, stage)
        if not os.path.lexists(path):
            continue
        raw = _regular_bytes(path, f"stage={stage} rollback materialization")
        document = _json_document(raw, path)
        restored = _rebase_json_paths(
            document,
            ((str(paths.canonical_root), str(paths.staging_root)),),
        )
        _atomic_write(path, _canonical_json_bytes(restored))


def _validate_published_topology(
    paths: Node17HssdTransferPaths,
) -> None:
    canonical = paths.canonical_root
    _validate_safe_tree(canonical, "published canonical HSSD")
    top_level = {entry.name for entry in os.scandir(canonical)}
    expected = _STAGE_TOP_LEVEL | {"publication", "training_data.json"}
    if top_level != expected:
        raise ValueError(
            "published canonical HSSD topology is not exact: "
            f"expected={sorted(expected)} actual={sorted(top_level)}"
        )
    publication = canonical / "publication"
    _safe_directory(publication, "published HSSD publication")
    publication_entries = {
        entry.name for entry in os.scandir(publication)
    }
    if publication_entries != {"report.json", "handoff.json"}:
        raise ValueError(
            "published HSSD publication topology is not exact: "
            f"actual={sorted(publication_entries)}"
        )
    _regular_bytes(
        canonical / "training_data.json", "published HSSD training data"
    )


def _promote_and_publish(
    paths: Node17HssdTransferPaths,
    spec: object,
    results: Mapping[str, StagePreflight],
) -> SourceTrainingData:
    if os.path.lexists(paths.canonical_root):
        raise FileExistsError(
            f"canonical HSSD appeared before promotion: {paths.canonical_root}"
        )
    os.rename(paths.staging_root, paths.canonical_root)
    try:
        publication = paths.canonical_root / "publication"
        training_data = paths.canonical_root / "training_data.json"
        training_preflight.publish_source_handoff(
            spec,
            results,
            publication / "report.json",
            publication / "handoff.json",
            training_data,
        )
        validated = validate_source_training_data("HSSD", training_data)
        _validate_published_topology(paths)
        return validated
    except Exception:
        if os.path.lexists(paths.canonical_root):
            if os.path.lexists(paths.staging_root):
                raise RuntimeError(
                    "cannot roll canonical HSSD back because staging "
                    f"reappeared: {paths.staging_root}"
                )
            os.rename(paths.canonical_root, paths.staging_root)
            _restore_staging_materializations(paths)
        raise


def _validate_reuse_provenance(
    paths: Node17HssdTransferPaths,
    source_inventory: TreeInventory,
    originals: Mapping[str, bytes],
) -> tuple[TreeInventory, dict[str, str], dict[str, str]]:
    original_digests, expected_canonical = (
        _prepare_original_materializations(
            paths, originals, paths.canonical_root
        )
    )
    current_inventory = _tree_inventory(paths.canonical_root)
    transferred_logical_bytes = current_inventory.logical_bytes
    canonical_digests = {}
    for stage in STAGES:
        path = (
            paths.canonical_root
            / stage
            / "active/materialization.json"
        )
        canonical_raw = _regular_bytes(
            path, f"canonical stage={stage} materialization"
        )
        canonical_document = _json_document(canonical_raw, path)
        if canonical_raw != _canonical_json_bytes(canonical_document):
            raise ValueError(
                "canonical materialization bytes do not use canonical "
                f"serialization: stage={stage}"
            )
        if canonical_raw != expected_canonical[stage]:
            raise ValueError(
                "remote original evidence does not transform to current "
                f"canonical materialization: stage={stage}"
            )
        transferred_logical_bytes += (
            len(originals[stage]) - len(canonical_raw)
        )
        canonical_digests[stage] = sha256(canonical_raw).hexdigest()
    target_inventory = TreeInventory(
        current_inventory.file_count, transferred_logical_bytes
    )
    if target_inventory != source_inventory:
        raise ValueError(
            "remote source and evidence-backed target inventories differ: "
            f"source={source_inventory} target={target_inventory}"
        )
    return target_inventory, original_digests, canonical_digests


def _reuse_existing(
    paths: Node17HssdTransferPaths,
    validated: SourceTrainingData,
    source_inventory: TreeInventory,
    originals: Mapping[str, bytes],
    started: float,
) -> HssdTransferResult:
    (
        target_inventory,
        original_digests,
        canonical_digests,
    ) = _validate_reuse_provenance(
        paths, source_inventory, originals
    )
    reuse_elapsed = time.monotonic() - started
    return HssdTransferResult(
        source_inventory=source_inventory,
        target_inventory=target_inventory,
        original_materialization_sha256=original_digests,
        canonical_materialization_sha256=canonical_digests,
        training_data=paths.canonical_root / "training_data.json",
        training_data_sha256=validated.sha256,
        stage_counts={
            stage: validated.stages[stage].total_count
            for stage in STAGES
        },
        elapsed_seconds={
            "inventory": reuse_elapsed,
            "transfer": 0.0,
            "verification": 0.0,
            "evidence_rebase": 0.0,
            "strict_preflight": 0.0,
            "promotion": 0.0,
            "total": time.monotonic() - started,
        },
    )


def transfer_and_publish_hssd(
    paths: Node17HssdTransferPaths,
    runtime_configs: Mapping[str, Path],
    command_runner: CommandRunner = subprocess.run,
) -> HssdTransferResult:
    """Transfer, fully preflight, promote, and publish HSSD fail-closed."""
    started = time.monotonic()
    _validate_paths(paths)
    configs = _validate_runtime_configs(runtime_configs)
    existing = _validate_existing_canonical(paths)
    if existing is not None:
        source_inventory = _remote_inventory(paths, command_runner)
        originals = _remote_materializations(paths, command_runner)
        return _reuse_existing(
            paths, existing, source_inventory, originals, started
        )

    phase_started = time.monotonic()
    source_inventory = _remote_inventory(paths, command_runner)
    _admit_free_space(paths, source_inventory)
    inventory_elapsed = time.monotonic() - phase_started

    paths.staging_root.mkdir(exist_ok=True)
    phase_started = time.monotonic()
    _run_transfer(_rsync_base_command(paths), command_runner)
    transfer_elapsed = time.monotonic() - phase_started

    phase_started = time.monotonic()
    _run_verification(
        _rsync_verification_command(paths), command_runner
    )
    target_inventory = _tree_inventory(paths.staging_root)
    if target_inventory != source_inventory:
        raise ValueError(
            "source and target HSSD inventories differ: "
            f"source={source_inventory} target={target_inventory}"
        )
    verification_elapsed = time.monotonic() - phase_started

    phase_started = time.monotonic()
    original_digests = _rebase_staged_materializations(paths)
    rebase_elapsed = time.monotonic() - phase_started

    phase_started = time.monotonic()
    spec, staged_results = _preflight_staging(paths, configs)
    preflight_elapsed = time.monotonic() - phase_started

    phase_started = time.monotonic()
    promoted_results = {
        stage: _promote_stage_preflight(paths, staged_results[stage])
        for stage in STAGES
    }
    _write_promoted_materializations(paths, promoted_results)
    validated = _promote_and_publish(paths, spec, promoted_results)
    promotion_elapsed = time.monotonic() - phase_started

    canonical_digests = {
        stage: promoted_results[stage].materialization_sha256
        for stage in STAGES
    }
    return HssdTransferResult(
        source_inventory=source_inventory,
        target_inventory=target_inventory,
        original_materialization_sha256=original_digests,
        canonical_materialization_sha256=canonical_digests,
        training_data=paths.canonical_root / "training_data.json",
        training_data_sha256=validated.sha256,
        stage_counts={
            stage: promoted_results[stage].asset_count
            for stage in STAGES
        },
        elapsed_seconds={
            "inventory": inventory_elapsed,
            "transfer": transfer_elapsed,
            "verification": verification_elapsed,
            "evidence_rebase": rebase_elapsed,
            "strict_preflight": preflight_elapsed,
            "promotion": promotion_elapsed,
            "total": time.monotonic() - started,
        },
    )
