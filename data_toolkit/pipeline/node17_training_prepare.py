"""Create-only CPU orchestration for Node17 three-source training inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time
from typing import Mapping

from data_toolkit.pipeline.node17_hssd_transfer import (
    SOURCE_HOST,
    SOURCE_PORT,
    SOURCE_ROOT,
    HssdTransferResult,
    Node17HssdTransferPaths,
    transfer_and_publish_hssd,
    validate_hssd_transfer_paths,
)
from data_toolkit.pipeline.training_config_policy import (
    create_node17_runtime_configs,
    node17_runtime_config_evidence,
    validate_finetuning_configs,
)
from data_toolkit.pipeline.training_manifest import (
    SAMPLING,
    STAGES,
    SourceTrainingData,
    publish_combined_training_data,
    resolve_training_data,
    validate_source_training_data,
)


GIB = 1024**3
MINIMUM_FREE_BYTES = 10 * GIB
PYTHON = Path("/opt/conda/envs/pixal3d/bin/python")
CONFIGS = {
    "ss64": Path(
        "configs/gen/"
        "ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json"
    ),
    "shape512": Path(
        "configs/gen/"
        "slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json"
    ),
    "shape1024": Path(
        "configs/gen/"
        "slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json"
    ),
    "pbr1024": Path(
        "configs/gen/"
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json"
    ),
}
SOURCE_DIRECTORIES = {
    "ABO": "abo",
    "3D-FUTURE": "3d-future",
    "HSSD": "hssd",
}
SOURCE_ORDER = tuple(SOURCE_DIRECTORIES)
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_HISTORICAL_DELIVERY_DOC_PATHS = frozenset(
    {
        "docs/node17_three_source_training_runbook_ko.md",
        (
            "docs/superpowers/reports/"
            "2026-07-30-node17-three-source-training-readiness.md"
        ),
    }
)
_HISTORICAL_VALIDATOR_BOOTSTRAP_PATHS = frozenset(
    {
        "data_toolkit/pipeline/node17_hssd_transfer.py",
        "data_toolkit/pipeline/node17_training_prepare.py",
        "scripts/prepare_node17_training.py",
        "tests/multiview/test_node17_hssd_transfer.py",
        "tests/multiview/test_node17_training_prepare.py",
    }
)


@dataclass(frozen=True)
class Node17PreparationPaths:
    data2_root: Path
    local_root: Path
    repo_root: Path
    production_root: Path
    runtime_config_root: Path
    evidence_root: Path
    source_host: str
    source_port: int
    source_root: Path

    @classmethod
    def from_roots(
        cls,
        data2_root: Path,
        local_root: Path,
        repo_root: Path,
        *,
        source_host: str = SOURCE_HOST,
        source_port: int = SOURCE_PORT,
        source_root: Path = SOURCE_ROOT,
    ) -> "Node17PreparationPaths":
        local = Path(local_root)
        production = local / "train/production"
        return cls(
            data2_root=Path(data2_root),
            local_root=local,
            repo_root=Path(repo_root),
            production_root=production,
            runtime_config_root=local / "train/runtime-configs",
            evidence_root=production / "node17-preparation-evidence",
            source_host=source_host,
            source_port=source_port,
            source_root=Path(source_root),
        )

    @property
    def hssd_training_data(self) -> Path:
        return self.production_root / "hssd/training_data.json"

    @property
    def combined_training_data(self) -> Path:
        return (
            self.production_root
            / "abo-3d-future-hssd/training_data.json"
        )


def require_cpu_only_environment() -> None:
    """Require explicit CUDA hiding without importing a CUDA library."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be explicitly set to empty"
        )


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


def validate_roots(paths: Node17PreparationPaths) -> None:
    """Validate Node17-owned roots without creating any path."""
    if not isinstance(paths, Node17PreparationPaths):
        raise TypeError("paths must be Node17PreparationPaths")
    for name in ("data2_root", "local_root", "repo_root"):
        _canonical_absolute(Path(getattr(paths, name)), name)
    expected = {
        "production_root": paths.local_root / "train/production",
        "runtime_config_root": paths.local_root / "train/runtime-configs",
        "evidence_root": (
            paths.local_root
            / "train/production/node17-preparation-evidence"
        ),
    }
    for name, expected_path in expected.items():
        selected = _canonical_absolute(Path(getattr(paths, name)), name)
        if selected != expected_path:
            raise ValueError(
                f"{name} does not match local_root: "
                f"expected={expected_path} actual={selected}"
            )
    for selected, label in (
        (paths.hssd_training_data, "HSSD training data"),
        (paths.combined_training_data, "combined training data"),
    ):
        _canonical_absolute(selected, label)
    validate_hssd_transfer_paths(_hssd_transfer_paths(paths))


def clean_git_revision(repo_root: Path) -> str:
    """Return HEAD only when the selected worktree is completely clean."""
    root = _canonical_absolute(Path(repo_root), "repo root")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"repo root is not a safe directory: {root}")
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"failed to inspect Git worktree: {root}") from error
    if status.stdout:
        raise ValueError(
            f"Node17 preparation requires a clean Git worktree: {root}"
        )
    revision = head.stdout.strip()
    if _REVISION.fullmatch(revision) is None:
        raise ValueError(f"Git HEAD is not a full revision: {revision!r}")
    return revision


def _existing_git_commit(repo_root: Path, revision: str, label: str) -> None:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError(f"{label} revision is invalid or missing")
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{label} revision is missing: {revision}") from error


def _git_is_ancestor(
    repo_root: Path, ancestor: str, descendant: str
) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise ValueError("failed to inspect Git revision ancestry")
    return completed.returncode == 0


def _git_changed_paths(
    repo_root: Path, old_revision: str, new_revision: str
) -> list[str]:
    output = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{old_revision}..{new_revision}",
            "--",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    return sorted(
        path.decode("utf-8")
        for path in output.split(b"\0")
        if path
    )


def validate_historical_evidence_revision(
    repo_root: Path,
    evidence_revision: str,
    delivery_revision: str,
    validator_revision: str,
) -> dict[str, object]:
    """Bind historical evidence to a docs-only delivered descendant."""
    root = _canonical_absolute(Path(repo_root), "repo root")
    current_revision = clean_git_revision(root)
    _existing_git_commit(root, evidence_revision, "evidence")
    _existing_git_commit(root, delivery_revision, "delivery")
    _existing_git_commit(root, validator_revision, "validator")
    if not _git_is_ancestor(root, evidence_revision, delivery_revision):
        raise ValueError(
            "evidence revision is not an ancestor of delivery revision"
        )
    if not _git_is_ancestor(root, delivery_revision, validator_revision):
        raise ValueError(
            "delivery revision is not an ancestor of validator revision"
        )
    if not _git_is_ancestor(root, validator_revision, current_revision):
        raise ValueError(
            "validator revision is not an ancestor of current revision"
        )
    delivery_paths = _git_changed_paths(
        root, evidence_revision, delivery_revision
    )
    disallowed_delivery = sorted(
        set(delivery_paths) - _HISTORICAL_DELIVERY_DOC_PATHS
    )
    if disallowed_delivery:
        raise ValueError(
            "historical delivery contains disallowed changed paths: "
            f"{disallowed_delivery}"
        )
    validator_paths = _git_changed_paths(
        root, delivery_revision, validator_revision
    )
    disallowed_validator = sorted(
        set(validator_paths) - _HISTORICAL_VALIDATOR_BOOTSTRAP_PATHS
    )
    if disallowed_validator:
        raise ValueError(
            "historical validator descendant contains disallowed "
            f"changed paths: {disallowed_validator}"
        )
    finalization_paths = _git_changed_paths(
        root, validator_revision, current_revision
    )
    disallowed_finalization = sorted(
        set(finalization_paths) - _HISTORICAL_DELIVERY_DOC_PATHS
    )
    if disallowed_finalization:
        raise ValueError(
            "historical finalization contains disallowed changed paths: "
            f"{disallowed_finalization}"
        )
    return {
        "evidence_revision": evidence_revision,
        "delivery_revision": delivery_revision,
        "validator_revision": validator_revision,
        "current_revision": current_revision,
        "delivery_changed_paths": delivery_paths,
        "validator_changed_paths": validator_paths,
        "finalization_changed_paths": finalization_paths,
    }


def _source_config_paths(
    paths: Node17PreparationPaths,
) -> dict[str, Path]:
    return {
        stage: paths.repo_root / relative
        for stage, relative in CONFIGS.items()
    }


def _runtime_config_paths(
    paths: Node17PreparationPaths,
    source_configs: Mapping[str, Path],
) -> dict[str, Path]:
    return {
        stage: (
            paths.runtime_config_root
            / f"{Path(source).stem}.node17.json"
        )
        for stage, source in source_configs.items()
    }


def _regular_bytes(path: Path, label: str) -> bytes:
    selected = Path(path)
    try:
        mode = os.lstat(selected).st_mode
    except OSError as error:
        raise FileExistsError(
            f"{label} is not a regular non-symlink file: {selected}"
        ) from error
    if not stat.S_ISREG(mode):
        raise FileExistsError(
            f"{label} is not a regular non-symlink file: {selected}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(selected, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FileExistsError(
                f"{label} is not a regular non-symlink file: {selected}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _file_digest(path: Path, label: str) -> str:
    return sha256(_regular_bytes(path, label)).hexdigest()


def _json_object(raw: bytes, path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _validate_existing_runtime_configs(
    paths: Node17PreparationPaths,
    source_configs: Mapping[str, Path],
) -> None:
    root = paths.runtime_config_root
    if not os.path.lexists(root):
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            f"runtime config root is not a safe directory: {root}"
        )
    entries = {entry.name: entry for entry in os.scandir(root)}
    if not entries:
        return
    runtime_configs = _runtime_config_paths(paths, source_configs)
    expected_names = {path.name for path in runtime_configs.values()}
    if set(entries) != expected_names or any(
        not entry.is_file(follow_symlinks=False)
        for entry in entries.values()
    ):
        raise ValueError(
            "partial runtime config output requires operator inspection: "
            f"{root}"
        )
    node17_runtime_config_evidence(runtime_configs, source_configs)


def _existing_ancestor(path: Path) -> Path:
    selected = Path(path)
    while not selected.exists():
        parent = selected.parent
        if parent == selected:
            break
        selected = parent
    return selected


def assert_free_space(
    path: Path, required_bytes: int
) -> dict[str, int | str]:
    """Admit a plan without creating its future local output root."""
    if type(required_bytes) is not int or required_bytes <= 0:
        raise ValueError("required bytes must be a positive integer")
    selected = _canonical_absolute(Path(path), "disk admission root")
    usage = shutil.disk_usage(_existing_ancestor(selected))
    if usage.free < required_bytes:
        raise ValueError(
            "insufficient local free space: "
            f"required={required_bytes} free={usage.free} path={selected}"
        )
    return {
        "path": str(selected),
        "free_bytes": usage.free,
        "required_bytes": required_bytes,
    }


def _assert_cpu_only() -> None:
    require_cpu_only_environment()
    import torch

    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preparation initialized CUDA")


def preflight_multisource_stage(
    training_data: Path, stage: str, config: Path
) -> dict[str, object]:
    """Lazy boundary that keeps plan-mode imports CUDA-free."""
    from scripts.preflight_multisource_training import (
        preflight_multisource_stage as configured_preflight,
    )

    return configured_preflight(training_data, stage, config)


def preflight_training_data(
    training_data: Path, runtime_configs: Mapping[str, Path]
) -> dict[str, object]:
    """Run configured CPU DataLoader checks with CUDA guards per stage."""
    if tuple(runtime_configs) != STAGES:
        raise ValueError(
            f"runtime configs must be ordered exactly as {STAGES}"
        )
    results = {}
    elapsed = {}
    total_started = time.monotonic()
    for stage in STAGES:
        _assert_cpu_only()
        started = time.monotonic()
        try:
            results[stage] = preflight_multisource_stage(
                Path(training_data),
                stage,
                Path(runtime_configs[stage]),
            )
            elapsed[stage] = time.monotonic() - started
        finally:
            _assert_cpu_only()
    elapsed["total"] = time.monotonic() - total_started
    return {"stages": results, "elapsed_seconds": elapsed}


def _source_training_path(
    paths: Node17PreparationPaths, source: str
) -> Path:
    return (
        paths.production_root
        / SOURCE_DIRECTORIES[source]
        / "training_data.json"
    )


def _hssd_transfer_paths(
    paths: Node17PreparationPaths,
) -> Node17HssdTransferPaths:
    return Node17HssdTransferPaths(
        source_host=paths.source_host,
        source_port=paths.source_port,
        source_root=paths.source_root,
        data2_root=paths.data2_root,
        production_root=paths.production_root,
        staging_root=(
            paths.production_root / ".hssd-node16-transfer"
        ),
    )


def publish_three_source_combined(
    paths: Node17PreparationPaths,
) -> Path:
    """Create or strictly reuse the ordered ABO/3D-FUTURE/HSSD manifest."""
    output = paths.combined_training_data
    parent = output.parent
    if os.path.lexists(parent):
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError(
                f"partial combined directory is unsafe: {parent}"
            )
        entries = {entry.name: entry for entry in os.scandir(parent)}
        manifest = entries.get(output.name)
        if (
            set(entries) != {output.name}
            or manifest is None
            or not manifest.is_file(follow_symlinks=False)
        ):
            raise ValueError(
                "partial combined directory requires operator "
                f"inspection: {parent}"
            )
        for stage in STAGES:
            resolve_training_data(output, stage)
    return publish_combined_training_data(
        {
            "ABO": _source_training_path(paths, "ABO"),
            "3D-FUTURE": _source_training_path(paths, "3D-FUTURE"),
            "HSSD": _source_training_path(paths, "HSSD"),
        },
        output,
    )


def training_scope_evidence(
    training_data: Path,
) -> dict[str, dict[str, object]]:
    """Strictly resolve and summarize every combined training stage."""
    return {
        stage: {
            "source_counts": resolved.source_counts,
            "total_count": resolved.total_count,
            "union_scope_sha256": resolved.union_scope_sha256,
        }
        for stage in STAGES
        for resolved in (
            resolve_training_data(Path(training_data), stage),
        )
    }


def _source_chain_evidence(
    validated: SourceTrainingData,
) -> dict[str, object]:
    return {
        "artifacts": {
            "report": {
                "path": str(validated.report_path),
                "sha256": validated.report_sha256,
            },
            "handoff": {
                "path": str(validated.handoff_path),
                "sha256": validated.handoff_sha256,
            },
            "training_data": {
                "path": str(validated.path),
                "sha256": validated.sha256,
            },
        },
        "stages": {
            stage: {
                "source_counts": {
                    validated.source:
                        validated.stages[stage].total_count
                },
                "total_count": validated.stages[stage].total_count,
                "union_scope_sha256":
                    validated.stages[stage].union_scope_sha256,
            }
            for stage in STAGES
        },
    }


def _validated_source_chains(
    paths: Node17PreparationPaths,
    hssd_result: HssdTransferResult,
) -> tuple[dict[str, SourceTrainingData], dict[str, dict[str, object]]]:
    validated = {
        "HSSD": validate_source_training_data(
            "HSSD", paths.hssd_training_data
        ),
        "ABO": validate_source_training_data(
            "ABO", _source_training_path(paths, "ABO")
        ),
        "3D-FUTURE": validate_source_training_data(
            "3D-FUTURE",
            _source_training_path(paths, "3D-FUTURE"),
        ),
    }
    if (
        Path(hssd_result.training_data) != paths.hssd_training_data
        or hssd_result.training_data_sha256 != validated["HSSD"].sha256
    ):
        raise ValueError(
            "HSSD transfer result does not match validated publication"
        )
    for source in SOURCE_ORDER:
        selected = validated[source]
        if (
            selected.source != source
            or Path(selected.path) != _source_training_path(paths, source)
        ):
            raise ValueError(
                f"validated source chain is not canonical: {source}"
            )
    return validated, {
        source: _source_chain_evidence(validated[source])
        for source in SOURCE_ORDER
    }


def _transfer_evidence(
    paths: Node17PreparationPaths,
    result: HssdTransferResult,
    validated_hssd: SourceTrainingData,
) -> dict[str, object]:
    if (
        result.source_inventory != result.target_inventory
        or result.stage_counts
        != {
            stage: validated_hssd.stages[stage].total_count
            for stage in STAGES
        }
    ):
        raise ValueError("HSSD transfer inventory or stage counts changed")
    return {
        "source_inventory": asdict(result.source_inventory),
        "target_inventory": asdict(result.target_inventory),
        "original_materialization_sha256":
            dict(result.original_materialization_sha256),
        "canonical_materialization_sha256":
            dict(result.canonical_materialization_sha256),
        "canonical_materializations": {
            stage: {
                "path": str(
                    paths.production_root
                    / f"hssd/{stage}/active/materialization.json"
                ),
                "sha256":
                    result.canonical_materialization_sha256[stage],
            }
            for stage in STAGES
        },
        "training_data": {
            "path": str(result.training_data),
            "sha256": result.training_data_sha256,
        },
        "stage_counts": dict(result.stage_counts),
        "elapsed_seconds": dict(result.elapsed_seconds),
    }


def _launch_commands(
    paths: Node17PreparationPaths,
    runtime_configs: Mapping[str, Path],
) -> dict[str, str]:
    return {
        stage: (
            f"{PYTHON} train.py "
            f"--config {Path(runtime_configs[stage])} "
            f"--training_data {paths.combined_training_data} "
            "--num_gpus 6 --use_wandb"
        )
        for stage in STAGES
    }


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"preparation report has invalid {label} digest")
    return value


def _mapping(
    value: object, keys: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(
            f"preparation report has invalid {label} evidence"
        )
    return value


def _validate_artifact(
    record: object, path: Path, label: str
) -> None:
    artifact = _mapping(
        record, {"path", "sha256"}, f"{label} artifact"
    )
    if artifact["path"] != str(path):
        raise ValueError(
            f"preparation report has invalid {label} artifact path"
        )
    expected = _digest(artifact["sha256"], label)
    if _file_digest(path, label) != expected:
        raise ValueError(
            f"preparation report references changed {label} artifact"
        )


def _validate_inventory(value: object, label: str) -> dict[str, int]:
    inventory = _mapping(
        value, {"file_count", "logical_bytes"}, label
    )
    if (
        type(inventory["file_count"]) is not int
        or inventory["file_count"] <= 0
        or type(inventory["logical_bytes"]) is not int
        or inventory["logical_bytes"] <= 0
    ):
        raise ValueError(
            f"preparation report has invalid {label} values"
        )
    return dict(inventory)


def _validate_elapsed(
    value: object,
    label: str,
    expected_keys: set[str],
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or any(
            not isinstance(name, str)
            or type(seconds) not in (int, float)
            or seconds < 0
            or not math.isfinite(seconds)
            for name, seconds in value.items()
        )
    ):
        raise ValueError(
            f"preparation report has invalid {label} elapsed times"
        )


def _validate_preflight(
    value: object,
    counts: Mapping[str, Mapping[str, int]],
    label: str,
) -> None:
    preflight = _mapping(
        value, {"stages", "elapsed_seconds"}, label
    )
    stages = _mapping(
        preflight["stages"], set(STAGES), f"{label} stages"
    )
    _validate_elapsed(
        preflight["elapsed_seconds"], label, set(STAGES) | {"total"}
    )
    for stage in STAGES:
        expected = dict(counts[stage])
        record = _mapping(
            stages[stage],
            {
                "stage",
                "source_counts",
                "total_count",
                "sampling",
                "boundary_instances_checked",
                "collated_sources",
            },
            f"{label} stage={stage}",
        )
        if (
            record["stage"] != stage
            or record["source_counts"] != expected
            or record["total_count"] != sum(expected.values())
            or record["sampling"] != SAMPLING
            or record["boundary_instances_checked"]
            != sum(min(count, 2) for count in expected.values())
            or record["collated_sources"] != list(expected)
        ):
            raise ValueError(
                f"preparation report has invalid {label} stage={stage}"
            )


def _validate_report(
    paths: Node17PreparationPaths,
    report: Mapping[str, object],
    *,
    validated_revision: str | None = None,
) -> None:
    """Re-hash artifacts and re-resolve every immutable manifest chain."""
    top_level = {
        "schema_version",
        "cpu_only",
        "revision",
        "paths",
        "disk",
        "runtime_configs",
        "transfer",
        "sources",
        "hssd_standalone_preflight",
        "combined",
        "elapsed_seconds",
        "launch_commands",
    }
    report = _mapping(report, top_level, "top-level")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or report["cpu_only"] is not True
    ):
        raise ValueError("preparation report has invalid schema")
    revision = report["revision"]
    selected_revision = (
        clean_git_revision(paths.repo_root)
        if validated_revision is None
        else validated_revision
    )
    if (
        not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
        or selected_revision != revision
    ):
        raise ValueError(
            "preparation report has invalid or changed Git revision"
        )
    expected_paths = {
        "data2_root": str(paths.data2_root),
        "local_root": str(paths.local_root),
        "repo_root": str(paths.repo_root),
        "runtime_config_root": str(paths.runtime_config_root),
        "hssd_training_data": str(paths.hssd_training_data),
        "combined_training_data": str(paths.combined_training_data),
        "evidence_root": str(paths.evidence_root),
    }
    if report["paths"] != expected_paths:
        raise ValueError("preparation report has invalid path evidence")
    disk = _mapping(
        report["disk"],
        {"path", "free_bytes", "required_bytes"},
        "disk",
    )
    if (
        disk["path"] != str(paths.local_root)
        or type(disk["free_bytes"]) is not int
        or type(disk["required_bytes"]) is not int
        or disk["free_bytes"] < disk["required_bytes"]
        or disk["required_bytes"] != MINIMUM_FREE_BYTES
    ):
        raise ValueError("preparation report has invalid disk evidence")

    source_configs = _source_config_paths(paths)
    runtime_paths = _runtime_config_paths(paths, source_configs)
    runtime = node17_runtime_config_evidence(
        runtime_paths, source_configs
    )
    if report["runtime_configs"] != runtime:
        raise ValueError(
            "preparation report has changed runtime config evidence"
        )

    sources = _mapping(
        report["sources"], set(SOURCE_ORDER), "source"
    )
    validated_sources = {}
    source_counts = {stage: {} for stage in STAGES}
    for source in SOURCE_ORDER:
        validated = validate_source_training_data(
            source, _source_training_path(paths, source)
        )
        expected = _source_chain_evidence(validated)
        if sources[source] != expected:
            raise ValueError(
                f"preparation report has changed source={source} evidence"
            )
        validated_sources[source] = validated
        for stage in STAGES:
            source_counts[stage][source] = (
                validated.stages[stage].total_count
            )

    transfer = _mapping(
        report["transfer"],
        {
            "source_inventory",
            "target_inventory",
            "original_materialization_sha256",
            "canonical_materialization_sha256",
            "canonical_materializations",
            "training_data",
            "stage_counts",
            "elapsed_seconds",
        },
        "transfer",
    )
    source_inventory = _validate_inventory(
        transfer["source_inventory"], "source inventory"
    )
    target_inventory = _validate_inventory(
        transfer["target_inventory"], "target inventory"
    )
    if source_inventory != target_inventory:
        raise ValueError("preparation report transfer inventories differ")
    _validate_elapsed(
        transfer["elapsed_seconds"],
        "HSSD transfer",
        {
            "inventory",
            "transfer",
            "verification",
            "evidence_rebase",
            "strict_preflight",
            "promotion",
            "total",
        },
    )
    expected_hssd_counts = {
        stage: source_counts[stage]["HSSD"] for stage in STAGES
    }
    if transfer["stage_counts"] != expected_hssd_counts:
        raise ValueError(
            "preparation report has invalid HSSD transfer counts"
        )
    for name in (
        "original_materialization_sha256",
        "canonical_materialization_sha256",
    ):
        digests = _mapping(
            transfer[name], set(STAGES), name
        )
        for stage in STAGES:
            _digest(digests[stage], f"{name} stage={stage}")
    materializations = _mapping(
        transfer["canonical_materializations"],
        set(STAGES),
        "canonical materializations",
    )
    for stage in STAGES:
        _validate_artifact(
            materializations[stage],
            paths.production_root
            / f"hssd/{stage}/active/materialization.json",
            f"HSSD canonical materialization stage={stage}",
        )
        if (
            materializations[stage]["sha256"]
            != transfer["canonical_materialization_sha256"][stage]
        ):
            raise ValueError(
                "preparation report canonical HSSD digest changed "
                f"stage={stage}"
            )
    _validate_artifact(
        transfer["training_data"],
        paths.hssd_training_data,
        "HSSD transfer training_data",
    )
    if (
        transfer["training_data"]["sha256"]
        != validated_sources["HSSD"].sha256
    ):
        raise ValueError(
            "preparation report HSSD transfer chain digest changed"
        )
    _validate_preflight(
        report["hssd_standalone_preflight"],
        {
            stage: {"HSSD": source_counts[stage]["HSSD"]}
            for stage in STAGES
        },
        "HSSD standalone preflight",
    )

    combined = _mapping(
        report["combined"],
        {"path", "sha256", "stages", "preflight"},
        "combined",
    )
    _validate_artifact(
        {"path": combined["path"], "sha256": combined["sha256"]},
        paths.combined_training_data,
        "combined training_data",
    )
    current_stages = training_scope_evidence(
        paths.combined_training_data
    )
    if combined["stages"] != current_stages:
        raise ValueError(
            "preparation report has changed combined stage evidence"
        )
    for stage in STAGES:
        if current_stages[stage]["source_counts"] != source_counts[stage]:
            raise ValueError(
                "preparation report combined source counts changed "
                f"stage={stage}"
            )
    _validate_preflight(
        combined["preflight"], source_counts, "combined preflight"
    )
    _validate_elapsed(
        report["elapsed_seconds"],
        "orchestration",
        {
            "runtime_configs",
            "hssd_transfer",
            "source_validation",
            "hssd_standalone_preflight",
            "combined_publication",
            "combined_preflight",
            "total",
        },
    )
    if report["launch_commands"] != _launch_commands(
        paths, runtime_paths
    ):
        raise ValueError(
            "preparation report has invalid launch commands"
        )


def _report_invariants(report: Mapping[str, object]) -> object:
    invariant = json.loads(json.dumps(report))

    def normalize_elapsed(value: object) -> None:
        if isinstance(value, dict):
            elapsed = value.get("elapsed_seconds")
            if isinstance(elapsed, dict):
                for name in elapsed:
                    elapsed[name] = 0
            for name, item in value.items():
                if name != "elapsed_seconds":
                    normalize_elapsed(item)
        elif isinstance(value, list):
            for item in value:
                normalize_elapsed(item)

    normalize_elapsed(invariant)
    disk = invariant.get("disk")
    if isinstance(disk, dict):
        disk["free_bytes"] = 0
    return invariant


def _exclusive_create(path: Path, payload: bytes) -> None:
    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(
        selected, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def write_final_report(
    paths: Node17PreparationPaths, report: Mapping[str, object]
) -> Path:
    """Create once or validate an identical invariant report in place."""
    output = paths.evidence_root / "report.json"
    if os.path.lexists(paths.evidence_root):
        root = paths.evidence_root
        if root.is_symlink() or not root.is_dir():
            raise ValueError(
                f"partial evidence directory is unsafe: {root}"
            )
        entries = {entry.name: entry for entry in os.scandir(root)}
        selected = entries.get(output.name)
        if (
            set(entries) != {output.name}
            or selected is None
            or not selected.is_file(follow_symlinks=False)
        ):
            raise ValueError(
                "partial evidence directory requires operator "
                f"inspection: {root}"
            )
        existing = _json_object(
            _regular_bytes(output, "existing preparation report"),
            output,
            "existing preparation report",
        )
        _validate_report(paths, existing)
        _validate_report(paths, report)
        if _canonical_json_bytes(
            _report_invariants(existing)
        ) != _canonical_json_bytes(_report_invariants(report)):
            raise FileExistsError(
                "existing preparation report has a different invariant: "
                f"{output}"
            )
        return output
    _validate_report(paths, report)
    _exclusive_create(output, _canonical_json_bytes(report))
    return output


def validate_node17_preparation_report(
    paths: Node17PreparationPaths,
) -> Path:
    """Read-only revalidation entrypoint for the immutable final report."""
    validate_roots(paths)
    output = paths.evidence_root / "report.json"
    if (
        not paths.evidence_root.is_dir()
        or paths.evidence_root.is_symlink()
        or {entry.name for entry in os.scandir(paths.evidence_root)}
        != {"report.json"}
    ):
        raise ValueError(
            "partial evidence directory requires operator inspection: "
            f"{paths.evidence_root}"
        )
    report = _json_object(
        _regular_bytes(output, "preparation report"),
        output,
        "preparation report",
    )
    _validate_report(paths, report)
    return output


def validate_node17_historical_preparation_report(
    paths: Node17PreparationPaths,
    delivery_revision: str,
    validator_revision: str,
    report_sha256: str,
) -> dict[str, object]:
    """Revalidate immutable evidence from a trusted docs-only descendant."""
    validate_roots(paths)
    output = paths.evidence_root / "report.json"
    if (
        not paths.evidence_root.is_dir()
        or paths.evidence_root.is_symlink()
        or {entry.name for entry in os.scandir(paths.evidence_root)}
        != {"report.json"}
    ):
        raise ValueError(
            "partial evidence directory requires operator inspection: "
            f"{paths.evidence_root}"
        )
    if (
        not isinstance(report_sha256, str)
        or _DIGEST.fullmatch(report_sha256) is None
    ):
        raise ValueError(
            "trusted external report SHA-256 must be exactly 64 "
            "lowercase hexadecimal characters"
        )
    raw_report = _regular_bytes(output, "preparation report")
    actual_sha256 = sha256(raw_report).hexdigest()
    if not hmac.compare_digest(actual_sha256, report_sha256):
        raise ValueError(
            "trusted external report SHA-256 does not match immutable "
            f"report bytes: expected={report_sha256} "
            f"actual={actual_sha256}"
        )
    report = _json_object(
        raw_report,
        output,
        "preparation report",
    )
    revision = report.get("revision")
    if (
        not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
    ):
        raise ValueError(
            "preparation report has invalid historical Git revision"
        )
    git_evidence = validate_historical_evidence_revision(
        paths.repo_root,
        revision,
        delivery_revision,
        validator_revision,
    )
    _validate_report(
        paths, report, validated_revision=revision
    )
    return {
        "report": str(output),
        "report_sha256": actual_sha256,
        **git_evidence,
    }


def plan_node17_training(
    paths: Node17PreparationPaths,
) -> dict[str, object]:
    """Validate local admission without writes, network, or CUDA import."""
    validate_roots(paths)
    revision = clean_git_revision(paths.repo_root)
    source_configs = _source_config_paths(paths)
    validate_finetuning_configs(source_configs)
    _validate_existing_runtime_configs(paths, source_configs)
    disk = assert_free_space(paths.local_root, MINIMUM_FREE_BYTES)
    return {
        "execute": False,
        "revision": revision,
        "data2_root": str(paths.data2_root),
        "local_root": str(paths.local_root),
        "repo_root": str(paths.repo_root),
        "source_host": paths.source_host,
        "source_port": paths.source_port,
        "source_root": str(paths.source_root),
        "hssd_training_data": str(paths.hssd_training_data),
        "combined_training_data": str(paths.combined_training_data),
        "runtime_config_root": str(paths.runtime_config_root),
        "evidence_root": str(paths.evidence_root),
        "disk": disk,
    }


def prepare_node17_training(
    paths: Node17PreparationPaths,
) -> Path:
    """Execute the complete create-only Node17 preparation workflow."""
    require_cpu_only_environment()
    _assert_cpu_only()
    total_started = time.monotonic()
    elapsed = {}
    validate_roots(paths)
    revision = clean_git_revision(paths.repo_root)
    source_configs = _source_config_paths(paths)
    validate_finetuning_configs(source_configs)
    disk = assert_free_space(paths.local_root, MINIMUM_FREE_BYTES)

    started = time.monotonic()
    runtime_configs = create_node17_runtime_configs(
        source_configs, paths.runtime_config_root
    )
    runtime_evidence = node17_runtime_config_evidence(
        runtime_configs, source_configs
    )
    elapsed["runtime_configs"] = time.monotonic() - started

    started = time.monotonic()
    hssd_result = transfer_and_publish_hssd(
        _hssd_transfer_paths(paths), runtime_configs
    )
    elapsed["hssd_transfer"] = time.monotonic() - started

    started = time.monotonic()
    validated_sources, source_evidence = _validated_source_chains(
        paths, hssd_result
    )
    elapsed["source_validation"] = time.monotonic() - started

    started = time.monotonic()
    standalone = preflight_training_data(
        paths.hssd_training_data, runtime_configs
    )
    elapsed["hssd_standalone_preflight"] = (
        time.monotonic() - started
    )

    started = time.monotonic()
    combined_path = publish_three_source_combined(paths)
    elapsed["combined_publication"] = time.monotonic() - started

    started = time.monotonic()
    combined_preflight = preflight_training_data(
        combined_path, runtime_configs
    )
    elapsed["combined_preflight"] = time.monotonic() - started

    combined_sha256 = _file_digest(
        combined_path, "combined training data"
    )
    elapsed["total"] = time.monotonic() - total_started
    report = {
        "schema_version": 1,
        "cpu_only": True,
        "revision": revision,
        "paths": {
            "data2_root": str(paths.data2_root),
            "local_root": str(paths.local_root),
            "repo_root": str(paths.repo_root),
            "runtime_config_root": str(paths.runtime_config_root),
            "hssd_training_data": str(paths.hssd_training_data),
            "combined_training_data": str(
                paths.combined_training_data
            ),
            "evidence_root": str(paths.evidence_root),
        },
        "disk": disk,
        "runtime_configs": runtime_evidence,
        "transfer": _transfer_evidence(
            paths, hssd_result, validated_sources["HSSD"]
        ),
        "sources": source_evidence,
        "hssd_standalone_preflight": standalone,
        "combined": {
            "path": str(combined_path),
            "sha256": combined_sha256,
            "stages": training_scope_evidence(combined_path),
            "preflight": combined_preflight,
        },
        "elapsed_seconds": elapsed,
        "launch_commands": _launch_commands(paths, runtime_configs),
    }
    return write_final_report(paths, report)
