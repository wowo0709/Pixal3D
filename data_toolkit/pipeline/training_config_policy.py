"""Shared fine-tuning policy and machine-local runtime config generation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

from data_toolkit.pipeline.training_manifest import STAGES


@dataclass(frozen=True)
class StageTrainingPolicy:
    batch_size_per_gpu: int
    batch_split: int
    six_gpu_global_batch: int


STAGE_POLICIES = {
    "ss64": StageTrainingPolicy(8, 4, 48),
    "shape512": StageTrainingPolicy(8, 4, 48),
    "shape1024": StageTrainingPolicy(2, 1, 12),
    "pbr1024": StageTrainingPolicy(2, 1, 12),
}

NODE17_PATH_REPLACEMENTS = (
    ("/file2/youngwoo/pixal3d", "/root/data2/pixal3d"),
    ("/file3/youngwoo/pixal3d", "/root/data3/pixal3d"),
)

_GLOBAL_INTEGER_POLICY = {
    "max_steps": 20_000,
    "num_workers": 2,
    "i_print": 10,
    "i_log": 10,
    "i_sample": 1000,
    "i_save": 1000,
    "max_checkpoints": 3,
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


def _validate_output_root(root: Path) -> None:
    selected = Path(root)
    if (
        not selected.is_absolute()
        or selected != selected.resolve(strict=False)
    ):
        raise ValueError(
            f"runtime config root must be absolute and canonical: {selected}"
        )
    if os.path.lexists(selected) and (
        selected.is_symlink() or not selected.is_dir()
    ):
        raise ValueError(
            f"runtime config root is not a safe directory: {selected}"
        )


def _exclusive_create(path: Path, payload: bytes) -> None:
    selected = Path(path)
    _validate_output_root(selected.parent)
    selected.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(selected, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _validate_stage_order(configs: Mapping[str, Path], label: str) -> None:
    if tuple(configs) != STAGES:
        raise ValueError(f"{label} must be ordered exactly as {STAGES}")


def validate_finetuning_configs(
    configs: Mapping[str, Path],
) -> dict[str, dict[str, object]]:
    """Parse configs and enforce the complete current training policy."""
    _validate_stage_order(configs, "source configs")
    parsed = {}
    for stage, path in configs.items():
        selected = Path(path)
        value = _json_object(
            _regular_bytes(selected, "production config"),
            selected,
            "production config",
        )
        try:
            args = value["trainer"]["args"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"invalid source config semantics stage={stage}: {selected}"
            ) from error
        if not isinstance(args, dict):
            raise ValueError(
                f"invalid source config semantics stage={stage}: {selected}"
            )
        policy = STAGE_POLICIES[stage]
        expected = {
            "multiview_stage": stage,
            "batch_size_per_gpu": policy.batch_size_per_gpu,
            "batch_split": policy.batch_split,
            **_GLOBAL_INTEGER_POLICY,
        }
        actual = {name: args.get(name) for name in expected}
        integer_fields = set(expected) - {"multiview_stage"}
        snapshot_valid = (
            args.get("snapshot_dataset_on_start") is False
            if stage == "ss64"
            else "snapshot_dataset_on_start" not in args
        )
        if (
            actual != expected
            or any(type(actual[name]) is not int for name in integer_fields)
            or policy.batch_size_per_gpu * 6
            != policy.six_gpu_global_batch
            or snapshot_valid is not True
        ):
            raise ValueError(
                f"invalid source config semantics stage={stage}: "
                f"expected={expected} "
                f"six_gpu_global_batch={policy.six_gpu_global_batch} "
                f"actual={actual} path={selected}"
            )
        parsed[stage] = value
    return parsed


def rebase_json_paths(
    value: object,
    replacements: Sequence[tuple[str, str]],
) -> object:
    """Return a JSON-shaped copy with boundary-matched string paths rebased."""
    if isinstance(value, dict):
        return {
            key: rebase_json_paths(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rebase_json_paths(item, replacements) for item in value]
    if isinstance(value, str):
        for source_prefix, target_prefix in replacements:
            if value == source_prefix:
                return target_prefix
            if value.startswith(source_prefix + "/"):
                return target_prefix + value[len(source_prefix) :]
    return value


def create_node17_runtime_configs(
    configs: Mapping[str, Path], output_root: Path
) -> dict[str, Path]:
    """Create exact policy copies with only approved Node17 path rebasing."""
    originals = validate_finetuning_configs(configs)
    root = Path(output_root)
    _validate_output_root(root)
    outputs = {
        stage: root / f"{Path(source).stem}.node17.json"
        for stage, source in configs.items()
    }
    expected_bytes = {
        stage: _canonical_json_bytes(
            rebase_json_paths(originals[stage], NODE17_PATH_REPLACEMENTS)
        )
        for stage in STAGES
    }
    existing = {
        stage: output
        for stage, output in outputs.items()
        if os.path.lexists(output)
    }
    if existing and len(existing) != len(outputs):
        raise ValueError(
            "partial runtime config output requires operator inspection: "
            + ", ".join(str(existing[stage]) for stage in existing)
        )
    if os.path.lexists(root):
        with os.scandir(root) as stream:
            entries = {entry.name: entry for entry in stream}
        expected_names = {path.name for path in outputs.values()}
        if entries and set(entries) != expected_names:
            raise ValueError(
                "runtime config topology mismatch: "
                f"expected={sorted(expected_names)} "
                f"actual={sorted(entries)} root={root}"
            )
        for name, entry in entries.items():
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(
                    f"runtime config topology has unsafe file: {root / name}"
                )
    if existing:
        for stage, output in outputs.items():
            if (
                _regular_bytes(output, "existing runtime config")
                != expected_bytes[stage]
            ):
                raise FileExistsError(
                    "existing runtime config has different bytes: "
                    f"{output}"
                )
        return outputs
    for stage, output in outputs.items():
        _exclusive_create(output, expected_bytes[stage])
    return outputs


def _runtime_policy_evidence(
    stage: str, num_workers: int
) -> dict[str, object]:
    policy = STAGE_POLICIES[stage]
    return {
        "batch_size_per_gpu": policy.batch_size_per_gpu,
        "batch_split": policy.batch_split,
        "six_gpu_global_batch": policy.six_gpu_global_batch,
        "max_steps": 20_000,
        "save_interval": 1000,
        "retained_checkpoints": 3,
        "snapshot_interval": 1000,
        "startup_dataset_snapshot": stage != "ss64",
        "num_workers_per_rank": num_workers,
    }


def node17_runtime_config_evidence(
    runtime_configs: Mapping[str, Path],
    source_configs: Mapping[str, Path],
) -> dict[str, dict[str, object]]:
    """Prove each runtime config is exactly an approved source path rebase."""
    _validate_stage_order(runtime_configs, "runtime configs")
    originals = validate_finetuning_configs(source_configs)
    reverse_replacements = tuple(
        (target, source)
        for source, target in NODE17_PATH_REPLACEMENTS
    )
    evidence = {}
    for stage, path in runtime_configs.items():
        runtime_path = Path(path)
        raw = _regular_bytes(runtime_path, "runtime config")
        runtime = _json_object(raw, runtime_path, "runtime config")
        restored = rebase_json_paths(runtime, reverse_replacements)
        if restored != originals[stage]:
            raise ValueError(
                "runtime config is not the exact source path rebase "
                f"stage={stage}: {runtime_path}"
            )
        source_path = Path(source_configs[stage])
        source_raw = _regular_bytes(source_path, "production config")
        evidence[stage] = {
            "path": str(runtime_path),
            "sha256": sha256(raw).hexdigest(),
            "source_config": {
                "path": str(source_path),
                "sha256": sha256(source_raw).hexdigest(),
            },
            **_runtime_policy_evidence(stage, num_workers=2),
        }
    return evidence
