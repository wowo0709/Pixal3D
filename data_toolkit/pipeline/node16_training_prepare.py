"""Create-only CPU preparation of Node16 multi-source training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
from typing import Mapping

from data_toolkit.pipeline.training_manifest import (
    AUTHORIZATION,
    SAMPLING,
    STAGES,
    publish_combined_training_data,
    resolve_training_data,
    validate_source_training_data,
)
from data_toolkit.pipeline.training_materialization import (
    STAGE_FAMILIES,
    load_source_catalog,
    materialize_stage,
)
from data_toolkit.pipeline.training_source_profiles import (
    SOURCE_PROFILE_NAMES,
    ProductionSourceSpec,
    build_source_spec,
    source_output_root,
)


GIB = 1024**3
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
_APPROVED_RUNTIME = {
    "ss64": (8, 4, 48),
    "shape512": (8, 4, 48),
    "shape1024": (2, 1, 12),
    "pbr1024": (2, 1, 12),
}


@dataclass(frozen=True)
class PreparationPaths:
    data2_root: Path
    local_root: Path
    repo_root: Path
    production_root: Path
    runtime_config_root: Path
    evidence_root: Path

    @classmethod
    def from_roots(
        cls, data2_root: Path, local_root: Path, repo_root: Path
    ) -> "PreparationPaths":
        local = Path(local_root)
        production = local / "train/production"
        return cls(
            Path(data2_root),
            local,
            Path(repo_root),
            production,
            local / "runtime-configs",
            production / "node16-preparation-evidence",
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


@dataclass(frozen=True)
class DiskEstimate:
    stage_expanded_pack_bytes: int
    required_bytes: int


@dataclass(frozen=True)
class DeploymentBinding:
    expected_revision: str
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class SourcePreparation:
    spec: ProductionSourceSpec
    output_root: Path
    publication: Mapping[str, Path]
    results: Mapping[str, object]
    validated: object | None
    reused: bool


def require_cpu_only_environment() -> None:
    """Reject execution unless CUDA is explicitly hidden."""
    if (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        or os.environ["CUDA_VISIBLE_DEVICES"] != ""
    ):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be explicitly set to empty"
        )


def _validate_canonical_owned_root(root: Path, label: str) -> None:
    """Reject non-absolute roots and any current symlink traversal."""
    selected = Path(root)
    if not selected.is_absolute():
        raise ValueError(f"{label} must be absolute: {selected}")
    canonical = selected.resolve(strict=False)
    if selected != canonical:
        raise ValueError(
            f"{label} must be canonical and non-symlinked: "
            f"{selected} resolves to {canonical}"
        )


def _validate_source_derived_roots(output_root: Path) -> None:
    output = Path(output_root)
    _validate_canonical_owned_root(output, "source output root")
    for stage in STAGES:
        stage_root = output / stage
        _validate_canonical_owned_root(
            stage_root, f"source stage={stage} root"
        )
        _validate_canonical_owned_root(
            stage_root / "active", f"source stage={stage} active root"
        )
    _validate_canonical_owned_root(
        output / "publication", "source publication root"
    )


def _validate_derived_roots(paths: PreparationPaths) -> None:
    local = Path(paths.local_root)
    expected = {
        "production root": local / "train/production",
        "runtime config root": local / "runtime-configs",
        "evidence root": (
            local
            / "train/production/node16-preparation-evidence"
        ),
    }
    selected = {
        "production root": Path(paths.production_root),
        "runtime config root": Path(paths.runtime_config_root),
        "evidence root": Path(paths.evidence_root),
    }
    for label, expected_root in expected.items():
        if selected[label] != expected_root:
            raise ValueError(
                f"{label} does not match local-root derivation: "
                f"expected={expected_root} actual={selected[label]}"
            )
        _validate_canonical_owned_root(selected[label], label)
    for profile in SOURCE_PROFILE_NAMES:
        _validate_source_derived_roots(
            source_output_root(profile, paths.local_root)
        )
    _validate_canonical_owned_root(
        paths.combined_training_data.parent, "combined output root"
    )


def validate_roots(paths: PreparationPaths) -> None:
    """Require absolute, normalized roots with no symlink traversal."""
    for name in ("data2_root", "local_root", "repo_root"):
        _validate_canonical_owned_root(
            Path(getattr(paths, name)), f"{name} root"
        )
    _validate_derived_roots(paths)


def _assert_torch_cpu_only():
    require_cpu_only_environment()
    import torch

    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preparation initialized CUDA")
    return torch


def _config_paths(paths: PreparationPaths) -> dict[str, Path]:
    return {
        stage: paths.repo_root / relative
        for stage, relative in CONFIGS.items()
    }


def _source_publication_paths(output_root: Path) -> dict[str, Path]:
    root = Path(output_root)
    return {
        "report": root / "publication/report.json",
        "handoff": root / "publication/handoff.json",
        "training-data": root / "training_data.json",
    }


def estimate_required_bytes(paths: PreparationPaths) -> DiskEstimate:
    """Validate all selected catalogs and count stage-expanded pack bytes."""
    catalogs = {
        profile: load_source_catalog(
            build_source_spec(profile, paths.data2_root),
            paths.data2_root / "prepared",
        )
        for profile in SOURCE_PROFILE_NAMES
    }
    stage_expanded = sum(
        pack.pack.stat().st_size
        for profile in SOURCE_PROFILE_NAMES
        for stage in STAGES
        for family in STAGE_FAMILIES[stage]
        for pack in catalogs[profile][family]
    )
    return DiskEstimate(
        stage_expanded_pack_bytes=stage_expanded,
        required_bytes=stage_expanded * 2 + 10 * GIB,
    )


def _existing_ancestor(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return candidate


def assert_free_space(
    path: Path, required_bytes: int
) -> dict[str, int | str]:
    """Return disk evidence or fail before any local artifact creation."""
    usage = shutil.disk_usage(_existing_ancestor(Path(path)))
    if usage.free < required_bytes:
        raise ValueError(
            "insufficient local free space: "
            f"required={required_bytes} free={usage.free} path={path}"
        )
    return {
        "path": str(Path(path)),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_bytes": required_bytes,
    }


def _regular_bytes(path: Path, label: str) -> bytes:
    path = Path(path)
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise FileExistsError(
            f"{label} is not a regular non-symlink file: {path}"
        ) from error
    if not stat.S_ISREG(mode):
        raise FileExistsError(
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
            raise FileExistsError(
                f"{label} is not a regular non-symlink file: {path}"
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


def _deployment_error(message: str) -> ValueError:
    return ValueError(f"deployment manifest {message}")


def _deployment_record(
    repo_root: Path, path: Path
) -> dict[str, object]:
    relative = path.relative_to(repo_root).as_posix()
    raw = _regular_bytes(path, "deployment file")
    mode = os.lstat(path).st_mode
    return {
        "path": relative,
        "mode": "100755" if mode & 0o111 else "100644",
        "size": len(raw),
        "sha256": sha256(raw).hexdigest(),
    }


def verify_deployment_manifest(
    repo_root: Path, binding: DeploymentBinding
) -> dict[str, object]:
    """Verify an extracted reviewed archive without consulting Git metadata."""
    if not isinstance(binding, DeploymentBinding):
        raise TypeError("deployment binding must be a DeploymentBinding")
    revision = binding.expected_revision
    manifest_digest = binding.manifest_sha256
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or not isinstance(manifest_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_digest) is None
    ):
        raise _deployment_error("binding is invalid")
    root = Path(repo_root)
    _validate_canonical_owned_root(root, "deployment repo root")
    if not root.is_dir() or root.is_symlink():
        raise _deployment_error(f"repo root is not a safe directory: {root}")
    manifest_path = Path(binding.manifest_path)
    if (
        not manifest_path.is_absolute()
        or manifest_path != manifest_path.resolve(strict=False)
    ):
        raise _deployment_error("path must be absolute and canonical")
    try:
        if manifest_path.is_relative_to(root):
            raise _deployment_error("must be stored outside the repo root")
        raw = _regular_bytes(manifest_path, "deployment manifest")
    except (FileExistsError, OSError) as error:
        raise _deployment_error("is not a regular non-symlink file") from error
    if sha256(raw).hexdigest() != manifest_digest:
        raise _deployment_error("digest does not match expected SHA-256")
    try:
        manifest = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise _deployment_error("is not valid JSON") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {"schema_version", "revision", "hash_algorithm", "files"}
        or manifest["schema_version"] != 1
        or type(manifest["schema_version"]) is not int
        or manifest["revision"] != revision
        or manifest["hash_algorithm"] != "sha256"
        or not isinstance(manifest["files"], list)
    ):
        raise _deployment_error("schema or reviewed revision is invalid")
    expected_records = []
    expected_paths: set[str] = set()
    for value in manifest["files"]:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "mode",
            "size",
            "sha256",
        }:
            raise _deployment_error("file record shape is invalid")
        relative = value["path"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or "\\" in relative
            or relative in expected_paths
            or value["mode"] not in {"100644", "100755"}
            or type(value["size"]) is not int
            or value["size"] < 0
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
        ):
            raise _deployment_error("file record is invalid")
        expected_paths.add(relative)
        expected_records.append(value)
    if expected_records != sorted(
        expected_records, key=lambda record: record["path"]
    ):
        raise _deployment_error("file inventory is not sorted")
    actual_records = []
    for directory, directory_names, file_names in os.walk(
        root, followlinks=False
    ):
        parent = Path(directory)
        for name in directory_names:
            candidate = parent / name
            if candidate.is_symlink():
                raise _deployment_error(
                    f"tree contains a symlink: {candidate}"
                )
        for name in file_names:
            candidate = parent / name
            try:
                actual_records.append(_deployment_record(root, candidate))
            except (FileExistsError, OSError) as error:
                raise _deployment_error(
                    f"tree contains an unsafe file: {candidate}"
                ) from error
    actual_records.sort(key=lambda record: record["path"])
    if actual_records != expected_records:
        raise _deployment_error("file inventory does not match reviewed archive")
    return {
        "revision": revision,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_digest,
        },
    }


def _exclusive_create(path: Path, payload: bytes) -> None:
    path = Path(path)
    _validate_canonical_owned_root(path.parent, "create-only output root")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _validate_owned_topology(
    root: Path,
    expected: Mapping[str, str],
    label: str,
) -> None:
    root = Path(root)
    _validate_canonical_owned_root(root, label)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root is not a safe directory: {root}")
    with os.scandir(root) as stream:
        entries = {entry.name: entry for entry in stream}
    if set(entries) != set(expected):
        raise ValueError(
            f"{label} topology mismatch: expected={sorted(expected)} "
            f"actual={sorted(entries)} root={root}"
        )
    for name, kind in expected.items():
        entry = entries[name]
        valid = (
            entry.is_file(follow_symlinks=False)
            if kind == "file"
            else entry.is_dir(follow_symlinks=False)
        )
        if not valid:
            raise ValueError(
                f"{label} topology has unsafe {kind}: {root / name}"
            )


def _validate_source_topology(output_root: Path) -> None:
    output = Path(output_root)
    _validate_source_derived_roots(output)
    _validate_owned_topology(
        output,
        {
            **dict.fromkeys(STAGES, "dir"),
            "publication": "dir",
            "training_data.json": "file",
        },
        "source output",
    )
    for stage in STAGES:
        _validate_owned_topology(
            output / stage, {"active": "dir"}, f"source stage={stage}"
        )
    _validate_owned_topology(
        output / "publication",
        {"report.json": "file", "handoff.json": "file"},
        "source publication",
    )


def validate_source_configs(
    configs: Mapping[str, Path],
) -> dict[str, dict[str, object]]:
    """Parse all source configs and enforce approved training semantics."""
    if tuple(configs) != STAGES:
        raise ValueError(f"source configs must be ordered exactly as {STAGES}")
    parsed = {}
    for stage, source in configs.items():
        source = Path(source)
        value = _json_object(
            _regular_bytes(source, "production config"),
            source,
            "production config",
        )
        try:
            args = value["trainer"]["args"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"invalid source config semantics: {source}"
            ) from error
        if not isinstance(args, dict):
            raise ValueError(f"invalid source config semantics: {source}")
        batch, split, global_batch = _APPROVED_RUNTIME[stage]
        expected = {
            "multiview_stage": stage,
            "batch_size_per_gpu": batch,
            "batch_split": split,
            "max_steps": 20_000,
            "i_save": 2_000,
            "max_checkpoints": 5,
            "i_sample": -1,
        }
        actual = {name: args.get(name) for name in expected}
        numeric_names = set(expected) - {"multiview_stage"}
        if (
            actual != expected
            or any(type(actual[name]) is not int for name in numeric_names)
            or batch * 6 != global_batch
        ):
            raise ValueError(
                f"invalid source config semantics stage={stage}: "
                f"expected={expected} global_batch={global_batch} "
                f"actual={actual} path={source}"
            )
        parsed[stage] = value
    return parsed


def create_runtime_configs(
    configs: Mapping[str, Path], output_root: Path
) -> dict[str, Path]:
    """Create semantic copies changing only trainer.args.num_workers."""
    _validate_canonical_owned_root(
        Path(output_root), "runtime config root"
    )
    originals = validate_source_configs(configs)
    planned_outputs = {
        stage: (
            Path(output_root)
            / f"{Path(source).stem}.node16-workers1.json"
        )
        for stage, source in configs.items()
    }
    existing_outputs = {
        stage: path
        for stage, path in planned_outputs.items()
        if os.path.lexists(path)
    }
    if existing_outputs and len(existing_outputs) != len(planned_outputs):
        raise ValueError(
            "partial runtime config output requires operator inspection: "
            + ", ".join(
                str(existing_outputs[stage])
                for stage in STAGES
                if stage in existing_outputs
            )
        )
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        _validate_owned_topology(
            output_root,
            {path.name: "file" for path in planned_outputs.values()},
            "runtime config",
        )
    outputs = {}
    for stage, source in configs.items():
        source = Path(source)
        original = originals[stage]
        runtime = json.loads(json.dumps(original))
        runtime["trainer"]["args"]["num_workers"] = 1
        output = planned_outputs[stage]
        if os.path.lexists(output):
            existing = _json_object(
                _regular_bytes(output, "existing runtime config"),
                output,
                "existing runtime config",
            )
            if (
                _canonical_json_bytes(existing)
                != _canonical_json_bytes(runtime)
            ):
                raise FileExistsError(
                    "existing runtime config has different content: "
                    f"{output}"
                )
        else:
            _exclusive_create(output, _canonical_json_bytes(runtime))
        outputs[stage] = output
    return outputs


def _discovered_paths(root: Path) -> list[Path]:
    root = Path(root)
    if not os.path.lexists(root):
        return []
    discovered = [root]
    if root.is_dir() and not root.is_symlink():
        for directory, names, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            discovered.extend(parent / name for name in sorted(names))
            discovered.extend(parent / name for name in sorted(files))
    return sorted(set(discovered), key=str)


def refuse_partial_source(output_root: Path) -> None:
    """Accept only an absent output root; never repair partial state."""
    output = Path(output_root)
    _validate_canonical_owned_root(output, "owned output root")
    discovered = _discovered_paths(output)
    if discovered:
        raise ValueError(
            "partial source output requires operator inspection: "
            + ", ".join(str(path) for path in discovered)
        )


def preflight_all_source_stages(
    spec: ProductionSourceSpec,
    output_root: Path,
    config_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Run the existing strict source preflight for every stage."""
    torch = _assert_torch_cpu_only()
    from data_toolkit.pipeline import training_preflight

    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preparation initialized CUDA")
    if spec.fixed_count_contract is not None:
        from scripts import preflight_multiview_production

        results = {}
        for stage in STAGES:
            _assert_torch_cpu_only()
            results[stage] = preflight_multiview_production.preflight_stage(
                stage,
                Path(output_root) / stage / "active",
                Path(config_paths[stage]),
            )
            _assert_torch_cpu_only()
    else:
        results = {}
        for stage in STAGES:
            _assert_torch_cpu_only()
            results[stage] = training_preflight.preflight_stage(
                spec,
                stage,
                Path(output_root) / stage / "active",
                Path(config_paths[stage]),
            )
            _assert_torch_cpu_only()
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preparation initialized CUDA")
    return results


def _validate_selected_publication(
    validated: object, publication: Mapping[str, Path]
) -> None:
    if set(publication) != {"report", "handoff", "training-data"}:
        raise ValueError("source publication paths are incomplete")
    expected = {
        "report": validated.report_path,
        "handoff": validated.handoff_path,
        "training-data": validated.path,
    }
    for label, expected_path in expected.items():
        selected = Path(publication[label])
        if selected != selected.resolve():
            raise ValueError(
                f"selected {label} path must be canonical: {selected}"
            )
        if selected != expected_path:
            raise ValueError(
                f"selected {label} path does not match validated chain: "
                f"{selected}"
            )


def verify_existing_source(
    spec: ProductionSourceSpec,
    publication: Mapping[str, Path],
    config_paths: Mapping[str, Path],
) -> SourcePreparation:
    """Validate a complete source chain and all materialized stage evidence."""
    output_root = Path(publication["training-data"]).parent
    _validate_source_topology(output_root)
    validated = validate_source_training_data(
        spec.source, Path(publication["training-data"])
    )
    _validate_selected_publication(validated, publication)
    results = preflight_all_source_stages(
        spec, output_root, config_paths
    )
    return SourcePreparation(
        spec=spec,
        output_root=output_root,
        publication=dict(publication),
        results=results,
        validated=validated,
        reused=True,
    )


def materialize_source(
    profile: str,
    paths: PreparationPaths,
    config_paths: Mapping[str, Path],
) -> SourcePreparation:
    """Materialize an absent source or fully validate a complete one."""
    _validate_canonical_owned_root(
        paths.production_root, "production root"
    )
    spec = build_source_spec(profile, paths.data2_root)
    output = source_output_root(profile, paths.local_root)
    _validate_source_derived_roots(output)
    publication = _source_publication_paths(output)
    if os.path.lexists(publication["training-data"]):
        _validate_source_topology(output)
        return verify_existing_source(spec, publication, config_paths)
    refuse_partial_source(output)
    catalog = load_source_catalog(spec, paths.data2_root / "prepared")
    for stage in STAGES:
        _validate_canonical_owned_root(output, "source output root")
        _validate_canonical_owned_root(
            output / stage, f"source stage={stage} root"
        )
        _validate_canonical_owned_root(
            output / stage / "active",
            f"source stage={stage} active root",
        )
        materialize_stage(spec, stage, catalog, output)
    results = preflight_all_source_stages(spec, output, config_paths)
    return SourcePreparation(
        spec=spec,
        output_root=output,
        publication=publication,
        results=results,
        validated=None,
        reused=False,
    )


def _source_evidence(prepared: SourcePreparation) -> dict[str, object]:
    validated = prepared.validated
    if validated is None:
        raise RuntimeError("source publication was not validated")
    report = _json_object(
        _regular_bytes(validated.report_path, "validated source report"),
        validated.report_path,
        "validated source report",
    )
    counts = report.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("validated source report counts are invalid")
    return {
        "source": validated.source,
        "reused": prepared.reused,
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
                "asset_count": validated.stages[stage].total_count,
                "asset_scope_sha256":
                    validated.stages[stage].union_scope_sha256,
                "eligibility_exclusion_count": (
                    counts["training_exclusions"][stage]
                ),
            }
            for stage in STAGES
        },
    }


def publish_source(
    profile: str,
    paths: PreparationPaths,
    prepared: SourcePreparation,
) -> dict[str, object]:
    """Publish a new source or report a previously verified source."""
    _validate_source_derived_roots(prepared.output_root)
    if prepared.spec != build_source_spec(profile, paths.data2_root):
        raise ValueError(f"source preparation does not match profile={profile}")
    if not prepared.reused:
        _assert_torch_cpu_only()
        if prepared.spec.fixed_count_contract is not None:
            from scripts import preflight_multiview_production

            evidence_from_result = (
                preflight_multiview_production.
                _materialization_evidence_from_result
            )
            materializations = {
                stage: evidence_from_result(prepared.results[stage])
                for stage in STAGES
            }
            created_at = (
                datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
            preflight_multiview_production.publish_handoff(
                prepared.spec.indexes[0],
                prepared.results,
                materializations,
                prepared.publication["report"],
                prepared.publication["handoff"],
                prepared.publication["training-data"],
                created_at,
            )
        else:
            from data_toolkit.pipeline.training_preflight import (
                publish_source_handoff,
            )

            publish_source_handoff(
                prepared.spec,
                prepared.results,
                prepared.publication["report"],
                prepared.publication["handoff"],
                prepared.publication["training-data"],
            )
        validated = validate_source_training_data(
            prepared.spec.source,
            prepared.publication["training-data"],
        )
        _validate_selected_publication(validated, prepared.publication)
        prepared = SourcePreparation(
            spec=prepared.spec,
            output_root=prepared.output_root,
            publication=prepared.publication,
            results=prepared.results,
            validated=validated,
            reused=False,
        )
    return _source_evidence(prepared)


def preflight_training_data(
    training_data: Path, config_paths: Mapping[str, Path]
) -> dict[str, object]:
    """Run existing configured Dataset/DataLoader preflight for all stages."""
    torch = _assert_torch_cpu_only()
    from scripts.preflight_multisource_training import (
        preflight_multisource_stage,
    )

    results = {}
    for stage in STAGES:
        _assert_torch_cpu_only()
        results[stage] = preflight_multisource_stage(
            Path(training_data), stage, Path(config_paths[stage])
        )
        _assert_torch_cpu_only()
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preparation initialized CUDA")
    return {"stages": results}


def publish_combined(paths: PreparationPaths) -> Path:
    """Publish the canonical three-source manifest create-only."""
    _validate_canonical_owned_root(
        paths.production_root, "production root"
    )
    _validate_canonical_owned_root(
        paths.combined_training_data.parent, "combined output root"
    )
    if os.path.lexists(paths.combined_training_data):
        _validate_owned_topology(
            paths.combined_training_data.parent,
            {"training_data.json": "file"},
            "combined output",
        )
    else:
        refuse_partial_source(paths.combined_training_data.parent)
    return publish_combined_training_data(
        {
            "ABO": paths.production_root / "abo/training_data.json",
            "3D-FUTURE": (
                paths.production_root / "3d-future/training_data.json"
            ),
            "HSSD": paths.hssd_training_data,
        },
        paths.combined_training_data,
    )


def training_scope_evidence(
    training_data: Path,
) -> dict[str, dict[str, object]]:
    """Report validated counts and union-scope digests for every stage."""
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


def runtime_config_evidence(
    configs: Mapping[str, Path],
) -> dict[str, dict[str, object]]:
    """Validate and report the approved six-GPU runtime semantics."""
    if tuple(configs) != STAGES:
        raise ValueError(f"runtime configs must be ordered exactly as {STAGES}")
    evidence = {}
    for stage, path in configs.items():
        value = _json_object(
            _regular_bytes(Path(path), "runtime config"),
            Path(path),
            "runtime config",
        )
        try:
            args = value["trainer"]["args"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid runtime config: {path}") from error
        batch, split, global_batch = _APPROVED_RUNTIME[stage]
        expected = {
            "batch_size_per_gpu": batch,
            "batch_split": split,
            "max_steps": 20_000,
            "i_save": 2_000,
            "max_checkpoints": 5,
            "i_sample": -1,
            "num_workers": 1,
            "multiview_stage": stage,
        }
        actual = {name: args.get(name) for name in expected}
        numeric_names = set(expected) - {"multiview_stage"}
        if (
            actual != expected
            or any(
                type(actual[name]) is not int
                for name in numeric_names
            )
        ):
            raise ValueError(
                f"runtime config changes approved semantics: {path}: "
                f"expected={expected} actual={actual}"
            )
        evidence[stage] = {
            "path": str(Path(path)),
            "sha256": sha256(
                _regular_bytes(Path(path), "runtime config")
            ).hexdigest(),
            "batch_size_per_gpu": batch,
            "batch_split": split,
            "six_gpu_global_batch": global_batch,
            "max_steps": 20_000,
            "save_interval": 2_000,
            "retained_checkpoints": 5,
            "snapshots_disabled": True,
            "num_workers_per_rank": 1,
        }
    return evidence


def _launch_commands(
    paths: PreparationPaths, configs: Mapping[str, Path]
) -> dict[str, str]:
    return {
        stage: (
            f"cd {paths.repo_root} && python train.py "
            f"--config {config} "
            f"--training_data {paths.combined_training_data} "
            "--num_gpus 6 --use_wandb"
        )
        for stage, config in configs.items()
    }


def _report_invariants(
    report: Mapping[str, object],
) -> dict[str, object]:
    invariant = json.loads(json.dumps(report))
    disk = invariant.get("disk")
    if isinstance(disk, dict):
        for name in ("total_bytes", "used_bytes", "free_bytes"):
            disk.pop(name, None)
    sources = invariant.get("sources")
    if isinstance(sources, dict):
        for source in sources.values():
            if isinstance(source, dict):
                source.pop("reused", None)
    return invariant


def _report_mapping(
    value: object, keys: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(
            f"preparation report has invalid {label} evidence"
        )
    return value


def _report_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(
            f"preparation report has invalid {label} digest"
        )
    return value


def _report_artifact(
    value: object, expected_path: Path, label: str
) -> None:
    record = _report_mapping(
        value, {"path", "sha256"}, f"{label} artifact"
    )
    expected = Path(expected_path)
    selected = record["path"]
    if (
        not isinstance(selected, str)
        or selected != str(expected)
        or expected != expected.resolve(strict=False)
    ):
        raise ValueError(
            f"preparation report has invalid {label} artifact path"
        )
    digest = _report_digest(record["sha256"], label)
    try:
        actual = sha256(_regular_bytes(expected, label)).hexdigest()
    except (FileExistsError, OSError) as error:
        raise ValueError(
            f"preparation report has invalid {label} artifact"
        ) from error
    if actual != digest:
        raise ValueError(
            f"preparation report has changed {label} artifact"
        )


def _validate_report_deployment(
    paths: PreparationPaths, value: object
) -> None:
    deployment = _report_mapping(
        value, {"revision", "manifest"}, "deployment"
    )
    manifest = _report_mapping(
        deployment["manifest"],
        {"path", "sha256"},
        "deployment manifest",
    )
    revision = deployment["revision"]
    path = manifest["path"]
    digest = manifest["sha256"]
    if not isinstance(path, str):
        raise ValueError(
            "preparation report has invalid deployment manifest path"
        )
    try:
        actual = verify_deployment_manifest(
            paths.repo_root,
            DeploymentBinding(
                expected_revision=revision,
                manifest_path=Path(path),
                manifest_sha256=digest,
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "preparation report has invalid deployment evidence"
        ) from error
    if dict(deployment) != actual:
        raise ValueError(
            "preparation report has invalid deployment evidence"
        )


def _validate_report_sources(
    paths: PreparationPaths, value: object
) -> dict[str, dict[str, int]]:
    sources = _report_mapping(
        value, set(SOURCE_PROFILE_NAMES), "source"
    )
    source_counts = {stage: {} for stage in STAGES}
    for profile in SOURCE_PROFILE_NAMES:
        source = _report_mapping(
            sources[profile],
            {"source", "reused", "artifacts", "stages"},
            f"source={profile}",
        )
        spec = build_source_spec(profile, paths.data2_root)
        if (
            source["source"] != spec.source
            or type(source["reused"]) is not bool
        ):
            raise ValueError(
                f"preparation report has invalid source={profile} identity"
            )
        output = source_output_root(profile, paths.local_root)
        artifacts = _report_mapping(
            source["artifacts"],
            {"report", "handoff", "training_data"},
            f"source={profile} artifacts",
        )
        for label, relative in {
            "report": Path("publication/report.json"),
            "handoff": Path("publication/handoff.json"),
            "training_data": Path("training_data.json"),
        }.items():
            _report_artifact(
                artifacts[label],
                output / relative,
                f"source={profile} {label}",
            )
        stages = _report_mapping(
            source["stages"], set(STAGES), f"source={profile} stages"
        )
        for stage in STAGES:
            record = _report_mapping(
                stages[stage],
                {
                    "asset_count",
                    "asset_scope_sha256",
                    "eligibility_exclusion_count",
                },
                f"source={profile} stage={stage}",
            )
            count = record["asset_count"]
            exclusions = record["eligibility_exclusion_count"]
            if (
                type(count) is not int
                or count <= 0
                or type(exclusions) is not int
                or exclusions < 0
            ):
                raise ValueError(
                    "preparation report has invalid source stage counts"
                )
            if spec.fixed_count_contract is None:
                expected_count = (
                    spec.expected_candidate_stages[stage] - exclusions
                )
            else:
                expected_count = spec.fixed_count_contract["stages"][stage]
                if (
                    exclusions
                    != spec.fixed_count_contract["training_exclusions"][
                        stage
                    ]
                ):
                    raise ValueError(
                        "preparation report has invalid fixed source "
                        "exclusions"
                    )
            if (
                count != expected_count
                or exclusions > spec.expected_candidate_stages[stage]
            ):
                raise ValueError(
                    "preparation report has invalid source count contract"
                )
            _report_digest(
                record["asset_scope_sha256"],
                f"source={profile} stage={stage} scope",
            )
            source_counts[stage][spec.source] = count
        try:
            report_document = _json_object(
                _regular_bytes(
                    output / "publication/report.json",
                    f"source={profile} report",
                ),
                output / "publication/report.json",
                f"source={profile} report",
            )
            handoff_document = _json_object(
                _regular_bytes(
                    output / "publication/handoff.json",
                    f"source={profile} handoff",
                ),
                output / "publication/handoff.json",
                f"source={profile} handoff",
            )
            training_document = _json_object(
                _regular_bytes(
                    output / "training_data.json",
                    f"source={profile} training_data",
                ),
                output / "training_data.json",
                f"source={profile} training_data",
            )
        except (FileExistsError, OSError, ValueError) as error:
            raise ValueError(
                f"preparation report has invalid source={profile} "
                "publication artifacts"
            ) from error
        expected_handoff = {
            **report_document,
            "report": dict(artifacts["report"]),
        }
        expected_training = {
            **handoff_document,
            "handoff": dict(artifacts["handoff"]),
        }
        if (
            report_document.get("source") != spec.source
            or handoff_document != expected_handoff
            or training_document != expected_training
        ):
            raise ValueError(
                f"preparation report has invalid source={profile} "
                "publication chain"
            )
        report_counts = report_document.get("counts")
        report_stages = report_document.get("stages")
        if (
            not isinstance(report_counts, Mapping)
            or not isinstance(report_stages, Mapping)
            or set(report_stages) != set(STAGES)
            or not isinstance(
                report_counts.get("training_exclusions"), Mapping
            )
            or not isinstance(report_counts.get("stages"), Mapping)
        ):
            raise ValueError(
                f"preparation report has invalid source={profile} "
                "publication counts"
            )
        for stage in STAGES:
            published_stage = report_stages[stage]
            if (
                not isinstance(published_stage, Mapping)
                or published_stage.get("asset_count")
                != stages[stage]["asset_count"]
                or published_stage.get("asset_scope_sha256")
                != stages[stage]["asset_scope_sha256"]
                or report_counts["training_exclusions"].get(stage)
                != stages[stage]["eligibility_exclusion_count"]
                or report_counts["stages"].get(stage)
                != stages[stage]["asset_count"]
            ):
                raise ValueError(
                    f"preparation report has invalid source={profile} "
                    f"published stage={stage} evidence"
                )
        try:
            validated = validate_source_training_data(
                spec.source, output / "training_data.json"
            )
        except (FileExistsError, OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"preparation report source={profile} source trust "
                f"validation failed: {error}"
            ) from error
        if (
            validated.source != spec.source
            or validated.path != output / "training_data.json"
            or validated.sha256 != artifacts["training_data"]["sha256"]
            or validated.report_path != output / "publication/report.json"
            or validated.report_sha256 != artifacts["report"]["sha256"]
            or validated.handoff_path
            != output / "publication/handoff.json"
            or validated.handoff_sha256 != artifacts["handoff"]["sha256"]
        ):
            raise ValueError(
                f"preparation report has invalid source={profile} "
                "validated artifact evidence"
            )
        for stage in STAGES:
            validated_stage = validated.stages[stage]
            if (
                validated_stage.total_count
                != stages[stage]["asset_count"]
                or validated_stage.union_scope_sha256
                != stages[stage]["asset_scope_sha256"]
            ):
                raise ValueError(
                    f"preparation report has invalid source={profile} "
                    f"validated stage={stage} evidence"
                )
    return source_counts


def _validate_report_preflight(
    value: object,
    expected_counts: Mapping[str, Mapping[str, int]],
    label: str,
) -> None:
    container = _report_mapping(value, {"stages"}, label)
    stages = _report_mapping(
        container["stages"], set(STAGES), f"{label} stages"
    )
    for stage in STAGES:
        record = _report_mapping(
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
        counts = expected_counts[stage]
        total = sum(counts.values())
        if (
            record["stage"] != stage
            or record["source_counts"] != counts
            or not isinstance(record["source_counts"], Mapping)
            or type(record["total_count"]) is not int
            or record["total_count"] != total
            or record["sampling"] != SAMPLING
            or type(record["boundary_instances_checked"]) is not int
            or record["boundary_instances_checked"]
            != sum(min(count, 2) for count in counts.values())
            or record["collated_sources"] != list(counts)
        ):
            raise ValueError(
                f"preparation report has invalid {label} stage={stage}"
            )


def _validate_report_runtime(
    paths: PreparationPaths, value: object
) -> dict[str, Path]:
    runtime = _report_mapping(
        value, set(STAGES), "runtime config"
    )
    paths_by_stage = {
        stage: (
            paths.runtime_config_root
            / f"{source.stem}.node16-workers1.json"
        )
        for stage, source in CONFIGS.items()
    }
    for stage, expected_path in paths_by_stage.items():
        record = runtime[stage]
        if (
            not isinstance(record, Mapping)
            or record.get("path") != str(expected_path)
        ):
            raise ValueError(
                "preparation report has invalid runtime config path"
            )
    try:
        expected = runtime_config_evidence(paths_by_stage)
    except (FileExistsError, OSError, ValueError) as error:
        raise ValueError(
            "preparation report has invalid runtime config evidence"
        ) from error
    if runtime != expected:
        raise ValueError(
            "preparation report has invalid runtime config semantics"
        )
    return paths_by_stage


def _validate_report_combined(
    paths: PreparationPaths,
    value: object,
    source_counts: Mapping[str, Mapping[str, int]],
) -> None:
    combined = _report_mapping(
        value, {"path", "sha256", "stages", "preflight"}, "combined"
    )
    _report_artifact(
        {"path": combined["path"], "sha256": combined["sha256"]},
        paths.combined_training_data,
        "combined training_data",
    )
    try:
        combined_document = _json_object(
            _regular_bytes(
                paths.combined_training_data,
                "combined training_data",
            ),
            paths.combined_training_data,
            "combined training_data",
        )
    except (FileExistsError, OSError, ValueError) as error:
        raise ValueError(
            "preparation report has invalid combined artifact"
        ) from error
    expected_source_references = {}
    for profile in SOURCE_PROFILE_NAMES:
        spec = build_source_spec(profile, paths.data2_root)
        output = source_output_root(profile, paths.local_root)
        training_path = output / "training_data.json"
        handoff_path = output / "publication/handoff.json"
        expected_source_references[spec.source] = {
            "training_data": {
                "path": str(training_path),
                "sha256": sha256(
                    _regular_bytes(
                        training_path,
                        f"source={profile} training_data",
                    )
                ).hexdigest(),
            },
            "handoff": {
                "path": str(handoff_path),
                "sha256": sha256(
                    _regular_bytes(
                        handoff_path,
                        f"source={profile} handoff",
                    )
                ).hexdigest(),
            },
        }
    if (
        set(combined_document)
        != {
            "schema_version",
            "authorization",
            "sampling",
            "sources",
            "stages",
        }
        or combined_document.get("schema_version") != 1
        or type(combined_document.get("schema_version")) is not int
        or combined_document.get("authorization") != AUTHORIZATION
        or combined_document.get("sampling") != SAMPLING
        or combined_document.get("sources")
        != expected_source_references
    ):
        raise ValueError(
            "preparation report has invalid combined artifact semantics"
        )
    published_stages = combined_document.get("stages")
    if (
        not isinstance(published_stages, Mapping)
        or set(published_stages) != set(STAGES)
    ):
        raise ValueError(
            "preparation report has invalid combined artifact stages"
        )
    stages = _report_mapping(
        combined["stages"], set(STAGES), "combined stages"
    )
    for stage in STAGES:
        record = _report_mapping(
            stages[stage],
            {"source_counts", "total_count", "union_scope_sha256"},
            f"combined stage={stage}",
        )
        expected = source_counts[stage]
        if (
            record["source_counts"] != expected
            or not isinstance(record["source_counts"], Mapping)
            or type(record["total_count"]) is not int
            or record["total_count"] != sum(expected.values())
        ):
            raise ValueError(
                "preparation report has invalid combined stage counts"
            )
        _report_digest(
            record["union_scope_sha256"],
            f"combined stage={stage} scope",
        )
        published = published_stages[stage]
        if (
            not isinstance(published, Mapping)
            or set(published)
            != {
                "source_counts",
                "total_count",
                "union_scope_sha256",
                "data_dir",
            }
            or published.get("source_counts") != record["source_counts"]
            or published.get("total_count") != record["total_count"]
            or published.get("union_scope_sha256")
            != record["union_scope_sha256"]
            or not isinstance(published.get("data_dir"), Mapping)
        ):
            raise ValueError(
                "preparation report has invalid combined published "
                f"stage={stage} evidence"
            )
        try:
            resolved = resolve_training_data(
                paths.combined_training_data, stage
            )
        except (FileExistsError, OSError, TypeError, ValueError) as error:
            raise ValueError(
                "preparation report combined trust validation failed "
                f"for stage={stage}: {error}"
            ) from error
        if (
            resolved.path != paths.combined_training_data
            or resolved.manifest_sha256 != combined["sha256"]
            or resolved.source_counts != record["source_counts"]
            or resolved.total_count != record["total_count"]
            or resolved.union_scope_sha256
            != record["union_scope_sha256"]
        ):
            raise ValueError(
                "preparation report has invalid combined validated "
                f"stage={stage} evidence"
            )
    _validate_report_preflight(
        combined["preflight"], source_counts, "combined preflight"
    )


def _validate_preparation_report_shape(
    paths: PreparationPaths,
    report: Mapping[str, object],
) -> None:
    top_level = {
        "schema_version",
        "cpu_only",
        "deployment",
        "paths",
        "disk",
        "sources",
        "hssd_standalone_preflight",
        "combined",
        "runtime_configs",
        "launch_commands",
    }
    if not isinstance(report, Mapping) or set(report) != top_level:
        raise ValueError("preparation report has invalid top-level shape")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != 2
        or report["cpu_only"] is not True
    ):
        raise ValueError("preparation report has invalid schema evidence")
    expected_paths = {
        "data2_root": str(paths.data2_root),
        "local_root": str(paths.local_root),
        "repo_root": str(paths.repo_root),
        "hssd_training_data": str(paths.hssd_training_data),
        "combined_training_data": str(paths.combined_training_data),
    }
    if report["paths"] != expected_paths:
        raise ValueError("preparation report has invalid path evidence")
    _validate_report_deployment(paths, report["deployment"])
    disk = report["disk"]
    disk_keys = {
        "path",
        "total_bytes",
        "used_bytes",
        "free_bytes",
        "required_bytes",
        "stage_expanded_pack_bytes",
    }
    if not isinstance(disk, Mapping) or set(disk) != disk_keys:
        raise ValueError("preparation report has invalid disk evidence")
    numeric_names = disk_keys - {"path"}
    if (
        disk["path"] != str(paths.local_root)
        or any(
            type(disk[name]) is not int or disk[name] < 0
            for name in numeric_names
        )
        or disk["total_bytes"] != disk["used_bytes"] + disk["free_bytes"]
        or disk["free_bytes"] < disk["required_bytes"]
        or disk["required_bytes"]
        != disk["stage_expanded_pack_bytes"] * 2 + 10 * GIB
    ):
        raise ValueError("preparation report has invalid disk invariants")
    source_counts = _validate_report_sources(paths, report["sources"])
    hssd_counts = {
        stage: {"HSSD": source_counts[stage]["HSSD"]}
        for stage in STAGES
    }
    _validate_report_preflight(
        report["hssd_standalone_preflight"],
        hssd_counts,
        "HSSD standalone preflight",
    )
    _validate_report_combined(paths, report["combined"], source_counts)
    runtime_paths = _validate_report_runtime(
        paths, report["runtime_configs"]
    )
    expected_launch = _launch_commands(paths, runtime_paths)
    if report["launch_commands"] != expected_launch:
        raise ValueError(
            "preparation report has invalid launch command evidence"
        )


def write_final_report(
    paths: PreparationPaths, report: Mapping[str, object]
) -> Path:
    """Create or validate one immutable canonical preparation report."""
    _validate_canonical_owned_root(
        paths.production_root, "production root"
    )
    _validate_canonical_owned_root(paths.evidence_root, "evidence root")
    output = paths.evidence_root / "report.json"
    if os.path.lexists(output):
        _validate_owned_topology(
            paths.evidence_root,
            {"report.json": "file"},
            "evidence output",
        )
        raw = _regular_bytes(output, "existing preparation report")
        existing = _json_object(
            raw, output, "existing preparation report"
        )
        _validate_preparation_report_shape(paths, existing)
        _validate_preparation_report_shape(paths, report)
        if (
            _canonical_json_bytes(_report_invariants(existing))
            != _canonical_json_bytes(_report_invariants(report))
        ):
            raise FileExistsError(
                f"existing preparation report has different content: {output}"
            )
    else:
        refuse_partial_source(paths.evidence_root)
        _validate_preparation_report_shape(paths, report)
        payload = _canonical_json_bytes(report)
        _exclusive_create(output, payload)
    return output


def plan_node16_training(
    paths: PreparationPaths, deployment: DeploymentBinding
) -> dict[str, object]:
    """Validate shared contracts and disk admission without local writes."""
    validate_roots(paths)
    deployment_evidence = verify_deployment_manifest(
        paths.repo_root, deployment
    )
    estimate = estimate_required_bytes(paths)
    disk = assert_free_space(paths.local_root, estimate.required_bytes)
    return {
        "execute": False,
        "data2_root": str(paths.data2_root),
        "local_root": str(paths.local_root),
        "repo_root": str(paths.repo_root),
        "hssd_training_data": str(paths.hssd_training_data),
        "combined_training_data": str(paths.combined_training_data),
        "runtime_config_root": str(paths.runtime_config_root),
        "evidence_root": str(paths.evidence_root),
        "deployment": deployment_evidence,
        "disk": {
            **disk,
            "stage_expanded_pack_bytes":
                estimate.stage_expanded_pack_bytes,
            "required_bytes": estimate.required_bytes,
        },
    }


def prepare_node16_training(
    paths: PreparationPaths,
    deployment: DeploymentBinding,
) -> Path:
    """Execute the complete create-only Node16 preparation workflow."""
    require_cpu_only_environment()
    validate_roots(paths)
    deployment_evidence = verify_deployment_manifest(
        paths.repo_root, deployment
    )
    source_configs = _config_paths(paths)
    validate_source_configs(source_configs)
    estimate = estimate_required_bytes(paths)
    disk = assert_free_space(paths.local_root, estimate.required_bytes)
    runtime_configs = create_runtime_configs(
        source_configs, paths.runtime_config_root
    )
    sources = {}
    for profile in SOURCE_PROFILE_NAMES:
        prepared = materialize_source(profile, paths, runtime_configs)
        sources[profile] = publish_source(profile, paths, prepared)
    standalone = preflight_training_data(
        paths.hssd_training_data, runtime_configs
    )
    combined_path = publish_combined(paths)
    combined = preflight_training_data(
        paths.combined_training_data, runtime_configs
    )
    report = {
        "schema_version": 2,
        "cpu_only": True,
        "deployment": deployment_evidence,
        "paths": {
            "data2_root": str(paths.data2_root),
            "local_root": str(paths.local_root),
            "repo_root": str(paths.repo_root),
            "hssd_training_data": str(paths.hssd_training_data),
            "combined_training_data": str(paths.combined_training_data),
        },
        "disk": {
            **disk,
            "stage_expanded_pack_bytes":
                estimate.stage_expanded_pack_bytes,
            "required_bytes": estimate.required_bytes,
        },
        "sources": sources,
        "hssd_standalone_preflight": standalone,
        "combined": {
            "path": str(combined_path),
            "sha256": sha256(
                _regular_bytes(
                    paths.combined_training_data,
                    "combined training data",
                )
            ).hexdigest()
            if os.path.lexists(paths.combined_training_data)
            else None,
            "stages": training_scope_evidence(
                paths.combined_training_data
            ),
            "preflight": combined,
        },
        "runtime_configs": runtime_config_evidence(runtime_configs),
        "launch_commands": _launch_commands(paths, runtime_configs),
    }
    return write_final_report(paths, report)
