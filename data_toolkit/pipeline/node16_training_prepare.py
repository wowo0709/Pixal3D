"""Create-only CPU preparation of Node16 multi-source training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Mapping

from data_toolkit.pipeline.training_manifest import (
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


def validate_roots(paths: PreparationPaths) -> None:
    """Require absolute, normalized roots with no symlink traversal."""
    for name in ("data2_root", "local_root", "repo_root"):
        root = Path(getattr(paths, name))
        if not root.is_absolute():
            raise ValueError(f"{name} root must be absolute: {root}")
        canonical = root.resolve(strict=False)
        if root != canonical:
            raise ValueError(
                f"{name} root must be canonical and non-symlinked: "
                f"{root} resolves to {canonical}"
            )


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


def _exclusive_create(path: Path, payload: bytes) -> None:
    path = Path(path)
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
    discovered = _discovered_paths(Path(output_root))
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
    _validate_source_topology(
        Path(publication["training-data"]).parent
    )
    validated = validate_source_training_data(
        spec.source, Path(publication["training-data"])
    )
    _validate_selected_publication(validated, publication)
    output_root = Path(publication["training-data"]).parent
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
    spec = build_source_spec(profile, paths.data2_root)
    output = source_output_root(profile, paths.local_root)
    publication = _source_publication_paths(output)
    if os.path.lexists(publication["training-data"]):
        _validate_source_topology(output)
        return verify_existing_source(spec, publication, config_paths)
    refuse_partial_source(output)
    catalog = load_source_catalog(spec, paths.data2_root / "prepared")
    for stage in STAGES:
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
        if actual != expected:
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


def _validate_preparation_report_shape(
    paths: PreparationPaths,
    report: Mapping[str, object],
) -> None:
    top_level = {
        "schema_version",
        "cpu_only",
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
        or report["schema_version"] != 1
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
    sources = report["sources"]
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(SOURCE_PROFILE_NAMES)
    ):
        raise ValueError("preparation report has invalid source evidence")
    for profile in SOURCE_PROFILE_NAMES:
        source = sources[profile]
        if (
            not isinstance(source, Mapping)
            or "reused" not in source
            or type(source["reused"]) is not bool
        ):
            raise ValueError(
                "preparation report has invalid source reused evidence: "
                f"{profile}"
            )
    for name in (
        "hssd_standalone_preflight",
        "combined",
        "runtime_configs",
        "launch_commands",
    ):
        if not isinstance(report[name], Mapping):
            raise ValueError(
                f"preparation report has invalid {name} evidence"
            )


def write_final_report(
    paths: PreparationPaths, report: Mapping[str, object]
) -> Path:
    """Create or validate one immutable canonical preparation report."""
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


def plan_node16_training(paths: PreparationPaths) -> dict[str, object]:
    """Validate shared contracts and disk admission without local writes."""
    validate_roots(paths)
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
        "disk": {
            **disk,
            "stage_expanded_pack_bytes":
                estimate.stage_expanded_pack_bytes,
            "required_bytes": estimate.required_bytes,
        },
    }


def prepare_node16_training(
    paths: PreparationPaths,
) -> Path:
    """Execute the complete create-only Node16 preparation workflow."""
    require_cpu_only_environment()
    validate_roots(paths)
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
        "schema_version": 1,
        "cpu_only": True,
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
