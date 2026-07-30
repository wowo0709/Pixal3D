from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline import node16_training_prepare as core
from data_toolkit.pipeline.node16_training_prepare import (
    CONFIGS,
    GIB,
    DiskEstimate,
    PreparationPaths,
    assert_free_space,
    create_runtime_configs,
    estimate_required_bytes,
    materialize_source,
    prepare_node16_training,
    refuse_partial_source,
    verify_existing_source,
)


def _paths(tmp_path: Path) -> PreparationPaths:
    return PreparationPaths.from_roots(
        data2_root=tmp_path / "shared",
        local_root=tmp_path / "local",
        repo_root=tmp_path / "repo",
    )


def _deployment_binding(
    tmp_path: Path,
    repo_root: Path,
    *,
    revision: str = "a" * 40,
) -> object:
    files = []
    if repo_root.exists():
        for path in sorted(
            candidate
            for candidate in repo_root.rglob("*")
            if candidate.is_file()
        ):
            raw = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(repo_root).as_posix(),
                    "mode": (
                        "100755"
                        if path.stat().st_mode & 0o111
                        else "100644"
                    ),
                    "size": len(raw),
                    "sha256": sha256(raw).hexdigest(),
                }
            )
    manifest = tmp_path / "deployment-manifest.json"
    payload = core._canonical_json_bytes(
        {
            "schema_version": 1,
            "revision": revision,
            "hash_algorithm": "sha256",
            "files": files,
        }
    )
    manifest.write_bytes(payload)
    return core.DeploymentBinding(
        expected_revision=revision,
        manifest_path=manifest,
        manifest_sha256=sha256(payload).hexdigest(),
    )


def _valid_final_report(paths: PreparationPaths) -> dict[str, object]:
    paths.repo_root.mkdir(parents=True, exist_ok=True)
    reviewed_marker = paths.repo_root / "reviewed.txt"
    if not reviewed_marker.exists():
        reviewed_marker.write_text("reviewed\n")
    source_configs = _copy_production_configs(paths.repo_root)
    binding = _deployment_binding(
        paths.local_root.parent, paths.repo_root
    )
    stage_bytes = 50
    required = stage_bytes * 2 + 10 * GIB
    free = required + 1_000
    used = 200
    source_names = {
        "abo": "ABO",
        "3d-future": "3D-FUTURE",
        "hssd": "HSSD",
    }
    sources = {}
    for profile, source_name in source_names.items():
        output = core.source_output_root(profile, paths.local_root)
        spec = core.build_source_spec(profile, paths.data2_root)
        stages = {}
        for stage in core.STAGES:
            if spec.fixed_count_contract is not None:
                final_count = spec.fixed_count_contract["stages"][stage]
                exclusion_count = spec.fixed_count_contract[
                    "training_exclusions"
                ][stage]
            else:
                exclusion_count = 0
                final_count = spec.expected_candidate_stages[stage]
            stages[stage] = {
                "asset_count": final_count,
                "asset_scope_sha256": sha256(
                    f"{source_name}:{stage}:scope".encode()
                ).hexdigest(),
                "eligibility_exclusion_count": exclusion_count,
            }
        report_path = output / "publication/report.json"
        handoff_path = output / "publication/handoff.json"
        training_path = output / "training_data.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_document = {
            "source": source_name,
            "counts": {
                "training_exclusions": {
                    stage: stages[stage][
                        "eligibility_exclusion_count"
                    ]
                    for stage in core.STAGES
                },
                "stages": {
                    stage: stages[stage]["asset_count"]
                    for stage in core.STAGES
                },
            },
            "stages": {
                stage: {
                    "asset_count": stages[stage]["asset_count"],
                    "asset_scope_sha256": stages[stage][
                        "asset_scope_sha256"
                    ],
                }
                for stage in core.STAGES
            },
        }
        report_path.write_bytes(
            core._canonical_json_bytes(report_document)
        )
        handoff_document = {
            **report_document,
            "report": {
                "path": str(report_path),
                "sha256": sha256(report_path.read_bytes()).hexdigest(),
            },
        }
        handoff_path.write_bytes(
            core._canonical_json_bytes(handoff_document)
        )
        training_document = {
            **handoff_document,
            "handoff": {
                "path": str(handoff_path),
                "sha256": sha256(handoff_path.read_bytes()).hexdigest(),
            },
        }
        training_path.write_bytes(
            core._canonical_json_bytes(training_document)
        )
        artifacts = {
            "report": {
                "path": str(report_path),
                "sha256": sha256(report_path.read_bytes()).hexdigest(),
            },
            "handoff": {
                "path": str(handoff_path),
                "sha256": sha256(handoff_path.read_bytes()).hexdigest(),
            },
            "training_data": {
                "path": str(training_path),
                "sha256": sha256(training_path.read_bytes()).hexdigest(),
            },
        }
        sources[profile] = {
            "source": source_name,
            "reused": False,
            "artifacts": artifacts,
            "stages": stages,
        }
    combined_path = paths.combined_training_data
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_stages = {}
    standalone_preflight = {"stages": {}}
    combined_preflight = {"stages": {}}
    for stage in core.STAGES:
        source_counts = {
            source_names[profile]:
                sources[profile]["stages"][stage]["asset_count"]
            for profile in source_names
        }
        total = sum(source_counts.values())
        combined_stages[stage] = {
            "source_counts": source_counts,
            "total_count": total,
            "union_scope_sha256": sha256(
                f"combined:{stage}:scope".encode()
            ).hexdigest(),
        }
        hssd_count = source_counts["HSSD"]
        standalone_preflight["stages"][stage] = {
            "stage": stage,
            "source_counts": {"HSSD": hssd_count},
            "total_count": hssd_count,
            "sampling": core.SAMPLING,
            "boundary_instances_checked": min(hssd_count, 2),
            "collated_sources": ["HSSD"],
        }
        combined_preflight["stages"][stage] = {
            "stage": stage,
            "source_counts": source_counts,
            "total_count": total,
            "sampling": core.SAMPLING,
            "boundary_instances_checked": sum(
                min(count, 2) for count in source_counts.values()
            ),
            "collated_sources": list(source_counts),
        }
    combined_document = {
        "schema_version": 1,
        "authorization": core.AUTHORIZATION,
        "sampling": core.SAMPLING,
        "sources": {
            source_names[profile]: {
                "training_data": dict(
                    sources[profile]["artifacts"]["training_data"]
                ),
                "handoff": dict(
                    sources[profile]["artifacts"]["handoff"]
                ),
            }
            for profile in source_names
        },
        "stages": {
            stage: {
                **combined_stages[stage],
                "data_dir": {},
            }
            for stage in core.STAGES
        },
    }
    combined_path.write_bytes(
        core._canonical_json_bytes(combined_document)
    )
    runtime_paths = {}
    paths.runtime_config_root.mkdir(parents=True, exist_ok=True)
    for stage, source_path in source_configs.items():
        value = json.loads(source_path.read_text())
        value["trainer"]["args"]["num_workers"] = 1
        output = (
            paths.runtime_config_root
            / f"{source_path.stem}.node16-workers1.json"
        )
        output.write_bytes(core._canonical_json_bytes(value))
        runtime_paths[stage] = output
    runtime = core.runtime_config_evidence(runtime_paths, source_configs)
    return {
        "schema_version": 2,
        "cpu_only": True,
        "deployment": core.verify_deployment_manifest(
            paths.repo_root, binding
        ),
        "paths": {
            "data2_root": str(paths.data2_root),
            "local_root": str(paths.local_root),
            "repo_root": str(paths.repo_root),
            "hssd_training_data": str(paths.hssd_training_data),
            "combined_training_data": str(paths.combined_training_data),
        },
        "disk": {
            "path": str(paths.local_root),
            "required_bytes": required,
            "stage_expanded_pack_bytes": stage_bytes,
            "total_bytes": used + free,
            "used_bytes": used,
            "free_bytes": free,
        },
        "sources": sources,
        "hssd_standalone_preflight": standalone_preflight,
        "combined": {
            "path": str(combined_path),
            "sha256": sha256(combined_path.read_bytes()).hexdigest(),
            "stages": combined_stages,
            "preflight": combined_preflight,
        },
        "runtime_configs": runtime,
        "launch_commands": core._launch_commands(paths, runtime_paths),
    }


def _patch_final_report_trust(
    monkeypatch, report: dict[str, object]
) -> None:
    sources_by_name = {
        evidence["source"]: evidence
        for evidence in report["sources"].values()
    }

    def validate_source(source, path):
        evidence = sources_by_name[source]
        artifacts = evidence["artifacts"]
        return SimpleNamespace(
            source=source,
            path=Path(artifacts["training_data"]["path"]),
            sha256=artifacts["training_data"]["sha256"],
            report_path=Path(artifacts["report"]["path"]),
            report_sha256=artifacts["report"]["sha256"],
            handoff_path=Path(artifacts["handoff"]["path"]),
            handoff_sha256=artifacts["handoff"]["sha256"],
            stages={
                stage: SimpleNamespace(
                    total_count=record["asset_count"],
                    union_scope_sha256=record["asset_scope_sha256"],
                )
                for stage, record in evidence["stages"].items()
            },
        )

    def resolve_combined(path, stage):
        record = report["combined"]["stages"][stage]
        return SimpleNamespace(
            path=Path(path),
            manifest_sha256=report["combined"]["sha256"],
            source_counts=record["source_counts"],
            total_count=record["total_count"],
            union_scope_sha256=record["union_scope_sha256"],
        )

    monkeypatch.setattr(
        core, "validate_source_training_data", validate_source
    )
    monkeypatch.setattr(core, "resolve_training_data", resolve_combined)


def _copy_production_configs(repo_root: Path) -> dict[str, Path]:
    outputs = {}
    for stage, relative in CONFIGS.items():
        output = repo_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(relative, output)
        outputs[stage] = output
    return outputs


def _create_complete_source_topology(output: Path) -> None:
    for stage in core.STAGES:
        (output / stage / "active").mkdir(parents=True)
    publication = output / "publication"
    publication.mkdir()
    (publication / "report.json").write_text("{}")
    (publication / "handoff.json").write_text("{}")
    (output / "training_data.json").write_text("{}")


def test_node16_paths_are_local_and_three_source():
    paths = PreparationPaths.from_roots(
        data2_root=Path("/file2/youngwoo/pixal3d"),
        local_root=Path("/home/youngwoo/data/pixal3d"),
        repo_root=Path("/home/youngwoo/Pixal3D-training-hssd"),
    )

    assert paths.combined_training_data == Path(
        "/home/youngwoo/data/pixal3d/train/production/"
        "abo-3d-future-hssd/training_data.json"
    )
    assert paths.hssd_training_data == Path(
        "/home/youngwoo/data/pixal3d/train/production/"
        "hssd/training_data.json"
    )
    assert paths.runtime_config_root == Path(
        "/home/youngwoo/data/pixal3d/runtime-configs"
    )
    assert paths.evidence_root == Path(
        "/home/youngwoo/data/pixal3d/train/production/"
        "node16-preparation-evidence"
    )


def test_deployment_manifest_verifies_archive_tree_without_git(tmp_path):
    repo = tmp_path / "archive"
    repo.mkdir()
    tracked = repo / "scripts/entrypoint.py"
    tracked.parent.mkdir()
    tracked.write_text("#!/usr/bin/env python3\nprint('ok')\n")
    tracked.chmod(0o755)
    binding = _deployment_binding(tmp_path, repo)

    evidence = core.verify_deployment_manifest(repo, binding)

    assert evidence == {
        "revision": "a" * 40,
        "manifest": {
            "path": str(binding.manifest_path.resolve()),
            "sha256": binding.manifest_sha256,
        },
    }
    assert not (repo / ".git").exists()


@pytest.mark.parametrize("mutation", ("file", "extra", "digest", "revision"))
def test_deployment_manifest_fails_closed_on_binding_or_tree_change(
    tmp_path, mutation
):
    repo = tmp_path / "archive"
    repo.mkdir()
    tracked = repo / "tracked.txt"
    tracked.write_text("reviewed\n")
    binding = _deployment_binding(tmp_path, repo)
    if mutation == "file":
        tracked.write_text("changed\n")
    elif mutation == "extra":
        (repo / "unreviewed.txt").write_text("extra\n")
    elif mutation == "digest":
        binding = core.DeploymentBinding(
            binding.expected_revision,
            binding.manifest_path,
            "b" * 64,
        )
    else:
        binding = core.DeploymentBinding(
            "b" * 40,
            binding.manifest_path,
            binding.manifest_sha256,
        )

    with pytest.raises(ValueError, match="deployment"):
        core.verify_deployment_manifest(repo, binding)


@pytest.mark.parametrize("entrypoint", ("plan", "execute"))
def test_entrypoints_verify_reviewed_deployment_before_admission_or_mutation(
    tmp_path, monkeypatch, entrypoint
):
    paths = _paths(tmp_path)
    paths.repo_root.mkdir(parents=True)
    tracked = paths.repo_root / "reviewed.py"
    tracked.write_text("reviewed = True\n")
    binding = _deployment_binding(tmp_path, paths.repo_root)
    tracked.write_text("reviewed = False\n")
    monkeypatch.setattr(
        core,
        "estimate_required_bytes",
        lambda _paths: pytest.fail(
            "deployment mismatch must abort before disk admission"
        ),
    )
    function = (
        core.plan_node16_training
        if entrypoint == "plan"
        else core.prepare_node16_training
    )

    with pytest.raises(ValueError, match="deployment"):
        function(paths, binding)

    assert not paths.local_root.exists()


def test_insufficient_space_aborts_before_materialization(monkeypatch):
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(100, 99, 1),
    )

    with pytest.raises(ValueError, match="insufficient local free space"):
        assert_free_space(Path("/local"), required_bytes=2)


def test_disk_estimate_counts_repeated_stage_family_consumption(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    sizes = {
        "common": 11,
        "SS-64": 13,
        "shape-512": 17,
        "shape-1024": 19,
        "PBR-1024": 23,
    }
    catalog = {}
    for family, size in sizes.items():
        pack = tmp_path / f"{family}.tar"
        pack.write_bytes(b"x" * size)
        catalog[family] = (SimpleNamespace(pack=pack),)
    loaded = []

    def fake_load(spec, prepared_root):
        loaded.append((spec.source, prepared_root))
        return catalog

    monkeypatch.setattr(core, "load_source_catalog", fake_load)

    estimate = estimate_required_bytes(paths)

    one_source = 4 * 11 + 13 + 17 + 2 * 19 + 23
    assert loaded == [
        ("ABO", paths.data2_root / "prepared"),
        ("3D-FUTURE", paths.data2_root / "prepared"),
        ("HSSD", paths.data2_root / "prepared"),
    ]
    assert estimate.stage_expanded_pack_bytes == 3 * one_source
    assert estimate.required_bytes == 2 * 3 * one_source + 10 * GIB
    assert not paths.production_root.exists()


def test_runtime_configs_change_only_num_workers(tmp_path):
    outputs = create_runtime_configs(CONFIGS, tmp_path)

    assert tuple(outputs) == tuple(CONFIGS)
    for stage, source in CONFIGS.items():
        output = outputs[stage]
        original = json.loads(source.read_text())
        runtime = json.loads(output.read_text())
        assert output.name == f"{source.stem}.node16-workers1.json"
        assert runtime["trainer"]["args"]["num_workers"] == 1
        runtime["trainer"]["args"]["num_workers"] = (
            original["trainer"]["args"]["num_workers"]
        )
        assert runtime == original


@pytest.mark.parametrize(
    ("stage", "field"),
    (("shape1024", "batch_split"), ("ss64", "num_workers")),
)
def test_runtime_evidence_rejects_integer_boolean_substitution(
    tmp_path, stage, field
):
    outputs = create_runtime_configs(CONFIGS, tmp_path)
    selected = outputs[stage]
    value = json.loads(selected.read_text())
    value["trainer"]["args"][field] = True
    selected.write_bytes(core._canonical_json_bytes(value))

    with pytest.raises(ValueError, match="runtime config"):
        core.runtime_config_evidence(outputs, CONFIGS)


@pytest.mark.parametrize(
    ("stage", "mutation"),
    (
        ("ss64", "dataset_num_views"),
        ("shape512", "model_type"),
        ("shape1024", "optimizer_lr"),
        ("pbr1024", "checkpoint_path"),
        ("ss64", "conditioning_model"),
        ("shape512", "extra_top_level_key"),
        ("shape1024", "missing_model_field"),
    ),
)
def test_low_level_final_report_rejects_runtime_drift_from_reviewed_config(
    tmp_path, monkeypatch, stage, mutation
):
    """A recomputed digest/summary must not launder any non-worker drift."""
    paths = _paths(tmp_path)
    report = _valid_final_report(paths)
    _patch_final_report_trust(monkeypatch, report)
    runtime_paths = {
        selected_stage: Path(record["path"])
        for selected_stage, record in report["runtime_configs"].items()
    }
    selected = runtime_paths[stage]
    value = json.loads(selected.read_text())
    if mutation == "dataset_num_views":
        value["dataset"]["args"]["num_views"] = 1
    elif mutation == "model_type":
        value["models"]["denoiser"]["args"]["num_heads"] = "12"
    elif mutation == "optimizer_lr":
        value["trainer"]["args"]["optimizer"]["args"]["lr"] = 0.01
    elif mutation == "checkpoint_path":
        value["trainer"]["args"]["finetune_ckpt"]["denoiser"] = (
            "/tmp/unreviewed.pt"
        )
    elif mutation == "conditioning_model":
        value["trainer"]["args"]["image_cond_model"]["args"][
            "model_name"
        ] = "unreviewed/model"
    elif mutation == "extra_top_level_key":
        value["unreviewed"] = True
    else:
        value["models"]["denoiser"]["args"].pop("num_blocks")
    selected.write_bytes(core._canonical_json_bytes(value))
    report["runtime_configs"][stage]["sha256"] = sha256(
        selected.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="preparation report"):
        core.write_final_report(paths, report)

    assert not (paths.evidence_root / "report.json").exists()


def test_runtime_evidence_binds_exact_reviewed_source_transform(tmp_path):
    repo_root = tmp_path / "repo"
    source_configs = _copy_production_configs(repo_root)
    outputs = create_runtime_configs(
        source_configs, tmp_path / "runtime-configs"
    )
    reformatted = outputs["shape512"]
    reformatted.write_text(
        json.dumps(json.loads(reformatted.read_text()), indent=4)
    )

    evidence = core.runtime_config_evidence(outputs, source_configs)

    for stage in core.STAGES:
        source = source_configs[stage]
        runtime = outputs[stage]
        assert evidence[stage]["source_config"] == {
            "path": str(source),
            "sha256": sha256(source.read_bytes()).hexdigest(),
        }
        original = json.loads(source.read_text())
        transformed = json.loads(runtime.read_text())
        transformed["trainer"]["args"]["num_workers"] = original[
            "trainer"
        ]["args"]["num_workers"]
        assert transformed == original


def test_invalid_last_source_config_aborts_before_any_local_mutation(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    paths.data2_root.mkdir()
    paths.local_root.mkdir()
    paths.repo_root.mkdir()
    source_configs = _copy_production_configs(paths.repo_root)
    invalid = json.loads(source_configs["pbr1024"].read_text())
    invalid["trainer"]["args"]["max_steps"] = 1
    source_configs["pbr1024"].write_text(json.dumps(invalid))
    monkeypatch.setattr(
        core,
        "estimate_required_bytes",
        lambda _paths: DiskEstimate(1, 10 * GIB + 2),
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda root, required: {
            "path": str(root),
            "required_bytes": required,
            "total_bytes": required + 2,
            "used_bytes": 1,
            "free_bytes": required + 1,
        },
    )
    monkeypatch.setattr(
        core,
        "materialize_source",
        lambda *_args: pytest.fail(
            "invalid config must abort before materialization"
        ),
    )
    deployment = _deployment_binding(tmp_path, paths.repo_root)

    with pytest.raises(ValueError, match="source config semantics"):
        prepare_node16_training(paths, deployment)

    assert not paths.runtime_config_root.exists()
    assert not paths.production_root.exists()


def test_runtime_config_reuses_semantically_equal_existing_json(tmp_path):
    outputs = create_runtime_configs(CONFIGS, tmp_path)
    selected = outputs["ss64"]
    expected = json.loads(selected.read_text())
    selected.write_text(json.dumps(expected, separators=(",", ":")))
    before = selected.read_bytes()

    reused = create_runtime_configs(CONFIGS, tmp_path)

    assert reused["ss64"] == selected
    assert selected.read_bytes() == before


def test_runtime_config_refuses_mismatch_without_replacement(tmp_path):
    outputs = create_runtime_configs(CONFIGS, tmp_path)
    selected = outputs["shape512"]
    changed = json.loads(selected.read_text())
    changed["trainer"]["args"]["max_steps"] = 1
    selected.write_text(json.dumps(changed))
    before = selected.read_bytes()

    with pytest.raises(
        FileExistsError, match="existing runtime config has different content"
    ):
        create_runtime_configs(CONFIGS, tmp_path)

    assert selected.read_bytes() == before


def test_runtime_configs_refuse_partial_existing_output_set(tmp_path):
    outputs = create_runtime_configs(CONFIGS, tmp_path)
    retained = outputs["ss64"]
    retained_bytes = retained.read_bytes()
    for stage in ("shape512", "shape1024", "pbr1024"):
        outputs[stage].unlink()

    with pytest.raises(ValueError, match="partial runtime config output"):
        create_runtime_configs(CONFIGS, tmp_path)

    assert retained.read_bytes() == retained_bytes
    assert {
        path.name for path in tmp_path.iterdir()
    } == {retained.name}


def test_runtime_config_reuse_rejects_unexpected_top_level_sibling(
    tmp_path,
):
    outputs = create_runtime_configs(CONFIGS, tmp_path)
    unexpected = tmp_path / ".runtime.lock"
    unexpected.write_text("inspect")

    with pytest.raises(ValueError, match="runtime config topology"):
        create_runtime_configs(CONFIGS, tmp_path)

    assert unexpected.read_text() == "inspect"
    assert all(path.exists() for path in outputs.values())


@pytest.mark.parametrize(
    "owned_root", ("runtime", "source", "combined", "evidence")
)
def test_empty_symlinked_owned_root_cannot_redirect_writes(
    tmp_path, monkeypatch, owned_root
):
    paths = _paths(tmp_path)
    outside = tmp_path / f"outside-{owned_root}"
    outside.mkdir()

    if owned_root == "runtime":
        target = paths.runtime_config_root
        target.parent.mkdir(parents=True)

        def operation():
            return create_runtime_configs(CONFIGS, target)
    elif owned_root == "source":
        target = paths.production_root / "abo"
        target.parent.mkdir(parents=True)
        monkeypatch.setattr(
            core,
            "load_source_catalog",
            lambda *_args: pytest.fail(
                "symlinked source root must abort before catalog load"
            ),
        )

        def operation():
            return materialize_source("abo", paths, CONFIGS)
    elif owned_root == "combined":
        target = paths.combined_training_data.parent
        target.parent.mkdir(parents=True)
        monkeypatch.setattr(
            core,
            "publish_combined_training_data",
            lambda *_args: pytest.fail(
                "symlinked combined root must abort before publication"
            ),
        )

        def operation():
            return core.publish_combined(paths)
    else:
        target = paths.evidence_root
        target.parent.mkdir(parents=True)

        def operation():
            return core.write_final_report(
                paths, _valid_final_report(paths)
            )

    target.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        ValueError, match="canonical and non-symlinked"
    ):
        operation()

    assert target.is_symlink()
    assert list(outside.iterdir()) == []


def test_partial_source_reports_every_path_without_mutation(
    tmp_path, monkeypatch
):
    output = tmp_path / "production" / "hssd"
    active = output / "ss64" / "active"
    report = output / "publication" / "report.json"
    active.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    before = sorted(str(path.relative_to(output)) for path in output.rglob("*"))
    monkeypatch.setattr(
        shutil,
        "rmtree",
        lambda *_args, **_kwargs: pytest.fail("rmtree must not be called"),
    )
    monkeypatch.setattr(
        os,
        "replace",
        lambda *_args, **_kwargs: pytest.fail("replace must not be called"),
    )

    with pytest.raises(ValueError) as error:
        refuse_partial_source(output)

    message = str(error.value)
    assert str(active) in message
    assert str(report) in message
    assert sorted(
        str(path.relative_to(output)) for path in output.rglob("*")
    ) == before


def test_existing_source_reuse_runs_chain_and_stage_verification(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    output = paths.production_root / "hssd"
    _create_complete_source_topology(output)
    publication = {
        "report": output / "publication/report.json",
        "handoff": output / "publication/handoff.json",
        "training-data": output / "training_data.json",
    }
    calls = []
    validated = SimpleNamespace(
        source="HSSD",
        report_path=publication["report"],
        handoff_path=publication["handoff"],
        path=publication["training-data"],
    )
    monkeypatch.setattr(
        core,
        "validate_source_training_data",
        lambda source, path: (
            calls.append(("chain", source, path)) or validated
        ),
    )
    monkeypatch.setattr(
        core,
        "preflight_all_source_stages",
        lambda spec, root, configs: (
            calls.append(("stages", spec.source, root, configs)) or {}
        ),
    )

    result = verify_existing_source(
        core.build_source_spec("hssd", paths.data2_root),
        publication,
        CONFIGS,
    )

    assert result.reused is True
    assert result.validated is validated
    assert [call[0] for call in calls] == ["chain", "stages"]


def test_materialize_source_reuses_complete_existing_source(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    _create_complete_source_topology(paths.production_root / "abo")
    calls = []
    expected = SimpleNamespace(reused=True)
    monkeypatch.setattr(
        core,
        "verify_existing_source",
        lambda spec, publication, configs: (
            calls.append((spec.source, publication, configs)) or expected
        ),
    )
    monkeypatch.setattr(
        core,
        "load_source_catalog",
        lambda *_args: pytest.fail("complete source must not rematerialize"),
    )

    assert materialize_source("abo", paths, CONFIGS) is expected
    assert calls[0][0] == "ABO"


def test_source_reuse_rejects_unexpected_top_level_sibling(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    output = paths.production_root / "hssd"
    for stage in core.STAGES:
        (output / stage / "active").mkdir(parents=True)
    publication = output / "publication"
    publication.mkdir()
    (publication / "report.json").write_text("{}")
    (publication / "handoff.json").write_text("{}")
    (output / "training_data.json").write_text("{}")
    unexpected = output / ".ss64.staging"
    unexpected.mkdir()
    monkeypatch.setattr(
        core,
        "verify_existing_source",
        lambda *_args: pytest.fail(
            "unexpected source topology must abort before trust validation"
        ),
    )

    with pytest.raises(ValueError, match="source output topology"):
        materialize_source("hssd", paths, CONFIGS)

    assert unexpected.is_dir()


def test_materialize_source_refuses_partial_root_before_catalog_load(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    partial = paths.production_root / "3d-future" / "shape512"
    partial.mkdir(parents=True)
    monkeypatch.setattr(
        core,
        "load_source_catalog",
        lambda *_args: pytest.fail("partial source must abort first"),
    )

    with pytest.raises(ValueError, match="partial source output"):
        materialize_source("3d-future", paths, CONFIGS)


def test_publish_abo_uses_existing_fixed_count_publisher(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    spec = core.build_source_spec("abo", paths.data2_root)
    output = paths.production_root / "abo"
    publication = core._source_publication_paths(output)
    results = {stage: SimpleNamespace(stage=stage) for stage in core.STAGES}
    prepared = core.SourcePreparation(
        spec=spec,
        output_root=output,
        publication=publication,
        results=results,
        validated=None,
        reused=False,
    )
    calls = []
    validated = SimpleNamespace()
    from data_toolkit.pipeline import training_preflight
    from scripts import preflight_multiview_production

    monkeypatch.setattr(core, "_assert_torch_cpu_only", lambda: None)
    monkeypatch.setattr(
        training_preflight,
        "publish_source_handoff",
        lambda *_args: pytest.fail(
            "ABO must not use the observed-count schema-2 publisher"
        ),
    )
    monkeypatch.setattr(
        preflight_multiview_production,
        "_materialization_evidence_from_result",
        lambda result: {"stage": result.stage},
    )
    monkeypatch.setattr(
        preflight_multiview_production,
        "publish_handoff",
        lambda index, selected, materializations, report, handoff,
        training_data, created_at: calls.append(
            (
                index,
                selected,
                materializations,
                report,
                handoff,
                training_data,
                created_at,
            )
        ),
    )
    monkeypatch.setattr(
        core,
        "validate_source_training_data",
        lambda source, path: validated,
    )
    monkeypatch.setattr(
        core, "_validate_selected_publication", lambda *_args: None
    )
    monkeypatch.setattr(
        core, "_source_evidence", lambda selected: {"source": "ABO"}
    )

    evidence = core.publish_source("abo", paths, prepared)

    assert evidence == {"source": "ABO"}
    assert len(calls) == 1
    assert calls[0][0] == spec.indexes[0]
    assert calls[0][1] is results
    assert calls[0][2] == {
        stage: {"stage": stage} for stage in core.STAGES
    }
    assert calls[0][3:6] == (
        publication["report"],
        publication["handoff"],
        publication["training-data"],
    )


def test_publish_combined_refuses_partial_output_root(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    partial = paths.combined_training_data.parent / "operator-note.txt"
    partial.parent.mkdir(parents=True)
    partial.write_text("inspect")
    monkeypatch.setattr(
        core,
        "publish_combined_training_data",
        lambda *_args: pytest.fail("partial combined root must abort first"),
    )

    with pytest.raises(ValueError, match="partial source output"):
        core.publish_combined(paths)

    assert partial.read_text() == "inspect"
    assert not paths.combined_training_data.exists()


def test_combined_reuse_rejects_unexpected_top_level_sibling(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    paths.combined_training_data.parent.mkdir(parents=True)
    paths.combined_training_data.write_text("{}")
    unexpected = paths.combined_training_data.parent / "staging"
    unexpected.mkdir()
    monkeypatch.setattr(
        core,
        "publish_combined_training_data",
        lambda _sources, output: output,
    )

    with pytest.raises(ValueError, match="combined output topology"):
        core.publish_combined(paths)

    assert unexpected.is_dir()


def test_final_report_refuses_partial_evidence_root(tmp_path):
    paths = _paths(tmp_path)
    partial = paths.evidence_root / "operator-note.txt"
    partial.parent.mkdir(parents=True)
    partial.write_text("inspect")

    with pytest.raises(ValueError, match="partial source output"):
        core.write_final_report(paths, {"schema_version": 1})

    assert partial.read_text() == "inspect"
    assert not (paths.evidence_root / "report.json").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "source_identity",
        "source_artifact_path",
        "source_artifact_digest",
        "source_artifact_semantics",
        "source_missing_stage",
        "source_stage_count",
        "source_stage_scope",
        "standalone_missing_stage",
        "standalone_source_counts",
        "standalone_sampling",
        "combined_path",
        "combined_digest",
        "combined_artifact_semantics",
        "combined_stage_counts",
        "combined_stage_scope",
        "combined_preflight",
        "runtime_missing_stage",
        "runtime_path",
        "runtime_digest",
        "runtime_semantics",
        "launch_missing_stage",
        "launch_swapped_config",
    ),
)
def test_low_level_final_report_rejects_malformed_nested_evidence(
    tmp_path, monkeypatch, mutation
):
    paths = _paths(tmp_path)
    report = _valid_final_report(paths)
    _patch_final_report_trust(monkeypatch, report)
    if mutation == "source_identity":
        report["sources"]["hssd"]["source"] = "3D-FUTURE"
    elif mutation == "source_artifact_path":
        report["sources"]["abo"]["artifacts"]["report"]["path"] = str(
            paths.production_root / "abo/other.json"
        )
    elif mutation == "source_artifact_digest":
        report["sources"]["3d-future"]["artifacts"]["handoff"][
            "sha256"
        ] = "0" * 64
    elif mutation == "source_artifact_semantics":
        artifacts = report["sources"]["hssd"]["artifacts"]
        report_path = Path(artifacts["report"]["path"])
        handoff_path = Path(artifacts["handoff"]["path"])
        training_path = Path(artifacts["training_data"]["path"])
        source_report = json.loads(report_path.read_text())
        source_report["source"] = "Other"
        report_path.write_bytes(
            core._canonical_json_bytes(source_report)
        )
        artifacts["report"]["sha256"] = sha256(
            report_path.read_bytes()
        ).hexdigest()
        handoff = {
            **source_report,
            "report": dict(artifacts["report"]),
        }
        handoff_path.write_bytes(core._canonical_json_bytes(handoff))
        artifacts["handoff"]["sha256"] = sha256(
            handoff_path.read_bytes()
        ).hexdigest()
        training = {
            **handoff,
            "handoff": dict(artifacts["handoff"]),
        }
        training_path.write_bytes(
            core._canonical_json_bytes(training)
        )
        artifacts["training_data"]["sha256"] = sha256(
            training_path.read_bytes()
        ).hexdigest()
    elif mutation == "source_missing_stage":
        report["sources"]["hssd"]["stages"].pop("pbr1024")
    elif mutation == "source_stage_count":
        report["sources"]["hssd"]["stages"]["ss64"]["asset_count"] -= 1
    elif mutation == "source_stage_scope":
        report["sources"]["abo"]["stages"]["shape512"][
            "asset_scope_sha256"
        ] = "not-a-digest"
    elif mutation == "standalone_missing_stage":
        report["hssd_standalone_preflight"]["stages"].pop("shape1024")
    elif mutation == "standalone_source_counts":
        report["hssd_standalone_preflight"]["stages"]["ss64"][
            "source_counts"
        ] = {"ABO": 1}
    elif mutation == "standalone_sampling":
        report["hssd_standalone_preflight"]["stages"]["ss64"][
            "sampling"
        ] = "weighted"
    elif mutation == "combined_path":
        report["combined"]["path"] = str(
            paths.combined_training_data.parent / "other.json"
        )
    elif mutation == "combined_digest":
        report["combined"]["sha256"] = "0" * 64
    elif mutation == "combined_artifact_semantics":
        combined_path = Path(report["combined"]["path"])
        combined_document = json.loads(combined_path.read_text())
        combined_document["sampling"] = "weighted"
        combined_path.write_bytes(
            core._canonical_json_bytes(combined_document)
        )
        report["combined"]["sha256"] = sha256(
            combined_path.read_bytes()
        ).hexdigest()
    elif mutation == "combined_stage_counts":
        report["combined"]["stages"]["ss64"]["source_counts"]["HSSD"] -= 1
    elif mutation == "combined_stage_scope":
        report["combined"]["stages"]["shape512"][
            "union_scope_sha256"
        ] = "invalid"
    elif mutation == "combined_preflight":
        report["combined"]["preflight"]["stages"]["pbr1024"][
            "collated_sources"
        ] = ["HSSD", "ABO", "3D-FUTURE"]
    elif mutation == "runtime_missing_stage":
        report["runtime_configs"].pop("pbr1024")
    elif mutation == "runtime_path":
        report["runtime_configs"]["ss64"]["path"] = str(
            paths.runtime_config_root / "other.json"
        )
    elif mutation == "runtime_digest":
        report["runtime_configs"]["shape512"]["sha256"] = "0" * 64
    elif mutation == "runtime_semantics":
        report["runtime_configs"]["shape1024"][
            "six_gpu_global_batch"
        ] = 48
    elif mutation == "launch_missing_stage":
        report["launch_commands"].pop("shape512")
    else:
        report["launch_commands"]["ss64"] = report[
            "launch_commands"
        ]["shape512"]

    with pytest.raises(ValueError, match="preparation report"):
        core.write_final_report(paths, report)

    assert not (paths.evidence_root / "report.json").exists()


def test_low_level_final_report_requires_source_trust_validation(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    report = _valid_final_report(paths)
    monkeypatch.setattr(
        core,
        "validate_source_training_data",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("source trust sentinel")
        ),
    )

    with pytest.raises(ValueError, match="source trust"):
        core.write_final_report(paths, report)

    assert not (paths.evidence_root / "report.json").exists()


def test_low_level_final_report_requires_combined_trust_validation(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    report = _valid_final_report(paths)
    _patch_final_report_trust(monkeypatch, report)
    monkeypatch.setattr(
        core,
        "resolve_training_data",
        lambda *_args: (_ for _ in ()).throw(
            ValueError("combined trust sentinel")
        ),
    )

    with pytest.raises(ValueError, match="combined trust"):
        core.write_final_report(paths, report)

    assert not (paths.evidence_root / "report.json").exists()


def test_report_reuse_rejects_unexpected_top_level_sibling(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    report = _valid_final_report(paths)
    _patch_final_report_trust(monkeypatch, report)
    output = core.write_final_report(paths, report)
    unexpected = paths.evidence_root / ".report.lock"
    unexpected.write_text("inspect")

    with pytest.raises(ValueError, match="evidence output topology"):
        core.write_final_report(paths, report)

    assert output.exists()
    assert unexpected.read_text() == "inspect"


def test_final_report_reuses_historical_disk_and_source_state(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    original = _valid_final_report(paths)
    _patch_final_report_trust(monkeypatch, original)
    output = core.write_final_report(paths, original)
    before = output.read_bytes()
    rerun = deepcopy(original)
    rerun["disk"]["total_bytes"] += 100
    rerun["disk"]["free_bytes"] += 100
    for source in rerun["sources"].values():
        source["reused"] = True

    assert core.write_final_report(paths, rerun) == output
    assert output.read_bytes() == before

    changed = deepcopy(rerun)
    changed["disk"]["stage_expanded_pack_bytes"] += 1
    changed["disk"]["required_bytes"] += 2
    with pytest.raises(
        FileExistsError, match="preparation report has different content"
    ):
        core.write_final_report(paths, changed)
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    ("missing_free", "integer_reused", "inconsistent_usage", "low_free"),
)
def test_existing_report_rejects_invalid_mandatory_dynamic_fields(
    tmp_path, monkeypatch, mutation
):
    paths = _paths(tmp_path)
    valid = _valid_final_report(paths)
    _patch_final_report_trust(monkeypatch, valid)
    output = core.write_final_report(paths, valid)
    invalid = deepcopy(valid)
    if mutation == "missing_free":
        invalid["disk"].pop("free_bytes")
    elif mutation == "integer_reused":
        invalid["sources"]["hssd"]["reused"] = 1
    elif mutation == "inconsistent_usage":
        invalid["disk"]["used_bytes"] += 1
    else:
        invalid["disk"]["free_bytes"] = invalid["disk"]["required_bytes"] - 1
        invalid["disk"]["total_bytes"] = (
            invalid["disk"]["used_bytes"] + invalid["disk"]["free_bytes"]
        )
    output.write_text(json.dumps(invalid))
    before = output.read_bytes()

    with pytest.raises(ValueError, match="preparation report"):
        core.write_final_report(paths, valid)

    assert output.read_bytes() == before


def test_prepare_orders_sources_then_standalone_and_combined_validation(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    calls = []
    _copy_production_configs(paths.repo_root)
    runtime_configs = {
        stage: tmp_path / f"{stage}.json" for stage in CONFIGS
    }
    monkeypatch.setattr(
        core,
        "estimate_required_bytes",
        lambda _paths: DiskEstimate(123, 10 * GIB + 246),
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda root, required_bytes: {
            "path": str(root),
            "required_bytes": required_bytes,
            "total_bytes": required_bytes + 2,
            "used_bytes": 1,
            "free_bytes": required_bytes + 1,
        },
    )
    monkeypatch.setattr(
        core,
        "create_runtime_configs",
        lambda configs, root: runtime_configs,
    )
    monkeypatch.setattr(
        core,
        "materialize_source",
        lambda name, _paths, _configs: (
            calls.append(("materialize", name))
            or SimpleNamespace(reused=False)
        ),
    )
    monkeypatch.setattr(
        core,
        "publish_source",
        lambda name, _paths, prepared: (
            calls.append(("source", name)) or {"profile": name}
        ),
    )
    monkeypatch.setattr(
        core,
        "preflight_training_data",
        lambda path, configs: (
            calls.append(("preflight", path)) or {"stages": {}}
        ),
    )
    monkeypatch.setattr(
        core,
        "publish_combined",
        lambda _paths: (
            calls.append(("combined", None))
            or paths.combined_training_data
        ),
    )
    monkeypatch.setattr(
        core,
        "runtime_config_evidence",
        lambda configs, source_configs: {
            "stages": sorted(configs),
            "sources": sorted(source_configs),
        },
    )
    scope_evidence = {
        stage: {
            "source_counts": {},
            "total_count": 0,
            "union_scope_sha256": stage * 8,
        }
        for stage in core.STAGES
    }
    monkeypatch.setattr(
        core,
        "training_scope_evidence",
        lambda path: scope_evidence,
    )
    monkeypatch.setattr(
        core,
        "write_final_report",
        lambda _paths, report: (
            calls.append(("report", None)) or report
        ),
    )
    deployment = _deployment_binding(tmp_path, paths.repo_root)

    report = prepare_node16_training(paths, deployment)

    assert calls[:6] == [
        ("materialize", "abo"),
        ("source", "abo"),
        ("materialize", "3d-future"),
        ("source", "3d-future"),
        ("materialize", "hssd"),
        ("source", "hssd"),
    ]
    assert calls[6:] == [
        ("preflight", paths.hssd_training_data),
        ("combined", None),
        ("preflight", paths.combined_training_data),
        ("report", None),
    ]
    assert report["disk"]["stage_expanded_pack_bytes"] == 123
    assert report["disk"]["required_bytes"] == 10 * GIB + 246
    assert report["combined"]["stages"] == scope_evidence
    assert report["deployment"]["revision"] == (
        deployment.expected_revision
    )
    assert report["deployment"]["manifest"]["sha256"] == (
        deployment.manifest_sha256
    )


@pytest.mark.parametrize("tamper_target", ("runtime", "source"))
def test_prepare_rejects_config_tamper_during_long_materialization(
    tmp_path, monkeypatch, tamper_target
):
    paths = _paths(tmp_path)
    source_configs = _copy_production_configs(paths.repo_root)
    deployment = _deployment_binding(tmp_path, paths.repo_root)
    monkeypatch.setattr(
        core,
        "estimate_required_bytes",
        lambda _paths: DiskEstimate(1, 10 * GIB + 2),
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda root, required: {
            "path": str(root),
            "required_bytes": required,
            "total_bytes": required + 2,
            "used_bytes": 1,
            "free_bytes": required + 1,
        },
    )
    state = {"tampered": False}

    def materialize(profile, _paths, runtime_configs):
        if not state["tampered"]:
            if tamper_target == "runtime":
                target = runtime_configs["ss64"]
                value = json.loads(target.read_text())
                value["dataset"]["args"]["num_views"] = 1
            else:
                target = source_configs["ss64"]
                value = json.loads(target.read_text())
                value["trainer"]["args"]["optimizer"]["args"]["lr"] = 0.01
            target.write_bytes(core._canonical_json_bytes(value))
            state["tampered"] = True
        return SimpleNamespace(reused=False)

    monkeypatch.setattr(core, "materialize_source", materialize)
    monkeypatch.setattr(
        core,
        "publish_source",
        lambda profile, _paths, prepared: {
            "profile": profile,
            "reused": prepared.reused,
        },
    )
    monkeypatch.setattr(
        core,
        "preflight_training_data",
        lambda *_args: {"stages": {}},
    )
    monkeypatch.setattr(
        core,
        "publish_combined",
        lambda _paths: paths.combined_training_data,
    )
    monkeypatch.setattr(
        core,
        "training_scope_evidence",
        lambda _path: {
            stage: {
                "source_counts": {},
                "total_count": 0,
                "union_scope_sha256": "0" * 64,
            }
            for stage in core.STAGES
        },
    )
    monkeypatch.setattr(
        core,
        "write_final_report",
        lambda *_args: pytest.fail(
            "tampered config must not reach final report publication"
        ),
    )

    with pytest.raises(
        ValueError, match="exact reviewed source transformation"
    ):
        prepare_node16_training(paths, deployment)

    assert state["tampered"] is True
    assert not (paths.evidence_root / "report.json").exists()


def test_source_preflight_stops_after_first_stage_initializes_cuda(
    tmp_path, monkeypatch
):
    import torch
    from data_toolkit.pipeline import training_preflight

    paths = _paths(tmp_path)
    spec = core.build_source_spec("hssd", paths.data2_root)
    state = {"initialized": False}
    calls = []

    def stage_preflight(
        _spec, stage, _root, _config
    ):
        calls.append(stage)
        state["initialized"] = True
        return object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "is_initialized", lambda: state["initialized"]
    )
    monkeypatch.setattr(
        training_preflight, "preflight_stage", stage_preflight
    )

    with pytest.raises(RuntimeError, match="CPU-only preparation"):
        core.preflight_all_source_stages(
            spec, paths.production_root / "hssd", CONFIGS
        )

    assert calls == ["ss64"]


def test_abo_preflight_accepts_bound_node16_runtime_configs(
    tmp_path, monkeypatch
):
    import torch
    from scripts import preflight_multiview_production as production_preflight

    paths = _paths(tmp_path)
    spec = core.build_source_spec("abo", paths.data2_root)
    runtime_configs = create_runtime_configs(
        CONFIGS, paths.runtime_config_root
    )
    index_path = spec.indexes[0]
    index_bytes = core._canonical_json_bytes(
        {
            "gate": "production",
            "source": "ABO",
            "shard_id": "ABO-00000",
            "batches": {
                batch: {}
                for batch in spec.expected_batches["ABO-00000"]
            },
        }
    )
    materialization_bytes = core._canonical_json_bytes(
        {
            "source_index": {
                "path": str(index_path),
                "sha256": sha256(index_bytes).hexdigest(),
            }
        }
    )
    calls = []

    def fixture_bytes(path):
        selected = Path(path)
        if selected == index_path:
            return index_bytes
        if selected.name == "materialization.json":
            return materialization_bytes
        raise AssertionError(f"unexpected read: {selected}")

    def fixture_preflight(stage, root, config):
        calls.append((stage, Path(root), Path(config)))
        return stage

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        production_preflight._core,
        "_existing_regular_bytes",
        fixture_bytes,
    )
    monkeypatch.setattr(
        production_preflight,
        "_preflight_stage_for_fixture",
        fixture_preflight,
    )

    results = core.preflight_all_source_stages(
        spec,
        paths.production_root / "abo",
        runtime_configs,
    )

    assert results == {stage: stage for stage in CONFIGS}
    assert calls == [
        (
            stage,
            paths.production_root / "abo" / stage / "active",
            runtime_configs[stage],
        )
        for stage in CONFIGS
    ]


def test_training_preflight_stops_after_first_stage_initializes_cuda(
    tmp_path, monkeypatch
):
    import torch
    import scripts.preflight_multisource_training as preflight

    state = {"initialized": False}
    calls = []

    def stage_preflight(_training_data, stage, _config):
        calls.append(stage)
        state["initialized"] = True
        return {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "is_initialized", lambda: state["initialized"]
    )
    monkeypatch.setattr(
        preflight, "preflight_multisource_stage", stage_preflight
    )

    with pytest.raises(RuntimeError, match="CPU-only preparation"):
        core.preflight_training_data(tmp_path / "training_data.json", CONFIGS)

    assert calls == ["ss64"]


def test_plan_validates_inputs_without_creating_local_roots(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    _copy_production_configs(paths.repo_root)
    deployment = _deployment_binding(tmp_path, paths.repo_root)
    monkeypatch.setattr(
        core,
        "estimate_required_bytes",
        lambda _paths: DiskEstimate(7, 10 * GIB + 14),
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda root, required_bytes: {
            "path": str(root),
            "required_bytes": required_bytes,
            "total_bytes": 20 * GIB,
            "used_bytes": 1,
            "free_bytes": 20 * GIB - 1,
        },
    )

    plan = core.plan_node16_training(paths, deployment)

    assert plan["execute"] is False
    assert plan["combined_training_data"] == str(
        paths.combined_training_data
    )
    assert plan["deployment"] == {
        "revision": deployment.expected_revision,
        "manifest": {
            "path": str(deployment.manifest_path.resolve()),
            "sha256": deployment.manifest_sha256,
        },
    }
    assert not paths.local_root.exists()


def test_plan_rejects_unapproved_reviewed_source_config_before_disk_admission(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    configs = _copy_production_configs(paths.repo_root)
    changed = json.loads(configs["ss64"].read_text())
    changed["trainer"]["args"]["max_steps"] = 1
    configs["ss64"].write_bytes(core._canonical_json_bytes(changed))
    deployment = _deployment_binding(tmp_path, paths.repo_root)
    monkeypatch.setattr(
        core,
        "estimate_required_bytes",
        lambda _paths: pytest.fail(
            "invalid config must abort before disk admission"
        ),
    )

    with pytest.raises(ValueError, match="source config semantics"):
        core.plan_node16_training(paths, deployment)

    assert not paths.local_root.exists()


def test_plan_rejects_existing_runtime_drift_before_disk_admission(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    configs = _copy_production_configs(paths.repo_root)
    runtime = create_runtime_configs(configs, paths.runtime_config_root)
    changed = json.loads(runtime["ss64"].read_text())
    changed["dataset"]["args"]["num_views"] = 1
    runtime["ss64"].write_bytes(core._canonical_json_bytes(changed))
    deployment = _deployment_binding(tmp_path, paths.repo_root)
    monkeypatch.setattr(
        core,
        "estimate_required_bytes",
        lambda _paths: pytest.fail(
            "runtime drift must abort before disk admission"
        ),
    )

    with pytest.raises(
        ValueError, match="exact reviewed source transformation"
    ):
        core.plan_node16_training(paths, deployment)

    assert not paths.production_root.exists()


@pytest.mark.parametrize("entrypoint", ("plan", "execute"))
@pytest.mark.parametrize("root_name", ("data2_root", "local_root", "repo_root"))
@pytest.mark.parametrize("mutation", ("relative", "dotdot", "symlink"))
def test_entrypoints_reject_noncanonical_roots_before_admission_or_mutation(
    tmp_path, monkeypatch, entrypoint, root_name, mutation
):
    roots = {
        "data2_root": tmp_path / "data2",
        "local_root": tmp_path / "local",
        "repo_root": tmp_path / "repo",
    }
    for root in roots.values():
        root.mkdir()
    if mutation == "relative":
        invalid = Path(f"relative-{root_name}")
    elif mutation == "dotdot":
        nested = tmp_path / "nested"
        nested.mkdir()
        invalid = nested / ".." / roots[root_name].name
    else:
        invalid = tmp_path / f"{root_name}-link"
        invalid.symlink_to(roots[root_name], target_is_directory=True)
    roots[root_name] = invalid
    paths = PreparationPaths.from_roots(**roots)
    monkeypatch.setattr(
        core,
        "estimate_required_bytes",
        lambda _paths: pytest.fail(
            "invalid roots must abort before disk admission"
        ),
    )

    function = (
        core.plan_node16_training
        if entrypoint == "plan"
        else core.prepare_node16_training
    )
    deployment = core.DeploymentBinding(
        expected_revision="a" * 40,
        manifest_path=tmp_path / "unused-deployment-manifest.json",
        manifest_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="root"):
        function(paths, deployment)

    assert not (tmp_path / "local" / "train").exists()
    assert not (tmp_path / "local" / "runtime-configs").exists()


@pytest.mark.parametrize("cuda_value", [None, "0"])
def test_cli_rejects_invalid_cuda_environment_before_torch_import(
    tmp_path, cuda_value
):
    marker = tmp_path / "torch-imported"
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(
        "import builtins, os\n"
        "_original_import = builtins.__import__\n"
        "def _guard(name, *args, **kwargs):\n"
        "    if name == 'torch' or name.startswith('torch.'):\n"
        "        open(os.environ['TORCH_IMPORT_MARKER'], 'w').write(name)\n"
        "    return _original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _guard\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path.cwd()))
    )
    env["TORCH_IMPORT_MARKER"] = str(marker)
    if cuda_value is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = cuda_value

    result = __import__("subprocess").run(
        [
            os.environ.get("PYTHON", "/opt/conda/envs/pixal3d/bin/python"),
            "scripts/prepare_node16_training.py",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "CUDA_VISIBLE_DEVICES must be explicitly set to empty" in (
        result.stderr + result.stdout
    )
    assert not marker.exists()


def test_cli_rejects_bytecode_writes_before_project_import(tmp_path):
    marker = tmp_path / "project-imported"
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(
        "import builtins, os\n"
        "_original_import = builtins.__import__\n"
        "def _guard(name, *args, **kwargs):\n"
        "    if name.startswith('data_toolkit.pipeline.node16'):\n"
        "        open(os.environ['PROJECT_IMPORT_MARKER'], 'w').write(name)\n"
        "    return _original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _guard\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path.cwd()))
    )
    env["PROJECT_IMPORT_MARKER"] = str(marker)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("PYTHONDONTWRITEBYTECODE", None)

    result = __import__("subprocess").run(
        [
            os.environ.get("PYTHON", "/opt/conda/envs/pixal3d/bin/python"),
            "scripts/prepare_node16_training.py",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "PYTHONDONTWRITEBYTECODE must be explicitly set to 1" in (
        result.stderr + result.stdout
    )
    assert not marker.exists()


def test_cli_requires_explicit_reviewed_deployment_binding():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = __import__("subprocess").run(
        [
            os.environ.get("PYTHON", "/opt/conda/envs/pixal3d/bin/python"),
            "scripts/prepare_node16_training.py",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    output = result.stderr + result.stdout
    for flag in (
        "--expected-revision",
        "--deployment-manifest",
        "--deployment-manifest-sha256",
    ):
        assert flag in output
