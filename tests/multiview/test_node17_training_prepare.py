from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline import training_manifest
from data_toolkit.pipeline import node17_training_prepare as core
from data_toolkit.pipeline.node17_hssd_transfer import (
    NODE17_DATA2_ROOT,
    HssdTransferResult,
    TreeInventory,
)
from data_toolkit.pipeline.training_manifest import SAMPLING, STAGES


CONFIG_NAMES = {
    "ss64": (
        "ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.node17.json"
    ),
    "shape512": (
        "slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512"
        ".node17.json"
    ),
    "shape1024": (
        "slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024"
        ".node17.json"
    ),
    "pbr1024": (
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024"
        ".node17.json"
    ),
}
REVISION = "a" * 40


def _paths(tmp_path: Path) -> core.Node17PreparationPaths:
    data2 = tmp_path / "data2/pixal3d"
    local = tmp_path / "node17/data/pixal3d"
    repo = tmp_path / "repo"
    for root in (data2, local, repo):
        root.mkdir(parents=True)
    return core.Node17PreparationPaths.from_roots(data2, local, repo)


def _runtime_paths(
    paths: core.Node17PreparationPaths,
) -> dict[str, Path]:
    return {
        stage: paths.runtime_config_root / CONFIG_NAMES[stage]
        for stage in STAGES
    }


def _validated_source(
    paths: core.Node17PreparationPaths, source: str
) -> SimpleNamespace:
    directory = {
        "ABO": "abo",
        "3D-FUTURE": "3d-future",
        "HSSD": "hssd",
    }[source]
    root = paths.production_root / directory
    count = {"ABO": 3, "3D-FUTURE": 5, "HSSD": 7}[source]
    stages = {
        stage: SimpleNamespace(
            total_count=count,
            union_scope_sha256=(source[0].lower() * 64),
        )
        for stage in STAGES
    }
    return SimpleNamespace(
        source=source,
        path=root / "training_data.json",
        sha256=(source[-1].lower() * 64),
        report_path=root / "publication/report.json",
        report_sha256="1" * 64,
        handoff_path=root / "publication/handoff.json",
        handoff_sha256="2" * 64,
        stages=stages,
    )


def _transfer_result(
    paths: core.Node17PreparationPaths,
) -> HssdTransferResult:
    return HssdTransferResult(
        source_inventory=TreeInventory(11, 12_000),
        target_inventory=TreeInventory(11, 12_000),
        original_materialization_sha256={
            stage: str(index + 1) * 64
            for index, stage in enumerate(STAGES)
        },
        canonical_materialization_sha256={
            stage: chr(ord("a") + index) * 64
            for index, stage in enumerate(STAGES)
        },
        training_data=paths.hssd_training_data,
        training_data_sha256="d" * 64,
        stage_counts=dict.fromkeys(STAGES, 7),
        elapsed_seconds={"transfer": 2.0, "total": 3.0},
    )


def _preflight(source_counts: dict[str, int]) -> dict[str, object]:
    return {
        "stages": {
            stage: {
                "stage": stage,
                "source_counts": dict(source_counts),
                "total_count": sum(source_counts.values()),
                "sampling": SAMPLING,
                "boundary_instances_checked": sum(
                    min(count, 2) for count in source_counts.values()
                ),
                "collated_sources": list(source_counts),
            }
            for stage in STAGES
        },
        "elapsed_seconds": {
            **dict.fromkeys(STAGES, 0.25),
            "total": 1.0,
        },
    }


def _scope_digest(scope: list[str]) -> str:
    return sha256("\n".join(scope).encode()).hexdigest()


def _stage_data_dir(
    source: str, stage: str, root: Path
) -> dict[str, dict[str, str]]:
    values = {
        "base": str(root),
        "render_cond": str(root / "renders_cond"),
    }
    if stage == "ss64":
        values["ss_latent"] = str(
            root / "ss_latents/ss_enc_conv3d_16l8_fp16_64_view"
        )
    elif stage == "shape512":
        values["shape_latent"] = str(
            root
            / "shape_latents/"
            "shape_enc_next_dc_f16c32_fp16_512_view"
        )
    else:
        values["shape_latent"] = str(
            root
            / "shape_latents/"
            "shape_enc_next_dc_f16c32_fp16_1024_view"
        )
        if stage == "pbr1024":
            values["pbr_latent"] = str(
                root
                / "pbr_latents/"
                "tex_enc_next_dc_f16c32_fp16_1024_view_fix"
            )
    return {source: values}


def _write_source_publication(
    root: Path,
    source: str,
    schema_version: int,
    count: int,
) -> Path:
    stages = {}
    materialization_evidence = {}
    for stage in STAGES:
        active = root / stage / "active"
        active.mkdir(parents=True)
        scope = [
            f"{source.lower()}-{stage}-{index}"
            for index in range(count)
        ]
        materialization = {
            "schema_version": 1,
            "source": source,
            "stage": stage,
            "stage_root": str(active),
            "asset_count": count,
            "stage_scope": scope,
            "stage_scope_sha256": _scope_digest(scope),
        }
        materialization_path = active / "materialization.json"
        materialization_path.write_bytes(
            core._canonical_json_bytes(materialization)
        )
        stages[stage] = {
            "root": str(active),
            "asset_count": count,
            "asset_scope_sha256": _scope_digest(scope),
            "anchors_checked": count * 2,
            "validation_counts": {"assets": count},
            "data_dir": _stage_data_dir(source, stage, active),
        }
        materialization_evidence[stage] = {
            "sha256": sha256(
                materialization_path.read_bytes()
            ).hexdigest(),
            "tool_commits": [f"{source}-{stage}-tool"],
        }

    publication = root / "publication"
    publication.mkdir()
    report_path = publication / "report.json"
    report = {
        "schema_version": schema_version,
        "created_at": "2026-07-30T00:00:00Z",
        "source": source,
        "acceptance_mode": (
            "production_gate"
            if source == "HSSD"
            else "valid_subset_user_waiver"
        ),
        "original_90_percent_gate_passed": source == "HSSD",
        "authorization": "training-input use only",
        "counts": {"stages": dict.fromkeys(STAGES, count)},
        "eligibility_policy": {"schema_version": 1},
        "stages": stages,
        "materialization_evidence": materialization_evidence,
        "observed_tool_commits": [f"{source}-tool"],
    }
    if schema_version == 1:
        index_path = root / "ABO-00000.json"
        index_path.write_bytes(
            core._canonical_json_bytes(
                {"shard_id": "ABO-00000", "assets": ["abo-asset"]}
            )
        )
        report["shard_id"] = "ABO-00000"
        report["source_index"] = {
            "path": str(index_path),
            "sha256": sha256(index_path.read_bytes()).hexdigest(),
        }
    else:
        index_path = root / f"{source}-00000.json"
        index_path.write_bytes(
            core._canonical_json_bytes(
                {
                    "shard_id": f"{source}-00000",
                    "assets": [f"{source.lower()}-asset"],
                }
            )
        )
        report["source_indexes"] = [
            {
                "shard_id": f"{source}-00000",
                "path": str(index_path),
                "sha256": sha256(index_path.read_bytes()).hexdigest(),
            }
        ]
    report_path.write_bytes(core._canonical_json_bytes(report))
    handoff_path = publication / "handoff.json"
    handoff = {
        **report,
        "report": {
            "path": str(report_path),
            "sha256": sha256(report_path.read_bytes()).hexdigest(),
        },
    }
    handoff_path.write_bytes(core._canonical_json_bytes(handoff))
    training_path = root / "training_data.json"
    training = {
        **handoff,
        "handoff": {
            "path": str(handoff_path),
            "sha256": sha256(handoff_path.read_bytes()).hexdigest(),
        },
    }
    training_path.write_bytes(core._canonical_json_bytes(training))
    return training_path


def _copy_source_configs(repo_root: Path) -> None:
    for relative in core.CONFIGS.values():
        target = repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(relative), target)


def _real_report_fixture(
    tmp_path: Path, monkeypatch
) -> tuple[
    core.Node17PreparationPaths,
    Path,
    dict[str, object],
]:
    paths = _compatible_paths(tmp_path)
    _copy_source_configs(paths.repo_root)
    source_paths = {
        "ABO": _write_source_publication(
            paths.production_root / "abo", "ABO", 1, 2
        ),
        "3D-FUTURE": _write_source_publication(
            paths.production_root / "3d-future",
            "3D-FUTURE",
            2,
            5,
        ),
        "HSSD": _write_source_publication(
            paths.production_root / "hssd", "HSSD", 2, 3
        ),
    }
    hssd = source_paths["HSSD"]
    canonical_digests = {
        stage: sha256(
            (
                paths.production_root
                / f"hssd/{stage}/active/materialization.json"
            ).read_bytes()
        ).hexdigest()
        for stage in STAGES
    }
    transfer = HssdTransferResult(
        source_inventory=TreeInventory(20, 50_000),
        target_inventory=TreeInventory(20, 50_000),
        original_materialization_sha256={
            stage: str(index + 1) * 64
            for index, stage in enumerate(STAGES)
        },
        canonical_materialization_sha256=canonical_digests,
        training_data=hssd,
        training_data_sha256=sha256(hssd.read_bytes()).hexdigest(),
        stage_counts=dict.fromkeys(STAGES, 3),
        elapsed_seconds={
            "inventory": 0.1,
            "transfer": 0.2,
            "verification": 0.1,
            "evidence_rebase": 0.1,
            "strict_preflight": 0.2,
            "promotion": 0.1,
            "total": 0.8,
        },
    )
    monkeypatch.setattr(
        core, "clean_git_revision", lambda _root: REVISION
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda path, required: {
            "path": str(path),
            "free_bytes": required + 1_000,
            "required_bytes": required,
        },
    )
    monkeypatch.setattr(
        core,
        "validate_source_training_data",
        training_manifest._validate_source_training_data_for_fixture,
    )
    monkeypatch.setattr(
        core,
        "resolve_training_data",
        training_manifest._resolve_training_data_for_fixture,
    )
    monkeypatch.setattr(
        core,
        "publish_combined_training_data",
        training_manifest._publish_combined_training_data_for_fixture,
    )
    monkeypatch.setattr(
        core, "transfer_and_publish_hssd", lambda *_args: transfer
    )

    def preflight(training_data, _runtime_configs):
        counts = (
            {"HSSD": 3}
            if Path(training_data) == hssd
            else {"ABO": 2, "3D-FUTURE": 5, "HSSD": 3}
        )
        return _preflight(counts)

    monkeypatch.setattr(core, "preflight_training_data", preflight)
    report_path = core.prepare_node17_training(paths)
    report = json.loads(report_path.read_text())
    return paths, report_path, report


def test_node17_paths_derive_canonical_training_outputs():
    paths = core.Node17PreparationPaths.from_roots(
        Path("/root/data2/pixal3d"),
        Path("/root/node17/data/pixal3d"),
        Path(
            "/root/dev/Pixal3D/.worktrees/multiview-model-extension"
        ),
    )

    assert paths.production_root == Path(
        "/root/node17/data/pixal3d/train/production"
    )
    assert paths.runtime_config_root == Path(
        "/root/node17/data/pixal3d/train/runtime-configs"
    )
    assert paths.evidence_root == (
        paths.production_root / "node17-preparation-evidence"
    )
    assert paths.hssd_training_data == (
        paths.production_root / "hssd/training_data.json"
    )
    assert paths.combined_training_data == (
        paths.production_root
        / "abo-3d-future-hssd/training_data.json"
    )


def test_plan_is_read_only_and_validates_revision_configs_and_disk(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    events = []
    source_configs = {
        stage: paths.repo_root / f"{stage}.json" for stage in STAGES
    }
    monkeypatch.setattr(
        core, "validate_roots", lambda selected: events.append("roots")
    )
    monkeypatch.setattr(
        core,
        "clean_git_revision",
        lambda repo: events.append("git") or "a" * 40,
    )
    monkeypatch.setattr(
        core, "_source_config_paths", lambda selected: source_configs
    )
    monkeypatch.setattr(
        core,
        "validate_finetuning_configs",
        lambda configs: events.append("configs") or {},
    )
    monkeypatch.setattr(
        core,
        "_validate_existing_runtime_configs",
        lambda selected, configs: events.append("runtime-existing"),
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda path, required: events.append("disk")
        or {
            "path": str(path),
            "free_bytes": required + 1,
            "required_bytes": required,
        },
    )
    monkeypatch.setattr(
        core,
        "create_node17_runtime_configs",
        lambda *_args: pytest.fail("plan created runtime configs"),
    )
    monkeypatch.setattr(
        core,
        "transfer_and_publish_hssd",
        lambda *_args: pytest.fail("plan contacted Node16"),
    )
    monkeypatch.setattr(
        core,
        "publish_combined_training_data",
        lambda *_args: pytest.fail("plan published combined data"),
    )
    monkeypatch.setattr(
        core,
        "_assert_cpu_only",
        lambda: pytest.fail("plan imported or queried CUDA"),
    )

    plan = core.plan_node17_training(paths)

    assert events == [
        "roots",
        "git",
        "configs",
        "runtime-existing",
        "disk",
    ]
    assert plan["execute"] is False
    assert plan["revision"] == "a" * 40
    assert plan["combined_training_data"] == str(
        paths.combined_training_data
    )
    assert not paths.production_root.exists()
    assert not paths.runtime_config_root.exists()
    assert not paths.evidence_root.exists()


def test_plan_module_import_does_not_import_torch():
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import data_toolkit.pipeline.node17_training_prepare; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def _compatible_paths(tmp_path: Path) -> core.Node17PreparationPaths:
    local = tmp_path / "node17/data/pixal3d"
    repo = tmp_path / "repo"
    local.mkdir(parents=True)
    repo.mkdir()
    return core.Node17PreparationPaths.from_roots(
        NODE17_DATA2_ROOT, local, repo
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("data2_root", Path("/root/data2/other"), "data2 root"),
        ("source_host", "other@n16.unist.info", "source host"),
        ("source_port", 22, "source port"),
        (
            "source_root",
            Path("/home/youngwoo/data/pixal3d/train/production/other"),
            "source root",
        ),
    ),
)
@pytest.mark.parametrize("operation", ("plan", "execute"))
def test_plan_and_execute_reject_task4_incompatible_identity_before_mutation(
    tmp_path, monkeypatch, field, value, message, operation
):
    paths = replace(_compatible_paths(tmp_path), **{field: value})
    monkeypatch.setattr(
        core, "clean_git_revision", lambda _root: "a" * 40
    )
    monkeypatch.setattr(
        core, "validate_finetuning_configs", lambda _configs: {}
    )
    monkeypatch.setattr(
        core,
        "_validate_existing_runtime_configs",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda path, required: {
            "path": str(path),
            "free_bytes": required + 1,
            "required_bytes": required,
        },
    )
    monkeypatch.setattr(core, "_assert_cpu_only", lambda: None)
    monkeypatch.setattr(
        core,
        "create_node17_runtime_configs",
        lambda *_args: pytest.fail("runtime config mutation occurred"),
    )
    monkeypatch.setattr(
        core,
        "transfer_and_publish_hssd",
        lambda *_args: pytest.fail("transfer/network occurred"),
    )

    selected = (
        core.plan_node17_training
        if operation == "plan"
        else core.prepare_node17_training
    )
    with pytest.raises(ValueError, match=message):
        selected(paths)

    assert not paths.runtime_config_root.exists()
    assert not paths.production_root.exists()


def test_prepare_rejects_preinitialized_cuda_before_any_mutation(
    tmp_path, monkeypatch
):
    import torch

    paths = _compatible_paths(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        core,
        "create_node17_runtime_configs",
        lambda *_args: pytest.fail("runtime config mutation occurred"),
    )
    monkeypatch.setattr(
        core,
        "transfer_and_publish_hssd",
        lambda *_args: pytest.fail("transfer/network occurred"),
    )

    with pytest.raises(
        RuntimeError, match="CPU-only preparation initialized CUDA"
    ):
        core.prepare_node17_training(paths)

    assert not paths.runtime_config_root.exists()
    assert not paths.production_root.exists()


def test_preflight_rechecks_cuda_when_a_stage_raises(
    tmp_path, monkeypatch
):
    runtime_paths = {
        stage: tmp_path / f"{stage}.json" for stage in STAGES
    }
    initialized = False
    checks = []

    def cpu_check():
        checks.append(initialized)
        if initialized:
            raise RuntimeError("CPU-only preparation initialized CUDA")

    def failing_stage(*_args):
        nonlocal initialized
        initialized = True
        raise ValueError("configured stage failed")

    monkeypatch.setattr(core, "_assert_cpu_only", cpu_check)
    monkeypatch.setattr(
        core, "preflight_multisource_stage", failing_stage
    )

    with pytest.raises(
        RuntimeError, match="CPU-only preparation initialized CUDA"
    ):
        core.preflight_training_data(
            tmp_path / "training_data.json", runtime_paths
        )

    assert checks == [False, True]


def test_prepare_uses_existing_task_boundaries_in_required_order(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    runtime_paths = _runtime_paths(paths)
    transfer_result = _transfer_result(paths)
    events = []
    source_by_name = {
        source: _validated_source(paths, source)
        for source in ("ABO", "3D-FUTURE", "HSSD")
    }
    monkeypatch.setattr(core, "validate_roots", lambda _paths: None)
    monkeypatch.setattr(
        core,
        "clean_git_revision",
        lambda _root: events.append("git") or "a" * 40,
    )
    monkeypatch.setattr(
        core, "validate_finetuning_configs", lambda _configs: {}
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda path, required: {
            "path": str(path),
            "free_bytes": required + 1,
            "required_bytes": required,
        },
    )
    monkeypatch.setattr(core, "_assert_cpu_only", lambda: None)
    monkeypatch.setattr(
        core,
        "create_node17_runtime_configs",
        lambda _configs, _root: events.append("runtime") or runtime_paths,
    )
    monkeypatch.setattr(
        core,
        "node17_runtime_config_evidence",
        lambda _runtime, _source: {
            stage: {
                "path": str(runtime_paths[stage]),
                "sha256": "a" * 64,
                "source_config": {
                    "path": str(paths.repo_root / f"{stage}.json"),
                    "sha256": "b" * 64,
                },
            }
            for stage in STAGES
        },
    )
    monkeypatch.setattr(
        core,
        "transfer_and_publish_hssd",
        lambda _paths, _configs: (
            events.append("transfer") or transfer_result
        ),
    )

    def validate_source(source, path):
        if source in {"ABO", "3D-FUTURE"}:
            events.append(f"validate-{source}")
        return source_by_name[source]

    monkeypatch.setattr(
        core, "validate_source_training_data", validate_source
    )

    def preflight(training_data, configs):
        if training_data == paths.hssd_training_data:
            events.append("preflight-hssd")
            return _preflight({"HSSD": 7})
        events.append("preflight-combined")
        return _preflight({"ABO": 3, "3D-FUTURE": 5, "HSSD": 7})

    monkeypatch.setattr(core, "preflight_training_data", preflight)
    monkeypatch.setattr(
        core,
        "publish_three_source_combined",
        lambda selected: (
            events.append("publish-combined")
            or selected.combined_training_data
        ),
    )
    monkeypatch.setattr(
        core,
        "training_scope_evidence",
        lambda _path: {
            stage: {
                "source_counts": {
                    "ABO": 3,
                    "3D-FUTURE": 5,
                    "HSSD": 7,
                },
                "total_count": 15,
                "union_scope_sha256": "c" * 64,
            }
            for stage in STAGES
        },
    )
    monkeypatch.setattr(
        core, "_file_digest", lambda _path, _label: "f" * 64
    )
    monkeypatch.setattr(
        core,
        "write_final_report",
        lambda selected, report: (
            events.append("report")
            or selected.evidence_root / "report.json"
        ),
    )

    result = core.prepare_node17_training(paths)

    assert result == paths.evidence_root / "report.json"
    assert events == [
        "git",
        "runtime",
        "transfer",
        "validate-ABO",
        "validate-3D-FUTURE",
        "preflight-hssd",
        "publish-combined",
        "preflight-combined",
        "report",
    ]
    assert not hasattr(core, "materialize_stage")
    assert not hasattr(core, "preflight_stage")


def test_report_contains_all_readiness_evidence_and_exact_launch_commands(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    runtime_paths = _runtime_paths(paths)
    captured = {}
    source_by_name = {
        source: _validated_source(paths, source)
        for source in ("ABO", "3D-FUTURE", "HSSD")
    }
    monkeypatch.setattr(core, "validate_roots", lambda _paths: None)
    monkeypatch.setattr(
        core, "clean_git_revision", lambda _root: "a" * 40
    )
    monkeypatch.setattr(
        core, "validate_finetuning_configs", lambda _configs: {}
    )
    monkeypatch.setattr(
        core,
        "assert_free_space",
        lambda path, required: {
            "path": str(path),
            "free_bytes": required + 100,
            "required_bytes": required,
        },
    )
    monkeypatch.setattr(core, "_assert_cpu_only", lambda: None)
    monkeypatch.setattr(
        core,
        "create_node17_runtime_configs",
        lambda _configs, _root: runtime_paths,
    )
    runtime_evidence = {
        stage: {
            "path": str(runtime_paths[stage]),
            "sha256": chr(97 + index) * 64,
            "source_config": {
                "path": str(paths.repo_root / f"{stage}.json"),
                "sha256": chr(102 + index) * 64,
            },
        }
        for index, stage in enumerate(STAGES)
    }
    monkeypatch.setattr(
        core,
        "node17_runtime_config_evidence",
        lambda _runtime, _source: runtime_evidence,
    )
    transfer = _transfer_result(paths)
    monkeypatch.setattr(
        core, "transfer_and_publish_hssd", lambda *_args: transfer
    )
    monkeypatch.setattr(
        core,
        "validate_source_training_data",
        lambda source, _path: source_by_name[source],
    )
    standalone = _preflight({"HSSD": 7})
    combined_preflight = _preflight(
        {"ABO": 3, "3D-FUTURE": 5, "HSSD": 7}
    )
    monkeypatch.setattr(
        core,
        "preflight_training_data",
        lambda path, _configs: (
            standalone
            if path == paths.hssd_training_data
            else combined_preflight
        ),
    )
    monkeypatch.setattr(
        core,
        "publish_three_source_combined",
        lambda selected: selected.combined_training_data,
    )
    scopes = {
        stage: {
            "source_counts": {
                "ABO": 3,
                "3D-FUTURE": 5,
                "HSSD": 7,
            },
            "total_count": 15,
            "union_scope_sha256": "e" * 64,
        }
        for stage in STAGES
    }
    monkeypatch.setattr(
        core, "training_scope_evidence", lambda _path: scopes
    )
    monkeypatch.setattr(
        core, "_file_digest", lambda _path, _label: "f" * 64
    )

    def capture(_paths, report):
        captured.update(report)
        return paths.evidence_root / "report.json"

    monkeypatch.setattr(core, "write_final_report", capture)

    core.prepare_node17_training(paths)

    assert captured["revision"] == "a" * 40
    assert captured["runtime_configs"] == runtime_evidence
    assert captured["transfer"]["source_inventory"] == {
        "file_count": 11,
        "logical_bytes": 12_000,
    }
    assert captured["transfer"]["target_inventory"] == {
        "file_count": 11,
        "logical_bytes": 12_000,
    }
    assert captured["transfer"]["original_materialization_sha256"]
    assert captured["transfer"]["canonical_materialization_sha256"]
    assert set(captured["sources"]) == {"ABO", "3D-FUTURE", "HSSD"}
    for source in captured["sources"].values():
        assert set(source["artifacts"]) == {
            "report",
            "handoff",
            "training_data",
        }
    assert captured["hssd_standalone_preflight"] == standalone
    assert captured["combined"]["stages"] == scopes
    assert captured["combined"]["preflight"] == combined_preflight
    assert captured["elapsed_seconds"]

    common = (
        "/opt/conda/envs/pixal3d/bin/python train.py "
        "--config {config} "
        f"--training_data {paths.combined_training_data} "
        "--num_gpus 6 --use_wandb"
    )
    assert captured["launch_commands"] == {
        stage: common.format(config=runtime_paths[stage])
        for stage in STAGES
    }
    assert not paths.local_root.joinpath("training-logs").exists()


def test_combined_publication_rejects_partial_directory(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    paths.combined_training_data.parent.mkdir(parents=True)
    (paths.combined_training_data.parent / "partial").write_text("stop")
    monkeypatch.setattr(
        core,
        "publish_combined_training_data",
        lambda *_args: pytest.fail("partial directory was reused"),
    )

    with pytest.raises(ValueError, match="partial combined"):
        core.publish_three_source_combined(paths)


def test_assembled_report_passes_real_validation_and_immutable_reuse(
    tmp_path, monkeypatch
):
    paths, output, report = _real_report_fixture(
        tmp_path, monkeypatch
    )
    inode = output.stat().st_ino
    core._validate_report(paths, report)
    rerun = json.loads(json.dumps(report))
    rerun["disk"]["free_bytes"] -= 1
    rerun["elapsed_seconds"]["total"] = 7.0
    rerun["transfer"]["elapsed_seconds"]["total"] = 6.0

    assert core.write_final_report(paths, rerun) == output
    assert output.stat().st_ino == inode


def test_real_report_validation_rejects_referenced_artifact_byte_change(
    tmp_path, monkeypatch
):
    paths, _output, report = _real_report_fixture(
        tmp_path, monkeypatch
    )
    source_report = Path(
        report["sources"]["ABO"]["artifacts"]["report"]["path"]
    )
    source_report.write_bytes(source_report.read_bytes() + b" ")

    with pytest.raises(ValueError):
        core._validate_report(paths, report)


@pytest.mark.parametrize("layer", ("source", "combined"))
def test_real_report_validation_rejects_source_and_combined_resolution_change(
    tmp_path, monkeypatch, layer
):
    paths, _output, report = _real_report_fixture(
        tmp_path, monkeypatch
    )
    if layer == "source":
        materialization = (
            paths.production_root
            / "3d-future/ss64/active/materialization.json"
        )
        materialization.write_bytes(
            materialization.read_bytes() + b" "
        )
    else:
        combined_path = paths.combined_training_data
        combined = json.loads(combined_path.read_text())
        combined["stages"]["ss64"]["total_count"] += 1
        combined_path.write_bytes(
            core._canonical_json_bytes(combined)
        )
        report["combined"]["sha256"] = sha256(
            combined_path.read_bytes()
        ).hexdigest()

    with pytest.raises(ValueError):
        core._validate_report(paths, report)


def test_real_report_reuse_rejects_changed_invariant(
    tmp_path, monkeypatch
):
    paths, _output, report = _real_report_fixture(
        tmp_path, monkeypatch
    )
    changed = deepcopy(report)
    changed["transfer"]["original_materialization_sha256"][
        "ss64"
    ] = "9" * 64

    with pytest.raises(FileExistsError, match="different invariant"):
        core.write_final_report(paths, changed)


def test_final_report_rejects_partial_evidence_directory(
    tmp_path,
):
    paths = _paths(tmp_path)
    paths.evidence_root.mkdir(parents=True)
    (paths.evidence_root / "partial.json").write_text("{}")

    with pytest.raises(ValueError, match="partial evidence"):
        core.write_final_report(paths, {"revision": "a" * 40})


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("CUDA_VISIBLE_DEVICES", None, "CUDA_VISIBLE_DEVICES"),
        ("CUDA_VISIBLE_DEVICES", "0", "CUDA_VISIBLE_DEVICES"),
        ("PYTHONDONTWRITEBYTECODE", None, "PYTHONDONTWRITEBYTECODE"),
        ("PYTHONDONTWRITEBYTECODE", "0", "PYTHONDONTWRITEBYTECODE"),
    ),
)
def test_cli_rejects_unsafe_import_environment(
    monkeypatch, name, value, message
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        module = importlib.import_module("scripts.prepare_node17_training")
        importlib.reload(module)


def test_cli_defaults_to_read_only_plan(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    module = importlib.import_module("scripts.prepare_node17_training")
    module = importlib.reload(module)
    captured = {}

    def plan(paths):
        captured["paths"] = paths
        return {"execute": False}

    monkeypatch.setattr(module, "plan_node17_training", plan)
    monkeypatch.setattr(
        module,
        "prepare_node17_training",
        lambda _paths: pytest.fail("default CLI executed mutations"),
    )

    assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out) == {"execute": False}
    paths = captured["paths"]
    assert paths.data2_root == Path("/root/data2/pixal3d")
    assert paths.local_root == Path("/root/node17/data/pixal3d")
    assert paths.repo_root == Path(
        "/root/dev/Pixal3D/.worktrees/multiview-model-extension"
    )
    assert paths.source_host == "youngwoo@n16.unist.info"
    assert paths.source_port == 55555
    assert paths.source_root == Path(
        "/home/youngwoo/data/pixal3d/train/production/hssd"
    )


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _historical_git_fixture(
    tmp_path: Path,
) -> tuple[Path, str, str]:
    repo = tmp_path / "historical-repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "train.py").write_text("print('training')\n")
    _git(repo, "add", "train.py")
    _git(repo, "commit", "-m", "execution")
    evidence_revision = _git(repo, "rev-parse", "HEAD")

    readiness = (
        repo
        / "docs/superpowers/reports/"
        "2026-07-30-node17-three-source-training-readiness.md"
    )
    readiness.parent.mkdir(parents=True)
    readiness.write_text("# readiness\n")
    _git(repo, "add", str(readiness.relative_to(repo)))
    _git(repo, "commit", "-m", "delivery docs")
    delivery_revision = _git(repo, "rev-parse", "HEAD")
    return repo, evidence_revision, delivery_revision


def test_historical_revision_accepts_docs_only_delivery_descendant(
    tmp_path,
):
    repo, evidence_revision, delivery_revision = (
        _historical_git_fixture(tmp_path)
    )

    result = core.validate_historical_evidence_revision(
        repo,
        evidence_revision,
        delivery_revision,
        delivery_revision,
    )

    assert result == {
        "evidence_revision": evidence_revision,
        "delivery_revision": delivery_revision,
        "validator_revision": delivery_revision,
        "current_revision": delivery_revision,
        "delivery_changed_paths": [
            "docs/superpowers/reports/"
            "2026-07-30-node17-three-source-training-readiness.md"
        ],
        "validator_changed_paths": [],
        "finalization_changed_paths": [],
    }


def test_historical_revision_accepts_exact_validator_bootstrap_paths(
    tmp_path,
):
    repo, evidence_revision, delivery_revision = (
        _historical_git_fixture(tmp_path)
    )
    paths = (
        "docs/node17_three_source_training_runbook_ko.md",
        (
            "docs/superpowers/reports/"
            "2026-07-30-node17-three-source-training-readiness.md"
        ),
        "data_toolkit/pipeline/node17_hssd_transfer.py",
        "data_toolkit/pipeline/node17_training_prepare.py",
        "scripts/prepare_node17_training.py",
        "tests/multiview/test_node17_hssd_transfer.py",
        "tests/multiview/test_node17_training_prepare.py",
    )
    for relative in paths:
        selected = repo / relative
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(f"{relative}\n")
    _git(repo, "add", *paths)
    _git(repo, "commit", "-m", "validator bootstrap")
    validator_revision = _git(repo, "rev-parse", "HEAD")

    result = core.validate_historical_evidence_revision(
        repo,
        evidence_revision,
        delivery_revision,
        validator_revision,
    )

    assert result["validator_revision"] == validator_revision
    assert result["current_revision"] == validator_revision
    assert result["validator_changed_paths"] == sorted(paths)
    assert result["finalization_changed_paths"] == []


def test_historical_revision_rejects_code_change_after_delivery(
    tmp_path,
):
    repo, evidence_revision, delivery_revision = (
        _historical_git_fixture(tmp_path)
    )
    (repo / "train.py").write_text("print('changed training')\n")
    _git(repo, "add", "train.py")
    _git(repo, "commit", "-m", "code changed")

    with pytest.raises(ValueError, match="disallowed"):
        core.validate_historical_evidence_revision(
            repo,
            evidence_revision,
            delivery_revision,
            _git(repo, "rev-parse", "HEAD"),
        )


def test_historical_revision_rejects_non_descendant_delivery(
    tmp_path,
):
    repo, evidence_revision, delivery_revision = (
        _historical_git_fixture(tmp_path)
    )
    _git(repo, "checkout", "-b", "unrelated", evidence_revision)
    (repo / "other.md").write_text("unrelated\n")
    _git(repo, "add", "other.md")
    _git(repo, "commit", "-m", "unrelated")

    with pytest.raises(ValueError, match="ancestor"):
        core.validate_historical_evidence_revision(
            repo,
            evidence_revision,
            delivery_revision,
            delivery_revision,
        )


def test_historical_revision_rejects_missing_recorded_revision(
    tmp_path,
):
    repo, _evidence_revision, delivery_revision = (
        _historical_git_fixture(tmp_path)
    )

    with pytest.raises(ValueError, match="missing"):
        core.validate_historical_evidence_revision(
            repo,
            "f" * 40,
            delivery_revision,
            delivery_revision,
        )


def test_historical_revision_rejects_dirty_worktree(
    tmp_path,
):
    repo, evidence_revision, delivery_revision = (
        _historical_git_fixture(tmp_path)
    )
    (repo / "dirty.txt").write_text("dirty\n")

    with pytest.raises(ValueError, match="clean"):
        core.validate_historical_evidence_revision(
            repo,
            evidence_revision,
            delivery_revision,
            delivery_revision,
        )


@pytest.mark.parametrize(
    "relative",
    (
        "train.py",
        "configs/gen/unrelated.json",
        "tests/multiview/unrelated.py",
        "docs/unrelated.md",
    ),
)
def test_historical_revision_rejects_unrelated_change_after_validator(
    tmp_path, relative
):
    repo, evidence_revision, delivery_revision = (
        _historical_git_fixture(tmp_path)
    )
    validator_revision = delivery_revision
    selected = repo / relative
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text("changed\n")
    _git(repo, "add", relative)
    _git(repo, "commit", "-m", "unrelated finalization")

    with pytest.raises(ValueError, match="disallowed"):
        core.validate_historical_evidence_revision(
            repo,
            evidence_revision,
            delivery_revision,
            validator_revision,
        )


def test_historical_revision_accepts_task7_docs_after_validator(
    tmp_path,
):
    repo, evidence_revision, delivery_revision = (
        _historical_git_fixture(tmp_path)
    )
    validator_revision = delivery_revision
    runbook = repo / "docs/node17_three_source_training_runbook_ko.md"
    runbook.parent.mkdir(parents=True, exist_ok=True)
    runbook.write_text("# validation\n")
    _git(repo, "add", str(runbook.relative_to(repo)))
    _git(repo, "commit", "-m", "final Task7 docs")
    current_revision = _git(repo, "rev-parse", "HEAD")

    result = core.validate_historical_evidence_revision(
        repo,
        evidence_revision,
        delivery_revision,
        validator_revision,
    )

    assert result["current_revision"] == current_revision
    assert result["finalization_changed_paths"] == [
        "docs/node17_three_source_training_runbook_ko.md"
    ]


def test_historical_report_revalidates_artifacts_at_recorded_revision(
    tmp_path, monkeypatch
):
    paths, output, _report = _real_report_fixture(
        tmp_path, monkeypatch
    )
    report_sha256 = sha256(output.read_bytes()).hexdigest()
    current_revision = "c" * 40
    delivery_revision = "b" * 40
    validator_revision = "d" * 40
    revision_evidence = {
        "evidence_revision": REVISION,
        "delivery_revision": delivery_revision,
        "validator_revision": validator_revision,
        "current_revision": current_revision,
        "delivery_changed_paths": ["docs/readiness.md"],
        "validator_changed_paths": ["scripts/validator.py"],
        "finalization_changed_paths": ["docs/runbook.md"],
    }
    monkeypatch.setattr(
        core, "clean_git_revision", lambda _root: current_revision
    )
    monkeypatch.setattr(
        core,
        "validate_historical_evidence_revision",
        lambda repo, evidence, delivery, validator: (
            revision_evidence
            if (
                repo == paths.repo_root
                and evidence == REVISION
                and delivery == delivery_revision
                and validator == validator_revision
            )
            else pytest.fail("historical revision policy mismatch")
        ),
    )

    result = core.validate_node17_historical_preparation_report(
        paths,
        delivery_revision,
        validator_revision,
        report_sha256,
    )

    assert result == {
        "report": str(output),
        "report_sha256": report_sha256,
        **revision_evidence,
    }


@pytest.mark.parametrize(
    "report_sha256",
    (
        None,
        "",
        "a" * 63,
        "A" * 64,
        "g" * 64,
        "0" * 64,
    ),
)
def test_historical_report_requires_matching_lowercase_external_digest(
    tmp_path, monkeypatch, report_sha256
):
    paths, output, _report = _real_report_fixture(
        tmp_path, monkeypatch
    )

    with pytest.raises(ValueError, match="report SHA-256"):
        core.validate_node17_historical_preparation_report(
            paths,
            "b" * 40,
            "d" * 40,
            report_sha256,
        )


def test_historical_report_rejects_raw_tamper_before_json_parsing(
    tmp_path, monkeypatch
):
    paths, output, _report = _real_report_fixture(
        tmp_path, monkeypatch
    )
    trusted_sha256 = sha256(output.read_bytes()).hexdigest()
    raw = bytearray(output.read_bytes())
    raw[-2] ^= 1
    output.write_bytes(raw)
    monkeypatch.setattr(
        core.json,
        "loads",
        lambda *_args, **_kwargs: pytest.fail(
            "report JSON was parsed before external digest verification"
        ),
    )

    with pytest.raises(ValueError, match="report SHA-256"):
        core.validate_node17_historical_preparation_report(
            paths,
            "b" * 40,
            "d" * 40,
            trusted_sha256,
        )


def test_historical_report_rejects_valid_looking_internal_mutation(
    tmp_path, monkeypatch
):
    paths, output, report = _real_report_fixture(
        tmp_path, monkeypatch
    )
    trusted_sha256 = sha256(output.read_bytes()).hexdigest()
    report["revision"] = "b" * 40
    output.write_bytes(core._canonical_json_bytes(report))
    monkeypatch.setattr(
        core,
        "validate_historical_evidence_revision",
        lambda *_args: pytest.fail(
            "mutated report fields were accepted before pin verification"
        ),
    )

    with pytest.raises(ValueError, match="report SHA-256"):
        core.validate_node17_historical_preparation_report(
            paths,
            "b" * 40,
            "d" * 40,
            trusted_sha256,
        )


def test_existing_report_validator_keeps_exact_current_revision_requirement(
    tmp_path, monkeypatch
):
    paths, _output, _report = _real_report_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        core, "clean_git_revision", lambda _root: "c" * 40
    )

    with pytest.raises(ValueError, match="changed Git revision"):
        core.validate_node17_preparation_report(paths)


def test_cli_historical_validation_is_read_only(
    monkeypatch, capsys
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    module = importlib.import_module("scripts.prepare_node17_training")
    module = importlib.reload(module)
    delivery_revision = "b" * 40
    validator_revision = "d" * 40
    report_sha256 = "e" * 64
    captured = {}

    def validate(paths, delivery, validator, trusted_report_sha256):
        captured["paths"] = paths
        captured["delivery_revision"] = delivery
        captured["validator_revision"] = validator
        captured["report_sha256"] = trusted_report_sha256
        return {"report": "/immutable/report.json"}

    monkeypatch.setattr(
        module,
        "validate_node17_historical_preparation_report",
        validate,
    )
    monkeypatch.setattr(
        module,
        "prepare_node17_training",
        lambda _paths: pytest.fail("validation executed mutation"),
    )

    assert module.main(
        [
            "--validate-evidence",
            "--delivery-revision",
            delivery_revision,
            "--validator-revision",
            validator_revision,
            "--report-sha256",
            report_sha256,
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out) == {
        "report": "/immutable/report.json"
    }
    assert captured["delivery_revision"] == delivery_revision
    assert captured["validator_revision"] == validator_revision
    assert captured["report_sha256"] == report_sha256


def test_cli_historical_validation_requires_report_digest(
    monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    module = importlib.import_module("scripts.prepare_node17_training")
    module = importlib.reload(module)

    with pytest.raises(ValueError, match="--report-sha256"):
        module.main(
            [
                "--validate-evidence",
                "--delivery-revision",
                "b" * 40,
                "--validator-revision",
                "d" * 40,
            ]
        )
