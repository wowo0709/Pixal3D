from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline import node17_training_prepare as core
from data_toolkit.pipeline.node17_hssd_transfer import (
    HssdTransferResult,
    TreeInventory,
)
from data_toolkit.pipeline.training_manifest import STAGES


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
                "sampling": "uniform-over-union",
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


def test_final_report_reuses_only_when_invariants_match(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    validations = []
    monkeypatch.setattr(
        core,
        "_validate_report",
        lambda selected, report: validations.append(
            (selected, report["revision"])
        ),
    )
    report = {
        "revision": "a" * 40,
        "disk": {"free_bytes": 100, "required_bytes": 50},
        "elapsed_seconds": {"total": 1.0},
        "transfer": {"elapsed_seconds": {"total": 0.5}},
    }

    output = core.write_final_report(paths, report)
    inode = output.stat().st_ino
    rerun = json.loads(json.dumps(report))
    rerun["disk"]["free_bytes"] = 90
    rerun["elapsed_seconds"]["total"] = 7.0
    rerun["transfer"]["elapsed_seconds"]["total"] = 6.0

    assert core.write_final_report(paths, rerun) == output
    assert output.stat().st_ino == inode
    assert validations == [
        (paths, "a" * 40),
        (paths, "a" * 40),
        (paths, "a" * 40),
    ]

    changed = json.loads(json.dumps(rerun))
    changed["revision"] = "b" * 40
    with pytest.raises(FileExistsError, match="different invariant"):
        core.write_final_report(paths, changed)


def test_final_report_rejects_partial_evidence_directory(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    paths.evidence_root.mkdir(parents=True)
    (paths.evidence_root / "partial.json").write_text("{}")
    monkeypatch.setattr(core, "_validate_report", lambda *_args: None)

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
