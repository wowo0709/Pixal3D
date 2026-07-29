from copy import deepcopy
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
    publication = {
        "report": output / "publication/report.json",
        "handoff": output / "publication/handoff.json",
        "training-data": output / "training_data.json",
    }
    for path in publication.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
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
    training_data = paths.production_root / "abo" / "training_data.json"
    training_data.parent.mkdir(parents=True)
    training_data.write_text("{}")
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


def test_final_report_refuses_partial_evidence_root(tmp_path):
    paths = _paths(tmp_path)
    partial = paths.evidence_root / "operator-note.txt"
    partial.parent.mkdir(parents=True)
    partial.write_text("inspect")

    with pytest.raises(ValueError, match="partial source output"):
        core.write_final_report(paths, {"schema_version": 1})

    assert partial.read_text() == "inspect"
    assert not (paths.evidence_root / "report.json").exists()


def test_final_report_reuses_historical_disk_and_source_state(tmp_path):
    paths = _paths(tmp_path)
    original = {
        "schema_version": 1,
        "disk": {
            "path": str(paths.local_root),
            "required_bytes": 100,
            "stage_expanded_pack_bytes": 50,
            "total_bytes": 1_000,
            "used_bytes": 200,
            "free_bytes": 800,
        },
        "sources": {
            profile: {
                "reused": False,
                "artifacts": {"training_data": {"sha256": profile * 8}},
            }
            for profile in ("abo", "3d-future", "hssd")
        },
        "combined": {"sha256": "a" * 64},
    }
    output = core.write_final_report(paths, original)
    before = output.read_bytes()
    rerun = deepcopy(original)
    rerun["disk"]["used_bytes"] = 300
    rerun["disk"]["free_bytes"] = 700
    for source in rerun["sources"].values():
        source["reused"] = True

    assert core.write_final_report(paths, rerun) == output
    assert output.read_bytes() == before

    changed = deepcopy(rerun)
    changed["combined"]["sha256"] = "b" * 64
    with pytest.raises(
        FileExistsError, match="preparation report has different content"
    ):
        core.write_final_report(paths, changed)
    assert output.read_bytes() == before


def test_prepare_orders_sources_then_standalone_and_combined_validation(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    calls = []
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
        lambda configs: {"stages": sorted(configs)},
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

    report = prepare_node16_training(paths)

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


def test_plan_validates_inputs_without_creating_local_roots(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
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

    plan = core.plan_node16_training(paths)

    assert plan["execute"] is False
    assert plan["combined_training_data"] == str(
        paths.combined_training_data
    )
    assert not paths.local_root.exists()


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
