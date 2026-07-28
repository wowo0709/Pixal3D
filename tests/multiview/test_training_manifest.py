import hashlib
import json
from pathlib import Path
import subprocess
import sys

from easydict import EasyDict as edict
import pytest
import torch

from data_toolkit.pipeline.training_manifest import (
    CANONICAL_SOURCES,
    STAGES,
    ResolvedTrainingData,
    build_combined_training_data,
    publish_combined_training_data,
    resolve_training_data,
    resolve_training_input,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MULTIVIEW_CONFIGS = (
    (
        "configs/gen/"
        "ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json",
        "ss64",
    ),
    (
        "configs/gen/"
        "slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json",
        "shape512",
    ),
    (
        "configs/gen/"
        "slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json",
        "shape1024",
    ),
    (
        "configs/gen/"
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json",
        "pbr1024",
    ),
)


def _canonical_json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _ordered_json_bytes(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def _scope_digest(scope):
    return hashlib.sha256("\n".join(scope).encode()).hexdigest()


def _stage_data_dir(source, stage, root):
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
            root / "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"
        )
    elif stage in {"shape1024", "pbr1024"}:
        values["shape_latent"] = str(
            root / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"
        )
        if stage == "pbr1024":
            values["pbr_latent"] = str(
                root
                / "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix"
            )
    return {source: values}


def _write_source(tmp_path, source, schema_version, count):
    stages = {}
    materialization_evidence = {}
    for stage in STAGES:
        root = tmp_path / source / stage / "active"
        root.mkdir(parents=True)
        scope = [f"{source.lower()}-{stage}-{index}" for index in range(count)]
        materialization = {
            "schema_version": 1,
            "source": source,
            "stage": stage,
            "stage_root": str(root),
            "asset_count": count,
            "stage_scope": scope,
            "stage_scope_sha256": _scope_digest(scope),
        }
        materialization_path = root / "materialization.json"
        materialization_path.write_bytes(_canonical_json_bytes(materialization))
        stages[stage] = {
            "root": str(root),
            "asset_count": count,
            "asset_scope_sha256": _scope_digest(scope),
            "anchors_checked": count * 2,
            "validation_counts": {"assets": count},
            "data_dir": _stage_data_dir(source, stage, root),
        }
        materialization_evidence[stage] = {
            "sha256": hashlib.sha256(materialization_path.read_bytes()).hexdigest(),
            "tool_commits": [f"{source}-{stage}-tool"],
        }

    report_path = tmp_path / source / "report.json"
    report = {
        "schema_version": schema_version,
        "created_at": "2026-07-28T00:00:00Z",
        "source": source,
        "acceptance_mode": "valid_subset_user_waiver",
        "original_90_percent_gate_passed": False,
        "authorization": "training-input use only",
        "counts": {"stages": {stage: count for stage in STAGES}},
        "eligibility_policy": {"schema_version": 1},
        "stages": stages,
        "materialization_evidence": materialization_evidence,
        "observed_tool_commits": [f"{source}-tool"],
    }
    if schema_version == 1:
        report["shard_id"] = "ABO-00000"
        report["source_index"] = {
            "path": str(tmp_path / source / "ABO-00000.json"),
            "sha256": "b" * 64,
        }
    else:
        report["source_indexes"] = [
            {
                "shard_id": f"{source}-00000",
                "path": str(tmp_path / source / f"{source}-00000.json"),
                "sha256": "b" * 64,
            }
        ]
    report_path.write_bytes(_canonical_json_bytes(report))
    handoff_path = tmp_path / source / "handoff.json"
    handoff = {
        **report,
        "report": {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
    }
    handoff_path.write_bytes(_canonical_json_bytes(handoff))
    training_path = tmp_path / source / "training_data.json"
    training = {
        **handoff,
        "handoff": {
            "path": str(handoff_path),
            "sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
        },
    }
    training_path.write_bytes(_canonical_json_bytes(training))
    return training_path


def _rewrite_source_chain(training_path, mutate):
    training = json.loads(training_path.read_text())
    handoff_path = Path(training["handoff"]["path"])
    handoff = json.loads(handoff_path.read_text())
    report_path = Path(handoff["report"]["path"])
    report = json.loads(report_path.read_text())
    mutate(report)
    report_path.write_bytes(_canonical_json_bytes(report))
    handoff = {
        **report,
        "report": {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
    }
    handoff_path.write_bytes(_canonical_json_bytes(handoff))
    training = {
        **handoff,
        "handoff": {
            "path": str(handoff_path),
            "sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
        },
    }
    training_path.write_bytes(_canonical_json_bytes(training))


def _rewrite_report_reference_only(training_path, mutate):
    """Mutate the report and repin it without updating the handoff projection."""
    training = json.loads(training_path.read_text())
    handoff_path = Path(training["handoff"]["path"])
    handoff = json.loads(handoff_path.read_text())
    report_path = Path(handoff["report"]["path"])
    report = json.loads(report_path.read_text())
    mutate(report)
    report_path.write_bytes(_canonical_json_bytes(report))
    handoff["report"]["sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    handoff_path.write_bytes(_canonical_json_bytes(handoff))
    training = {
        **handoff,
        "handoff": {
            "path": str(handoff_path),
            "sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
        },
    }
    training_path.write_bytes(_canonical_json_bytes(training))


def _rewrite_materialization(training_path, stage, mutate):
    training = json.loads(training_path.read_text())
    materialization_path = (
        Path(training["stages"][stage]["root"]) / "materialization.json"
    )
    materialization = json.loads(materialization_path.read_text())
    mutate(materialization)
    materialization_path.write_bytes(_canonical_json_bytes(materialization))
    digest = hashlib.sha256(materialization_path.read_bytes()).hexdigest()

    def update_handoff(handoff):
        handoff["materialization_evidence"][stage]["sha256"] = digest

    _rewrite_source_chain(training_path, update_handoff)


@pytest.fixture
def source_inputs(tmp_path):
    return {
        "ABO": _write_source(tmp_path, "ABO", schema_version=1, count=2),
        "3D-FUTURE": _write_source(
            tmp_path, "3D-FUTURE", schema_version=2, count=5
        ),
    }


@pytest.fixture
def manifest(source_inputs, tmp_path):
    path = tmp_path / "combined" / "training_data.json"
    publish_combined_training_data(source_inputs, path)
    return path


@pytest.fixture
def config():
    return edict({"trainer": {"args": {"multiview_stage": "ss64"}}})


def test_combined_manifest_has_exact_sources_and_proportional_counts(
    source_inputs,
):
    value = build_combined_training_data(source_inputs)
    stage = value["stages"]["ss64"]
    assert tuple(value["sources"]) == CANONICAL_SOURCES
    assert list(stage["data_dir"]) == ["ABO", "3D-FUTURE"]
    assert stage["source_counts"] == {"ABO": 2, "3D-FUTURE": 5}
    assert stage["total_count"] == 7
    assert "source_weights" not in stage
    assert "split" not in stage


def test_combined_manifest_rejects_cross_source_asset_overlap(source_inputs):
    abo_scope = json.loads(
        (
            Path(
                json.loads(source_inputs["ABO"].read_text())["stages"][
                    "shape1024"
                ]["root"]
            )
            / "materialization.json"
        ).read_text()
    )["stage_scope"]
    def overlap(evidence):
        evidence["stage_scope"][0] = abo_scope[0]
        evidence["stage_scope"].sort()
        evidence["stage_scope_sha256"] = _scope_digest(evidence["stage_scope"])

    _rewrite_materialization(
        source_inputs["3D-FUTURE"], "shape1024", overlap
    )
    changed = json.loads(source_inputs["3D-FUTURE"].read_text())
    changed_digest = json.loads(
        (
            Path(changed["stages"]["shape1024"]["root"])
            / "materialization.json"
        ).read_text()
    )["stage_scope_sha256"]
    _rewrite_source_chain(
        source_inputs["3D-FUTURE"],
        lambda handoff: handoff["stages"]["shape1024"].__setitem__(
            "asset_scope_sha256", changed_digest
        ),
    )
    with pytest.raises(ValueError, match="cross-source asset overlap"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_rejects_changed_handoff_digest(source_inputs):
    training = json.loads(source_inputs["ABO"].read_text())
    training["handoff"]["sha256"] = "0" * 64
    source_inputs["ABO"].write_bytes(_canonical_json_bytes(training))
    with pytest.raises(ValueError, match="handoff digest"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_rejects_report_digest_drift(source_inputs):
    training = json.loads(source_inputs["ABO"].read_text())
    handoff = json.loads(Path(training["handoff"]["path"]).read_text())
    report_path = Path(handoff["report"]["path"])
    report_path.write_bytes(report_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="report digest"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_rejects_handoff_not_projected_from_report(
    source_inputs,
):
    _rewrite_report_reference_only(
        source_inputs["3D-FUTURE"],
        lambda report: report["counts"]["stages"].__setitem__("ss64", 6),
    )

    with pytest.raises(ValueError, match="handoff.*report"):
        build_combined_training_data(source_inputs)


@pytest.mark.parametrize("unsafe", ("symlink", "noncanonical"))
def test_combined_manifest_rejects_unsafe_report_reference(
    source_inputs, tmp_path, unsafe
):
    training_path = source_inputs["ABO"]
    training = json.loads(training_path.read_text())
    handoff_path = Path(training["handoff"]["path"])
    handoff = json.loads(handoff_path.read_text())
    report_path = Path(handoff["report"]["path"])
    if unsafe == "symlink":
        target = tmp_path / "report-copy.json"
        target.write_bytes(report_path.read_bytes())
        report_path.unlink()
        report_path.symlink_to(target)
    else:
        handoff["report"]["path"] = str(
            report_path.parent / "nested" / ".." / report_path.name
        )
        handoff_path.write_bytes(_canonical_json_bytes(handoff))
        training = {
            **handoff,
            "handoff": {
                "path": str(handoff_path),
                "sha256": hashlib.sha256(
                    handoff_path.read_bytes()
                ).hexdigest(),
            },
        }
        training_path.write_bytes(_canonical_json_bytes(training))

    with pytest.raises(ValueError, match="report.*(regular|canonical)"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_rejects_symlinked_handoff(source_inputs, tmp_path):
    training = json.loads(source_inputs["ABO"].read_text())
    handoff_path = Path(training["handoff"]["path"])
    target = tmp_path / "handoff-copy.json"
    target.write_bytes(handoff_path.read_bytes())
    handoff_path.unlink()
    handoff_path.symlink_to(target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_rejects_missing_source_stage(source_inputs):
    _rewrite_source_chain(
        source_inputs["3D-FUTURE"],
        lambda handoff: handoff["stages"].pop("pbr1024"),
    )
    with pytest.raises(ValueError, match="all four stages"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_rejects_wrong_component_key(source_inputs):
    def mutate(handoff):
        values = handoff["stages"]["ss64"]["data_dir"]["ABO"]
        values["wrong_latent"] = values.pop("ss_latent")

    _rewrite_source_chain(source_inputs["ABO"], mutate)
    with pytest.raises(ValueError, match="component keys"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_rejects_noncanonical_stage_root(source_inputs):
    def mutate(handoff):
        root = Path(handoff["stages"]["shape512"]["root"])
        handoff["stages"]["shape512"]["root"] = str(
            root.parent / "nested" / ".." / root.name
        )

    _rewrite_source_chain(source_inputs["ABO"], mutate)
    with pytest.raises(ValueError, match="canonical"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_rejects_changed_materialized_scope(source_inputs):
    _rewrite_materialization(
        source_inputs["ABO"],
        "ss64",
        lambda evidence: evidence["stage_scope"].append("unbound-asset"),
    )
    with pytest.raises(ValueError, match="scope"):
        build_combined_training_data(source_inputs)


def test_combined_manifest_requires_canonical_source_set(source_inputs):
    with pytest.raises(ValueError, match="exactly.*ABO.*3D-FUTURE"):
        build_combined_training_data({"ABO": source_inputs["ABO"]})


def test_combined_manifest_requires_schema_one_abo_and_schema_two_future(
    source_inputs,
):
    _rewrite_source_chain(
        source_inputs["3D-FUTURE"],
        lambda handoff: handoff.__setitem__("schema_version", 1),
    )
    with pytest.raises(ValueError, match="schema_version"):
        build_combined_training_data(source_inputs)


def test_publish_combined_training_data_is_create_only_and_idempotent(
    source_inputs, tmp_path
):
    output = tmp_path / "combined" / "training_data.json"
    assert publish_combined_training_data(source_inputs, output) == output
    published = json.loads(output.read_text())
    assert published == build_combined_training_data(source_inputs)
    assert list(published["sources"]) == ["ABO", "3D-FUTURE"]
    assert list(published["stages"]["ss64"]["data_dir"]) == [
        "ABO",
        "3D-FUTURE",
    ]
    original_inode = output.stat().st_ino
    original_bytes = output.read_bytes()

    assert publish_combined_training_data(source_inputs, output) == output

    assert output.stat().st_ino == original_inode
    assert output.read_bytes() == original_bytes
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_publish_combined_training_data_preserves_different_existing_manifest(
    source_inputs, tmp_path
):
    output = tmp_path / "combined" / "training_data.json"
    output.parent.mkdir()
    output.write_text('{"stale":true}\n')
    original_inode = output.stat().st_ino
    original_bytes = output.read_bytes()

    with pytest.raises(ValueError, match="different"):
        publish_combined_training_data(source_inputs, output)

    assert output.stat().st_ino == original_inode
    assert output.read_bytes() == original_bytes


def test_publish_script_accepts_explicit_safe_paths(source_inputs, tmp_path):
    output = tmp_path / "script-output" / "training_data.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/publish_multisource_training.py",
            "--abo",
            str(source_inputs["ABO"]),
            "--3d-future",
            str(source_inputs["3D-FUTURE"]),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["stages"]["ss64"] == {
        "source_counts": {"ABO": 2, "3D-FUTURE": 5},
        "total_count": 7,
    }
    assert json.loads(output.read_text())["sampling"] == (
        "proportional-unweighted-concatenation"
    )


def test_resolve_training_data_returns_verified_stage(source_inputs, manifest):
    resolved = resolve_training_data(manifest, "shape1024")
    assert isinstance(resolved, ResolvedTrainingData)
    assert resolved.stage == "shape1024"
    assert list(resolved.data_dir) == ["ABO", "3D-FUTURE"]
    assert resolved.source_counts == {"ABO": 2, "3D-FUTURE": 5}
    assert resolved.total_count == 7
    assert resolved.source_scopes["ABO"][0] == "abo-shape1024-0"
    assert resolved.sampling == "proportional-unweighted-concatenation"


def test_resolve_training_data_rejects_source_changed_after_publication(
    source_inputs, manifest
):
    source_inputs["ABO"].write_bytes(source_inputs["ABO"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="training data digest"):
        resolve_training_data(manifest, "ss64")


def test_resolve_training_data_rechecks_report_digest_at_launch(
    source_inputs, manifest
):
    training = json.loads(source_inputs["3D-FUTURE"].read_text())
    handoff = json.loads(Path(training["handoff"]["path"]).read_text())
    report_path = Path(handoff["report"]["path"])
    report_path.write_bytes(report_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="report digest"):
        resolve_training_data(manifest, "ss64")


def test_resolve_training_data_rejects_symlinked_manifest(manifest, tmp_path):
    target = tmp_path / "manifest-copy.json"
    target.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        resolve_training_data(manifest, "ss64")


def test_resolve_training_data_rejects_missing_stage(manifest):
    value = json.loads(manifest.read_text())
    value["stages"].pop("shape512")
    manifest.write_bytes(_ordered_json_bytes(value))
    with pytest.raises(ValueError, match="all four stages"):
        resolve_training_data(manifest, "ss64")


def test_resolve_training_data_rejects_noncanonical_source_order(manifest):
    value = json.loads(manifest.read_text())
    value["sources"] = {
        "3D-FUTURE": value["sources"]["3D-FUTURE"],
        "ABO": value["sources"]["ABO"],
    }
    manifest.write_bytes(_ordered_json_bytes(value))
    with pytest.raises(ValueError, match="ABO then 3D-FUTURE"):
        resolve_training_data(manifest, "ss64")


def test_resolve_training_data_rejects_unknown_stage(manifest):
    with pytest.raises(ValueError, match="unknown stage"):
        resolve_training_data(manifest, "unknown")


def test_resolve_training_input_rejects_both_interfaces(tmp_path, config):
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_training_input(config, "{}", tmp_path / "training_data.json")


def test_resolve_training_input_uses_manifest_stage(manifest, config):
    data_dir, evidence = resolve_training_input(config, None, manifest)
    assert list(json.loads(data_dir)) == ["ABO", "3D-FUTURE"]
    assert evidence["stage"] == "ss64"
    assert evidence["source_counts"] == {"ABO": 2, "3D-FUTURE": 5}


@pytest.mark.parametrize(("config_path", "expected_stage"), MULTIVIEW_CONFIGS)
def test_resolve_training_input_uses_real_nested_config_stage_before_cuda(
    manifest, monkeypatch, config_path, expected_stage
):
    config = json.loads((REPO_ROOT / config_path).read_text())
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("CUDA queried before manifest resolution"),
    )
    _data_dir, evidence = resolve_training_input(config, None, manifest)
    assert evidence["stage"] == expected_stage
    assert evidence["stage"] == config["trainer"]["args"]["multiview_stage"]


def test_resolve_training_input_rejects_unknown_multiview_stage(
    manifest, config
):
    config.trainer.args.multiview_stage = "unknown"
    with pytest.raises(ValueError, match="unknown multiview_stage"):
        resolve_training_input(config, None, manifest)


@pytest.mark.parametrize(
    ("cli_data_dir", "configured_data_dir", "expected"),
    [
        ("/cli/data", "/config/data", "/cli/data"),
        (None, "/config/data", "/config/data"),
        (None, None, "./data/"),
    ],
)
def test_resolve_training_input_preserves_legacy_data_dir_behavior(
    cli_data_dir, configured_data_dir, expected
):
    config = {}
    if configured_data_dir is not None:
        config["data_dir"] = configured_data_dir
    assert resolve_training_input(config, cli_data_dir, None) == (
        expected,
        None,
    )
