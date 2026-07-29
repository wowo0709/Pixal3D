import hashlib
import json
from pathlib import Path
import subprocess
import sys

from easydict import EasyDict as edict
import pytest
import torch

import data_toolkit.pipeline.training_manifest as training_manifest
from scripts import preflight_multiview_production as production_preflight
from data_toolkit.pipeline.training_manifest import (
    CANONICAL_SOURCES,
    STAGES,
    ResolvedTrainingData,
)
from data_toolkit.pipeline.training_source_profiles import build_source_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
build_combined_training_data = (
    training_manifest._build_combined_training_data_for_fixture
)
publish_combined_training_data = (
    training_manifest._publish_combined_training_data_for_fixture
)
resolve_training_data = training_manifest._resolve_training_data_for_fixture
resolve_training_input = (
    training_manifest._resolve_training_input_for_fixture
)
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


def _write_source(
    tmp_path,
    source,
    schema_version,
    count,
    *,
    acceptance_mode=None,
    original_90_percent_gate_passed=None,
):
    if acceptance_mode is None:
        acceptance_mode = (
            "production_gate"
            if source == "HSSD"
            else "valid_subset_user_waiver"
        )
    if original_90_percent_gate_passed is None:
        original_90_percent_gate_passed = source == "HSSD"
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
        "acceptance_mode": acceptance_mode,
        "original_90_percent_gate_passed":
            original_90_percent_gate_passed,
        "authorization": "training-input use only",
        "counts": {"stages": {stage: count for stage in STAGES}},
        "eligibility_policy": {"schema_version": 1},
        "stages": stages,
        "materialization_evidence": materialization_evidence,
        "observed_tool_commits": [f"{source}-tool"],
    }
    if schema_version == 1:
        index_path = tmp_path / source / "ABO-00000.json"
        index_path.write_bytes(
            _canonical_json_bytes(
                {"shard_id": "ABO-00000", "assets": ["abo-asset"]}
            )
        )
        report["shard_id"] = "ABO-00000"
        report["source_index"] = {
            "path": str(index_path),
            "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        }
    else:
        index_path = tmp_path / source / f"{source}-00000.json"
        index_path.write_bytes(
            _canonical_json_bytes(
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
                "sha256": hashlib.sha256(
                    index_path.read_bytes()
                ).hexdigest(),
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


def _source_chain_paths(training_path):
    training = json.loads(training_path.read_text())
    handoff_path = Path(training["handoff"]["path"])
    handoff = json.loads(handoff_path.read_text())
    report_path = Path(handoff["report"]["path"])
    return report_path, handoff_path, training_path


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


def _rewrite_acceptance_gate_layer(
    training_path, layer, integer_gate
):
    training = json.loads(training_path.read_text())
    handoff_path = Path(training["handoff"]["path"])
    handoff = json.loads(handoff_path.read_text())
    report_path = Path(handoff["report"]["path"])
    if layer == "report":
        report = json.loads(report_path.read_text())
        report["original_90_percent_gate_passed"] = integer_gate
        report_path.write_bytes(_canonical_json_bytes(report))
        handoff["report"]["sha256"] = hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest()
        training["report"]["sha256"] = handoff["report"]["sha256"]
    elif layer == "handoff":
        handoff["original_90_percent_gate_passed"] = integer_gate
    else:
        training["original_90_percent_gate_passed"] = integer_gate
        training_path.write_bytes(_canonical_json_bytes(training))
        return
    handoff_path.write_bytes(_canonical_json_bytes(handoff))
    training["handoff"]["sha256"] = hashlib.sha256(
        handoff_path.read_bytes()
    ).hexdigest()
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
def hssd_training_data(tmp_path):
    return _write_source(
        tmp_path, "HSSD", schema_version=2, count=3
    )


@pytest.fixture
def manifest(source_inputs, tmp_path):
    path = tmp_path / "combined" / "training_data.json"
    publish_combined_training_data(source_inputs, path)
    return path


@pytest.fixture
def config():
    return edict({"trainer": {"args": {"multiview_stage": "ss64"}}})


@pytest.mark.parametrize(
    ("source", "profile"),
    (("ABO", "abo"), ("3D-FUTURE", "3d-future")),
)
@pytest.mark.parametrize(
    "tamper",
    (
        "report-bytes",
        "semantic-count",
        "source-index-bytes",
        "selected-report-path",
    ),
)
def test_verify_existing_cli_rejects_tampered_source_chain(
    source_inputs, monkeypatch, source, profile, tamper
):
    monkeypatch.setattr(
        production_preflight,
        "validate_source_training_data",
        training_manifest._validate_source_training_data_for_fixture,
    )
    training_path = source_inputs[source]
    report_path, handoff_path, _training_path = _source_chain_paths(
        training_path
    )
    if tamper == "report-bytes":
        report_path.write_bytes(report_path.read_bytes() + b" ")
        expected = "report digest"
    elif tamper == "semantic-count":

        def change_count(report):
            report["counts"]["stages"]["ss64"] += 1

        _rewrite_source_chain(training_path, change_count)
        expected = "count evidence"
    elif tamper == "source-index-bytes":
        training = json.loads(training_path.read_text())
        reference = (
            training["source_index"]
            if source == "ABO"
            else training["source_indexes"][0]
        )
        index_path = Path(reference["path"])
        index_path.write_bytes(index_path.read_bytes() + b" ")
        expected = "source index digest"
    else:
        selected_report = report_path.with_name("selected-report.json")
        selected_report.write_bytes(report_path.read_bytes())
        report_path = selected_report
        expected = "report path"

    with pytest.raises(ValueError, match=expected):
        production_preflight._verify_existing(
            source,
            {
                "report": report_path,
                "handoff": handoff_path,
                "training-data": training_path,
            },
        )


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


def test_hssd_source_training_data_resolves_as_one_source(
    hssd_training_data,
):
    resolved = resolve_training_data(hssd_training_data, "ss64")
    assert tuple(resolved.data_dir) == ("HSSD",)
    assert resolved.source_counts == {"HSSD": 3}
    assert resolved.total_count == 3
    assert resolved.sampling == "proportional-unweighted-concatenation"


@pytest.mark.parametrize("source", ("ABO", "3D-FUTURE", "HSSD"))
def test_public_launch_rejects_self_consistent_noncanonical_index_identity(
    source_inputs, hssd_training_data, source
):
    """A repinned fixture index must not impersonate a production source."""
    training_path = (
        hssd_training_data
        if source == "HSSD"
        else source_inputs[source]
    )

    with pytest.raises(ValueError, match="canonical production profile"):
        training_manifest.resolve_training_data(training_path, "ss64")


def _pin_canonical_hssd_indexes(
    training_data, tmp_path, *, omit_last_batch=False
):
    spec = build_source_spec("hssd", tmp_path / "shared")

    def canonical_indexes(report):
        records = []
        for path in spec.indexes:
            shard = path.stem
            path.parent.mkdir(parents=True, exist_ok=True)
            batches = {
                batch: {}
                for batch in spec.expected_batches[shard]
            }
            if omit_last_batch and shard == "HSSD-00001":
                batches.pop(spec.expected_batches[shard][-1])
            path.write_bytes(
                _canonical_json_bytes(
                    {
                        "source": "HSSD",
                        "shard_id": shard,
                        "batches": batches,
                    }
                )
            )
            records.append(
                {
                    "shard_id": shard,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        report["source_indexes"] = records

    _rewrite_source_chain(training_data, canonical_indexes)


def test_public_launch_rejects_noncanonical_hssd_source_counts(
    hssd_training_data, tmp_path
):
    """Canonical index names cannot authorize a three-asset HSSD profile."""
    _pin_canonical_hssd_indexes(hssd_training_data, tmp_path)

    with pytest.raises(ValueError, match="source count contract"):
        training_manifest.resolve_training_data(
            hssd_training_data, "ss64"
        )


def test_public_launch_rejects_noncanonical_index_batch_set(
    hssd_training_data, tmp_path
):
    """Canonical paths cannot conceal a missing approved source batch."""
    _pin_canonical_hssd_indexes(
        hssd_training_data, tmp_path, omit_last_batch=True
    )

    with pytest.raises(ValueError, match="batch set"):
        training_manifest.resolve_training_data(
            hssd_training_data, "ss64"
        )


@pytest.mark.parametrize("source", ("ABO", "3D-FUTURE"))
def test_legacy_waiver_source_training_data_remains_valid(
    source_inputs, source
):
    resolved = resolve_training_data(source_inputs[source], "ss64")
    assert tuple(resolved.data_dir) == (source,)
    assert resolved.source_counts[source] in (2, 5)


def test_hssd_self_consistent_waiver_chain_is_rejected(
    hssd_training_data,
):
    _rewrite_source_chain(
        hssd_training_data,
        lambda report: report.update(
            {
                "acceptance_mode": "valid_subset_user_waiver",
                "original_90_percent_gate_passed": False,
            }
        ),
    )
    with pytest.raises(ValueError, match="acceptance contract"):
        resolve_training_data(hssd_training_data, "ss64")


def test_hssd_materialization_waiver_is_rejected(
    hssd_training_data,
):
    def add_waiver(report):
        root = Path(report["stages"]["ss64"]["root"])
        path = root / "materialization.json"
        evidence = json.loads(path.read_text())
        evidence["waiver"] = "production-valid-subset"
        path.write_bytes(_canonical_json_bytes(evidence))
        report["materialization_evidence"]["ss64"]["sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )

    _rewrite_source_chain(hssd_training_data, add_waiver)

    with pytest.raises(ValueError, match="waiver"):
        resolve_training_data(hssd_training_data, "ss64")


@pytest.mark.parametrize(
    ("source", "integer_gate"),
    (("ABO", 0), ("3D-FUTURE", 0), ("HSSD", 1)),
)
def test_source_acceptance_contract_rejects_integer_gate(
    source_inputs,
    hssd_training_data,
    source,
    integer_gate,
):
    training_path = (
        hssd_training_data
        if source == "HSSD"
        else source_inputs[source]
    )
    _rewrite_source_chain(
        training_path,
        lambda report: report.__setitem__(
            "original_90_percent_gate_passed", integer_gate
        ),
    )
    with pytest.raises(ValueError, match="acceptance contract"):
        resolve_training_data(training_path, "ss64")


@pytest.mark.parametrize(
    ("source", "integer_gate"),
    (("ABO", 0), ("3D-FUTURE", 0), ("HSSD", 1)),
)
@pytest.mark.parametrize("layer", ("report", "handoff", "training_data"))
def test_source_acceptance_contract_rejects_integer_at_each_layer(
    source_inputs,
    hssd_training_data,
    source,
    integer_gate,
    layer,
):
    training_path = (
        hssd_training_data
        if source == "HSSD"
        else source_inputs[source]
    )
    _rewrite_acceptance_gate_layer(
        training_path, layer, integer_gate
    )
    with pytest.raises(ValueError, match="acceptance contract"):
        resolve_training_data(training_path, "ss64")


@pytest.mark.parametrize(
    ("source", "mode", "integer_gate"),
    (
        ("ABO", "valid_subset_user_waiver", 0),
        ("3D-FUTURE", "valid_subset_user_waiver", 0),
        ("HSSD", "production_gate", 1),
    ),
)
def test_production_materialization_acceptance_rejects_integer_gate(
    source_inputs,
    hssd_training_data,
    source,
    mode,
    integer_gate,
):
    training_path = (
        hssd_training_data
        if source == "HSSD"
        else source_inputs[source]
    )
    training = json.loads(training_path.read_text())
    stage = training["stages"]["ss64"]
    root = Path(stage["root"])
    path = root / "materialization.json"
    evidence = json.loads(path.read_text())
    evidence.update(
        {
            "acceptance_mode": mode,
            "original_90_percent_gate_passed": integer_gate,
        }
    )
    if source != "HSSD":
        evidence["waiver"] = "production-valid-subset"
    path.write_bytes(_canonical_json_bytes(evidence))

    with pytest.raises(ValueError, match="acceptance contract"):
        training_manifest._validate_materialization(
            source,
            "ss64",
            root,
            stage["asset_count"],
            stage["asset_scope_sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
            require_production_profile=True,
        )


def test_three_source_bundle_preserves_canonical_order(
    source_inputs, tmp_path
):
    source_inputs["HSSD"] = _write_source(
        tmp_path, "HSSD", schema_version=2, count=3
    )
    value = build_combined_training_data(source_inputs)
    assert list(value["sources"]) == ["ABO", "3D-FUTURE", "HSSD"]
    assert value["stages"]["ss64"]["source_counts"] == {
        "ABO": 2,
        "3D-FUTURE": 5,
        "HSSD": 3,
    }
    assert value["stages"]["ss64"]["total_count"] == 10


def test_three_source_pairwise_overlap_is_rejected(
    source_inputs, tmp_path
):
    hssd_path = _write_source(
        tmp_path, "HSSD", schema_version=2, count=3
    )
    abo = json.loads(source_inputs["ABO"].read_text())
    abo_scope = json.loads(
        (
            Path(abo["stages"]["ss64"]["root"])
            / "materialization.json"
        ).read_text()
    )["stage_scope"]

    def overlap(evidence):
        evidence["stage_scope"][0] = abo_scope[0]
        evidence["stage_scope"].sort()
        evidence["stage_scope_sha256"] = _scope_digest(
            evidence["stage_scope"]
        )

    _rewrite_materialization(hssd_path, "ss64", overlap)
    changed = json.loads(hssd_path.read_text())
    changed_digest = json.loads(
        (
            Path(changed["stages"]["ss64"]["root"])
            / "materialization.json"
        ).read_text()
    )["stage_scope_sha256"]
    _rewrite_source_chain(
        hssd_path,
        lambda handoff: handoff["stages"]["ss64"].__setitem__(
            "asset_scope_sha256", changed_digest
        ),
    )
    source_inputs["HSSD"] = hssd_path

    with pytest.raises(ValueError, match="cross-source asset overlap"):
        build_combined_training_data(source_inputs)


def test_existing_two_source_bundle_remains_valid(source_inputs):
    value = build_combined_training_data(
        {
            "ABO": source_inputs["ABO"],
            "3D-FUTURE": source_inputs["3D-FUTURE"],
        }
    )
    assert list(value["sources"]) == ["ABO", "3D-FUTURE"]


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
    with pytest.raises(ValueError, match="source order must be one of"):
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


def test_publish_combined_training_data_preserves_concurrent_different_creator(
    source_inputs, tmp_path, monkeypatch
):
    output = tmp_path / "combined" / "training_data.json"
    racer_bytes = b'{"racer":true}\n'
    racer_inode = None

    def create_racer_then_fail_link(_temporary, destination):
        nonlocal racer_inode
        destination = Path(destination)
        destination.write_bytes(racer_bytes)
        racer_inode = destination.stat().st_ino
        raise FileExistsError("synthetic concurrent creator")

    monkeypatch.setattr(
        training_manifest.os,
        "link",
        create_racer_then_fail_link,
    )

    with pytest.raises(ValueError, match="different"):
        publish_combined_training_data(source_inputs, output)

    assert output.stat().st_ino == racer_inode
    assert output.read_bytes() == racer_bytes
    assert not list(output.parent.glob(f".{output.name}.*"))


@pytest.mark.parametrize("unsafe", ("symlink", "directory"))
def test_publish_combined_training_data_preserves_unsafe_existing_node(
    source_inputs, tmp_path, unsafe
):
    output = tmp_path / "combined" / "training_data.json"
    output.parent.mkdir()
    if unsafe == "symlink":
        target = tmp_path / "target.json"
        target.write_bytes(
            _ordered_json_bytes(
                build_combined_training_data(source_inputs)
            )
        )
        target_inode = target.stat().st_ino
        target_bytes = target.read_bytes()
        output.symlink_to(target)
    else:
        output.mkdir()
        (output / "sentinel").write_text("keep")
        output_inode = output.stat().st_ino

    with pytest.raises(ValueError, match="regular non-symlink"):
        publish_combined_training_data(source_inputs, output)

    if unsafe == "symlink":
        assert output.is_symlink()
        assert target.stat().st_ino == target_inode
        assert target.read_bytes() == target_bytes
    else:
        assert output.is_dir() and not output.is_symlink()
        assert output.stat().st_ino == output_inode
        assert (output / "sentinel").read_text() == "keep"


def test_publish_script_rejects_noncanonical_fixture_profiles(
    source_inputs, tmp_path
):
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
    assert result.returncode != 0
    assert "canonical production profile" in result.stderr
    assert not output.exists()


def test_publish_script_hssd_rejects_noncanonical_fixture_profiles(
    source_inputs, hssd_training_data, tmp_path
):
    output = tmp_path / "script-output" / "training_data.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/publish_multisource_training.py",
            "--abo",
            str(source_inputs["ABO"]),
            "--3d-future",
            str(source_inputs["3D-FUTURE"]),
            "--hssd",
            str(hssd_training_data),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "canonical production profile" in completed.stderr
    assert not output.exists()


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


def test_resolve_training_data_does_not_fallback_after_combined_validation(
    manifest,
):
    value = json.loads(manifest.read_text())
    value["sources"] = {}
    manifest.write_bytes(_ordered_json_bytes(value))
    with pytest.raises(
        ValueError, match="combined training manifest sources"
    ):
        resolve_training_data(manifest, "ss64")


def test_resolve_training_data_requires_exact_source_schema_keys(
    hssd_training_data,
):
    value = json.loads(hssd_training_data.read_text())
    value["unexpected"] = True
    hssd_training_data.write_bytes(_ordered_json_bytes(value))
    with pytest.raises(ValueError, match="unrecognized training_data schema"):
        resolve_training_data(hssd_training_data, "ss64")


def test_resolve_training_data_rejects_noncanonical_source_order(manifest):
    value = json.loads(manifest.read_text())
    value["sources"] = {
        "3D-FUTURE": value["sources"]["3D-FUTURE"],
        "ABO": value["sources"]["ABO"],
    }
    manifest.write_bytes(_ordered_json_bytes(value))
    with pytest.raises(ValueError, match="source order must be one of"):
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
