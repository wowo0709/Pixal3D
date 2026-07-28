import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from data_toolkit.pipeline.training_eligibility import (
    policy_evidence,
)
from data_toolkit.pipeline.training_materialization import ProductionSourceSpec
from data_toolkit.pipeline import training_preflight
from data_toolkit.pipeline.training_preflight import (
    StagePreflight,
    build_source_report,
    preflight_stage,
    publish_source_handoff,
    stage_data_dir,
    validate_direct_loader,
    write_create_only_json,
)


STAGES = ("ss64", "shape512", "shape1024", "pbr1024")


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


@pytest.fixture
def two_index_preflight(tmp_path):
    indexes = []
    source_indexes = []
    for shard in ("Fixture-00000", "Fixture-00001"):
        path = tmp_path / "indexes" / f"{shard}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"source": "Fixture", "shard_id": shard}) + "\n")
        indexes.append(path)
        source_indexes.append(
            {
                "shard_id": shard,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    candidates = {stage: 2 for stage in STAGES}
    exclusions = {stage: int(stage == "shape512") for stage in STAGES}
    spec = ProductionSourceSpec(
        source="Fixture",
        indexes=tuple(indexes),
        expected_batches={
            "Fixture-00000": ("batch000",),
            "Fixture-00001": ("batch000",),
        },
        expected_frozen=2,
        expected_candidate_stages=candidates,
        fixed_count_contract=None,
        acceptance_mode="valid_subset_user_waiver",
        original_90_percent_gate_passed=False,
    )
    results = {}
    materializations = {}
    for stage in STAGES:
        root = tmp_path / "production" / stage / "active"
        root.mkdir(parents=True)
        exclusion_count = exclusions[stage]
        counts = {
            "frozen": 2,
            "global_quarantine": 0,
            "shape512_family_exclusions": 0,
            "candidate_stages": {stage: 2},
            "pack_exclusions": {stage: 0},
            "training_exclusions": {stage: exclusion_count},
            "stages": {stage: 2 - exclusion_count},
        }
        final_scope = [f"{stage}-asset-0"]
        if stage != "shape512":
            final_scope.append(f"{stage}-asset-1")
        candidate_scope = sorted(
            final_scope
            + ([f"{stage}-excluded"] if stage == "shape512" else [])
        )
        excluded = sorted(set(candidate_scope) - set(final_scope))
        evidence = {
            "schema_version": 1,
            "created_at": "2026-07-28T00:00:00Z",
            "source": "Fixture",
            "source_indexes": source_indexes,
            "acceptance_mode": "valid_subset_user_waiver",
            "original_90_percent_gate_passed": False,
            "counts": counts,
            "candidate_asset_count": len(candidate_scope),
            "candidate_stage_scope": candidate_scope,
            "candidate_stage_scope_sha256": hashlib.sha256(
                "\n".join(candidate_scope).encode()
            ).hexdigest(),
            "asset_count": len(final_scope),
            "packs": [
                {
                    "shard_id": source_index["shard_id"],
                    "tool_commit": f"{source_index['shard_id']}-tool",
                }
                for source_index in source_indexes
            ],
            "stage": stage,
            "stage_root": str(root.resolve()),
            "stage_scope": final_scope,
            "stage_scope_sha256": hashlib.sha256(
                "\n".join(final_scope).encode()
            ).hexdigest(),
            "training_exclusion_count": len(excluded),
            "training_exclusions": [
                {
                    "asset": asset,
                    "reasons": ["shape_tokens_view00_exceed_8192"],
                }
                for asset in excluded
            ],
            "training_exclusion_reason_counts": (
                {"shape_tokens_view00_exceed_8192": 1} if excluded else {}
            ),
            "frozen_assets": 2,
            "quarantined_assets": 0,
            "shape512_exclusions": 0,
            "eligibility_policy": policy_evidence(),
            "tool_commits": [f"{stage}-tool"],
        }
        raw = canonical_json_bytes(evidence)
        (root / "materialization.json").write_bytes(raw)
        materializations[stage] = evidence
        results[stage] = StagePreflight(
            source="Fixture",
            stage=stage,
            root=root,
            asset_count=len(final_scope),
            asset_scope_sha256=evidence["stage_scope_sha256"],
            anchors_checked=len(final_scope) * 2,
            validation_counts={"assets": len(final_scope)},
            materialization_bytes=raw,
        )
    return spec, results, materializations, "2026-07-28T00:00:00Z"


def test_stage_data_dir_preserves_exact_source_name(tmp_path):
    assert stage_data_dir("3D-FUTURE", "ss64", tmp_path) == {
        "3D-FUTURE": {
            "base": str(tmp_path),
            "render_cond": str(tmp_path / "renders_cond"),
            "ss_latent": str(
                tmp_path / "ss_latents/ss_enc_conv3d_16l8_fp16_64_view"
            ),
        }
    }


def test_source_handoff_binds_both_index_digests(two_index_preflight):
    report = build_source_report(*two_index_preflight)
    assert [entry["shard_id"] for entry in report["source_indexes"]] == [
        "Fixture-00000",
        "Fixture-00001",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in report["source_indexes"])


def test_source_indexes_follow_index_order_not_expected_batch_mapping_order(
    two_index_preflight,
):
    spec, results, materializations, created_at = two_index_preflight
    reordered = replace(
        spec,
        expected_batches={
            "Fixture-00001": ("batch000",),
            "Fixture-00000": ("batch000",),
        },
    )
    report = build_source_report(
        reordered, results, materializations, created_at
    )
    assert [
        (entry["shard_id"], Path(entry["path"]).name)
        for entry in report["source_indexes"]
    ] == [
        ("Fixture-00000", "Fixture-00000.json"),
        ("Fixture-00001", "Fixture-00001.json"),
    ]


def test_preflight_stage_preserves_source_identity(
    two_index_preflight, monkeypatch
):
    spec, existing_results, _materializations, _created_at = two_index_preflight
    expected = existing_results["ss64"]
    monkeypatch.setattr(
        training_preflight,
        "validate_stage_structure",
        lambda source, stage, root, assets: {
            "assets": len(assets),
            "source_checked": int(source == "Fixture"),
        },
    )
    monkeypatch.setattr(
        training_preflight,
        "validate_direct_loader",
        lambda source, stage, root, assets, config: len(assets) * 2,
    )
    result = preflight_stage(
        spec, "ss64", expected.root, Path("unused-config.json")
    )
    assert result.source == "Fixture"
    assert result.asset_scope_sha256 == expected.asset_scope_sha256
    assert result.validation_counts == {
        "assets": 2,
        "source_checked": 1,
    }


@pytest.mark.parametrize(
    ("stage", "expected_asset_count"),
    [
        ("ss64", 2),
        ("shape512", 1),
        ("shape1024", 2),
        ("pbr1024", 2),
    ],
)
def test_preflight_stage_accepts_stage_local_materializer_count_evidence(
    two_index_preflight, monkeypatch, stage, expected_asset_count
):
    spec, existing_results, _materializations, _created_at = (
        two_index_preflight
    )
    expected = existing_results[stage]
    monkeypatch.setattr(
        training_preflight,
        "validate_stage_structure",
        lambda source, received_stage, root, assets: {
            "assets": len(assets),
        },
    )
    monkeypatch.setattr(
        training_preflight,
        "validate_direct_loader",
        lambda source, received_stage, root, assets, config: len(assets) * 2,
    )

    result = preflight_stage(
        spec, stage, expected.root, Path("unused-config.json")
    )

    assert result.asset_count == expected_asset_count
    assert result.anchors_checked == expected_asset_count * 2


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing", r"source=Fixture.*stage=ss64"),
        (
            "invalid_observed_count",
            r"source=Fixture.*training exclusion exceeds candidate count",
        ),
    ],
)
def test_preflight_stage_count_failures_include_source_context(
    two_index_preflight, mutation, message
):
    spec, results, _materializations, _created_at = two_index_preflight
    result = results["ss64"]
    evidence = json.loads(result.materialization_bytes)
    if mutation == "missing":
        evidence["counts"].pop("training_exclusions")
    else:
        evidence["counts"]["training_exclusions"]["ss64"] = 3
    (result.root / "materialization.json").write_bytes(
        canonical_json_bytes(evidence)
    )
    with pytest.raises(ValueError, match=message):
        preflight_stage(
            spec, "ss64", result.root, Path("unused-config.json")
        )


def test_observed_source_handoff_uses_materialization_counts(two_index_preflight):
    report = build_source_report(*two_index_preflight)
    assert report["counts"]["candidate_stages"]["shape512"] == 2
    assert report["counts"]["training_exclusions"]["shape512"] == 1
    assert report["counts"]["stages"]["shape512"] == 1
    assert report["schema_version"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "materialization_bytes",
        "missing_stage",
        "policy",
        "reordered_scope",
    ],
)
def test_source_report_fails_closed_on_unbound_evidence(
    two_index_preflight, mutation
):
    spec, results, materializations, created_at = two_index_preflight
    if mutation == "source":
        materializations["ss64"]["source"] = "Other"
    elif mutation == "materialization_bytes":
        results["ss64"] = replace(
            results["ss64"],
            materialization_bytes=results["ss64"].materialization_bytes + b" ",
        )
    elif mutation == "missing_stage":
        results.pop("pbr1024")
    elif mutation == "policy":
        materializations["ss64"]["eligibility_policy"]["schema_version"] = 2
    else:
        materializations["ss64"]["stage_scope"] = list(
            reversed(materializations["ss64"]["stage_scope"])
        )
    with pytest.raises(ValueError):
        build_source_report(spec, results, materializations, created_at)


def test_source_report_rejects_index_bytes_changed_after_materialization(
    two_index_preflight,
):
    spec, results, materializations, created_at = two_index_preflight
    spec.indexes[1].write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="index"):
        build_source_report(spec, results, materializations, created_at)


def test_source_report_rejects_noncanonical_materialization_index_path(
    two_index_preflight,
):
    spec, results, materializations, created_at = two_index_preflight
    materializations["ss64"]["source_indexes"][0]["path"] = str(
        spec.indexes[0].parent / "nested" / ".." / spec.indexes[0].name
    )
    with pytest.raises(ValueError, match="index"):
        build_source_report(spec, results, materializations, created_at)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("index", r"source=Fixture.*stage=ss64"),
        ("evidence", r"source=Fixture.*stage=ss64"),
        (
            "changed_materialization",
            r"source=Fixture.*materialization evidence bytes changed",
        ),
    ],
)
def test_source_publication_validation_failures_include_source_context(
    two_index_preflight, mutation, message
):
    spec, results, materializations, created_at = two_index_preflight
    if mutation == "index":
        spec.indexes[1].write_text('{"changed":true}\n')
    elif mutation == "evidence":
        materializations["ss64"]["stage"] = "other"
    else:
        path = results["ss64"].root / "materialization.json"
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=message):
        build_source_report(
            spec, results, materializations, created_at
        )


def test_source_report_rejects_invalid_source_policy(two_index_preflight):
    spec, results, materializations, created_at = two_index_preflight
    invalid = replace(spec, acceptance_mode="unapproved")
    with pytest.raises(ValueError, match="acceptance"):
        build_source_report(invalid, results, materializations, created_at)


def test_create_only_publication_rejects_different_existing_content(tmp_path):
    path = tmp_path / "report.json"
    write_create_only_json(path, {"schema_version": 2})
    with pytest.raises(FileExistsError, match="different"):
        write_create_only_json(path, {"schema_version": 3})


def test_source_handoff_publication_cross_links_schema_two_artifacts(
    two_index_preflight, tmp_path
):
    spec, results, materializations, _created_at = two_index_preflight
    report_path = tmp_path / "shared" / "report.json"
    handoff_path = tmp_path / "shared" / "handoff.json"
    training_path = tmp_path / "local" / "training_data.json"
    assert publish_source_handoff(
        spec,
        results,
        report_path,
        handoff_path,
        training_path,
    ) == (report_path, handoff_path, training_path)
    report = json.loads(report_path.read_text())
    handoff = json.loads(handoff_path.read_text())
    training = json.loads(training_path.read_text())
    assert report["schema_version"] == handoff["schema_version"] == 2
    assert training["schema_version"] == 2
    assert training["source_indexes"] == report["source_indexes"]
    assert handoff["report"]["sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()


def test_direct_loader_rejects_instances_with_wrong_source_name(
    tmp_path, monkeypatch
):
    class WrongSourceDataset:
        def __init__(self, _roots, **_kwargs):
            self.instances = [({"base": str(tmp_path)}, "asset", "Wrong")]

    from pixal3d import datasets

    monkeypatch.setattr(datasets, "WrongSourceDataset", WrongSourceDataset, raising=False)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "name": "WrongSourceDataset",
                    "args": {"image_size": 512},
                }
            }
        )
    )
    with pytest.raises(
        ValueError,
        match=r"source=3D-FUTURE.*instance set",
    ):
        validate_direct_loader(
            "3D-FUTURE", "ss64", tmp_path, ["asset"], config
        )
