import csv
import io
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from data_toolkit.pipeline.training_eligibility import (
    SCALE_ATOL,
    SCALE_RTOL,
    EligibilityExclusion,
    evaluate_stage_asset,
    filter_stage_scope,
    policy_evidence,
)
from data_toolkit.pipeline.training_materialization import (
    ProductionSourceSpec,
)
from data_toolkit.pipeline.training_source_profiles import build_source_spec
import data_toolkit.pipeline.training_materialization as training_materializer
import scripts.materialize_multiview_production as materializer
import scripts.preflight_multiview_production as strict_preflight
from scripts.materialize_multiview_production import (
    FamilyPack,
    compute_stage_scopes,
    load_production_catalog,
)


FAMILIES = (
    "common", "SS-64", "shape-256", "shape-512", "shape-1024",
    "PBR-256", "PBR-512", "PBR-1024",
)
DIRS = {
    "common": "common", "SS-64": "ss/64", "shape-256": "shape/256",
    "shape-512": "shape/512", "shape-1024": "shape/1024",
    "PBR-256": "pbr/256", "PBR-512": "pbr/512", "PBR-1024": "pbr/1024",
}
ASSET_A = "a" * 64
ASSET_B = "b" * 64
FIXTURE_WAIVER = {
    "frozen_assets": 2,
    "quarantined_assets": 0,
    "shape512_exclusions": 1,
}
FIXTURE_STAGE_COUNTS = {
    "ss64": 2,
    "shape512": 1,
    "shape1024": 2,
    "pbr1024": 2,
}
FIXTURE_TRAINING_EXCLUSION_COUNTS = {
    "ss64": 0,
    "shape512": 0,
    "shape1024": 0,
    "pbr1024": 0,
}
load_source_catalog = training_materializer._load_source_catalog
materialize_stage = materializer._materialize_stage_for_fixture


def _materialize_fixture_source_stage(
    spec, stage, catalog, output_root
):
    training_materializer._validate_catalog_identity(spec, catalog)
    return training_materializer._materialize_stage(
        stage,
        catalog,
        output_root,
        index_path=spec.indexes[0],
        spec=spec,
        expected_counts=spec.expected_candidate_stages,
    )


@pytest.mark.parametrize(
    ("profile", "integer_gate"),
    (("abo", 0), ("3d-future", 0), ("hssd", 1)),
)
def test_materializer_spec_validator_rejects_integer_booleans(
    profile, integer_gate
):
    spec = replace(
        build_source_spec(profile, Path("/file2/youngwoo/pixal3d")),
        original_90_percent_gate_passed=integer_gate,
    )

    with pytest.raises(ValueError, match="canonical production profile"):
        training_materializer._validate_source_spec(spec)


_ELIGIBILITY_ROOTS = {
    "shape512": ("shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",),
    "shape1024": ("shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",),
    "pbr1024": (
        "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
        "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
    ),
}


def _npz_payload(count, *, coords=None, feats=None):
    coords = np.zeros((count, 3), dtype=np.float32) if coords is None else coords
    feats = np.zeros((count, 32), dtype=np.float32) if feats is None else feats
    stream = io.BytesIO()
    np.savez(stream, coords=coords, feats=feats)
    return stream.getvalue()


def write_eligibility_stage(
    tmp_path,
    stage,
    *,
    asset="e" * 64,
    shape_counts=(1, 1),
    pbr_counts=None,
    shape_coords=None,
    pbr_coords=None,
    shape_scales=(0.5, 0.5),
    pbr_scales=None,
):
    """Build real extracted latent files for one policy input asset."""
    root = tmp_path / "stage"
    pbr_counts = shape_counts if pbr_counts is None else pbr_counts
    pbr_scales = shape_scales if pbr_scales is None else pbr_scales
    for family_root in _ELIGIBILITY_ROOTS[stage]:
        counts = pbr_counts if family_root.startswith("pbr_latents") else shape_counts
        coords = pbr_coords if family_root.startswith("pbr_latents") else shape_coords
        scales = pbr_scales if family_root.startswith("pbr_latents") else shape_scales
        for anchor, count in enumerate(counts):
            anchor_coords = None if coords is None else coords[anchor]
            (root / family_root / asset).mkdir(parents=True, exist_ok=True)
            (root / family_root / asset / f"view{anchor:02d}.npz").write_bytes(
                _npz_payload(count, coords=anchor_coords)
            )
            (root / family_root / asset / f"view{anchor:02d}_scale.json").write_text(
                json.dumps({"total_scale": scales[anchor]})
            )
    return root, asset


def test_training_eligibility_uses_both_anchor_token_limits(tmp_path):
    """Dropping the second Shape-512 anchor token check must reject this asset."""
    root, asset = write_eligibility_stage(
        tmp_path, "shape512", shape_counts=(8192, 8193)
    )
    assert evaluate_stage_asset("shape512", root, asset) == (
        "shape_tokens_view01_exceed_8192",
    )


@pytest.mark.parametrize(
    ("stage", "counts", "expected"),
    [
        ("shape512", (8192, 8192), ()),
        ("shape512", (8193, 8192), ("shape_tokens_view00_exceed_8192",)),
        ("shape1024", (32768, 32768), ()),
        ("shape1024", (32768, 32769), ("shape_tokens_view01_exceed_32768",)),
    ],
)
def test_training_eligibility_enforces_exact_shape_token_boundaries(tmp_path, stage, counts, expected):
    """Changing either configured Shape token boundary must alter these real NPZ outcomes."""
    root, asset = write_eligibility_stage(tmp_path, stage, shape_counts=counts)
    assert evaluate_stage_asset(stage, root, asset) == expected


def test_training_eligibility_applies_pbr1024_limit_to_shape_and_pbr_anchors(tmp_path):
    """Omitting either PBR-stage latent family from token inspection must fail this result."""
    root, asset = write_eligibility_stage(
        tmp_path, "pbr1024", shape_counts=(32769, 32768), pbr_counts=(32768, 32769)
    )
    assert evaluate_stage_asset("pbr1024", root, asset) == (
        "pbr_shape_coords_view00_mismatch",
        "pbr_shape_coords_view01_mismatch",
        "pbr_tokens_view01_exceed_32768",
        "shape_tokens_view00_exceed_32768",
    )


def test_training_eligibility_requires_exact_pbr_shape_coordinates(tmp_path):
    """Comparing only coordinate shapes must reject value-mismatched PBR anchors."""
    shape_coords = (np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32))
    pbr_coords = (np.ones((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32))
    root, asset = write_eligibility_stage(
        tmp_path, "pbr1024", shape_counts=(2, 2), shape_coords=shape_coords, pbr_coords=pbr_coords
    )
    assert evaluate_stage_asset("pbr1024", root, asset) == (
        "pbr_shape_coords_view00_mismatch",
    )


def test_training_eligibility_accepts_scale_drift_at_float32_tolerance(tmp_path):
    """Tightening approved float32 scale tolerance must reject this PBR anchor pair."""
    root, asset = write_eligibility_stage(
        tmp_path, "pbr1024", shape_scales=(0.5, 0.5), pbr_scales=(0.5000002, 0.5)
    )
    assert evaluate_stage_asset("pbr1024", root, asset) == ()


def test_training_eligibility_rejects_scale_drift_above_float32_tolerance(tmp_path):
    """Ignoring PBR scale mismatches must fail to exclude this above-tolerance anchor."""
    root, asset = write_eligibility_stage(
        tmp_path, "pbr1024", shape_scales=(0.5, 0.5), pbr_scales=(0.5000003, 0.5)
    )
    assert evaluate_stage_asset("pbr1024", root, asset) == (
        "pbr_shape_scale_view00_mismatch",
    )


@pytest.mark.parametrize(
    "scale_document",
    [
        0.5,
        {},
        {"total_scale": "0.5"},
        {"total_scale": [0.5]},
        {"total_scale": True},
        {"total_scale": float("nan")},
        {"total_scale": 0},
        {"total_scale": -1},
    ],
)
def test_training_eligibility_rejects_invalid_total_scale_schema_with_asset_context(
    tmp_path, scale_document
):
    """Permitting malformed, non-finite, or non-positive scales must stop policy evaluation."""
    root, asset = write_eligibility_stage(tmp_path, "pbr1024")
    scale_path = root / _ELIGIBILITY_ROOTS["pbr1024"][0] / asset / "view00_scale.json"
    scale_path.write_text(json.dumps(scale_document))
    with pytest.raises(ValueError, match=asset):
        evaluate_stage_asset("pbr1024", root, asset)


@pytest.mark.parametrize(
    ("ulp_count", "expected"),
    [(3, ()), (4, ("pbr_shape_scale_view00_mismatch",))],
)
def test_training_eligibility_uses_exact_float32_ulp_tolerance_boundary(
    tmp_path, ulp_count, expected
):
    """Changing float32 conversion or the 3-ULP policy boundary must alter this result."""
    base = np.float32(0.5)
    shifted = base
    for _ in range(ulp_count):
        shifted = np.nextafter(shifted, np.float32(np.inf), dtype=np.float32)
    assert shifted.view(np.uint32) - base.view(np.uint32) == ulp_count
    expected_difference = (
        np.float32(1.7881393432617188e-7)
        if ulp_count == 3
        else np.float32(2.384185791015625e-7)
    )
    assert shifted - base == expected_difference
    root, asset = write_eligibility_stage(
        tmp_path,
        "pbr1024",
        shape_scales=(float(base), float(base)),
        pbr_scales=(float(shifted), float(base)),
    )
    assert evaluate_stage_asset("pbr1024", root, asset) == expected


def test_training_eligibility_filters_in_canonical_order_with_sorted_unique_reasons(tmp_path):
    """Unsorted candidates or duplicate reasons must not produce nondeterministic exclusion evidence."""
    root, rejected = write_eligibility_stage(
        tmp_path, "pbr1024", shape_counts=(32769, 32768), pbr_counts=(32768, 32769),
        shape_scales=(0.5, 0.5), pbr_scales=(0.5000003, 0.5),
    )
    valid = "d" * 64
    write_eligibility_stage(tmp_path, "pbr1024", asset=valid)
    final, exclusions = filter_stage_scope("pbr1024", root, (rejected, valid, rejected))
    assert final == (valid,)
    assert exclusions == (EligibilityExclusion(rejected, (
        "pbr_shape_coords_view00_mismatch",
        "pbr_shape_coords_view01_mismatch",
        "pbr_shape_scale_view00_mismatch",
        "pbr_tokens_view01_exceed_32768",
        "shape_tokens_view00_exceed_32768",
    )),)
    assert policy_evidence() == {
        "schema_version": 1,
        "token_limits": {"shape512": 8192, "shape1024": 32768, "pbr1024": 32768},
        "pbr_shape_coordinates": "exact",
        "pbr_shape_scale": {"dtype": "float32", "rtol": SCALE_RTOL, "atol": SCALE_ATOL},
    }


def _members(family, assets, latent_specs=None):
    values = {}
    for asset in assets:
        if family == "common":
            for frame in range(8):
                values[f"renders_cond/{asset}/{frame:03d}.png"] = f"{asset}-{frame}".encode()
            values[f"renders_cond/{asset}/transforms.json"] = b"{}"
        else:
            root = {
                "SS-64": "ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
                "shape-256": "shape_latents/shape_enc_next_dc_f16c32_fp16_256_view",
                "shape-512": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
                "shape-1024": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
                "PBR-256": "pbr_latents/tex_enc_next_dc_f16c32_fp16_256_view_fix",
                "PBR-512": "pbr_latents/tex_enc_next_dc_f16c32_fp16_512_view_fix",
                "PBR-1024": "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
            }[family]
            for anchor in range(2):
                spec = (latent_specs or {}).get((family, asset, anchor), {})
                values[f"{root}/{asset}/view{anchor:02d}.npz"] = _npz_payload(
                    spec.get("count", 1), coords=spec.get("coords")
                )
                values[f"{root}/{asset}/view{anchor:02d}_scale.json"] = json.dumps(
                    {"total_scale": spec.get("scale", 0.5)}
                ).encode()
    return values


def write_pack(
    path,
    *,
    batch,
    family,
    frozen,
    included,
    latent_specs=None,
    shard="ABO-00000",
):
    members = _members(family, frozen, latent_specs)
    manifest_members = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as bundle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            bundle.addfile(info, io.BytesIO(payload))
            manifest_members.append({"path": name, "size": len(payload), "sha256": sha256(payload).hexdigest()})
    manifest = {
        "schema_version": 2, "shard_id": shard, "batch_id": batch,
        "family": family, "config_hash": "c" * 64, "tool_commit": "deadbeef",
        "asset_sha256s": sorted(frozen), "included_asset_sha256s": sorted(included),
        "completed_count": len(included), "quarantined_count": len(frozen) - len(included),
        "created_at": "2026-07-25T00:00:00+00:00", "validated_at": "2026-07-25T00:01:00+00:00",
        "pack_sha256": sha256(path.read_bytes()).hexdigest(), "members": manifest_members,
        "gate": "production",
    }
    manifest_path = path.with_suffix(".tar.manifest.json")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return manifest_path


def write_catalog_fixture(tmp_path, *, shape512_includes_all=False, latent_specs=None):
    prepared = tmp_path / "prepared"
    index = {"gate": "production", "source": "ABO", "shard_id": "ABO-00000", "batches": {}}
    for batch, asset in (("batch000", ASSET_A), ("batch001", ASSET_B)):
        records = {}
        for family in FAMILIES:
            frozen = (asset,)
            included = frozen if family in {"common", "SS-64", "shape-1024", "PBR-1024"} else ()
            if family == "shape-512" and (batch == "batch000" or shape512_includes_all):
                included = frozen
            pack = prepared / DIRS[family] / "ABO" / "ABO-00000" / f"{batch}.tar"
            manifest = write_pack(
                pack, batch=batch, family=family, frozen=frozen, included=included,
                latent_specs=latent_specs,
            )
            records[family] = {
                "pack": pack.relative_to(prepared).as_posix(),
                "pack_sha256": sha256(pack.read_bytes()).hexdigest(),
                "manifest": manifest.relative_to(prepared).as_posix(),
                "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
            }
        index["batches"][batch] = records
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index, sort_keys=True))
    return index_path, prepared, ASSET_A, ASSET_B


def load_fixture(tmp_path):
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    catalog = load_production_catalog(
        index, prepared, "ABO", "ABO-00000",
        expected_batches=("batch000", "batch001"),
    )
    return index, prepared, catalog


def update_manifest_index(index, prepared, batch="batch000", family="common"):
    value = json.loads(index.read_text())
    manifest = prepared / value["batches"][batch][family]["manifest"]
    value["batches"][batch][family]["manifest_sha256"] = sha256(manifest.read_bytes()).hexdigest()
    index.write_text(json.dumps(value))


def make_two_shard_source(
    tmp_path,
    *,
    overlapping_assets=False,
    source="Fixture",
    acceptance_mode="fixture",
    original_90_percent_gate_passed=True,
):
    """Build two real verified indexes whose batch names intentionally collide."""
    prepared = tmp_path / "prepared"
    indexes = []
    assets = (ASSET_A, ASSET_A if overlapping_assets else ASSET_B)
    shards = (f"{source}-00000", f"{source}-00001")
    for shard, asset in zip(shards, assets):
        index = {
            "gate": "production",
            "source": source,
            "shard_id": shard,
            "batches": {},
        }
        records = {}
        for family in FAMILIES:
            pack = (
                prepared / DIRS[family] / source / shard / "batch000.tar"
            )
            manifest = write_pack(
                pack,
                batch="batch000",
                family=family,
                frozen=(asset,),
                included=(asset,),
                shard=shard,
            )
            records[family] = {
                "pack": pack.relative_to(prepared).as_posix(),
                "pack_sha256": sha256(pack.read_bytes()).hexdigest(),
                "manifest": manifest.relative_to(prepared).as_posix(),
                "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
            }
        index["batches"]["batch000"] = records
        index_path = tmp_path / f"{shard}.json"
        index_path.write_text(json.dumps(index, sort_keys=True))
        indexes.append(index_path)
    spec = ProductionSourceSpec(
        source=source,
        indexes=tuple(indexes),
        expected_batches={
            shard: ("batch000",) for shard in shards
        },
        expected_frozen=2,
        expected_candidate_stages={
            "ss64": 2,
            "shape512": 2,
            "shape1024": 2,
            "pbr1024": 2,
        },
        fixed_count_contract=None,
        acceptance_mode=acceptance_mode,
        original_90_percent_gate_passed=original_90_percent_gate_passed,
    )
    return spec, prepared


def test_multi_index_catalog_accepts_duplicate_batch_names_across_shards(tmp_path):
    """Treating batch ID alone as identity must incorrectly reject this catalog."""
    spec, prepared = make_two_shard_source(tmp_path)
    catalog = load_source_catalog(spec, prepared)
    assert {
        (pack.shard_id, pack.batch_id) for pack in catalog["common"]
    } == {
        ("Fixture-00000", "batch000"),
        ("Fixture-00001", "batch000"),
    }


def test_legacy_public_materializer_rejects_noncanonical_fixture_profile(
    tmp_path,
):
    index, _prepared, catalog = load_fixture(tmp_path)

    with pytest.raises(ValueError, match="canonical production profile"):
        materializer.materialize_stage(
            "ss64",
            catalog,
            tmp_path / "output",
            index_path=index,
            expected_counts=FIXTURE_STAGE_COUNTS,
            expected_waiver=FIXTURE_WAIVER,
        )


def test_hssd_production_gate_materialization_does_not_claim_waiver(tmp_path):
    """A passed production gate must never emit waiver authorization."""
    spec, prepared = make_two_shard_source(
        tmp_path,
        source="HSSD",
        acceptance_mode="production_gate",
        original_90_percent_gate_passed=True,
    )
    catalog = load_source_catalog(spec, prepared)

    final = _materialize_fixture_source_stage(
        spec, "ss64", catalog, tmp_path / "output"
    )
    evidence = json.loads((final / "materialization.json").read_text())

    assert "waiver" not in evidence


def test_multi_index_catalog_rejects_asset_overlap_across_shards(tmp_path):
    """The same frozen asset in two source shards must never be materialized twice."""
    spec, prepared = make_two_shard_source(tmp_path, overlapping_assets=True)
    with pytest.raises(ValueError, match="asset overlap across shards"):
        load_source_catalog(spec, prepared)


def test_source_aware_materialize_stage_records_both_shards(tmp_path):
    """Losing source or shard provenance must alter observed-source evidence."""
    spec, prepared = make_two_shard_source(tmp_path)
    spec = replace(
        spec,
        expected_batches={
            "Fixture-00001": ("batch000",),
            "Fixture-00000": ("batch000",),
        },
    )
    catalog = load_source_catalog(spec, prepared)
    final = _materialize_fixture_source_stage(
        spec, "shape512", catalog, tmp_path / "output"
    )
    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["source"] == "Fixture"
    assert [entry["shard_id"] for entry in evidence["source_indexes"]] == [
        "Fixture-00000",
        "Fixture-00001",
    ]
    assert evidence["candidate_asset_count"] == 2
    assert evidence["asset_count"] == 2
    assert evidence["counts"]["candidate_stages"] == {"shape512": 2}
    assert evidence["counts"]["pack_exclusions"] == {"shape512": 0}
    assert evidence["counts"]["training_exclusions"] == {"shape512": 0}
    assert evidence["counts"]["stages"] == {"shape512": 2}
    assert {
        (pack["shard_id"], pack["batch_id"]) for pack in evidence["packs"]
    } == {
        ("Fixture-00000", "batch000"),
        ("Fixture-00001", "batch000"),
    }


def test_source_aware_materialize_all_publishes_each_stage(tmp_path):
    """Skipping any declared stage must leave this source publication incomplete."""
    spec, prepared = make_two_shard_source(tmp_path)
    catalog = load_source_catalog(spec, prepared)
    outputs = {
        stage: _materialize_fixture_source_stage(
            spec, stage, catalog, tmp_path / "output"
        )
        for stage in ("ss64", "shape512", "shape1024", "pbr1024")
    }
    assert outputs == {
        stage: tmp_path / "output" / stage / "active"
        for stage in ("ss64", "shape512", "shape1024", "pbr1024")
    }
    assert all(path.is_dir() for path in outputs.values())


def test_source_aware_materialize_stage_rejects_foreign_catalog(tmp_path):
    """A caller must not relabel verified packs with another source spec."""
    spec, prepared = make_two_shard_source(tmp_path)
    catalog = load_source_catalog(spec, prepared)
    with pytest.raises(ValueError, match="catalog identity"):
        _materialize_fixture_source_stage(
            replace(spec, source="Other"),
            "ss64",
            catalog,
            tmp_path / "output",
        )


def test_source_aware_materialize_stage_rejects_duplicate_pack_identity(tmp_path):
    """Every expected shard/batch identity must occur exactly once per family."""
    spec, prepared = make_two_shard_source(tmp_path)
    catalog = load_source_catalog(spec, prepared)
    tampered = {
        family: (*records, records[0])
        for family, records in catalog.items()
    }
    with pytest.raises(ValueError, match="catalog identity"):
        _materialize_fixture_source_stage(
            spec, "ss64", tampered, tmp_path / "output"
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(source="OTHER"), "index identity"),
        (lambda value: value.update(shard_id="Fixture-99999"), "index identity"),
        (lambda value: value["batches"].pop("batch000"), "batch set"),
        (
            lambda value: value["batches"]["batch000"]["common"].update(
                manifest_sha256="0" * 64
            ),
            "manifest digest mismatch",
        ),
    ],
)
def test_multi_index_catalog_rejects_unbound_index_or_manifest(
    tmp_path, mutation, match
):
    """Every shard identity, batch set, and manifest byte digest stays bound."""
    spec, prepared = make_two_shard_source(tmp_path)
    index = spec.indexes[0]
    value = json.loads(index.read_text())
    mutation(value)
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        load_source_catalog(spec, prepared)


def test_materializer_cli_exposes_source_profiles_and_repeatable_stages():
    """Removing either profile or stage selection must break the public CLI."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_multiview_production.py",
            "--help",
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--profile {abo,3d-future,hssd}" in result.stdout
    assert "--stage {ss64,shape512,shape1024,pbr1024}" in result.stdout


def test_materializer_legacy_argument_error_uses_configured_parser():
    """Legacy ABO errors must retain the full compatibility CLI usage."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_multiview_production.py",
            "--profile",
            "abo",
            "--prepared-root",
            "/unexpected/prepared",
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--prepared-root must match the selected profile" in result.stderr
    assert "--profile {abo,3d-future,hssd}" in result.stderr
    assert "--data2-root DATA2_ROOT" in result.stderr
    assert "--stage {ss64,shape512,shape1024,pbr1024}" in result.stderr


def test_materializer_cli_rejects_index_from_another_profile():
    """A selected profile must not be combined with an unrelated index."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_multiview_production.py",
            "--profile",
            "3d-future",
            "--index",
            "/root/data2/pixal3d/prepared/index/ABO/ABO-00000.json",
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--index must match the selected profile" in result.stderr


def test_catalog_computes_exact_family_intersections(tmp_path):
    """A wrong family union or cross-batch duplicate check must fail this scope contract."""
    index, prepared, asset_a, asset_b = write_catalog_fixture(tmp_path)
    catalog = load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))
    assert compute_stage_scopes(catalog) == {
        "ss64": (asset_a, asset_b), "shape512": (asset_a,),
        "shape1024": (asset_a, asset_b), "pbr1024": (asset_a, asset_b),
    }
    assert isinstance(catalog["common"][0], FamilyPack)


@pytest.mark.parametrize("mutation, match", [
    (lambda value: value.update(source="OTHER"), "index"),
    (lambda value: value["batches"].pop("batch001"), "batch"),
    (lambda value: value["batches"]["batch000"].pop("common"), "family"),
])
def test_catalog_rejects_invalid_index_identity_or_completeness(tmp_path, mutation, match):
    """A production index with a wrong identity or incomplete pack matrix must not be admitted."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); mutation(value); index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def test_catalog_rejects_manifest_digest_disagreement(tmp_path):
    """An index digest mutation must prevent using a manifest that changed after indexing."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); value["batches"]["batch000"]["common"]["manifest_sha256"] = "0" * 64; index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def test_catalog_rejects_symlinked_pack_even_when_target_is_valid(tmp_path):
    """A symlink substitution must not redirect production materialization to another pack."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text())
    record = value["batches"]["batch000"]["common"]
    pack = prepared / record["pack"]
    link = pack.with_name("linked.tar")
    link.symlink_to(pack.name)
    record["pack"] = link.relative_to(prepared).as_posix()
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="non-regular pack"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def test_catalog_rejects_symlinked_index_even_when_target_is_valid(tmp_path):
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    target = tmp_path / "index-target.json"
    target.write_bytes(index.read_bytes())
    index.unlink()
    index.symlink_to(target)

    with pytest.raises(ValueError, match="non-regular production index"):
        load_production_catalog(
            index,
            prepared,
            "ABO",
            "ABO-00000",
            expected_batches=("batch000", "batch001"),
        )


def test_catalog_rejects_noncanonical_included_scope(tmp_path):
    """A duplicate or unsorted included scope must not silently alter selected assets."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    index_value = json.loads(index.read_text())
    manifest = prepared / index_value["batches"]["batch000"]["common"]["manifest"]
    value = json.loads(manifest.read_text()); value["included_asset_sha256s"] = [ASSET_A, ASSET_A]; manifest.write_text(json.dumps(value))
    index_value["batches"]["batch000"]["common"]["manifest_sha256"] = sha256(manifest.read_bytes()).hexdigest()
    index.write_text(json.dumps(index_value))
    with pytest.raises(ValueError, match="included"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def test_materialize_stage_extracts_only_intersection_and_exact_metadata(tmp_path):
    """Extracting an excluded asset or wrong CSV row must fail the selected-publication contract."""
    index, prepared, asset_a, _ = write_catalog_fixture(tmp_path)
    catalog = load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))
    final = materialize_stage(
        "shape512", catalog, tmp_path / "output", index_path=index,
        expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER,
        expected_stage_counts={"shape512": 1},
        expected_training_exclusion_counts={"shape512": 0},
    )
    assert sorted(path.name for path in (final / "renders_cond").iterdir()) == [asset_a, "metadata.csv"]
    with (final / "renders_cond" / "metadata.csv").open() as stream:
        assert list(csv.DictReader(stream)) == [{"sha256": asset_a, "cond_rendered": "True"}]
    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["stage"] == "shape512" and evidence["asset_count"] == 1
    assert evidence["stage_root"] == str(final.resolve())
    assert evidence["waiver"] == "production-valid-subset"


def _metadata_assets(component):
    with (component / "metadata.csv").open() as stream:
        return [row["sha256"] for row in csv.DictReader(stream)]


def test_materialize_stage_removes_training_ineligible_assets_from_every_component(tmp_path):
    """Publishing the candidate instead of the Shape-512 eligible scope must fail this contract."""
    latent_specs = {("shape-512", ASSET_B, 1): {"count": 8193}}
    index, prepared, accepted, rejected = write_catalog_fixture(
        tmp_path, shape512_includes_all=True, latent_specs=latent_specs
    )
    catalog = load_production_catalog(
        index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001")
    )
    final = materialize_stage(
        "shape512", catalog, tmp_path / "output", index_path=index,
        expected_counts={"shape512": 2}, expected_waiver={
            "frozen_assets": 2, "quarantined_assets": 0, "shape512_exclusions": 0,
        },
        expected_stage_counts={"shape512": 1},
        expected_training_exclusion_counts={"shape512": 1},
    )
    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["candidate_asset_count"] == 2
    assert evidence["candidate_stage_scope"] == [accepted, rejected]
    assert evidence["candidate_stage_scope_sha256"] == sha256(
        f"{accepted}\n{rejected}".encode()
    ).hexdigest()
    assert evidence["asset_count"] == 1
    assert evidence["stage_scope"] == [accepted]
    assert evidence["stage_scope_sha256"] == sha256(accepted.encode()).hexdigest()
    assert evidence["training_exclusion_count"] == 1
    assert evidence["training_exclusions"] == [{
        "asset": rejected, "reasons": ["shape_tokens_view01_exceed_8192"],
    }]
    assert evidence["training_exclusion_reason_counts"] == {
        "shape_tokens_view01_exceed_8192": 1,
    }
    assert evidence["eligibility_policy"] == policy_evidence()
    for relative in (
        "renders_cond",
        "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    ):
        component = final / relative
        assert _metadata_assets(component) == [accepted]
        assert not (component / rejected).exists()


def test_materialize_stage_keeps_tolerant_pbr_scale_and_removes_coordinate_mismatch(tmp_path):
    """Skipping PBR coordinate checks or tightening approved scale tolerance must alter this scope."""
    pbr_mismatch = np.ones((1, 3), dtype=np.float32)
    latent_specs = {
        ("PBR-1024", ASSET_A, 0): {"scale": 0.5000002},
        ("PBR-1024", ASSET_B, 0): {"coords": pbr_mismatch},
        ("PBR-1024", ASSET_B, 1): {"coords": pbr_mismatch},
    }
    index, prepared, accepted, rejected = write_catalog_fixture(tmp_path, latent_specs=latent_specs)
    catalog = load_production_catalog(
        index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001")
    )
    final = materialize_stage(
        "pbr1024", catalog, tmp_path / "output", index_path=index,
        expected_counts={"pbr1024": 2}, expected_waiver=FIXTURE_WAIVER,
        expected_stage_counts={"pbr1024": 1},
        expected_training_exclusion_counts={"pbr1024": 1},
    )
    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["stage_scope"] == [accepted]
    assert evidence["training_exclusions"] == [{
        "asset": rejected,
        "reasons": [
            "pbr_shape_coords_view00_mismatch",
            "pbr_shape_coords_view01_mismatch",
        ],
    }]
    assert evidence["training_exclusion_reason_counts"] == {
        "pbr_shape_coords_view00_mismatch": 1,
        "pbr_shape_coords_view01_mismatch": 1,
    }
    for relative in (
        "renders_cond",
        "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
        "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
    ):
        component = final / relative
        assert _metadata_assets(component) == [accepted]
        assert not (component / rejected).exists()


def test_materialize_stage_rejects_final_count_and_preserves_attempt(tmp_path):
    """A wrong approved count must preserve the rejected staging attempt."""
    latent_specs = {("shape-512", ASSET_B, 1): {"count": 8193}}
    index, prepared, _, _ = write_catalog_fixture(
        tmp_path, shape512_includes_all=True, latent_specs=latent_specs
    )
    catalog = load_production_catalog(
        index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001")
    )
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    with pytest.raises(ValueError, match="final|training exclusion"):
        materialize_stage(
            "shape512", catalog, tmp_path / "output", index_path=index,
            expected_counts={"shape512": 2}, expected_waiver={
                "frozen_assets": 2, "quarantined_assets": 0, "shape512_exclusions": 0,
            },
            expected_stage_counts={"shape512": 2},
            expected_training_exclusion_counts={"shape512": 1},
        )
    parent = tmp_path / "output" / "shape512"
    assert not (parent / "active").exists()
    assert not list(parent.glob(".materializing-*"))
    assert len(list(rejected.iterdir())) == 1


def test_materialize_stage_refuses_existing_active_before_temporary_creation(tmp_path):
    """An existing active stage must never be overwritten or leave a staging sibling."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    catalog = load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))
    active = tmp_path / "output" / "shape512" / "active"; active.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        materialize_stage("shape512", catalog, tmp_path / "output", index_path=index, expected_counts={"shape512": 1})
    assert not list(active.parent.glob(".materializing-*"))


def test_materialize_rejects_catalog_without_required_waiver_before_temporary_creation(tmp_path):
    """A catalog with stage-sized fixtures but no required frozen/quarantine waiver must not publish."""
    index, _, catalog = load_fixture(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="waiver"):
        materialize_stage("shape512", catalog, output, index_path=index, expected_counts={"shape512": 1})
    assert not list((output / "shape512").glob(".materializing-*")) if (output / "shape512").exists() else True


def test_materialize_records_checked_waiver_and_sorted_evidence(tmp_path):
    """Evidence must retain independently checked waiver populations and deterministic ordering."""
    index, _, catalog = load_fixture(tmp_path)
    final = materialize_stage(
        "shape512", catalog, tmp_path / "output", index_path=index,
        expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER,
        expected_stage_counts={"shape512": 1},
        expected_training_exclusion_counts={"shape512": 0},
    )
    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["frozen_assets"] == 2
    assert evidence["quarantined_assets"] == 0
    assert evidence["shape512_exclusions"] == 1
    assert evidence["stage_scope"] == sorted(evidence["stage_scope"])
    assert evidence["packs"] == sorted(evidence["packs"], key=lambda row: (row["family"], row["batch_id"]))
    assert evidence["index_sha256"] == sha256(index.read_bytes()).hexdigest()
    assert evidence["stage_scope_sha256"] == sha256(ASSET_A.encode()).hexdigest()
    assert evidence["tool_commits"] == ["deadbeef"]
    assert len(evidence["packs"]) == 4
    assert all(set(pack) == {"batch_id", "family", "pack", "manifest", "pack_sha256", "manifest_sha256", "tool_commit"} for pack in evidence["packs"])
    assert evidence["schema_version"] == 1
    assert evidence["source"] == "ABO" and evidence["shard_id"] == "ABO-00000"
    assert evidence["source_index"] == {"path": str(index.resolve()), "sha256": sha256(index.read_bytes()).hexdigest()}
    assert evidence["acceptance_mode"] == "valid_subset_user_waiver"
    assert evidence["original_90_percent_gate_passed"] is False
    assert evidence["counts"] == {
        "frozen": 2,
        "global_quarantine": 0,
        "shape512_family_exclusions": 1,
        "candidate_stages": {
            "ss64": 2,
            "shape512": 1,
            "shape1024": 2,
            "pbr1024": 2,
        },
        "training_exclusions": {
            "ss64": 0,
            "shape512": 0,
            "shape1024": 0,
            "pbr1024": 0,
        },
        "stages": {"ss64": 2, "shape512": 1, "shape1024": 2, "pbr1024": 2},
    }
    assert isinstance(evidence["created_at"], str) and evidence["created_at"]
    assert not list(final.parent.glob(".materializing-*"))


def test_materialize_rejects_index_drift_before_staging(tmp_path):
    index, _, catalog = load_fixture(tmp_path)
    index.write_bytes(index.read_bytes() + b" ")

    with pytest.raises(ValueError, match="index bytes changed"):
        materialize_stage(
            "shape512",
            catalog,
            tmp_path / "output",
            index_path=index,
            expected_counts={"shape512": 1},
            expected_waiver=FIXTURE_WAIVER,
        )

    assert not (tmp_path / "output" / "shape512").exists()


def test_materialize_rejects_inconsistent_catalog_index_pin_before_staging(
    tmp_path,
):
    index, _, catalog = load_fixture(tmp_path)
    tampered = dict(catalog)
    tampered["common"] = (
        replace(
            catalog["common"][0],
            source_index_path=tmp_path / "other-index.json",
        ),
        *catalog["common"][1:],
    )

    with pytest.raises(ValueError, match="catalog index pin"):
        materialize_stage(
            "shape512",
            tampered,
            tmp_path / "output",
            index_path=index,
            expected_counts={"shape512": 1},
            expected_waiver=FIXTURE_WAIVER,
        )

    assert not (tmp_path / "output" / "shape512").exists()


def test_materialize_rejects_index_drift_immediately_before_publication(
    tmp_path, monkeypatch
):
    index, _, catalog = load_fixture(tmp_path)
    pinned_digest = sha256(index.read_bytes()).hexdigest()
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    original_copy = training_materializer._copy_selected
    changed = False

    def mutate_index_after_copy(pack, assets, temporary):
        nonlocal changed
        original_copy(pack, assets, temporary)
        if not changed:
            index.write_bytes(index.read_bytes() + b" ")
            changed = True

    monkeypatch.setattr(
        training_materializer, "_copy_selected", mutate_index_after_copy
    )

    with pytest.raises(ValueError, match="index bytes changed"):
        materialize_stage(
            "shape512",
            catalog,
            tmp_path / "output",
            index_path=index,
            expected_counts={"shape512": 1},
            expected_waiver=FIXTURE_WAIVER,
            expected_stage_counts={"shape512": 1},
            expected_training_exclusion_counts={"shape512": 0},
        )

    attempts = list(rejected.iterdir())
    assert len(attempts) == 1
    evidence = json.loads(
        (attempts[0] / "materialization.json").read_text()
    )
    assert evidence["source_index"]["sha256"] == pinned_digest
    assert not (tmp_path / "output" / "shape512" / "active").exists()


def test_materialization_evidence_uses_catalogued_index_digest(
    tmp_path, monkeypatch
):
    index, _, catalog = load_fixture(tmp_path)
    pinned_digest = sha256(index.read_bytes()).hexdigest()
    monkeypatch.setattr(
        training_materializer, "file_sha", lambda _path: "f" * 64
    )

    final = materialize_stage(
        "shape512",
        catalog,
        tmp_path / "output",
        index_path=index,
        expected_counts={"shape512": 1},
        expected_waiver=FIXTURE_WAIVER,
        expected_stage_counts={"shape512": 1},
        expected_training_exclusion_counts={"shape512": 0},
    )

    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["index_sha256"] == pinned_digest
    assert evidence["source_index"]["sha256"] == pinned_digest


def test_materializer_evidence_is_accepted_by_strict_materialization_scope(
    tmp_path, monkeypatch
):
    """Diverging count assemblers must not make real materializer evidence fail strict preflight."""
    index, _, catalog = load_fixture(tmp_path)
    final = materialize_stage(
        "shape512",
        catalog,
        tmp_path / "output",
        index_path=index,
        expected_counts={"shape512": 1},
        expected_waiver=FIXTURE_WAIVER,
        expected_stage_counts={"shape512": 1},
        expected_training_exclusion_counts={"shape512": 0},
    )
    evidence = json.loads((final / "materialization.json").read_text())
    monkeypatch.setattr(
        strict_preflight,
        "HANDOFF_CANDIDATE_STAGE_COUNTS",
        evidence["counts"]["candidate_stages"],
    )
    monkeypatch.setattr(
        strict_preflight,
        "HANDOFF_TRAINING_EXCLUSION_COUNTS",
        evidence["counts"]["training_exclusions"],
    )
    monkeypatch.setattr(
        strict_preflight,
        "HANDOFF_STAGE_COUNTS",
        evidence["counts"]["stages"],
    )
    monkeypatch.setattr(
        strict_preflight, "HANDOFF_FROZEN_COUNT", evidence["counts"]["frozen"]
    )
    monkeypatch.setattr(
        strict_preflight,
        "HANDOFF_GLOBAL_QUARANTINE_COUNT",
        evidence["counts"]["global_quarantine"],
    )
    monkeypatch.setattr(
        strict_preflight,
        "HANDOFF_SHAPE512_FAMILY_EXCLUSION_COUNT",
        evidence["counts"]["shape512_family_exclusions"],
    )
    assert strict_preflight._materialization_scope("shape512", final)[:2] == (
        tuple(evidence["stage_scope"]),
        evidence["stage_scope_sha256"],
    )


def test_materialize_rejects_dangling_active_symlink_lexically(tmp_path):
    """A dangling active symlink is an existing destination and must stop before staging."""
    index, _, catalog = load_fixture(tmp_path)
    active = tmp_path / "output" / "shape512" / "active"
    active.parent.mkdir(parents=True)
    active.symlink_to("missing-active")
    with pytest.raises(FileExistsError):
        materialize_stage(
            "shape512", catalog, tmp_path / "output", index_path=index,
            expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER,
            expected_stage_counts={"shape512": 1},
            expected_training_exclusion_counts={"shape512": 0},
        )
    assert active.is_symlink() and not list(active.parent.glob(".materializing-*"))


def test_materialize_never_replaces_active_created_during_publication(tmp_path, monkeypatch):
    """A racing active creation must win over publication and preserve its sentinel contents."""
    index, _, catalog = load_fixture(tmp_path)
    original = materializer._publish_no_replace
    rejected = tmp_path / "rejected"
    rejected.mkdir()

    def create_racer(temporary, final, **kwargs):
        parent_fd = kwargs.get("parent_fd")
        if parent_fd is None:
            final.mkdir()
            (final / "sentinel").write_text("racer")
        else:
            os.mkdir(final.name, dir_fd=parent_fd)
            descriptor = os.open(
                f"{final.name}/sentinel",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(descriptor, "w") as stream:
                stream.write("racer")
        original(temporary, final, **kwargs)

    monkeypatch.setattr(materializer, "_publish_no_replace", create_racer)
    with pytest.raises(FileExistsError):
        materialize_stage(
            "shape512", catalog, tmp_path / "output", index_path=index,
            expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER,
            expected_stage_counts={"shape512": 1},
            expected_training_exclusion_counts={"shape512": 0},
        )
    active = tmp_path / "output" / "shape512" / "active"
    assert (active / "sentinel").read_text() == "racer"
    assert not list(active.parent.glob(".materializing-*"))
    assert len(list(rejected.iterdir())) == 1


def test_materialize_success_parent_swap_publishes_exact_attempt_only_to_pinned_parent(
    tmp_path, monkeypatch
):
    index, _, catalog = load_fixture(tmp_path)
    stage_parent = tmp_path / "output" / "shape512"
    original_stage_parent = tmp_path / "original-shape512"
    original_publish = materializer._publish_no_replace

    def swap_parent_with_decoy_then_publish(
        temporary, final, **kwargs
    ):
        stage_parent.rename(original_stage_parent)
        stage_parent.mkdir()
        decoy = stage_parent / Path(temporary).name
        decoy.mkdir()
        (decoy / "sentinel").write_text("decoy")
        return original_publish(temporary, final, **kwargs)

    monkeypatch.setattr(
        materializer,
        "_publish_no_replace",
        swap_parent_with_decoy_then_publish,
    )

    materialize_stage(
        "shape512",
        catalog,
        tmp_path / "output",
        index_path=index,
        expected_counts={"shape512": 1},
        expected_waiver=FIXTURE_WAIVER,
        expected_stage_counts={"shape512": 1},
        expected_training_exclusion_counts={"shape512": 0},
    )

    active = original_stage_parent / "active"
    assert (
        active / "renders_cond" / ASSET_A / "000.png"
    ).read_bytes() == f"{ASSET_A}-0".encode()
    assert not (active / "sentinel").exists()
    assert not (stage_parent / "active").exists()
    decoys = list(stage_parent.glob(".materializing-*"))
    assert len(decoys) == 1
    assert (decoys[0] / "sentinel").read_text() == "decoy"


@pytest.mark.parametrize("field, replacement", [
    ("source", "OTHER"), ("shard_id", "OTHER-00000"), ("gate", "pilot"),
])
def test_catalog_rejects_each_wrong_index_identity_field(tmp_path, field, replacement):
    """Each production identity field must independently bind the catalog to ABO's shard."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); value[field] = replacement; index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="index"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def test_catalog_rejects_extra_batch_key(tmp_path):
    """An unexpected batch must not alter the fixed production population."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); value["batches"]["batch999"] = value["batches"]["batch000"]; index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="batch"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def test_catalog_rejects_extra_family_key(tmp_path):
    """An extra family entry must not conceal a malformed required eight-family matrix."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); value["batches"]["batch000"]["extra"] = {}; index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="family"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


@pytest.mark.parametrize("kind", ["symlink", "directory", "missing", "empty"])
@pytest.mark.parametrize("field", ["pack", "manifest"])
def test_catalog_rejects_each_invalid_pack_or_manifest_filesystem_entry(tmp_path, field, kind):
    """Every filesystem type mutation must fail before an index entry is trusted."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); record = value["batches"]["batch000"]["common"]
    original = prepared / record[field]
    replacement = original.with_name(f"bad-{field}-{kind}{original.suffix}")
    if kind == "symlink":
        replacement.symlink_to(original.name)
    elif kind == "directory":
        replacement.mkdir(parents=True)
    elif kind == "empty":
        replacement.parent.mkdir(parents=True, exist_ok=True); replacement.write_bytes(b"")
    else:
        replacement = replacement.with_name("missing-entry")
    record[field] = replacement.relative_to(prepared).as_posix(); index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="(non-regular|missing|empty)"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def test_catalog_rejects_index_pack_digest_disagreement(tmp_path):
    """Changing only the indexed pack digest must not relabel a verified manifest."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); value["batches"]["batch000"]["common"]["pack_sha256"] = "0" * 64; index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="pack digest disagreement"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


@pytest.mark.parametrize("field, replacement", [
    ("batch_id", "batch999"), ("family", "shape-512"), ("gate", "pilot"),
])
def test_catalog_rejects_each_wrong_manifest_identity_field(tmp_path, field, replacement):
    """A manifest's production identity must agree with its exact index record."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); manifest = prepared / value["batches"]["batch000"]["common"]["manifest"]
    manifest_value = json.loads(manifest.read_text()); manifest_value[field] = replacement; manifest.write_text(json.dumps(manifest_value))
    update_manifest_index(index, prepared)
    with pytest.raises(ValueError, match="identity"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def test_catalog_rejects_missing_manifest_validation_timestamp(tmp_path):
    """A pack that was never validated must never enter the production catalog."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    value = json.loads(index.read_text()); manifest = prepared / value["batches"]["batch000"]["common"]["manifest"]
    manifest_value = json.loads(manifest.read_text()); manifest_value["validated_at"] = ""; manifest.write_text(json.dumps(manifest_value))
    update_manifest_index(index, prepared)
    with pytest.raises(ValueError, match="validated_at"):
        load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))


def _rewrite_tar(path, entries):
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as bundle:
        for name, payload, kind in entries:
            info = tarfile.TarInfo(name); info.mtime = 0
            if kind == "file":
                info.size = len(payload); bundle.addfile(info, io.BytesIO(payload))
            elif kind == "link":
                info.type = tarfile.SYMTYPE; info.linkname = "target"; bundle.addfile(info)
            else:
                raise AssertionError(kind)


@pytest.mark.parametrize("fault", ["unsafe", "link", "unexpected", "size", "sha256"])
def test_materialize_rejects_each_selected_tar_fault_and_preserves_attempt(tmp_path, fault):
    """Selected tar faults must be rejected into one preserved attempt tree."""
    index, _, catalog = load_fixture(tmp_path)
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    record = catalog["common"][0]
    expected = materializer._expected_member_paths("common", ASSET_A)
    if fault == "unsafe":
        entries = [(f"renders_cond/{ASSET_A}/../escape", b"bad", "file")]
    elif fault == "link":
        entries = [(expected[0], b"", "link")]
    elif fault == "unexpected":
        entries = [(f"renders_cond/{ASSET_A}/extra.bin", b"bad", "file")]
    elif fault == "size":
        entries = [(expected[0], b"different-size", "file")]
    else:
        original = f"{ASSET_A}-0".encode()
        entries = [(expected[0], b"Z" * len(original), "file")]
    _rewrite_tar(record.pack, entries)
    with pytest.raises(ValueError, match="(unsafe|unexpected|digest|set mismatch)"):
        materialize_stage("shape512", catalog, tmp_path / "output", index_path=index, expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER)
    parent = tmp_path / "output" / "shape512"
    assert not list(parent.glob(".materializing-*"))
    assert len(list(rejected.iterdir())) == 1


def test_materialize_preserves_attempt_after_mid_extraction_failure(tmp_path):
    """A later-family extraction error must preserve earlier copied files."""
    index, _, catalog = load_fixture(tmp_path)
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    record = catalog["shape-512"][0]
    _rewrite_tar(record.pack, [(materializer._expected_member_paths("shape-512", ASSET_A)[0], b"wrong", "file")])
    with pytest.raises(ValueError, match="digest"):
        materialize_stage("shape512", catalog, tmp_path / "output", index_path=index, expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER)
    parent = tmp_path / "output" / "shape512"
    assert not list(parent.glob(".materializing-*")) and not (parent / "active").exists()
    attempts = list(rejected.iterdir())
    assert len(attempts) == 1
    assert (
        attempts[0] / "renders_cond" / ASSET_A / "000.png"
    ).read_bytes() == f"{ASSET_A}-0".encode()


def test_materialize_preserves_each_failed_attempt_in_unique_rejected_child(
    tmp_path,
):
    index, _, catalog = load_fixture(tmp_path)
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    record = catalog["shape-512"][0]
    _rewrite_tar(
        record.pack,
        [
            (
                materializer._expected_member_paths(
                    "shape-512", ASSET_A
                )[0],
                b"wrong",
                "file",
            )
        ],
    )

    for _attempt in range(2):
        with pytest.raises(ValueError, match="digest"):
            materialize_stage(
                "shape512",
                catalog,
                tmp_path / "output",
                index_path=index,
                expected_counts={"shape512": 1},
                expected_waiver=FIXTURE_WAIVER,
            )

    attempts = sorted(rejected.iterdir())
    assert len(attempts) == 2
    assert attempts[0].name != attempts[1].name
    assert all(attempt.is_dir() and not attempt.is_symlink() for attempt in attempts)
    assert all(
        (
            attempt
            / "renders_cond"
            / ASSET_A
            / "000.png"
        ).read_bytes()
        == f"{ASSET_A}-0".encode()
        for attempt in attempts
    )
    stage_parent = tmp_path / "output" / "shape512"
    assert not list(stage_parent.glob(".materializing-*"))
    assert not (stage_parent / "active").exists()


def test_materialize_preserves_interrupted_attempt_after_staging(
    tmp_path, monkeypatch
):
    index, _, catalog = load_fixture(tmp_path)
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    original_copy = training_materializer._copy_selected

    def interrupt_after_copy(pack, assets, temporary):
        original_copy(pack, assets, temporary)
        raise KeyboardInterrupt("synthetic interruption")

    monkeypatch.setattr(
        training_materializer, "_copy_selected", interrupt_after_copy
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic"):
        materialize_stage(
            "shape512",
            catalog,
            tmp_path / "output",
            index_path=index,
            expected_counts={"shape512": 1},
            expected_waiver=FIXTURE_WAIVER,
        )

    attempts = list(rejected.iterdir())
    assert len(attempts) == 1
    assert (
        attempts[0] / "renders_cond" / ASSET_A / "000.png"
    ).exists()
    assert not list(
        (tmp_path / "output" / "shape512").glob(".materializing-*")
    )


def test_materialize_rejected_parent_swap_cannot_redirect_attempt(
    tmp_path, monkeypatch
):
    """Replacing the validated reject pathname must not redirect preservation."""
    index, _, catalog = load_fixture(tmp_path)
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    original_rejected = tmp_path / "original-rejected"
    record = catalog["shape-512"][0]
    _rewrite_tar(
        record.pack,
        [
            (
                materializer._expected_member_paths(
                    "shape-512", ASSET_A
                )[0],
                b"wrong",
                "file",
            )
        ],
    )
    original_rename = training_materializer._rename_no_replace
    swapped = False

    def swap_parent_then_rename(source, destination, **kwargs):
        nonlocal swapped
        rejected.rename(original_rejected)
        rejected.mkdir()
        swapped = True
        return original_rename(source, destination, **kwargs)

    monkeypatch.setattr(
        training_materializer,
        "_rename_no_replace",
        swap_parent_then_rename,
    )

    with pytest.raises(ValueError, match="digest"):
        materialize_stage(
            "shape512",
            catalog,
            tmp_path / "output",
            index_path=index,
            expected_counts={"shape512": 1},
            expected_waiver=FIXTURE_WAIVER,
        )

    assert swapped
    attempts = list(original_rejected.iterdir())
    assert len(attempts) == 1
    assert (
        attempts[0] / "renders_cond" / ASSET_A / "000.png"
    ).read_bytes() == f"{ASSET_A}-0".encode()
    assert not list(rejected.iterdir())
    assert not list(
        (tmp_path / "output" / "shape512").glob(".materializing-*")
    )


def test_materialize_failure_parent_swap_preserves_exact_attempt(
    tmp_path, monkeypatch
):
    index, _, catalog = load_fixture(tmp_path)
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    stage_parent = tmp_path / "output" / "shape512"
    original_stage_parent = tmp_path / "original-shape512"
    original_copy = training_materializer._copy_selected

    def copy_then_swap_parent_and_fail(pack, assets, temporary):
        original_copy(pack, assets, temporary)
        attempt = next(stage_parent.glob(".materializing-*"))
        stage_parent.rename(original_stage_parent)
        stage_parent.mkdir()
        decoy = stage_parent / attempt.name
        decoy.mkdir()
        (decoy / "sentinel").write_text("decoy")
        raise ValueError("synthetic post-staging failure")

    monkeypatch.setattr(
        training_materializer,
        "_copy_selected",
        copy_then_swap_parent_and_fail,
    )

    with pytest.raises(ValueError, match="synthetic post-staging"):
        materialize_stage(
            "shape512",
            catalog,
            tmp_path / "output",
            index_path=index,
            expected_counts={"shape512": 1},
            expected_waiver=FIXTURE_WAIVER,
        )

    attempts = list(rejected.iterdir())
    assert len(attempts) == 1
    assert (
        attempts[0] / "renders_cond" / ASSET_A / "000.png"
    ).read_bytes() == f"{ASSET_A}-0".encode()
    assert not list(original_stage_parent.glob(".materializing-*"))
    decoys = list(stage_parent.glob(".materializing-*"))
    assert len(decoys) == 1
    assert (decoys[0] / "sentinel").read_text() == "decoy"


@pytest.mark.parametrize("unsafe_rejected", ("missing", "symlink"))
def test_materialize_preservation_failure_leaves_staging_tree_intact(
    tmp_path, unsafe_rejected
):
    index, _, catalog = load_fixture(tmp_path)
    rejected = tmp_path / "rejected"
    if unsafe_rejected == "symlink":
        target = tmp_path / "other-rejected"
        target.mkdir()
        rejected.symlink_to(target, target_is_directory=True)
    record = catalog["shape-512"][0]
    _rewrite_tar(
        record.pack,
        [
            (
                materializer._expected_member_paths(
                    "shape-512", ASSET_A
                )[0],
                b"wrong",
                "file",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="preserve.*staging retained"):
        materialize_stage(
            "shape512",
            catalog,
            tmp_path / "output",
            index_path=index,
            expected_counts={"shape512": 1},
            expected_waiver=FIXTURE_WAIVER,
        )

    staging = list(
        (tmp_path / "output" / "shape512").glob(".materializing-*")
    )
    assert len(staging) == 1
    assert (
        staging[0] / "renders_cond" / ASSET_A / "000.png"
    ).read_bytes() == f"{ASSET_A}-0".encode()
    if unsafe_rejected == "symlink":
        assert not list((tmp_path / "other-rejected").iterdir())


def test_materialize_sorts_multi_asset_metadata_and_evidence_scope(tmp_path):
    """A multi-asset stage must retain stable row and evidence ordering independent of pack order."""
    index, _, catalog = load_fixture(tmp_path)
    final = materialize_stage(
        "ss64", catalog, tmp_path / "output", index_path=index,
        expected_counts={"ss64": 2}, expected_waiver=FIXTURE_WAIVER,
        expected_stage_counts={"ss64": 2},
        expected_training_exclusion_counts={"ss64": 0},
    )
    with (final / "renders_cond" / "metadata.csv").open() as stream:
        assert [row["sha256"] for row in csv.DictReader(stream)] == [ASSET_A, ASSET_B]
    assert json.loads((final / "materialization.json").read_text())["stage_scope"] == [ASSET_A, ASSET_B]


def test_materializer_emits_exact_preflight_latent_metadata_headers(tmp_path):
    """Changing any producer header must fail before strict preflight sees drifted metadata."""
    index, _, catalog = load_fixture(tmp_path)
    cases = {
        "ss64": (
            ("ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
             "sha256,ss_latent_view_scale00_encoded,ss_latent_view_scale01_encoded"),
        ),
        "shape512": (
            ("shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
             "sha256,shape_latent_view00_encoded,shape_latent_view01_encoded"),
        ),
        "shape1024": (
            ("shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
             "sha256,shape_latent_view00_encoded,shape_latent_view01_encoded"),
        ),
        "pbr1024": (
            ("shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
             "sha256,shape_latent_view00_encoded,shape_latent_view01_encoded"),
            ("pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
             "sha256,pbr_latent_view00_encoded,pbr_latent_view01_encoded"),
        ),
    }
    for stage, expected_headers in cases.items():
        final = materialize_stage(
            stage, catalog, tmp_path / "output", index_path=index,
            expected_counts={stage: FIXTURE_STAGE_COUNTS[stage]},
            expected_waiver=FIXTURE_WAIVER,
            expected_stage_counts={stage: FIXTURE_STAGE_COUNTS[stage]},
            expected_training_exclusion_counts={stage: FIXTURE_TRAINING_EXCLUSION_COUNTS[stage]},
        )
        assert [
            (relative, (final / relative / "metadata.csv").read_text().splitlines()[0])
            for relative, _ in expected_headers
        ] == list(expected_headers)
