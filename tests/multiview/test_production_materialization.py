import csv
import io
import json
import tarfile
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.materialize_multiview_production import (
    FamilyPack,
    compute_stage_scopes,
    load_production_catalog,
    materialize_stage,
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


def _members(family, assets):
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
            for name in ("view00.npz", "view00_scale.json", "view01.npz", "view01_scale.json"):
                values[f"{root}/{asset}/{name}"] = f"{asset}-{name}".encode()
    return values


def write_pack(path, *, batch, family, frozen, included):
    members = _members(family, frozen)
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
        "schema_version": 2, "shard_id": "ABO-00000", "batch_id": batch,
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


def write_catalog_fixture(tmp_path):
    prepared = tmp_path / "prepared"
    index = {"gate": "production", "source": "ABO", "shard_id": "ABO-00000", "batches": {}}
    for batch, asset in (("batch000", ASSET_A), ("batch001", ASSET_B)):
        records = {}
        for family in FAMILIES:
            frozen = (asset,)
            included = frozen if family in {"common", "SS-64", "shape-1024", "PBR-1024"} else ()
            if family == "shape-512" and batch == "batch000":
                included = frozen
            pack = prepared / DIRS[family] / "ABO" / "ABO-00000" / f"{batch}.tar"
            manifest = write_pack(pack, batch=batch, family=family, frozen=frozen, included=included)
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
    final = materialize_stage("shape512", catalog, tmp_path / "output", index_path=index, expected_counts={"shape512": 1})
    assert sorted(path.name for path in (final / "renders_cond").iterdir()) == [asset_a, "metadata.csv"]
    with (final / "renders_cond" / "metadata.csv").open() as stream:
        assert list(csv.DictReader(stream)) == [{"sha256": asset_a, "cond_rendered": "True"}]
    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["stage"] == "shape512" and evidence["asset_count"] == 1
    assert evidence["waiver"] == "production-valid-subset"


def test_materialize_stage_refuses_existing_active_before_temporary_creation(tmp_path):
    """An existing active stage must never be overwritten or leave a staging sibling."""
    index, prepared, *_ = write_catalog_fixture(tmp_path)
    catalog = load_production_catalog(index, prepared, "ABO", "ABO-00000", expected_batches=("batch000", "batch001"))
    active = tmp_path / "output" / "shape512" / "active"; active.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        materialize_stage("shape512", catalog, tmp_path / "output", index_path=index, expected_counts={"shape512": 1})
    assert not list(active.parent.glob(".materializing-*"))
