import csv
import io
import json
import os
import tarfile
from hashlib import sha256
from pathlib import Path

import pytest

import scripts.materialize_multiview_production as materializer
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
FIXTURE_WAIVER = {
    "frozen_assets": 2,
    "quarantined_assets": 0,
    "shape512_exclusions": 1,
}


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
    final = materialize_stage(
        "shape512", catalog, tmp_path / "output", index_path=index,
        expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER,
    )
    assert sorted(path.name for path in (final / "renders_cond").iterdir()) == [asset_a, "metadata.csv"]
    with (final / "renders_cond" / "metadata.csv").open() as stream:
        assert list(csv.DictReader(stream)) == [{"sha256": asset_a, "cond_rendered": "True"}]
    evidence = json.loads((final / "materialization.json").read_text())
    assert evidence["stage"] == "shape512" and evidence["asset_count"] == 1
    assert evidence["stage_root"] == str(final.resolve())
    assert evidence["waiver"] == "production-valid-subset"


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
        "stages": {"ss64": 2, "shape512": 1, "shape1024": 2, "pbr1024": 2},
    }
    assert isinstance(evidence["created_at"], str) and evidence["created_at"]
    assert not list(final.parent.glob(".materializing-*"))


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
        )
    assert active.is_symlink() and not list(active.parent.glob(".materializing-*"))


def test_materialize_never_replaces_active_created_during_publication(tmp_path, monkeypatch):
    """A racing active creation must win over publication and preserve its sentinel contents."""
    index, _, catalog = load_fixture(tmp_path)
    original = materializer._publish_no_replace

    def create_racer(temporary, final):
        final.mkdir()
        (final / "sentinel").write_text("racer")
        original(temporary, final)

    monkeypatch.setattr(materializer, "_publish_no_replace", create_racer)
    with pytest.raises(FileExistsError):
        materialize_stage(
            "shape512", catalog, tmp_path / "output", index_path=index,
            expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER,
        )
    active = tmp_path / "output" / "shape512" / "active"
    assert (active / "sentinel").read_text() == "racer"
    assert not list(active.parent.glob(".materializing-*"))


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
def test_materialize_rejects_each_selected_tar_fault_and_cleans_temporary(tmp_path, fault):
    """Selected tar members must be safe, exact, and leave no failed materialization tree."""
    index, _, catalog = load_fixture(tmp_path)
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


def test_materialize_cleans_temporary_after_mid_extraction_failure(tmp_path):
    """A later-family extraction error must remove files already copied from the common pack."""
    index, _, catalog = load_fixture(tmp_path)
    record = catalog["shape-512"][0]
    _rewrite_tar(record.pack, [(materializer._expected_member_paths("shape-512", ASSET_A)[0], b"wrong", "file")])
    with pytest.raises(ValueError, match="digest"):
        materialize_stage("shape512", catalog, tmp_path / "output", index_path=index, expected_counts={"shape512": 1}, expected_waiver=FIXTURE_WAIVER)
    parent = tmp_path / "output" / "shape512"
    assert not list(parent.glob(".materializing-*")) and not (parent / "active").exists()


def test_materialize_sorts_multi_asset_metadata_and_evidence_scope(tmp_path):
    """A multi-asset stage must retain stable row and evidence ordering independent of pack order."""
    index, _, catalog = load_fixture(tmp_path)
    final = materialize_stage("ss64", catalog, tmp_path / "output", index_path=index, expected_counts={"ss64": 2}, expected_waiver=FIXTURE_WAIVER)
    with (final / "renders_cond" / "metadata.csv").open() as stream:
        assert [row["sha256"] for row in csv.DictReader(stream)] == [ASSET_A, ASSET_B]
    assert json.loads((final / "materialization.json").read_text())["stage_scope"] == [ASSET_A, ASSET_B]
