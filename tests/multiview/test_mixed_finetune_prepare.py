import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
import pytest

SCRIPT = Path("experiments/multiview_finetune/prepare.py")


def test_extraction_checks_members_and_does_not_copy_tar(tmp_path):
    source, destination = tmp_path / "nfs", tmp_path / "ssd"
    family = source / "common/ABO/ABO-00000"
    family.mkdir(parents=True)
    content = b"real archive payload"
    pack = family / "batch001.tar"
    with tarfile.open(pack, "w") as archive:
        member = tarfile.TarInfo("renders_cond/asset/transforms.json")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    manifest = {"members": [{"path": member.name, "size": member.size,
                 "sha256": hashlib.sha256(content).hexdigest()}],
                "included_asset_sha256s": ["asset"]}
    Path(str(pack) + ".manifest.json").write_text(json.dumps(manifest))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"source_root": str(source),
        "destination_root": str(destination), "families": ["common"],
        "batches": [{"source": "ABO", "shard": "ABO-00000",
                    "batch": "batch001", "split": "train"}]}))
    result = subprocess.run([sys.executable, str(SCRIPT), "extract", "--selection",
                             str(selection), "--workers", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    root = destination / "ABO/ABO-00000/batch001"
    assert (root / member.name).read_bytes() == content
    assert not list(destination.rglob("*.tar"))
    assert json.loads((root / "extraction_report.json").read_text())["packs"][0]["files_verified"] == 1


def test_corrupt_member_is_not_published(tmp_path):
    source, destination = tmp_path / "nfs", tmp_path / "ssd"
    family = source / "common/ABO/ABO-00000"
    family.mkdir(parents=True)
    pack = family / "batch001.tar"
    with tarfile.open(pack, "w") as archive:
        member = tarfile.TarInfo("renders_cond/asset/000.png")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    Path(str(pack) + ".manifest.json").write_text(json.dumps({"members": [
        {"path": member.name, "size": 1, "sha256": "0" * 64}],
        "included_asset_sha256s": ["asset"]}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"source_root": str(source),
        "destination_root": str(destination), "families": ["common"],
        "batches": [{"source": "ABO", "shard": "ABO-00000", "batch": "batch001", "split": "train"}]}))
    result = subprocess.run([sys.executable, str(SCRIPT), "extract", "--selection",
                             str(selection), "--workers", "1"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Member integrity mismatch" in result.stderr
    assert not (destination / "ABO/ABO-00000/batch001").exists()


@pytest.mark.parametrize('all_rejected', [False, True])
def test_coordinate_mismatch_is_recorded_and_excluded_from_all_metadata(tmp_path, all_rejected):
    destination = tmp_path / "ssd"
    root = destination / "HSSD/HSSD-00000/batch001"
    (root / "pack_manifests").mkdir(parents=True)
    (root / "renders_cond").mkdir()
    for family in ["common", "ss-64", "shape-512", "shape-1024", "pbr-1024"]:
        (root / "pack_manifests" / (family + ".manifest.json")).write_text(
            json.dumps({"included_asset_sha256s": ["valid", "broken"]}))
    latents = ["ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
               "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
               "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
               "pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix"]
    for subdir in latents:
        for asset in ["valid", "broken"]:
            path = root / subdir / asset
            path.mkdir(parents=True)
            for view in [0, 1]:
                coords = np.zeros((1, 3), dtype=np.int32)
                if subdir.startswith("pbr") and (asset == "broken" or all_rejected) and view == 1:
                    coords[0, 0] = 1
                np.savez(path / f"view{view:02d}.npz", coords=coords,
                         feats=np.zeros((1, 32), dtype=np.float32), z=np.zeros((1,)))
                (path / f"view{view:02d}_scale.json").write_text('{"total_scale":1}')
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"destination_root": str(destination),
        "batches": [{"source": "HSSD", "shard": "HSSD-00000", "batch": "batch001", "split": "train"}]}))
    result = subprocess.run([sys.executable, str(SCRIPT), "metadata", "--selection",
                             str(selection), "--workers", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((root / "metadata_report.json").read_text())
    expected = [] if all_rejected else ["valid"]
    assert report["eligible_assets"] == expected
    assert report["rejected_assets"]["broken"] == ["shape_pbr_coordinates_view01"]
    for subdir in latents:
        assert "broken" not in (root / subdir / "metadata.csv").read_text()
        if all_rejected:
            import pandas as pd
            metadata = pd.read_csv(root / subdir / "metadata.csv")
            assert len(metadata) == 0 and "sha256" in metadata.columns

    # A cached report from before the coordinate audit must be regenerated.
    del report["rejected_assets"]
    report["eligible_assets"].append("broken")
    (root / "metadata_report.json").write_text(json.dumps(report))
    result = subprocess.run([sys.executable, str(SCRIPT), "metadata", "--selection",
                             str(selection), "--workers", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((root / "metadata_report.json").read_text())
    assert report["eligible_assets"] == expected
    assert "rejected_assets" in report
