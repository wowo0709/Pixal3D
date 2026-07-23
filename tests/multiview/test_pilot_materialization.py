import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from hashlib import sha256

from scripts.materialize_multiview_pilot import (
    extract_verified_pack,
    write_metadata,
)


def write_pack(path, members):
    manifest_members = []
    with tarfile.open(path, "w") as bundle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
            manifest_members.append({
                "path": name,
                "size": len(payload),
                "sha256": sha256(payload).hexdigest(),
            })
    manifest = {
        "schema_version": 2,
        "shard_id": "ABO-00000",
        "batch_id": "batch000",
        "family": "common",
        "config_hash": "c" * 64,
        "tool_commit": "deadbeef",
        "pack_sha256": sha256(path.read_bytes()).hexdigest(),
        "asset_sha256s": ["a" * 64],
        "included_asset_sha256s": ["a" * 64],
        "completed_count": 1,
        "quarantined_count": 0,
        "created_at": "2026-07-23T00:00:00+00:00",
        "validated_at": "",
        "members": manifest_members,
        "gate": "pilot",
    }
    manifest_path = path.with_suffix(".tar.manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_verified_extract_and_metadata_are_non_destructive(tmp_path):
    pack = tmp_path / "one.tar"
    manifest = write_pack(pack, {"renders_cond/" + "a" * 64 + "/000.png": b"png"})
    destination = tmp_path / "active"
    destination.mkdir()
    extract_verified_pack(pack, manifest, destination)
    write_metadata(destination / "renders_cond", ["a" * 64], {"cond_rendered": True})
    with (destination / "renders_cond" / "metadata.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"sha256": "a" * 64, "cond_rendered": "True"}]


def test_verified_extract_rejects_path_traversal(tmp_path):
    pack = tmp_path / "unsafe.tar"
    manifest = write_pack(pack, {"../escape": b"bad"})
    try:
        extract_verified_pack(pack, manifest, tmp_path / "active")
    except ValueError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("unsafe tar member was extracted")


def test_materializer_script_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/materialize_multiview_pilot.py", "--help"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
