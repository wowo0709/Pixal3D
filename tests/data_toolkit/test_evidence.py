from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.evidence import (
    MEASUREMENT_COLUMNS,
    GateEvidenceCollector,
    allocate_bytes,
    select_complete_segments,
)
from data_toolkit.pipeline.packing import PACK_FAMILIES, build_pack
from data_toolkit.pipeline.runtime import ArtifactValidationError, SafeRegistryStore


def _telemetry(timestamp, command, *, local_free_gib, action="run"):
    return {
        "timestamp": timestamp.isoformat(),
        "shard_id": "Synthetic-00000",
        "command": command,
        "action": action,
        "local_free_gib": local_free_gib,
        "cpu_percent": 20.0,
        "available_ram_gib": 200.0,
        "gpu_metrics": [{"memory_used_mib": 100.0}],
    }


def _complete_segment(start, *, local_free_gib=100.0):
    return [
        _telemetry(start, "stage_raw", local_free_gib=local_free_gib),
        _telemetry(
            start + timedelta(seconds=10),
            "build_packs",
            local_free_gib=local_free_gib - 1,
        ),
        _telemetry(
            start + timedelta(seconds=20),
            "archive_raw",
            local_free_gib=local_free_gib - 2,
        ),
        _telemetry(
            start + timedelta(seconds=30),
            "cleanup_local",
            local_free_gib=local_free_gib - 3,
        ),
    ]


def test_select_complete_segments_uses_latest_exact_complete_sequences():
    start = datetime(2026, 7, 18, tzinfo=timezone.utc)
    first = _complete_segment(start)
    incomplete = [
        _telemetry(
            start + timedelta(minutes=1),
            "stage_raw",
            local_free_gib=90.0,
        ),
        _telemetry(
            start + timedelta(minutes=1, seconds=10),
            "build_packs",
            local_free_gib=89.0,
        ),
    ]
    latest = _complete_segment(start + timedelta(minutes=2), local_free_gib=80.0)

    assert select_complete_segments(first + incomplete + latest, 1) == (tuple(latest),)


def test_select_complete_segments_rejects_missing_boundaries():
    start = datetime(2026, 7, 18, tzinfo=timezone.utc)
    records = _complete_segment(start)[:-1]

    with pytest.raises(ArtifactValidationError, match="complete telemetry"):
        select_complete_segments(records, 1)


def test_allocate_bytes_preserves_exact_size_and_membership():
    assets = ("a" * 64, "b" * 64, "c" * 64)

    result = allocate_bytes(10, assets)

    assert result == {"a" * 64: 4, "b" * 64: 3, "c" * 64: 3}
    assert sum(result.values()) == 10
    assert "d" * 64 not in result


def _family_directory(family):
    if family == "common":
        return Path("common")
    prefix, resolution = family.split("-", 1)
    return Path(
        {"SS": "ss", "shape": "shape", "PBR": "pbr"}[prefix],
        resolution,
    )


def _publish_smoke_fixture(config, assets):
    gate = "smoke"
    source = "Synthetic"
    shard = "Synthetic-00000"
    batch = "batch000"
    completed = (assets[0],)
    batch_payload = "".join(f"{asset}\n" for asset in assets).encode("ascii")
    batch_sha = sha256(batch_payload).hexdigest()
    frozen = (
        config.paths.data2_root
        / "control/qualification/smoke/shards"
        / source
        / shard
    )
    frozen.mkdir(parents=True)
    (frozen / f"{batch}.txt").write_bytes(batch_payload)
    (frozen / "batches.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "gate": gate,
                "source": source,
                "shard_id": shard,
                "config_hash": config.config_hash(),
                "canonical_shard_sha256": batch_sha,
                "scope_sha256": batch_sha,
                "batches": [
                    {"name": f"{batch}.txt", "count": 2, "sha256": batch_sha}
                ],
            }
        )
    )

    prepared = config.paths.data2_root / "prepared"
    prefix = Path("qualification/smoke")
    empty_root = config.paths.local_root / "empty-pack-root"
    empty_root.mkdir(parents=True)
    validated_at = datetime.now(timezone.utc).isoformat()
    index_entries = {}
    for family in PACK_FAMILIES:
        relative = (
            prefix
            / _family_directory(family)
            / source
            / shard
            / f"{batch}.tar"
        )
        output = prepared / relative
        manifest = build_pack(
            empty_root,
            [],
            output,
            shard,
            batch_id=batch,
            family=family,
            config_hash=config.config_hash(),
            tool_commit="test-commit",
            asset_sha256s=assets,
            included_asset_sha256s=completed,
            completed_count=1,
            quarantined_count=1,
            gate=gate,
        )
        manifest_path = output.with_suffix(".tar.manifest.json")
        manifest_value = asdict(manifest)
        manifest_value["validated_at"] = validated_at
        manifest_path.write_text(json.dumps(manifest_value))
        index_entries[family] = {
            "pack": relative.as_posix(),
            "pack_sha256": sha256(output.read_bytes()).hexdigest(),
            "manifest": relative.with_suffix(".tar.manifest.json").as_posix(),
            "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        }
    index = prepared / prefix / "index" / source / f"{shard}.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps(
            {
                "gate": gate,
                "source": source,
                "shard_id": shard,
                "batches": {batch: index_entries},
            }
        )
    )

    archive = (
        config.paths.data3_root
        / "archive/qualification/smoke/raw"
        / source
        / shard
        / f"{batch}.tar"
    )
    manifest = build_pack(
        empty_root,
        [],
        archive,
        shard,
        batch_id=batch,
        family="raw",
        config_hash=config.config_hash(),
        tool_commit="test-commit",
        asset_sha256s=assets,
        included_asset_sha256s=completed,
        completed_count=1,
        quarantined_count=1,
        gate=gate,
    )
    archive_manifest = archive.with_suffix(".tar.manifest.json")
    manifest_value = asdict(manifest)
    manifest_value["validated_at"] = validated_at
    archive_manifest.write_text(json.dumps(manifest_value))

    ledger = (
        config.paths.data2_root
        / "control/qualification/smoke/quality"
        / source
        / f"{shard}.json"
    )
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "source": source,
                "shard_id": shard,
                "gate": gate,
                "batches": {
                    batch: {
                        "instances_sha256": batch_sha,
                        "admitted_prefix": 2,
                    }
                },
                "entries": [
                    {
                        "batch_id": batch,
                        "position": 0,
                        "asset_sha": assets[0],
                        "outcome": "completed",
                    },
                    {
                        "batch_id": batch,
                        "position": 1,
                        "asset_sha": assets[1],
                        "outcome": "failure",
                    },
                ],
                "quarantine": {
                    assets[1]: {
                        "category": "provider_asset_unavailable",
                        "stage": "download",
                        "reason": "fixture provider miss",
                        "attempts": 3,
                    }
                },
                "family_exclusions": {},
            }
        )
    )

    start = datetime.now(timezone.utc) - timedelta(minutes=1)
    telemetry = (
        config.paths.data2_root / "control/telemetry/resources.jsonl"
    )
    telemetry.parent.mkdir(parents=True)
    telemetry.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in _complete_segment(start)
        )
    )


def test_collects_checksum_bound_evidence_from_publications_and_ledger(
    synthetic_config,
):
    config = load_config(synthetic_config)
    registry = SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    ).load()
    assets = tuple(sorted(registry["sha256"]))
    _publish_smoke_fixture(config, assets)

    measurements_path, fp16_path, telemetry_path, manifest_path = (
        GateEvidenceCollector(config).collect("smoke")
    )

    measurements = pd.read_csv(measurements_path, dtype={"sha256": str})
    assert tuple(measurements.columns) == MEASUREMENT_COLUMNS
    by_asset = measurements.set_index("sha256")
    assert by_asset.loc[assets[0], "outcome"] == "completed"
    assert by_asset.loc[assets[0], "failure_category"] == "none"
    assert by_asset.loc[assets[1], "outcome"] == "failure"
    assert (
        by_asset.loc[assets[1], "failure_category"]
        == "provider_asset_unavailable"
    )
    assert by_asset.loc[assets[1], "final_data2_bytes"] == 0
    assert by_asset.loc[assets[1], "final_data3_bytes"] == 0
    assert by_asset.loc[assets[0], "final_data2_bytes"] > 0
    assert by_asset.loc[assets[0], "final_data3_bytes"] > 0
    assert set(by_asset["final_local_bytes"]) == {0}

    fp16 = pd.read_csv(fp16_path)
    assert fp16.empty
    assert set(fp16.columns) == {
        "sha256",
        "family",
        "resolution",
        "fp16_abs_error",
        "coordinates_match",
        "fp16_finite",
        "decode_degradation_percent",
    }
    telemetry_rows = [
        json.loads(line) for line in telemetry_path.read_text().splitlines()
    ]
    assert {row["gate"] for row in telemetry_rows} == {"smoke"}
    assert {row["source"] for row in telemetry_rows} == {"Synthetic"}

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["artifact_type"] == "gate_evidence_manifest"
    assert manifest["config_hash"] == config.config_hash()
    assert manifest["gate"] == "smoke"
    assert manifest["artifacts"] == {
        "measurements_sha256": sha256(measurements_path.read_bytes()).hexdigest(),
        "fp16_sha256": sha256(fp16_path.read_bytes()).hexdigest(),
        "telemetry_sha256": sha256(telemetry_path.read_bytes()).hexdigest(),
    }
