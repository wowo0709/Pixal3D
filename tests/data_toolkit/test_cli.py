from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from data_toolkit.pipeline.cli import main, parser
from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.orchestrator import (
    EscalationCategory,
    EscalationReport,
    PipelineStopped,
)
from data_toolkit.pipeline.packing import PACK_FAMILIES, build_pack
from data_toolkit.pipeline.reporting import ReportValidationError, write_report
from data_toolkit.pipeline.runtime import (
    ArtifactValidationError,
    CanonicalRegistryBuilder,
    FrozenReferenceCounter,
    PilotArtifactReader,
    NoFollowTelemetryWriter,
    RuntimeReportBuilder,
    SafeRegistryStore,
    build_mutating_services,
    read_gate_report,
    validate_family_memberships,
)
from test_reporting import CONFIG_HASH, valid_report_payload


def registry_source_inputs(config, *, partition="training"):
    sources = (
        config.sources
        if partition == "training"
        else config.evaluation_sources
    )
    return {
        source: {
            "path": f"/immutable/{source}.csv",
            "sha256": f"{index + 1:064x}",
            "rows": 1,
        }
        for index, source in enumerate(sources)
    }


def test_audit_rejects_pbr_identity_missing_from_matching_shape(tmp_config):
    config = load_config(tmp_config)
    included = {family: set() for family in PACK_FAMILIES}
    included["common"] = {"a" * 64, "b" * 64}
    included["shape-256"] = {"a" * 64}
    included["PBR-256"] = {"b" * 64}

    with pytest.raises(ArtifactValidationError, match="PBR-256.*shape-256"):
        validate_family_memberships(included, config)


def valid_hardware_payload(config_hash):
    return {
        "schema_version": 2,
        "artifact_type": "hardware_preflight_evidence",
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": {
            "cuda_version": "12.8",
            "torch_version": "2.8.0+cu128",
            "blender_version": "4.5.1",
            "optix_enabled": True,
        },
        "gpus": [
            {
                "index": index,
                "name": f"GPU-{index}",
                "cuda_visible_device": str(index),
                "cycles_device": "OPTIX",
                "cpu_fallback_detected": False,
                "cube_render_sha256": f"{index + 1:064x}",
            }
            for index in range(7)
        ],
        "storage": {
            root: {
                "fixture_bytes": 10 * 1024**3,
                "write_elapsed_seconds": 10.0,
                "read_elapsed_seconds": 8.0,
                "write_sha256": "c" * 64,
                "read_sha256": "c" * 64,
                "total_bytes": 100 * 1024**4,
                "free_bytes_before": 50 * 1024**4,
                "free_bytes_after": 50 * 1024**4,
                "fixture_removed": True,
            }
            for root in ("local", "data2", "data3")
        },
        "source_measurements": {
            source: [100, 150, 200, 250]
            for source in (
                "ObjaverseXL_sketchfab",
                "ObjaverseXL_github",
                "ABO",
                "HSSD",
                "3D-FUTURE",
            )
        },
    }


def _family_directory(family):
    if family == "common":
        return Path("common")
    prefix, resolution = family.split("-", 1)
    return Path({"SS": "ss", "shape": "shape", "PBR": "pbr"}[prefix], resolution)


def write_complete_gate_evidence(config, gate):
    training_rows = []
    source_assets = {}
    next_asset = 1
    for source in config.sources:
        assets = tuple(f"{index:064x}" for index in range(next_asset, next_asset + 20))
        next_asset += 20
        source_assets[source] = assets
        shard = f"{source}-00000"
        for position, asset in enumerate(assets):
            training_rows.append(
                {
                    "sha256": asset,
                    "owner_source": source,
                    "shard_id": shard,
                    "split": "validation" if position == 0 else "train",
                    "local_path": f"raw/{source}/{asset}.glb",
                }
            )
    training = pd.DataFrame(training_rows)
    SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    ).save(training, source_inputs=registry_source_inputs(config))
    evaluation_asset = "f" * 64
    SafeRegistryStore(
        config.paths.data2_root / "control/evaluation_assets.parquet",
        config,
        partition="evaluation",
    ).save(
        pd.DataFrame(
            {
                "sha256": [evaluation_asset],
                "owner_source": [config.evaluation_sources[0]],
                "shard_id": [f"{config.evaluation_sources[0]}-00000"],
                "split": ["evaluation"],
                "local_path": ["raw/Toys4k/evaluation.blend"],
            }
        ),
        source_inputs=registry_source_inputs(config, partition="evaluation"),
    )

    prepared = config.paths.data2_root / "prepared"
    prefix = Path() if gate == "production" else Path("qualification", gate)
    empty_root = config.paths.local_root / "empty-pack-root"
    empty_root.mkdir(parents=True)
    created_at = datetime.now(timezone.utc).isoformat()
    for source, assets in source_assets.items():
        shard = f"{source}-00000"
        write_frozen_batch(config, source, shard, "batch000", assets, gate=gate)
        entries = {}
        for family in PACK_FAMILIES:
            relative = (
                prefix
                / _family_directory(family)
                / source
                / shard
                / "batch000.tar"
            )
            output = prepared / relative
            manifest = build_pack(
                empty_root,
                [],
                output,
                shard,
                batch_id="batch000",
                family=family,
                config_hash=config.config_hash(),
                tool_commit="test-commit",
                asset_sha256s=assets,
                completed_count=len(assets),
                quarantined_count=0,
                gate=gate,
            )
            manifest_path = output.with_suffix(".tar.manifest.json")
            manifest_value = asdict(manifest)
            manifest_value["validated_at"] = created_at
            manifest_path.write_text(json.dumps(manifest_value))
            entries[family] = {
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
                    "batches": {"batch000": entries},
                }
            )
        )

        archive_root = config.paths.data3_root / "archive"
        archive_root = (
            archive_root / "raw"
            if gate == "production"
            else archive_root / "qualification" / gate / "raw"
        )
        archive = archive_root / source / shard / "batch000.tar"
        manifest = build_pack(
            empty_root,
            [],
            archive,
            shard,
            batch_id="batch000",
            family="raw",
            config_hash=config.config_hash(),
            tool_commit="test-commit",
            asset_sha256s=assets,
            completed_count=len(assets),
            quarantined_count=0,
            gate=gate,
        )
        manifest_value = asdict(manifest)
        manifest_value["validated_at"] = created_at
        archive.with_suffix(".tar.manifest.json").write_text(
            json.dumps(manifest_value)
        )

    hardware_input = config.paths.data2_root / "control/report_inputs/hardware.json"
    hardware_input.parent.mkdir(parents=True, exist_ok=True)
    hardware_input.write_text(json.dumps(valid_hardware_payload(config.config_hash())))
    RuntimeReportBuilder(config)(None, True)

    measurements = pd.DataFrame(
        {
            "sha256": training["sha256"],
            "source": training["owner_source"],
            "shard_id": training["shard_id"],
            "outcome": "completed",
            "failure_category": "none",
            "elapsed_seconds": 10.0,
            "peak_local_bytes": 100,
            "final_local_bytes": 50,
            "final_data2_bytes": 80,
            "final_data3_bytes": 20,
        }
    )
    fp16_rows = []
    if config.targets.latent_dtype == "float16":
        for family in ("shape", "PBR"):
            for resolution in (256, 512, 1024):
                for asset in training["sha256"]:
                    fp16_rows.append(
                        {
                            "sha256": asset,
                            "family": family,
                            "resolution": resolution,
                            "fp16_abs_error": 0.001,
                            "coordinates_match": True,
                            "fp16_finite": True,
                            "decode_degradation_percent": 0.05,
                        }
                    )
    fp16 = pd.DataFrame(
        fp16_rows,
        columns=(
            "sha256",
            "family",
            "resolution",
            "fp16_abs_error",
            "coordinates_match",
            "fp16_finite",
            "decode_degradation_percent",
        ),
    )
    telemetry = "".join(
        json.dumps(
            {
                "gate": gate,
                "source": source,
                "shard_id": f"{source}-00000",
                "timestamp": created_at,
                "cpu_percent": 20.0,
                "available_ram_gib": 200.0,
                "gpu_metrics": [{"memory_used_mib": 100.0}],
            }
        )
        + "\n"
        for source in config.sources
    ).encode()
    evidence_root = config.paths.data2_root / "control/report_evidence" / gate
    evidence_root.mkdir(parents=True)
    payloads = {
        "measurements.csv": measurements.to_csv(index=False).encode(),
        "fp16.csv": fp16.to_csv(index=False).encode(),
        "telemetry.jsonl": telemetry,
    }
    for name, payload in payloads.items():
        (evidence_root / name).write_bytes(payload)
    candidate = {
        "schema_version": 2,
        "artifact_type": "gate_evidence_manifest",
        "config_hash": config.config_hash(),
        "gate": gate,
        "created_at": created_at,
        "artifacts": {
            "measurements_sha256": sha256(payloads["measurements.csv"]).hexdigest(),
            "fp16_sha256": sha256(payloads["fp16.csv"]).hexdigest(),
            "telemetry_sha256": sha256(payloads["telemetry.jsonl"]).hexdigest(),
        },
    }
    (config.paths.data2_root / "control/report_inputs" / f"{gate}.json").write_text(
        json.dumps(candidate)
    )


def test_production_gate_derives_pass_and_complete_handoff(tmp_config):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "production")

    report_path, _ = RuntimeReportBuilder(config)("production", True)
    report = json.loads(report_path.read_text())
    admitted = read_gate_report(config, "production")
    handoff = json.loads(
        (
            config.paths.data2_root / "control/splits/training_handoff.json"
        ).read_text()
    )

    assert report["decision"] == admitted["decision"] == "passed"
    assert report["handoff"]["ready"] is True
    assert set(handoff["identities"]) == {"train", "validation", "evaluation"}
    assert handoff["families"] == list(PACK_FAMILIES)
    assert len(handoff["packs"]) == len(config.sources) * len(PACK_FAMILIES)


def test_gate_audit_requires_ledger_evidence_for_family_exclusion(
    tmp_config,
):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "production")
    source = config.sources[0]
    shard = f"{source}-00000"
    prepared = config.paths.data2_root / "prepared"
    index_path = prepared / "index" / source / f"{shard}.json"
    index = json.loads(index_path.read_text())
    entry = index["batches"]["batch000"]["PBR-256"]
    manifest_path = prepared / entry["manifest"]
    manifest = json.loads(manifest_path.read_text())
    manifest["included_asset_sha256s"] = manifest[
        "included_asset_sha256s"
    ][:-1]
    manifest["completed_count"] -= 1
    manifest["quarantined_count"] += 1
    manifest_path.write_text(json.dumps(manifest))
    entry["manifest_sha256"] = sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    index_path.write_text(json.dumps(index))

    with pytest.raises(ArtifactValidationError, match="family exclusion evidence"):
        RuntimeReportBuilder(config)("production", True)


def test_gate_report_derives_family_counts_from_manifest_identities(
    tmp_config,
):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "production")
    source = config.sources[0]
    shard = f"{source}-00000"
    prepared = config.paths.data2_root / "prepared"
    index_path = prepared / "index" / source / f"{shard}.json"
    index = json.loads(index_path.read_text())
    entry = index["batches"]["batch000"]["PBR-256"]
    manifest_path = prepared / entry["manifest"]
    manifest = json.loads(manifest_path.read_text())
    excluded = manifest["included_asset_sha256s"][-1]
    manifest["included_asset_sha256s"] = manifest[
        "included_asset_sha256s"
    ][:-1]
    manifest["completed_count"] -= 1
    manifest["quarantined_count"] += 1
    manifest_path.write_text(json.dumps(manifest))
    entry["manifest_sha256"] = sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    index_path.write_text(json.dumps(index))
    record = {
        "category": "unsupported_shader",
        "stage": "dump_pbr",
        "reason": "Material is not supported",
        "attempts": 1,
    }
    quality_path = (
        config.paths.data2_root / "control/quality" / source / f"{shard}.json"
    )
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "source": source,
                "shard_id": shard,
                "gate": "production",
                "batches": {},
                "entries": [],
                "quarantine": {},
                "family_exclusions": {
                    excluded: {"PBR-256": record}
                },
            }
        )
    )

    report_path, _ = RuntimeReportBuilder(config)("production", True)
    report = json.loads(report_path.read_text())
    handoff = json.loads(
        (
            config.paths.data2_root
            / "control/splits/training_handoff.json"
        ).read_text()
    )

    total_assets = len(config.sources) * 20
    assert report["quality"]["failures"] == 0
    assert report["handoff"]["family_counts"]["PBR-256"] == {
        "included": total_assets - 1,
        "excluded": 1,
    }
    assert handoff["family_counts"] == report["handoff"]["family_counts"]


def test_failed_production_republication_revokes_previous_handoff(tmp_config):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "production")
    builder = RuntimeReportBuilder(config)
    builder("production", True)
    handoff_path = (
        config.paths.data2_root / "control/splits/training_handoff.json"
    )
    assert handoff_path.is_file()

    measurements_path = (
        config.paths.data2_root
        / "control/report_evidence/production/measurements.csv"
    )
    measurements = pd.read_csv(measurements_path, dtype={"sha256": str})
    measurements["outcome"] = "failure"
    measurements["failure_category"] = "quality"
    measurements_path.write_text(measurements.to_csv(index=False))
    candidate_path = (
        config.paths.data2_root / "control/report_inputs/production.json"
    )
    candidate = json.loads(candidate_path.read_text())
    candidate["artifacts"]["measurements_sha256"] = sha256(
        measurements_path.read_bytes()
    ).hexdigest()
    candidate_path.write_text(json.dumps(candidate))

    report_path, _ = builder("production")

    assert json.loads(report_path.read_text())["decision"] == "failed"
    assert not handoff_path.exists()


def test_production_report_publication_failure_leaves_no_handoff(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "production")
    handoff_path = (
        config.paths.data2_root / "control/splits/training_handoff.json"
    )
    report_path = (
        config.paths.data2_root / "control/reports/gates/production.json"
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.write_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ReportValidationError("publication failed")
        ),
    )

    with pytest.raises(ReportValidationError, match="publication failed"):
        RuntimeReportBuilder(config)("production")

    assert not handoff_path.exists()
    assert not report_path.exists()


def test_gate_telemetry_must_cover_every_frozen_scope(tmp_config):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "production")
    evidence = (
        config.paths.data2_root
        / "control/report_evidence/production/telemetry.jsonl"
    )
    lines = evidence.read_bytes().splitlines(keepends=True)
    evidence.write_bytes(b"".join(lines[:-1]))
    candidate_path = (
        config.paths.data2_root / "control/report_inputs/production.json"
    )
    candidate = json.loads(candidate_path.read_text())
    candidate["artifacts"]["telemetry_sha256"] = sha256(
        evidence.read_bytes()
    ).hexdigest()
    candidate_path.write_text(json.dumps(candidate))

    with pytest.raises(ArtifactValidationError, match="telemetry"):
        RuntimeReportBuilder(config)("production")


def test_gate_telemetry_can_predate_fresh_evidence_manifest(tmp_config):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "production")
    evidence = (
        config.paths.data2_root
        / "control/report_evidence/production/telemetry.jsonl"
    )
    stale = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat()
    records = [json.loads(line) for line in evidence.read_text().splitlines()]
    for record in records:
        record["timestamp"] = stale
    evidence.write_text("".join(json.dumps(record) + "\n" for record in records))
    candidate_path = (
        config.paths.data2_root / "control/report_inputs/production.json"
    )
    candidate = json.loads(candidate_path.read_text())
    candidate["artifacts"]["telemetry_sha256"] = sha256(
        evidence.read_bytes()
    ).hexdigest()
    candidate_path.write_text(json.dumps(candidate))

    report_path, _ = RuntimeReportBuilder(config)("production")

    assert json.loads(report_path.read_text())["decision"] == "passed"


def test_smoke_gate_requires_zero_schema_failures(tmp_config):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "smoke")
    measurements_path = (
        config.paths.data2_root
        / "control/report_evidence/smoke/measurements.csv"
    )
    measurements = pd.read_csv(measurements_path, dtype={"sha256": str})
    measurements.loc[0, "outcome"] = "schema_failure"
    measurements.loc[0, "failure_category"] = "schema"
    measurements_path.write_text(measurements.to_csv(index=False))
    candidate_path = config.paths.data2_root / "control/report_inputs/smoke.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["artifacts"]["measurements_sha256"] = sha256(
        measurements_path.read_bytes()
    ).hexdigest()
    candidate_path.write_text(json.dumps(candidate))

    report_path, _ = RuntimeReportBuilder(config)("smoke")

    assert json.loads(report_path.read_text())["decision"] == "failed"


def test_smoke_gate_allows_ten_percent_asset_failures(tmp_config):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "smoke")
    measurements_path = (
        config.paths.data2_root
        / "control/report_evidence/smoke/measurements.csv"
    )
    measurements = pd.read_csv(measurements_path, dtype={"sha256": str})
    source = "ObjaverseXL_github"
    failed = measurements.index[measurements["source"] == source][:2]
    measurements.loc[failed, "outcome"] = "failure"
    measurements.loc[failed, "failure_category"] = (
        "provider_asset_unavailable"
    )
    measurements_path.write_text(measurements.to_csv(index=False))
    candidate_path = (
        config.paths.data2_root / "control/report_inputs/smoke.json"
    )
    candidate = json.loads(candidate_path.read_text())
    candidate["artifacts"]["measurements_sha256"] = sha256(
        measurements_path.read_bytes()
    ).hexdigest()
    candidate_path.write_text(json.dumps(candidate))

    report_path, _ = RuntimeReportBuilder(config)("smoke")

    report = json.loads(report_path.read_text())
    assert report["decision"] == "passed"
    assert report["quality"]["sources"][source]["failure_rate"] == 0.1


def test_fp32_gate_accepts_header_only_fp16_evidence(tmp_config):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "smoke")
    evidence_path = (
        config.paths.data2_root
        / "control/report_evidence/smoke/fp16.csv"
    )
    evidence_path.write_text(
        "sha256,family,resolution,fp16_abs_error,coordinates_match,"
        "fp16_finite,decode_degradation_percent\n"
    )
    candidate_path = (
        config.paths.data2_root / "control/report_inputs/smoke.json"
    )
    candidate = json.loads(candidate_path.read_text())
    candidate["artifacts"]["fp16_sha256"] = sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    candidate_path.write_text(json.dumps(candidate))

    report_path, _ = RuntimeReportBuilder(config)("smoke")

    report = json.loads(report_path.read_text())
    assert report["decision"] == "passed"
    assert report["fp16_parity"] == {
        "dtype": "float32",
        "required": False,
        "passed": True,
        "families": {},
    }


@pytest.mark.parametrize("artifact", ("frozen", "index", "manifest"))
def test_gate_rejects_unindexed_or_noncanonical_publication_state(
    tmp_config, artifact
):
    config = load_config(tmp_config)
    write_complete_gate_evidence(config, "production")
    source = config.sources[0]
    shard = f"{source}-00000"
    if artifact == "frozen":
        path = (
            config.paths.data2_root
            / "control/shards"
            / source
            / shard
            / "batch999.txt"
        )
        path.write_text("f" * 64 + "\n")
    elif artifact == "index":
        path = (
            config.paths.data2_root
            / "prepared/index"
            / source
            / f"{shard}.json"
        )
        value = json.loads(path.read_text())
        value["batches"]["batch999"] = value["batches"]["batch000"]
        path.write_text(json.dumps(value))
    else:
        path = (
            config.paths.data2_root
            / "prepared/common"
            / source
            / shard
            / "batch000.tar.manifest.json"
        )
        value = json.loads(path.read_text())
        value["operator_passed"] = True
        path.write_text(json.dumps(value))
        index_path = (
            config.paths.data2_root
            / "prepared/index"
            / source
            / f"{shard}.json"
        )
        index = json.loads(index_path.read_text())
        index["batches"]["batch000"]["common"]["manifest_sha256"] = sha256(
            path.read_bytes()
        ).hexdigest()
        index_path.write_text(json.dumps(index))

    with pytest.raises(ArtifactValidationError):
        RuntimeReportBuilder(config)("production")


def test_plan_is_read_only(tmp_config, capsys):
    assert main(["plan", "--config", str(tmp_config), "--gate", "smoke"]) == 0
    output = capsys.readouterr().out

    assert "dump_mesh" in output and "build_packs" in output
    assert not tmp_config.parent.joinpath("data2").exists()
    assert not tmp_config.parent.joinpath("data3").exists()
    assert not tmp_config.parent.joinpath("local").exists()


def test_plan_calls_service_with_explicit_freeze_false(
    tmp_config, monkeypatch
):
    calls = []

    class Services:
        def plan(self, *args, **kwargs):
            calls.append((args, kwargs))
            return ("dump_mesh",)

    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_read_only_services",
        lambda config: Services(),
    )

    assert main(["plan", "--config", str(tmp_config), "--gate", "smoke"]) == 0
    assert calls == [(('smoke', None, None, None), {"freeze": False})]


@pytest.mark.parametrize(
    "argv",
    [
        ["plan", "--gate", "smoke", "--source", "ABO"],
        ["plan", "--gate", "smoke", "--count", "2"],
        [
            "run",
            "--gate",
            "production",
            "--source",
            "ABO",
            "--shard",
            "ABO-00000",
            "--count",
            "2",
        ],
        ["resume", "--source", "ABO"],
        ["audit", "--shard", "ABO-00000"],
        ["resume", "--source", "ABO", "--shard", "ABO-00000"],
        ["audit", "--source", "ABO", "--shard", "ABO-00000"],
        ["report"],
    ],
)
def test_cli_rejects_invalid_command_combinations(tmp_config, argv):
    with pytest.raises(SystemExit) as raised:
        parser().parse_args([*argv, "--config", str(tmp_config)])

    assert raised.value.code == 2


def test_hardware_report_command_does_not_require_gate(tmp_config):
    args = parser().parse_args(
        ["report", "--hardware-check", "--config", str(tmp_config)]
    )

    assert args.gate is None
    assert args.hardware_check is True


def test_hardware_preflight_collects_explicit_bootstrap_reservation(
    tmp_config, monkeypatch, capsys
):
    calls = []
    evidence = tmp_config.parent / "data2/control/report_inputs/hardware.json"

    def collect(config, *, bootstrap_peak_local_bytes):
        calls.append((config.config_hash(), bootstrap_peak_local_bytes))
        evidence.parent.mkdir(parents=True)
        evidence.write_text("{}")
        return evidence

    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.collect_hardware_preflight", collect
    )

    assert main(
        [
            "hardware-preflight",
            "--config",
            str(tmp_config),
            "--bootstrap-peak-local-gib",
            "350",
        ]
    ) == 0
    assert calls == [(load_config(tmp_config).config_hash(), 350 * 1024**3)]
    assert str(evidence) in capsys.readouterr().out


def test_evidence_command_does_not_initialize_mutating_runtime(
    tmp_config, monkeypatch, capsys
):
    calls = []
    paths = tuple(tmp_config.parent / f"evidence-{index}" for index in range(4))

    class Collector:
        def __init__(self, config):
            calls.append(("init", config.config_hash()))

        def collect(self, gate):
            calls.append(("collect", gate))
            return paths

    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.GateEvidenceCollector", Collector
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: pytest.fail("evidence initialized mutating services"),
    )

    assert (
        main(
            [
                "evidence",
                "--config",
                str(tmp_config),
                "--gate",
                "smoke",
            ]
        )
        == 0
    )
    assert calls == [
        ("init", load_config(tmp_config).config_hash()),
        ("collect", "smoke"),
    ]
    assert capsys.readouterr().out.splitlines() == [str(path) for path in paths]


def test_cli_rejects_unknown_source_and_mismatched_shard(tmp_config):
    assert (
        main(
            [
                "plan",
                "--config",
                str(tmp_config),
                "--gate",
                "smoke",
                "--source",
                "Unknown",
                "--shard",
                "Unknown-00000",
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "run",
                "--config",
                str(tmp_config),
                "--gate",
                "smoke",
                "--source",
                "ABO",
                "--shard",
                "HSSD-00000",
            ]
        )
        == 2
    )


@pytest.mark.parametrize("command", ["plan", "run", "resume", "audit"])
def test_cli_rejects_evaluation_source_as_training_scope(
    tmp_config, command
):
    arguments = [
        command,
        "--config",
        str(tmp_config),
        "--gate",
        "smoke",
        "--source",
        "Toys4k",
        "--shard",
        "Toys4k-00000",
    ]
    if command == "plan":
        arguments.extend(("--count", "1"))

    assert main(arguments) == 2


def test_resume_and_audit_dispatch_explicit_gate(tmp_config, monkeypatch):
    calls = []

    class Runtime:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @property
        def services(self):
            return self

        def resume(self, *args):
            calls.append(("resume", args))

        def audit(self, *args):
            calls.append(("audit", args))

    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: Runtime(),
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.read_gate_report",
        lambda config, gate: {"config_hash": config.config_hash()},
    )

    for command in ("resume", "audit"):
        assert main(
            [
                command,
                "--config",
                str(tmp_config),
                "--gate",
                "pilot",
                "--source",
                "ABO",
                "--shard",
                "ABO-00000",
            ]
        ) == 0

    assert calls == [
        ("resume", ("pilot", "ABO", "ABO-00000")),
        ("audit", ("pilot", "ABO", "ABO-00000")),
    ]


def test_full_run_dispatches_inside_mutating_runtime(tmp_config, monkeypatch):
    calls = []

    class Runtime:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @property
        def services(self):
            return "held-services"

    class Runner:
        def __init__(self, config, services):
            calls.append(("init", config.config_hash(), services))

        def run(self):
            calls.append(("run",))

    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: Runtime(),
    )
    monkeypatch.setattr("data_toolkit.pipeline.cli.FullProductionRunner", Runner)

    assert main(["full-run", "--config", str(tmp_config)]) == 0
    assert calls == [
        ("init", load_config(tmp_config).config_hash(), "held-services"),
        ("run",),
    ]


@pytest.mark.parametrize(
    ("gate", "expected_reports"),
    [("pilot", ["smoke"]), ("production", ["smoke", "pilot"])],
)
def test_resume_requires_current_prerequisite_gates(
    gate, expected_reports, tmp_config, monkeypatch
):
    reports = []
    resumed = []

    class Runtime:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @property
        def services(self):
            return self

        def resume(self, *args):
            resumed.append(args)

    def read(config, report_gate):
        reports.append(report_gate)
        return {"config_hash": config.config_hash()}

    monkeypatch.setattr("data_toolkit.pipeline.cli.read_gate_report", read)
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: Runtime(),
    )

    assert main(
        [
            "resume",
            "--config",
            str(tmp_config),
            "--gate",
            gate,
            "--source",
            "ABO",
            "--shard",
            "ABO-00000",
        ]
    ) == 0
    assert reports == expected_reports
    assert resumed == [(gate, "ABO", "ABO-00000")]


@pytest.mark.parametrize("gate", ["pilot", "production"])
@pytest.mark.parametrize("invalid_evidence", ["missing", "stale", "failed"])
def test_resume_rejects_invalid_smoke_evidence_before_runtime(
    gate, invalid_evidence, tmp_config, monkeypatch
):
    built = []

    def reject(config, gate):
        raise ArtifactValidationError(f"{invalid_evidence} {gate} evidence")

    monkeypatch.setattr("data_toolkit.pipeline.cli.read_gate_report", reject)
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: built.append(config),
    )

    assert main(
        [
            "resume",
            "--config",
            str(tmp_config),
            "--gate",
            gate,
            "--source",
            "ABO",
            "--shard",
            "ABO-00000",
        ]
    ) == 2
    assert built == []


@pytest.mark.parametrize("invalid_evidence", ["missing", "stale", "failed"])
def test_production_resume_rejects_invalid_pilot_evidence_before_runtime(
    invalid_evidence, tmp_config, monkeypatch
):
    built = []

    def read(config, gate):
        if gate == "pilot":
            raise ArtifactValidationError(f"{invalid_evidence} pilot evidence")
        return {"config_hash": config.config_hash()}

    monkeypatch.setattr("data_toolkit.pipeline.cli.read_gate_report", read)
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: built.append(config),
    )

    assert main(
        [
            "resume",
            "--config",
            str(tmp_config),
            "--gate",
            "production",
            "--source",
            "ABO",
            "--shard",
            "ABO-00000",
        ]
    ) == 2
    assert built == []


def test_production_resume_rejects_cross_config_gate_evidence_before_runtime(
    tmp_config, monkeypatch
):
    built = []

    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.read_gate_report",
        lambda config, gate: {"config_hash": gate},
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: built.append(config),
    )

    assert main(
        [
            "resume",
            "--config",
            str(tmp_config),
            "--gate",
            "production",
            "--source",
            "ABO",
            "--shard",
            "ABO-00000",
        ]
    ) == 2
    assert built == []


def test_production_requires_strict_smoke_and_pilot_gates(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    report_root = config.paths.data2_root / "control/reports/gates"
    report_root.mkdir(parents=True)
    smoke = valid_report_payload("smoke")
    pilot = valid_report_payload("pilot")
    smoke["config_hash"] = config.config_hash()
    pilot["config_hash"] = config.config_hash()
    write_report(report_root, "smoke", smoke)
    write_report(report_root, "pilot", pilot)
    pilot["hardware"]["checked"] = "truthy"
    (report_root / "pilot.json").write_text(json.dumps(pilot))
    built = []
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: built.append(config),
    )

    result = main(
        [
            "run",
            "--config",
            str(tmp_config),
            "--gate",
            "production",
            "--source",
            "ABO",
            "--shard",
            "ABO-00000",
        ]
    )

    assert result == 2
    assert built == []


def test_production_gate_reports_must_share_active_config_hash(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    report_root = config.paths.data2_root / "control/reports/gates"
    report_root.mkdir(parents=True)
    for gate in ("smoke", "pilot"):
        payload = valid_report_payload(gate)
        payload["config_hash"] = config.config_hash()
        if gate == "pilot":
            payload["config_hash"] = "b" * 64
        write_report(report_root, gate, payload)
    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: pytest.fail("runtime must not be constructed"),
    )

    assert (
        main(
            [
                "run",
                "--config",
                str(tmp_config),
                "--gate",
                "production",
                "--source",
                "ABO",
                "--shard",
                "ABO-00000",
            ]
        )
        == 2
    )


def stopped(category, exit_code=99):
    report = EscalationReport(
        source="ABO",
        shard_id="ABO-00000",
        command="dump_mesh",
        category=category,
        reason="stopped",
        recent_telemetry=(),
        completed_counts={},
        safe_resume_command="resume",
        recovery_choices=(),
        created_at="2026-07-16T00:00:00+00:00",
        persistence_errors=(),
    )
    return PipelineStopped(report, exit_code)


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (EscalationCategory.INFRASTRUCTURE, 2),
        (EscalationCategory.COMMAND_FAILURE, 2),
        (EscalationCategory.RESOURCE, 3),
        (EscalationCategory.DATA_QUALITY, 4),
    ],
)
def test_cli_maps_stop_category_not_loose_exit_code(
    tmp_config, monkeypatch, category, expected
):
    class Runtime:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @property
        def services(self):
            return self

        def run(self, *args):
            raise stopped(category)

    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_mutating_services",
        lambda config: Runtime(),
    )

    assert (
        main(
            [
                "run",
                "--config",
                str(tmp_config),
                "--gate",
                "smoke",
                "--source",
                "ABO",
                "--shard",
                "ABO-00000",
            ]
        )
        == expected
    )


def test_cli_does_not_hide_programmer_errors(tmp_config, monkeypatch):
    class Services:
        def plan(self, *args, **kwargs):
            raise AssertionError("programmer defect")

    monkeypatch.setattr(
        "data_toolkit.pipeline.cli.build_read_only_services",
        lambda config: Services(),
    )

    with pytest.raises(AssertionError, match="programmer defect"):
        main(["plan", "--config", str(tmp_config), "--gate", "smoke"])


def test_pilot_reader_validates_schema_and_source(tmp_config):
    config = load_config(tmp_config)
    root = config.paths.data2_root / "control/report_inputs"
    root.mkdir(parents=True)
    payload = valid_hardware_payload(config.config_hash())
    (root / "hardware.json").write_text(json.dumps(payload))
    RuntimeReportBuilder(config)(None, True)
    reader = PilotArtifactReader(config)

    assert reader.p95_peak_local_bytes("ABO") == 243
    with pytest.raises(ArtifactValidationError, match="source"):
        reader.p95_peak_local_bytes("Unknown")


def test_pilot_reader_prefers_recomputed_pilot_measurement(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    report_root = config.paths.data2_root / "control/reports"
    gate_root = report_root / "gates"
    gate_root.mkdir(parents=True)
    (gate_root / "pilot.json").write_text("{}")
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.read_gate_report",
        lambda config, gate: {
            "capacity": {
                "sources": {"ABO": {"p95_peak_local_bytes": 300}}
            }
        },
    )

    assert PilotArtifactReader(config).p95_peak_local_bytes("ABO") == 300


def test_pilot_reader_uses_the_previous_gate_for_sizing(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    root = config.paths.data2_root / "control/report_inputs"
    root.mkdir(parents=True)
    (root / "hardware.json").write_text(
        json.dumps(valid_hardware_payload(config.config_hash()))
    )
    RuntimeReportBuilder(config)(None, True)
    reader = PilotArtifactReader(config)

    def gate_report(config, gate):
        return {
            "capacity": {
                "sources": {
                    "ABO": {
                        "p95_peak_local_bytes":
                        {"smoke": 115, "pilot": 180}[gate]
                    }
                }
            }
        }

    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.read_gate_report", gate_report
    )
    gate_root = config.paths.data2_root / "control/reports/gates"
    gate_root.mkdir(parents=True)
    (gate_root / "pilot.json").write_text("{}")
    assert reader.p95_peak_local_bytes_for_gate("ABO", "smoke") == 243
    assert reader.p95_peak_local_bytes_for_gate("ABO", "pilot") == 115
    assert reader.p95_peak_local_bytes_for_gate("ABO", "production") == 180


def test_gate_admission_recomputes_and_rejects_legacy_pass(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    report_root = config.paths.data2_root / "control/reports/gates"
    report_root.mkdir(parents=True)
    legacy = valid_report_payload("pilot")
    legacy["config_hash"] = config.config_hash()
    write_report(report_root, "pilot", legacy)
    derived = {
        "schema_version": 2,
        "report_type": "pipeline_gate_derived",
        "gate": "pilot",
        "decision": "failed",
        "config_hash": config.config_hash(),
    }
    monkeypatch.setattr(
        RuntimeReportBuilder,
        "_derive_gate",
        lambda self, gate: (derived, None, None),
    )

    with pytest.raises(ArtifactValidationError, match="held evidence"):
        read_gate_report(config, "pilot")


def test_gate_admission_verifies_canonical_training_handoff_path(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    report_root = config.paths.data2_root / "control/reports/gates"
    report_root.mkdir(parents=True)
    derived = {
        "schema_version": 2,
        "report_type": "pipeline_gate_derived",
        "gate": "production",
        "decision": "passed",
        "config_hash": config.config_hash(),
    }
    (report_root / "production.json").write_text(json.dumps(derived))
    handoff = {"artifact_type": "training_handoff"}
    handoff_payload = json.dumps(
        handoff, sort_keys=True, separators=(",", ":")
    ).encode()
    handoff_path = (
        config.paths.data2_root / "control/splits/training_handoff.json"
    )
    handoff_path.parent.mkdir(parents=True)
    handoff_path.write_bytes(handoff_payload)
    monkeypatch.setattr(
        RuntimeReportBuilder,
        "_derive_gate",
        lambda self, gate: (derived, handoff, handoff_payload),
    )

    assert read_gate_report(config, "production") == derived


def test_hardware_report_builder_validates_and_publishes(tmp_config):
    config = load_config(tmp_config)
    input_root = config.paths.data2_root / "control/report_inputs"
    input_root.mkdir(parents=True)
    (input_root / "hardware.json").write_text(
        json.dumps(valid_hardware_payload(config.config_hash()))
    )

    outputs = RuntimeReportBuilder(config)(None, True)

    assert [path.name for path in outputs] == ["hardware.json", "hardware.md"]


def test_hardware_candidate_cannot_supply_a_pass_decision(tmp_config):
    config = load_config(tmp_config)
    input_root = config.paths.data2_root / "control/report_inputs"
    input_root.mkdir(parents=True)
    payload = valid_hardware_payload(config.config_hash())
    payload["decision"] = "passed"
    (input_root / "hardware.json").write_text(json.dumps(payload))

    with pytest.raises(ArtifactValidationError, match="hardware"):
        RuntimeReportBuilder(config)(None, True)


def test_hardware_evidence_is_fresh_exact_and_math_is_derived(tmp_config):
    config = load_config(tmp_config)
    input_root = config.paths.data2_root / "control/report_inputs"
    input_root.mkdir(parents=True)
    payload = valid_hardware_payload(config.config_hash())
    (input_root / "hardware.json").write_text(json.dumps(payload))

    paths = RuntimeReportBuilder(config)(None, True)
    report = json.loads(paths[0].read_text())

    assert report["decision"] == "passed"
    assert report["gpu"]["gpu_count"] == 7
    assert report["storage"]["local"]["write_mib_per_second"] == pytest.approx(1024.0)
    assert report["pilot_sizing"]["sources"]["ABO"]["p95_peak_local_bytes"] == 243
    assert report["evidence_sha256"] == sha256(
        (input_root / "hardware.json").read_bytes()
    ).hexdigest()

    payload["created_at"] = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat()
    (input_root / "hardware.json").write_text(json.dumps(payload))
    with pytest.raises(ArtifactValidationError, match="fresh"):
        RuntimeReportBuilder(config)(None, True)


def test_hardware_inventory_allows_repeated_gpu_model_names(tmp_config):
    config = load_config(tmp_config)
    input_root = config.paths.data2_root / "control/report_inputs"
    input_root.mkdir(parents=True)
    payload = valid_hardware_payload(config.config_hash())
    for gpu in payload["gpus"]:
        gpu["name"] = "NVIDIA RTX PRO 6000 Blackwell"
    (input_root / "hardware.json").write_text(json.dumps(payload))

    report_path, _ = RuntimeReportBuilder(config)(None, True)

    assert json.loads(report_path.read_text())["decision"] == "passed"


def test_gate_candidate_cannot_supply_pass_fields(tmp_config):
    config = load_config(tmp_config)
    root = config.paths.data2_root / "control/report_inputs"
    root.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "artifact_type": "gate_evidence_manifest",
        "config_hash": config.config_hash(),
        "gate": "pilot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "measurements_sha256": "a" * 64,
            "fp16_sha256": "b" * 64,
            "telemetry_sha256": "c" * 64,
        },
        "decision": "passed",
    }
    (root / "pilot.json").write_text(json.dumps(payload))

    with pytest.raises(ArtifactValidationError, match="evidence"):
        RuntimeReportBuilder(config)("pilot")


def test_gate_evidence_digest_is_verified_before_report_math(tmp_config):
    config = load_config(tmp_config)
    input_root = config.paths.data2_root / "control/report_inputs"
    evidence_root = config.paths.data2_root / "control/report_evidence/pilot"
    input_root.mkdir(parents=True)
    evidence_root.mkdir(parents=True)
    for name in ("measurements.csv", "fp16.csv", "telemetry.jsonl"):
        (evidence_root / name).write_text("machine evidence\n")
    payload = {
        "schema_version": 2,
        "artifact_type": "gate_evidence_manifest",
        "config_hash": config.config_hash(),
        "gate": "pilot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "measurements_sha256": "a" * 64,
            "fp16_sha256": "b" * 64,
            "telemetry_sha256": "c" * 64,
        },
    }
    (input_root / "pilot.json").write_text(json.dumps(payload))

    with pytest.raises(ArtifactValidationError, match="checksum"):
        RuntimeReportBuilder(config)("pilot")

    payload["created_at"] = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat()
    payload["artifacts"] = {
        "measurements_sha256": sha256(
            (evidence_root / "measurements.csv").read_bytes()
        ).hexdigest(),
        "fp16_sha256": sha256((evidence_root / "fp16.csv").read_bytes()).hexdigest(),
        "telemetry_sha256": sha256(
            (evidence_root / "telemetry.jsonl").read_bytes()
        ).hexdigest(),
    }
    (input_root / "pilot.json").write_text(json.dumps(payload))
    with pytest.raises(ArtifactValidationError, match="fresh"):
        RuntimeReportBuilder(config)("pilot")


def test_safe_registry_roundtrip_binds_schema_checksum_and_config(tmp_config):
    config = load_config(tmp_config)
    frame = pd.DataFrame(
        {
            "sha256": ["a" * 64],
            "owner_source": ["ABO"],
            "shard_id": ["ABO-00000"],
        }
    )
    store = SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    )

    store.save(frame, source_inputs=registry_source_inputs(config))

    assert store.load().to_dict("records") == frame.to_dict("records")
    manifest = json.loads(store.manifest_path.read_text())
    manifest["config_hash"] = "b" * 64
    store.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactValidationError, match="config hash"):
        store.load()


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"sha256": ["a" * 64]}),
        pd.DataFrame(
            {
                "sha256": ["a" * 64, "a" * 64],
                "owner_source": ["ABO", "ABO"],
                "shard_id": ["ABO-00000", "ABO-00000"],
            }
        ),
        pd.DataFrame(
            {
                "sha256": ["a" * 64],
                "owner_source": ["ABO"],
                "shard_id": ["HSSD-00000"],
            }
        ),
    ],
)
def test_safe_registry_rejects_semantically_invalid_frames(tmp_config, frame):
    config = load_config(tmp_config)
    store = SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    )

    with pytest.raises(ArtifactValidationError, match="registry"):
        store.save(frame)


def test_registry_builder_uses_configured_source_order_and_no_network(
    tmp_config,
):
    config = load_config(tmp_config)
    calls = []
    frames = {
        source: pd.DataFrame(
            {
                "sha256": [f"{index + 1:064x}"],
                "file_identifier": [f"{source}.glb"],
            }
        )
        for index, source in enumerate(
            (*config.sources, *config.evaluation_sources)
        )
    }

    def loader(source):
        calls.append(source)
        return frames[source]

    builder = CanonicalRegistryBuilder(config, source_loader=loader)
    result = builder()

    assert calls == [*config.sources, *config.evaluation_sources]
    assert result["owner_source"].tolist() == list(config.sources)
    assert SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    ).load().shape[0] == len(config.sources)
    evaluation = SafeRegistryStore(
        config.paths.data2_root / "control/evaluation_assets.parquet",
        config,
        partition="evaluation",
    ).load()
    assert evaluation["owner_source"].tolist() == list(config.evaluation_sources)
    assert set(evaluation["split"]) == {"evaluation"}


def test_registry_builder_normalizes_adapter_raw_reference_paths(tmp_config):
    config = load_config(tmp_config)
    sketchfab_uid = "bGOGeXuHiDCTB33QjbQSBV6A2Fj"
    smithsonian_url = (
        "https://3d-api.si.edu/content/document/"
        "3d_package:3fa90364-e4f4-4d36-ac0b-d6f42c69bb54/"
        "2014_243_4_001-FlightSuitCharlesFBolden-"
        "RENDER-150k-2048-medium.glb"
    )
    smithsonian_uid = "7083426f-1d5f-5901-8528-c5a12dcb0c85"
    frames = {
        "ObjaverseXL_sketchfab": pd.DataFrame(
            {
                "sha256": [f"{1:064x}"],
                "file_identifier": [
                    f"https://sketchfab.com/3d-models/{sketchfab_uid}"
                ],
            }
        ),
        "ObjaverseXL_github": pd.DataFrame(
            {
                "sha256": [f"{2:064x}", f"{7:064x}"],
                "file_identifier": [
                    "https://github.com/example/repository/blob/"
                    "0123456789abcdef/models/Model #636156 With Spaces.fbx",
                    smithsonian_url,
                ],
            }
        ),
        "ABO": pd.DataFrame(
            {
                "sha256": [f"{3:064x}"],
                "file_identifier": ["3/B07YBH2SR3.glb"],
            }
        ),
        "HSSD": pd.DataFrame(
            {
                "sha256": [f"{4:064x}"],
                "file_identifier": ["objects/3/object.glb"],
            }
        ),
        "3D-FUTURE": pd.DataFrame(
            {
                "sha256": [f"{5:064x}"],
                "file_identifier": ["3D-FUTURE-model/object-id"],
            }
        ),
        "Toys4k": pd.DataFrame(
            {
                "sha256": [f"{6:064x}"],
                "file_identifier": ["hammer/hammer_001.blend"],
            }
        ),
    }

    CanonicalRegistryBuilder(
        config, source_loader=lambda source: frames[source]
    )()

    index = json.loads(
        (
            config.paths.data2_root / "control/raw_references.json"
        ).read_text()
    )["sources"]
    expected = {
        "ObjaverseXL_sketchfab": (
            f"raw/hf-objaverse-v1/by-uid/{sketchfab_uid}.glb",
        ),
        "ObjaverseXL_github": (
            "raw/github/repos/example/repository.zip",
            f"raw/smithsonian/objects/{smithsonian_uid}.glb",
        ),
        "ABO": ("raw/3dmodels/original/3/B07YBH2SR3.glb",),
        "HSSD": ("raw/objects/3/object.glb",),
        "3D-FUTURE": ("raw/3D-FUTURE-model/object-id/raw_model.obj",),
    }
    assert {
        source: tuple(paths) for source, paths in index.items()
    } == expected

    runtime_paths = (
        (
            "ObjaverseXL_sketchfab",
            "raw/hf-objaverse-v1/glbs/000-000/"
            f"{sketchfab_uid}.glb",
        ),
        (
            "ObjaverseXL_github",
            "raw/github/repos/example/repository.zip/"
            "models/Model #636156 With Spaces.fbx",
        ),
        (
            "ObjaverseXL_github",
            f"raw/smithsonian/objects/{smithsonian_uid}.glb",
        ),
        *(
            (source, paths[0])
            for source, paths in expected.items()
            if "Objaverse" not in source
        ),
    )
    counter = FrozenReferenceCounter(config)
    for source, path in runtime_paths:
        assert counter.pending_references(
            source,
            path,
            excluding_shard_id=f"{source}-00000",
            excluding_batch_id="batch000",
            gate="smoke",
        ) == 1


def test_registry_builder_refreshes_injected_input_digests_each_build(
    tmp_config,
):
    config = load_config(tmp_config)
    frames = {
        source: pd.DataFrame(
            {
                "sha256": [f"{index + 1:064x}"],
                "file_identifier": [f"{source}.glb"],
            }
        )
        for index, source in enumerate(
            (*config.sources, *config.evaluation_sources)
        )
    }
    builder = CanonicalRegistryBuilder(
        config, source_loader=lambda source: frames[source]
    )
    builder()
    manifest_path = (
        config.paths.data2_root / "control/assets.parquet.manifest.json"
    )
    first = json.loads(manifest_path.read_text())["source_inputs"]["ABO"]["sha256"]
    frames["ABO"].loc[0, "file_identifier"] = "changed.glb"

    builder()

    second = json.loads(manifest_path.read_text())["source_inputs"]["ABO"]["sha256"]
    assert second != first


def test_registry_builder_maps_known_loader_failure_but_not_programmer_defect(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)

    def unavailable(source):
        raise OSError("metadata unavailable")

    with pytest.raises(ArtifactValidationError, match="metadata unavailable"):
        CanonicalRegistryBuilder(config, source_loader=unavailable)()

    frames = {
        source: pd.DataFrame(
            {"sha256": [f"{index + 1:064x}"], "file_identifier": [source]}
        )
        for index, source in enumerate(
            (*config.sources, *config.evaluation_sources)
        )
    }
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.canonicalize_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("programmer defect")
        ),
    )

    with pytest.raises(AssertionError, match="programmer defect"):
        CanonicalRegistryBuilder(
            config, source_loader=lambda source: frames[source]
        )()


def test_registry_parquet_reader_does_not_hide_programmer_defect(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    store = SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    )
    store.save(
        pd.DataFrame(
            {
                "sha256": ["a" * 64],
                "owner_source": ["ABO"],
                "shard_id": ["ABO-00000"],
            }
        ),
        source_inputs=registry_source_inputs(config),
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.pd.read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("programmer defect")
        ),
    )

    with pytest.raises(AssertionError, match="programmer defect"):
        store.load()


def test_registry_builder_preserves_inputs_and_manifests_their_digests(
    tmp_config,
):
    config = load_config(tmp_config)
    before = {}
    for index, source in enumerate((*config.sources, *config.evaluation_sources)):
        path = config.paths.data2_root / "control/metadata" / source / "metadata.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            "sha256,file_identifier,local_path\n"
            f"{index + 1:064x},{source}.glb,raw/{source}.glb\n"
        ).encode()
        path.write_bytes(payload)
        before[path] = payload

    CanonicalRegistryBuilder(config)()

    assert {path: path.read_bytes() for path in before} == before
    manifest = json.loads(
        (
            config.paths.data2_root
            / "control/assets.parquet.manifest.json"
        ).read_text()
    )
    assert set(manifest["source_inputs"]) == set(config.sources)
    assert manifest["source_inputs"]["ABO"]["sha256"] == sha256(
        before[
            config.paths.data2_root
            / "control/metadata/ABO/metadata.csv"
        ]
    ).hexdigest()
    assert (
        config.paths.data2_root
        / "control/compatibility_metadata/training/ABO/metadata.csv"
    ).is_file()

    del manifest["source_inputs"]["ABO"]
    (
        config.paths.data2_root
        / "control/assets.parquet.manifest.json"
    ).write_text(json.dumps(manifest))
    with pytest.raises(ArtifactValidationError, match="source input"):
        SafeRegistryStore(
            config.paths.data2_root / "control/assets.parquet", config
        ).load()


def test_evaluation_overlap_is_excluded_from_training_registry(tmp_config):
    config = load_config(tmp_config)
    shared = "a" * 64
    unique = "b" * 64
    frames = {
        source: pd.DataFrame(
            {
                "sha256": [shared if source in {"ABO", "Toys4k"} else f"{index + 1:064x}"],
                "file_identifier": [f"{source}.glb"],
                "local_path": [f"raw/{source}.glb"],
            }
        )
        for index, source in enumerate((*config.sources, *config.evaluation_sources))
    }
    frames["HSSD"] = pd.DataFrame(
        {
            "sha256": [unique],
            "file_identifier": ["unique.glb"],
            "local_path": ["raw/unique.glb"],
        }
    )

    training = CanonicalRegistryBuilder(
        config, source_loader=lambda source: frames[source]
    )()

    assert shared not in set(training["sha256"])
    evaluation = SafeRegistryStore(
        config.paths.data2_root / "control/evaluation_assets.parquet",
        config,
        partition="evaluation",
    ).load()
    assert shared in set(evaluation["sha256"])


def test_reference_index_retains_future_unfrozen_shared_zip(tmp_config):
    config = load_config(tmp_config)
    source = "ObjaverseXL_github"
    first, future = "a" * 64, "b" * 64
    training = pd.DataFrame(
        {
            "sha256": [first, future],
            "owner_source": [source, source],
            "shard_id": [f"{source}-00000", f"{source}-00001"],
            "split": ["train", "train"],
            "local_path": [
                "raw/github/repos/repo.zip/models/first.glb",
                "raw/github/repos/repo.zip/models/future.glb",
            ],
        }
    )
    store = SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    )
    store.save(training, source_inputs=registry_source_inputs(config))
    manifest = json.loads(store.manifest_path.read_text())
    reference = {
        "schema_version": 2,
        "artifact_type": "canonical_raw_reference_index",
        "config_hash": config.config_hash(),
        "training_registry_sha256": manifest["sha256"],
        "sources": {
            configured_source: ({
                "raw/github/repos/repo.zip": [
                    {"sha256": first, "shard_id": f"{source}-00000"},
                    {"sha256": future, "shard_id": f"{source}-00001"},
                ]
            } if configured_source == source else {})
            for configured_source in config.sources
        },
    }
    index_path = config.paths.data2_root / "control/raw_references.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(reference))
    write_frozen_batch(
        config, source, f"{source}-00000", "batch000", (first,)
    )

    counter = FrozenReferenceCounter(config)

    assert counter.pending_references(
        source,
        "raw/github/repos/repo.zip",
        excluding_shard_id=f"{source}-00000",
        excluding_batch_id="batch000",
        gate="production",
    ) == 1


def test_reference_index_rejects_forged_raw_path(tmp_config):
    config = load_config(tmp_config)
    source = "ObjaverseXL_github"
    asset = "a" * 64
    write_reference_index(
        config,
        source,
        ((asset, f"{source}-00000", "raw/github/repos/repo.zip"),),
    )
    path = config.paths.data2_root / "control/raw_references.json"
    value = json.loads(path.read_text())
    entries = value["sources"][source].pop("raw/github/repos/repo.zip")
    value["sources"][source]["raw/github/repos/other.zip"] = entries
    path.write_text(json.dumps(value))

    with pytest.raises(ArtifactValidationError, match="training registry"):
        FrozenReferenceCounter(config).pending_references(
            source,
            "raw/github/repos/other.zip",
            excluding_shard_id=f"{source}-00000",
            excluding_batch_id="batch000",
        )


def write_frozen_batch(
    config, source, shard, batch, shas, *, gate="production", canonical_shas=None
):
    root = config.paths.data2_root / "control"
    root = (
        root / "shards"
        if gate == "production"
        else root / "qualification" / gate / "shards"
    )
    root = root / source / shard
    root.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{item}\n" for item in shas).encode("ascii")
    canonical_payload = "".join(
        f"{item}\n" for item in (canonical_shas or shas)
    ).encode("ascii")
    (root / f"{batch}.txt").write_bytes(payload)
    marker_path = root / "batches.json"
    marker = {
        "schema_version": 2,
        "gate": gate,
        "source": source,
        "shard_id": shard,
        "config_hash": config.config_hash(),
        "canonical_shard_sha256": sha256(canonical_payload).hexdigest(),
        "scope_sha256": sha256(payload).hexdigest(),
        "batches": [
            {
                "name": f"{batch}.txt",
                "count": len(shas),
                "sha256": sha256(payload).hexdigest(),
            }
        ],
    }
    marker_path.write_text(json.dumps(marker))


def write_reference_index(config, source, references):
    store = SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    )
    store.save(
        pd.DataFrame(
            {
                "sha256": [item[0] for item in references],
                "owner_source": [source] * len(references),
                "shard_id": [item[1] for item in references],
                "local_path": [item[2] for item in references],
            }
        ),
        source_inputs=registry_source_inputs(config),
    )
    manifest = json.loads(store.manifest_path.read_text())
    grouped = {}
    for asset, shard, raw_path in references:
        grouped.setdefault(raw_path, []).append(
            {"sha256": asset, "shard_id": shard}
        )
    path = config.paths.data2_root / "control/raw_references.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_type": "canonical_raw_reference_index",
                "config_hash": config.config_hash(),
                "training_registry_sha256": manifest["sha256"],
                "sources": {
                    configured: grouped if configured == source else {}
                    for configured in config.sources
                },
            }
        )
    )


def test_reference_counter_preserves_shared_zip_until_final_frozen_batch(
    tmp_config,
):
    config = load_config(tmp_config)
    source = "ObjaverseXL_github"
    first, second = "a" * 64, "b" * 64
    metadata = config.paths.data2_root / "raw" / source / "raw/metadata.csv"
    metadata.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "sha256": [first, second],
            "local_path": [
                "raw/github/repos/repo.zip/models/first.glb",
                "raw/github/repos/repo.zip/models/second.glb",
            ],
        }
    ).to_csv(metadata, index=False)
    write_reference_index(
        config,
        source,
        (
            (first, f"{source}-00000", "raw/github/repos/repo.zip"),
            (second, f"{source}-00001", "raw/github/repos/repo.zip"),
        ),
    )
    write_frozen_batch(config, source, f"{source}-00000", "batch000", (first,))
    write_frozen_batch(config, source, f"{source}-00001", "batch000", (second,))
    counter = FrozenReferenceCounter(config)

    assert (
        counter.pending_references(
            source,
            "raw/github/repos/repo.zip",
            excluding_shard_id=f"{source}-00000",
            excluding_batch_id="batch000",
        )
        == 1
    )


def test_reference_counter_rejects_corrupt_frozen_manifest(tmp_config):
    config = load_config(tmp_config)
    source = "ObjaverseXL_github"
    asset = "a" * 64
    metadata = config.paths.data2_root / "raw" / source / "raw/metadata.csv"
    metadata.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "sha256": [asset],
            "local_path": ["raw/github/repos/repo.zip/models/item.glb"],
        }
    ).to_csv(metadata, index=False)
    write_reference_index(
        config,
        source,
        ((asset, f"{source}-00000", "raw/github/repos/repo.zip"),),
    )
    write_frozen_batch(config, source, f"{source}-00000", "batch000", (asset,))
    marker = (
        config.paths.data2_root
        / "control/shards"
        / source
        / f"{source}-00000/batches.json"
    )
    marker.write_text("not json")

    with pytest.raises(ArtifactValidationError, match="frozen"):
        FrozenReferenceCounter(config).pending_references(
            source,
            "raw/github/repos/repo.zip",
            excluding_shard_id="other",
            excluding_batch_id="batch000",
        )


def test_reference_counter_releases_shared_zip_after_final_verified_archive(
    tmp_config,
):
    config = load_config(tmp_config)
    source = "ObjaverseXL_github"
    first = "a" * 64
    contents = b"second member"
    second = sha256(contents).hexdigest()
    metadata = config.paths.data2_root / "raw" / source / "raw/metadata.csv"
    metadata.parent.mkdir(parents=True)
    first_path = "raw/github/repos/repo.zip/models/first.glb"
    second_path = "raw/github/repos/repo.zip/models/second.glb"
    pd.DataFrame(
        {
            "sha256": [first, second],
            "local_path": [first_path, second_path],
        }
    ).to_csv(metadata, index=False)
    write_reference_index(
        config,
        source,
        (
            (first, f"{source}-00000", "raw/github/repos/repo.zip"),
            (second, f"{source}-00001", "raw/github/repos/repo.zip"),
        ),
    )
    write_frozen_batch(config, source, f"{source}-00000", "batch000", (first,))
    write_frozen_batch(config, source, f"{source}-00001", "batch000", (second,))
    archive_root = config.paths.local_root / "archive-source"
    member = archive_root / second_path
    member.parent.mkdir(parents=True)
    member.write_bytes(contents)
    archive = (
        config.paths.data3_root
        / "archive/raw"
        / source
        / f"{source}-00001/batch000.tar"
    )
    manifest = build_pack(
        archive_root,
        [Path(second_path)],
        archive,
        f"{source}-00001",
        batch_id="batch000",
        family="raw",
        config_hash=config.config_hash(),
        tool_commit="test-commit",
        asset_sha256s=(second,),
        completed_count=1,
        quarantined_count=0,
    )
    manifest_path = archive.with_suffix(".tar.manifest.json")
    payload = asdict(manifest)
    payload["validated_at"] = "2026-07-16T00:00:00+00:00"
    manifest_path.write_text(json.dumps(payload))

    assert (
        FrozenReferenceCounter(config).pending_references(
            source,
            "raw/github/repos/repo.zip",
            excluding_shard_id=f"{source}-00000",
            excluding_batch_id="batch000",
        )
        == 0
    )


def test_no_follow_telemetry_rejects_symlinked_artifact(tmp_path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged")
    path = tmp_path / "telemetry/resources.jsonl"
    path.parent.mkdir()
    path.symlink_to(outside)

    with pytest.raises(ArtifactValidationError, match="telemetry"):
        NoFollowTelemetryWriter(path)

    assert outside.read_text() == "unchanged"


def test_mutating_runtime_injects_all_task10_providers(tmp_config):
    config = load_config(tmp_config)
    for root in (
        config.paths.data2_root,
        config.paths.data3_root,
        config.paths.local_root,
    ):
        root.mkdir(parents=True)

    with build_mutating_services(config) as runtime:
        services = runtime.services
        assert type(services.pilot_reader).__name__ != "_MissingPilotReader"
        assert type(services.reference_counter).__name__ != "_MissingReferenceCounter"
        assert type(services.project_accounting).__name__ != "_MissingProjectAccounting"
        assert type(services.resource_guard).__name__ != "_MissingResourceGuard"
        assert services.registry_builder is not None
        assert services.report_builder is not None


def test_mutating_runtime_construction_is_side_effect_free(tmp_config):
    config = load_config(tmp_config)

    runtime = build_mutating_services(config)

    assert runtime._services is None
    assert not config.paths.data2_root.exists()
    assert not config.paths.data3_root.exists()
    assert not config.paths.local_root.exists()


@pytest.mark.parametrize(
    "failing_factory",
    (
        "ResourceGuard",
        "SafeRegistryStore",
        "PilotArtifactReader",
        "FrozenReferenceCounter",
        "CanonicalRegistryBuilder",
        "RuntimeReportBuilder",
        "PipelineServices",
    ),
)
def test_mutating_runtime_closes_telemetry_after_lazy_init_failure(
    tmp_config, monkeypatch, failing_factory
):
    config = load_config(tmp_config)

    class Telemetry:
        instances = []

        def __init__(self, path):
            self.closed = False
            self.instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime._ensure_runtime_roots", lambda config: None
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.initialize_project_accounting",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.ResourceSampler", lambda *args: object()
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.ResourcePolicy", lambda *args: object()
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime.NoFollowTelemetryWriter", Telemetry
    )
    for name in (
        "ResourceGuard",
        "SafeRegistryStore",
        "PilotArtifactReader",
        "FrozenReferenceCounter",
        "CanonicalRegistryBuilder",
        "RuntimeReportBuilder",
        "PipelineServices",
    ):
        factory = (
            (lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
            if name == failing_factory
            else (lambda *args, **kwargs: object())
        )
        monkeypatch.setattr(f"data_toolkit.pipeline.runtime.{name}", factory)

    runtime = build_mutating_services(config)
    with pytest.raises(RuntimeError, match="boom"):
        runtime.services

    assert len(Telemetry.instances) == 1
    assert Telemetry.instances[0].closed is True
    assert runtime._telemetry is None


def test_mutating_runtime_maps_root_initialization_failure(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    monkeypatch.setattr(
        "data_toolkit.pipeline.runtime._open_directory_nofollow",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(ArtifactValidationError, match="runtime root"):
        build_mutating_services(config).services
