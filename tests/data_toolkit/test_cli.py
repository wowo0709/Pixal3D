from dataclasses import asdict
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
from data_toolkit.pipeline.packing import build_pack
from data_toolkit.pipeline.reporting import write_report
from data_toolkit.pipeline.runtime import (
    ArtifactValidationError,
    CanonicalRegistryBuilder,
    FrozenReferenceCounter,
    PilotArtifactReader,
    NoFollowTelemetryWriter,
    RuntimeReportBuilder,
    SafeRegistryStore,
    build_mutating_services,
)
from test_reporting import CONFIG_HASH, valid_report_payload


def valid_hardware_payload(config_hash):
    return {
        "schema_version": 1,
        "artifact_type": "hardware_preflight",
        "config_hash": config_hash,
        "decision": "passed",
        "gpu": {
            "checked": True,
            "cycles_device": "OPTIX",
            "gpu_count": 7,
            "device_names": [f"GPU-{index}" for index in range(7)],
            "cpu_fallback_detected": False,
            "cube_render_sha256": "c" * 64,
        },
        "storage": {
            root: {
                "read_mib_per_second": 100.0,
                "write_mib_per_second": 100.0,
                "free_bytes_before": 10_000,
                "free_bytes_after": 10_000,
                "fixture_removed": True,
            }
            for root in ("local", "data2", "data3")
        },
        "pilot_sizing": {
            "sources": {"ABO": {"p95_peak_local_bytes": 200}}
        },
    }


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
    root = config.paths.data2_root / "control/reports"
    root.mkdir(parents=True)
    payload = valid_hardware_payload(config.config_hash())
    (root / "hardware.json").write_text(json.dumps(payload))
    reader = PilotArtifactReader(config)

    assert reader.p95_peak_local_bytes("ABO") == 200
    with pytest.raises(ArtifactValidationError, match="source"):
        reader.p95_peak_local_bytes("HSSD")


def test_pilot_reader_prefers_passed_pilot_measurement(tmp_config):
    config = load_config(tmp_config)
    report_root = config.paths.data2_root / "control/reports"
    gate_root = report_root / "gates"
    gate_root.mkdir(parents=True)
    hardware = valid_hardware_payload(config.config_hash())
    (report_root / "hardware.json").write_text(json.dumps(hardware))
    pilot = valid_report_payload("pilot")
    pilot["config_hash"] = config.config_hash()
    pilot["capacity"]["sources"]["ABO"]["p95_peak_local_bytes"] = 300
    write_report(gate_root, "pilot", pilot)

    assert PilotArtifactReader(config).p95_peak_local_bytes("ABO") == 300


def test_hardware_report_builder_validates_and_publishes(tmp_config):
    config = load_config(tmp_config)
    input_root = config.paths.data2_root / "control/report_inputs"
    input_root.mkdir(parents=True)
    (input_root / "hardware.json").write_text(
        json.dumps(valid_hardware_payload(config.config_hash()))
    )

    outputs = RuntimeReportBuilder(config)(None, True)

    assert [path.name for path in outputs] == ["hardware.json", "hardware.md"]


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

    store.save(frame)

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
    assert result["owner_source"].tolist() == [
        *config.sources,
        *config.evaluation_sources,
    ]
    assert SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    ).load().shape[0] == len(frames)


def write_frozen_batch(config, source, shard, batch, shas):
    root = config.paths.data2_root / "control/shards" / source / shard
    root.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{item}\n" for item in shas).encode("ascii")
    (root / f"{batch}.txt").write_bytes(payload)
    marker_path = root / "batches.json"
    marker = {
        "schema_version": 1,
        "source": source,
        "shard_id": shard,
        "config_hash": config.config_hash(),
        "batches": [
            {
                "name": f"{batch}.txt",
                "count": len(shas),
                "sha256": sha256(payload).hexdigest(),
            }
        ],
    }
    marker_path.write_text(json.dumps(marker))


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
