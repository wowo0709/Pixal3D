from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from data_toolkit.pipeline.cli import main
from data_toolkit.pipeline.commands import (
    ShardContext,
    build_preprocessing_dag,
)
from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.orchestrator import PipelineServices
from data_toolkit.pipeline.packing import PACK_FAMILIES, verify_pack
from data_toolkit.pipeline.resources import (
    ResourceAction,
    ResourceDecision,
    ResourceSampler,
)
from data_toolkit.pipeline.runtime import CanonicalRegistryBuilder, SafeRegistryStore


SOURCE = "Synthetic"
SHARD = "Synthetic-00000"
GATE = "smoke"


def _family_directory(family):
    if family == "common":
        return Path("common")
    prefix, resolution = family.split("-", 1)
    return Path(
        {"SS": "ss", "shape": "shape", "PBR": "pbr"}[prefix],
        resolution,
    )


def _expected_leaf_counts(context, config):
    counts = Counter()
    for command in build_preprocessing_dag(context, config):
        if "--original-script" not in command.argv:
            continue
        index = command.argv.index("--original-script")
        workers = (
            command.gpu_ranks * command.workers_per_gpu
            if command.gpu_ranks
            else 1
        )
        counts[command.argv[index + 1]] += workers
    return dict(counts)


def _synthetic_disk_usage(path):
    return SimpleNamespace(
        total=100 * 1024**4,
        used=50 * 1024**4,
        free=50 * 1024**4,
        percent=50.0,
    )


def _reject_host_resource_call(*args, **kwargs):
    raise AssertionError("integration test attempted a live host resource read")


class _SyntheticPsutil:
    def __init__(self):
        self.calls = Counter()
        self.disk_paths = []

    def _record(self, name, value):
        self.calls[name] += 1
        return value

    def cpu_times_percent(self, interval=None):
        assert interval is None
        return self._record("cpu_times_percent", SimpleNamespace(iowait=0.0))

    def virtual_memory(self):
        return self._record(
            "virtual_memory", SimpleNamespace(available=400 * 1024**3)
        )

    def swap_memory(self):
        return self._record("swap_memory", SimpleNamespace(sin=0))

    def cpu_percent(self, interval=None):
        assert interval is None
        return self._record("cpu_percent", 10.0)

    def getloadavg(self):
        return self._record("getloadavg", (1.0, 1.0, 1.0))

    def disk_usage(self, path):
        self.disk_paths.append(Path(path))
        return self._record("disk_usage", _synthetic_disk_usage(path))


@pytest.fixture(autouse=True)
def synthetic_voxel_reader(monkeypatch):
    fake_voxel = SimpleNamespace(
        io=SimpleNamespace(
            read_vxz_info=lambda _path: {
                "shape": (1, 1, 1),
                "attributes": ("value",),
            }
        )
    )
    monkeypatch.setitem(sys.modules, "o_voxel", fake_voxel)


@pytest.mark.integration
def test_two_asset_shard_runs_and_resumes(synthetic_config, monkeypatch):
    config = load_config(synthetic_config)
    synthetic_resources = _SyntheticPsutil()
    worker = Path("tests/data_toolkit/fixtures/fake_leaf_worker.py").resolve()
    monkeypatch.setenv("PIXAL3D_LEAF_WORKER", str(worker))
    monkeypatch.setattr(
        ResourceSampler,
        "_gpu_metrics",
        lambda self: ((), "disabled by synthetic integration test"),
    )
    for name in (
        "cpu_times_percent",
        "virtual_memory",
        "swap_memory",
        "cpu_percent",
        "getloadavg",
        "disk_usage",
    ):
        monkeypatch.setattr(
            f"data_toolkit.pipeline.resources.psutil.{name}",
            _reject_host_resource_call,
        )
    monkeypatch.setattr(
        "data_toolkit.pipeline.resources.os.getloadavg",
        _reject_host_resource_call,
    )
    monkeypatch.setitem(
        ResourceSampler.__init__.__kwdefaults__,
        "psutil_api",
        synthetic_resources,
    )
    monkeypatch.setitem(
        PipelineServices.__init__.__kwdefaults__,
        "disk_usage",
        _synthetic_disk_usage,
    )
    context = ShardContext.from_config(
        config, SOURCE, SHARD, "batch000", gate=GATE
    )
    assert not context.instances.exists()
    arguments = [
        "--config",
        str(synthetic_config),
        "--source",
        SOURCE,
        "--shard",
        SHARD,
        "--gate",
        GATE,
    ]

    assert main(["run", *arguments]) == 0

    sample_count = synthetic_resources.calls["cpu_percent"]
    assert sample_count > 0
    assert synthetic_resources.calls == Counter(
        {
            "cpu_times_percent": sample_count,
            "virtual_memory": sample_count,
            "swap_memory": sample_count,
            "cpu_percent": sample_count,
            "getloadavg": sample_count,
            "disk_usage": 3 * sample_count,
        }
    )
    assert Counter(synthetic_resources.disk_paths) == Counter(
        {
            config.paths.local_root: sample_count,
            config.paths.data2_root: sample_count,
            config.paths.data3_root: sample_count,
        }
    )

    counts_path = context.instances.parent / "leaf-command-counts.json"
    first_counts = json.loads(counts_path.read_text())
    assert first_counts == _expected_leaf_counts(context, config)

    prepared = config.paths.data2_root / "prepared"
    qualification = prepared / "qualification" / GATE
    expected_packs = {
        family: qualification
        / _family_directory(family)
        / SOURCE
        / SHARD
        / "batch000.tar"
        for family in PACK_FAMILIES
    }
    actual_packs = set(qualification.rglob("*.tar"))
    assert set(expected_packs) == set(PACK_FAMILIES)
    assert actual_packs == set(expected_packs.values())
    assert len(actual_packs) == 8
    for path in expected_packs.values():
        verify_pack(path, path.with_suffix(".tar.manifest.json"))
        with tarfile.open(path) as bundle:
            assert bundle.getmembers()

    index_path = qualification / "index" / SOURCE / f"{SHARD}.json"
    index = json.loads(index_path.read_text())
    assert set(index["batches"]["batch000"]) == set(PACK_FAMILIES)

    archive = (
        config.paths.data3_root
        / "archive"
        / "qualification"
        / GATE
        / "raw"
        / SOURCE
        / SHARD
        / "batch000.tar"
    )
    verify_pack(archive, archive.with_suffix(".tar.manifest.json"))
    with tarfile.open(archive) as bundle:
        assert len(bundle.getmembers()) == 2

    checkpoint = json.loads(
        (
            config.paths.data2_root
            / "control"
            / "qualification"
            / GATE
            / "checkpoints"
            / SOURCE
            / SHARD
            / "batch000.json"
        ).read_text()
    )
    assert checkpoint["gate"] == GATE
    assert set(checkpoint["completed_commands"]) == {
        command.name for command in build_preprocessing_dag(context, config)
    }
    assert checkpoint["quality_outcomes"] == {
        asset: "completed" for asset in context.instances.read_text().splitlines()
    }
    quality = json.loads(
        (
            config.paths.data2_root
            / "control"
            / "qualification"
            / GATE
            / "quality"
            / SOURCE
            / f"{SHARD}.json"
        ).read_text()
    )
    assert quality["gate"] == GATE
    assert [entry["outcome"] for entry in quality["entries"]] == [
        "completed",
        "completed",
    ]
    for root in (context.download_root, context.work_root, context.output_root):
        assert not root.exists()

    assert main(["resume", *arguments]) == 0
    assert json.loads(counts_path.read_text()) == first_counts


@pytest.fixture
def parallel_synthetic_config(tmp_config):
    raw = yaml.safe_load(tmp_config.read_text())
    raw["sources"] = [SOURCE]
    raw["evaluation_sources"] = [f"{SOURCE}Eval"]
    raw["shard_size"] = 65
    tmp_config.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = load_config(tmp_config)

    payloads = tuple(
        f"pixal3d parallel synthetic asset {index:03d}".encode("ascii")
        for index in range(65)
    )
    training = config.paths.data2_root / f"control/metadata/{SOURCE}/metadata.csv"
    training.parent.mkdir(parents=True)
    training.write_text(
        "sha256,file_identifier,fixture_payload\n"
        + "".join(
            f"{sha256(payload).hexdigest()},objects/asset-{index:03d}.glb,"
            f"{payload.decode('ascii')}\n"
            for index, payload in enumerate(payloads)
        )
    )
    evaluation = (
        config.paths.data2_root
        / f"control/metadata/{SOURCE}Eval/metadata.csv"
    )
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text(
        "sha256,file_identifier\n"
        f"{sha256(b'parallel evaluation').hexdigest()},evaluation/unused.glb\n"
    )
    CanonicalRegistryBuilder(config)()
    return config


class _AlwaysRunGuard:
    def wait_for_admission(self, _shard_id, _command):
        return None

    def last_five_minutes(self):
        return ()

    def check(self, _shard_id, _command):
        return ResourceDecision(ResourceAction.RUN, ())


class _OneMiBPilot:
    def p95_peak_local_bytes(self, _source):
        return 1024**2

    def p95_peak_local_bytes_for_gate(self, _source, _gate):
        return 1024**2


class _KeepRawReferences:
    def pending_references(self, *_args, **_kwargs):
        return 1


class _NoopAccounting:
    def record_registry_delta(self, _path, _delta):
        return None

    def reconcile_at_shard_boundary(self):
        return (0, 0)


@pytest.mark.integration
def test_65_asset_production_batch_runs_as_two_restartable_chunks(
    parallel_synthetic_config, monkeypatch
):
    config = parallel_synthetic_config
    worker = Path("tests/data_toolkit/fixtures/fake_leaf_worker.py").resolve()
    monkeypatch.setenv("PIXAL3D_LEAF_WORKER", str(worker))
    monkeypatch.setenv("PIXAL3D_FAKE_FAST", "1")
    monkeypatch.setenv(
        "PIXAL3D_FAKE_VXZ_TEMPLATE",
        str(config.paths.local_root / "fake-template.vxz"),
    )
    services = PipelineServices(
        config,
        resource_guard=_AlwaysRunGuard(),
        pilot_reader=_OneMiBPilot(),
        reference_counter=_KeepRawReferences(),
        project_accounting=_NoopAccounting(),
        registry_store=SafeRegistryStore(
            config.paths.data2_root / "control/assets.parquet", config
        ),
        disk_usage=_synthetic_disk_usage,
        tool_commit="parallel-integration-test",
    )
    services.runner.monitor_interval_seconds = 0.01

    services.run("production", SOURCE, SHARD)

    context = ShardContext.from_config(
        config, SOURCE, SHARD, "batch000", gate="production"
    )
    chunks = (
        config.paths.data2_root
        / f"control/checkpoints/{SOURCE}/{SHARD}/chunks/batch000"
    )
    manifest = json.loads((chunks / "manifest.json").read_text())
    assert [entry["count"] for entry in manifest["chunks"]] == [64, 1]
    assert all(
        json.loads((chunks / f"chunk{index:03d}/checkpoint.json").read_text())[
            "promoted"
        ]
        for index in range(2)
    )
    parent_checkpoint = json.loads(
        (config.paths.data2_root / f"control/checkpoints/{SOURCE}/{SHARD}/batch000.json").read_text()
    )
    assert len(parent_checkpoint["quality_outcomes"]) == 65
    assert set(parent_checkpoint["completed_commands"]) == {
        "download",
        "validate_outputs",
        "build_packs",
        "archive_raw",
        "cleanup_local",
    }
    prepared = config.paths.data2_root / "prepared"
    assert len(tuple(prepared.rglob("batch000.tar"))) == 8
    for root in (context.download_root, context.work_root, context.output_root):
        assert not root.exists()
    assert not (context.work_root.parent / "chunks").exists()

    command_counts = {
        path: json.loads(path.read_text())
        for path in chunks.rglob("leaf-command-counts.json")
    }
    assert len(command_counts) == 2
    services.resume("production", SOURCE, SHARD)
    assert {
        path: json.loads(path.read_text()) for path in command_counts
    } == command_counts


@pytest.mark.integration
def test_parallelism_benchmark_executes_frozen_64_asset_scope(
    parallel_synthetic_config, monkeypatch
):
    config = parallel_synthetic_config
    worker = Path("tests/data_toolkit/fixtures/fake_leaf_worker.py").resolve()
    monkeypatch.setenv("PIXAL3D_LEAF_WORKER", str(worker))
    monkeypatch.setenv("PIXAL3D_FAKE_FAST", "1")
    monkeypatch.setenv(
        "PIXAL3D_FAKE_VXZ_TEMPLATE",
        str(config.paths.local_root / "fake-template.vxz"),
    )
    registry = SafeRegistryStore(
        config.paths.data2_root / "control/assets.parquet", config
    )
    assets = tuple(
        sorted(
            registry.load().loc[
                lambda frame: frame["owner_source"] == SOURCE, "sha256"
            ]
        )[:64]
    )
    metadata = pd.read_csv(
        config.paths.data2_root / f"control/metadata/{SOURCE}/metadata.csv"
    ).set_index("sha256")
    raw_root = config.paths.data2_root / f"raw/{SOURCE}"
    raw_records = []
    for asset in assets:
        record = metadata.loc[asset]
        relative = Path(record["file_identifier"])
        destination = raw_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(record["fixture_payload"].encode("ascii"))
        raw_records.append({"sha256": asset, "local_path": relative.as_posix()})
    raw_metadata = raw_root / "raw/metadata.csv"
    raw_metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_records).to_csv(raw_metadata, index=False)

    services = PipelineServices(
        config,
        resource_guard=_AlwaysRunGuard(),
        pilot_reader=_OneMiBPilot(),
        registry_store=registry,
        tool_commit="parallel-benchmark-integration-test",
    )
    services.runner.monitor_interval_seconds = 0.01
    report_paths = services.benchmark_parallelism(
        SOURCE, SHARD, 64, dry_run=False
    )

    report = json.loads(report_paths[0].read_text())
    assert report["report_type"] == "parallelism_benchmark"
    assert report["count"] == 64
    assert report["quality"] == {
        "terminal_assets": 64,
        "completed_assets": 64,
        "quarantined_assets": 0,
    }
    assert report["audit"]["passed"] is True
    assert report["measurement_valid"] is False
    assert report["decision"] == "held"
    assert set(report["stage_seconds"]) == {
        command.name
        for command in build_preprocessing_dag(
            ShardContext.from_config(
                config, SOURCE, SHARD, "benchmark064", gate="pilot"
            ),
            config,
        )
        if command.name
        not in {"download", "build_packs", "archive_raw", "cleanup_local"}
    }
