from collections import Counter
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline.cli import main
from data_toolkit.pipeline.commands import (
    ShardContext,
    build_preprocessing_dag,
)
from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.orchestrator import PipelineServices
from data_toolkit.pipeline.packing import PACK_FAMILIES, verify_pack
from data_toolkit.pipeline.resources import ResourceSampler


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
