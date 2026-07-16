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
        counts[command.argv[index + 1]] += command.gpu_ranks or 1
    return dict(counts)


def _synthetic_disk_usage(path):
    return SimpleNamespace(
        total=100 * 1024**4,
        used=50 * 1024**4,
        free=50 * 1024**4,
        percent=50.0,
    )


@pytest.mark.integration
def test_two_asset_shard_runs_and_resumes(synthetic_config, monkeypatch):
    config = load_config(synthetic_config)
    worker = Path("tests/data_toolkit/fixtures/fake_leaf_worker.py").resolve()
    monkeypatch.setenv("PIXAL3D_LEAF_WORKER", str(worker))
    monkeypatch.setattr(
        ResourceSampler,
        "_gpu_metrics",
        lambda self: ((), "disabled by synthetic integration test"),
    )
    monkeypatch.setattr(
        "data_toolkit.pipeline.resources.psutil.disk_usage",
        _synthetic_disk_usage,
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
