import pandas as pd
import pytest

from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.full_run import FullProductionRunner


def _registry():
    rows = []
    counts = {
        "ObjaverseXL_sketchfab": 2,
        "ABO": 1,
        "HSSD": 1,
        "3D-FUTURE": 1,
    }
    for source, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "owner_source": source,
                    "shard_id": f"{source}-{index:05d}",
                }
            )
    return pd.DataFrame(rows)


class _Services:
    def __init__(self, *, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def run(self, gate, source, shard, count):
        self.calls.append(("run", gate, source, shard, count))
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("synthetic interruption")

    def audit(self, gate, source, shard):
        self.calls.append(("audit", gate, source, shard))


def _install_inputs(monkeypatch, config, gate_calls, parallelism_calls=None):
    import data_toolkit.pipeline.full_run as module

    if parallelism_calls is None:
        parallelism_calls = []

    monkeypatch.setattr(
        module,
        "read_gate_report",
        lambda held_config, gate: gate_calls.append((held_config, gate)),
    )
    monkeypatch.setattr(
        module,
        "read_parallelism_report",
        lambda held_config: parallelism_calls.append(held_config),
    )

    class Store:
        def __init__(self, path, held_config):
            assert path == held_config.paths.data2_root / "control/assets.parquet"
            assert held_config is config

        def load(self):
            return _registry()

    monkeypatch.setattr(module, "SafeRegistryStore", Store)


def test_full_runner_uses_fixed_source_order_and_audits_each_shard(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    gates = []
    parallelism = []
    _install_inputs(monkeypatch, config, gates, parallelism)
    services = _Services()

    FullProductionRunner(config, services).run()

    assert [gate for _, gate in gates] == ["smoke", "pilot"]
    assert parallelism == [config]
    expected_shards = [
        ("ABO", "ABO-00000"),
        ("HSSD", "HSSD-00000"),
        ("3D-FUTURE", "3D-FUTURE-00000"),
        ("ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00000"),
        ("ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00001"),
    ]
    assert services.calls == [
        call
        for source, shard in expected_shards
        for call in (
            ("run", "production", source, shard, None),
            ("audit", "production", source, shard),
        )
    ]


def test_full_runner_restart_revalidates_interrupted_shard_before_advancing(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    _install_inputs(monkeypatch, config, [])
    services = _Services(fail_first=True)
    runner = FullProductionRunner(config, services)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.run()
    runner.run()

    first = (
        "run",
        "production",
        "ABO",
        "ABO-00000",
        None,
    )
    assert services.calls[:3] == [
        first,
        first,
        (
            "audit",
            "production",
            "ABO",
            "ABO-00000",
        ),
    ]


def test_full_runner_dry_run_is_ordered_and_does_not_mutate(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    _install_inputs(monkeypatch, config, [])

    class Services(_Services):
        def plan(self, gate, source, shard, count, *, freeze):
            assert gate == "production"
            assert count is None
            assert freeze is False
            return ("batch000: 256 assets", "batch001: 17 assets")

    services = Services()

    lines = FullProductionRunner(config, services).plan()

    assert lines[0] == "ABO/ABO-00000/batch000: 256 assets, 4 chunks (max 64)"
    assert lines[1] == "ABO/ABO-00000/batch001: 17 assets, 1 chunks (max 64)"
    assert lines[-1].startswith(
        "ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00001/"
    )
    assert services.calls == []


def test_work_units_freeze_batches_and_round_robin_sources(
    tmp_config, monkeypatch
):
    config = load_config(tmp_config)
    _install_inputs(monkeypatch, config, [])

    class Services(_Services):
        def plan(self, gate, source, shard, count, *, freeze):
            assert (gate, count, freeze) == ("production", None, True)
            return (
                "batch000: 256 assets",
                "batch001: 17 assets",
            )

    units = FullProductionRunner(config, Services()).work_units(freeze=True)

    assert [unit.source for unit in units[:4]] == [
        "ABO",
        "HSSD",
        "3D-FUTURE",
        "ObjaverseXL_sketchfab",
    ]
    assert [unit.batch_id for unit in units[:4]] == ["batch000"] * 4
    assert units[4].source == "ABO"
    assert units[4].batch_id == "batch001"
    assert units[4].count == 17
