import pandas as pd
import pytest

from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.full_run import FullProductionRunner


def _registry():
    rows = []
    counts = {
        "ObjaverseXL_sketchfab": 2,
        "ObjaverseXL_github": 1,
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


def _install_inputs(monkeypatch, config, gate_calls):
    import data_toolkit.pipeline.full_run as module

    monkeypatch.setattr(
        module,
        "read_gate_report",
        lambda held_config, gate: gate_calls.append((held_config, gate)),
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
    _install_inputs(monkeypatch, config, gates)
    services = _Services()

    FullProductionRunner(config, services).run()

    assert [gate for _, gate in gates] == ["smoke", "pilot"]
    expected_shards = [
        ("ABO", "ABO-00000"),
        ("HSSD", "HSSD-00000"),
        ("3D-FUTURE", "3D-FUTURE-00000"),
        ("ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00000"),
        ("ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00001"),
        ("ObjaverseXL_github", "ObjaverseXL_github-00000"),
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
