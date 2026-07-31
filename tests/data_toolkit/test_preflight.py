from dataclasses import replace

from data_toolkit.pipeline.preflight import PreflightStatus, run_preflight


def test_missing_manual_archives_block(config, tmp_path):
    cfg = replace(
        config, paths=replace(config.paths, data2_root=tmp_path / "data2")
    )
    results = {
        item.source: item for item in run_preflight(cfg, check_remote=False)
    }
    assert results["3D-FUTURE"].status == PreflightStatus.BLOCKED
    assert "3D-FUTURE-model.zip" in results["3D-FUTURE"].message
    assert results["Toys4k"].status == PreflightStatus.BLOCKED


def test_present_manual_archives_are_ready(config, tmp_path):
    data2_root = tmp_path / "data2"
    future = data2_root / "raw/3D-FUTURE/3D-FUTURE-model.zip"
    toys = data2_root / "raw/Toys4k/toys4k_blend_files.zip"
    future.parent.mkdir(parents=True)
    toys.parent.mkdir(parents=True)
    future.touch()
    toys.touch()
    cfg = replace(config, paths=replace(config.paths, data2_root=data2_root))

    results = {
        item.source: item for item in run_preflight(cfg, check_remote=False)
    }

    assert results["3D-FUTURE"].status == PreflightStatus.READY
    assert results["Toys4k"].status == PreflightStatus.READY


def test_hssd_remote_failure_is_blocked(config, monkeypatch):
    import data_toolkit.pipeline.preflight as preflight

    def fail():
        raise RuntimeError("no credentials")

    monkeypatch.setattr(preflight.huggingface_hub, "whoami", fail)

    results = {
        item.source: item for item in run_preflight(config, check_remote=True)
    }

    assert results["HSSD"].status == PreflightStatus.BLOCKED
    assert "no credentials" in results["HSSD"].message
