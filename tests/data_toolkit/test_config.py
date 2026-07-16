from pathlib import Path

import pytest

from data_toolkit.pipeline.config import load_config


CONFIG = Path("data_toolkit/configs/multiview_preprocess.yaml")


def test_fixed_contract():
    cfg = load_config(CONFIG)
    assert cfg.paths.local_root == Path("/root/pixal3d-data")
    assert cfg.paths.data2_root == Path("/root/data2/pixal3d")
    assert cfg.paths.data3_root == Path("/root/data3/pixal3d")
    assert cfg.sources == (
        "ObjaverseXL_sketchfab",
        "ObjaverseXL_github",
        "ABO",
        "HSSD",
        "3D-FUTURE",
    )
    assert (cfg.render.num_views, cfg.render.resolution) == (8, 512)
    assert cfg.targets.views == (0, 1)
    assert cfg.targets.resolutions == (256, 512, 1024)
    assert cfg.limits.cpu_soft_percent == 80.0
    assert cfg.limits.ram_soft_available_gib == 96


def test_unknown_root_key_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(CONFIG.read_text() + "\nunknown_key: true\n")
    with pytest.raises(ValueError, match="unknown_key"):
        load_config(path)


def test_hash_is_stable():
    assert load_config(CONFIG).config_hash() == load_config(CONFIG).config_hash()
    assert len(load_config(CONFIG).config_hash()) == 64
