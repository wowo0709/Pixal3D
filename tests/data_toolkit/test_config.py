from pathlib import Path

import pytest
import yaml

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


def test_config_accepts_fp16_only_as_the_supported_post_pilot_dtype(tmp_path):
    config = load_config(
        _write_config(
            tmp_path,
            lambda raw: raw["targets"].update(latent_dtype="float16"),
        )
    )

    assert config.targets.latent_dtype == "float16"


def _write_config(tmp_path: Path, mutator) -> Path:
    raw = yaml.safe_load(CONFIG.read_text())
    mutator(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_config_read_rejects_symlink(tmp_path: Path):
    target = tmp_path / "target.yaml"
    target.write_text(CONFIG.read_text())
    link = tmp_path / "config.yaml"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="unsafe config"):
        load_config(link)


@pytest.mark.parametrize(
    "section,mutation",
    [
        ("paths", lambda value: value.update(extra="/tmp/extra")),
        ("render", lambda value: value.pop("camera_policy")),
        ("targets", lambda value: value.update(resolutions=[256, 512])),
        ("workers", lambda value: value.update(render_workers=0)),
        ("limits", lambda value: value.update(cpu_soft_percent=True)),
    ],
)
def test_config_rejects_invalid_nested_contract(tmp_path, section, mutation):
    def mutate(raw):
        mutation(raw[section])

    with pytest.raises(ValueError, match=section):
        load_config(_write_config(tmp_path, mutate))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: raw.update(sources=[]),
        lambda raw: raw.update(sources=["ABO", "ABO"]),
        lambda raw: raw.update(evaluation_sources=[]),
        lambda raw: raw.update(evaluation_sources=["Toys4k", "ABO"]),
        lambda raw: raw.update(sources=["ABO"], evaluation_sources=["ABO"]),
        lambda raw: raw.update(sources=["ABO", "unsafe/source"]),
    ],
)
def test_config_rejects_invalid_source_partition(tmp_path, mutator):
    with pytest.raises(ValueError, match="source"):
        load_config(_write_config(tmp_path, mutator))


@pytest.mark.parametrize(
    "paths",
    [
        {"data2_root": "relative", "data3_root": "/data3", "local_root": "/local"},
        {"data2_root": "/same", "data3_root": "/same", "local_root": "/local"},
        {"data2_root": "/project", "data3_root": "/project/archive", "local_root": "/local"},
        {"data2_root": "/data2\0bad", "data3_root": "/data3", "local_root": "/local"},
    ],
)
def test_config_rejects_unsafe_root_relationships(tmp_path, paths):
    with pytest.raises(ValueError, match="root"):
        load_config(_write_config(tmp_path, lambda raw: raw.update(paths=paths)))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: raw["render"].update(num_views=7),
        lambda raw: raw["render"].update(resolution=0),
        lambda raw: raw["render"].update(resolution=256),
        lambda raw: raw["render"].update(fov_min_degrees=80.0),
        lambda raw: raw["targets"].update(views=[1, 0]),
        lambda raw: raw["targets"].update(views=[False, 1]),
        lambda raw: raw["targets"].update(resolutions=[256, 512, 2048]),
        lambda raw: raw["targets"].update(ss_resolution=32),
        lambda raw: raw.update(shard_size=True),
        lambda raw: raw["workers"].update(cpu_threads=-1),
        lambda raw: raw["limits"].update(cpu_soft_percent=float("nan")),
    ],
)
def test_config_rejects_fixed_pipeline_invariant_changes(tmp_path, mutator):
    with pytest.raises(ValueError):
        load_config(_write_config(tmp_path, mutator))
