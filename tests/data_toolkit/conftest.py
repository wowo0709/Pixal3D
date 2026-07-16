from pathlib import Path

import pytest
import yaml

from data_toolkit.pipeline.config import load_config


@pytest.fixture
def config():
    return load_config(Path("data_toolkit/configs/multiview_preprocess.yaml"))


@pytest.fixture
def tmp_config(tmp_path):
    raw = yaml.safe_load(
        Path("data_toolkit/configs/multiview_preprocess.yaml").read_text()
    )
    raw["paths"] = {
        "data2_root": str(tmp_path / "data2"),
        "data3_root": str(tmp_path / "data3"),
        "local_root": str(tmp_path / "local"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path
