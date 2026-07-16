from pathlib import Path

import pytest

from data_toolkit.pipeline.config import load_config


@pytest.fixture
def config():
    return load_config(Path("data_toolkit/configs/multiview_preprocess.yaml"))
