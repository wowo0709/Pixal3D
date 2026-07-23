import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from pixal3d import datasets
from tests.multiview.test_configs import CONFIGS


PILOT = Path("/root/node17/data/pixal3d/train/development/abo-pilot64")
STAGE_ROOTS = {
    "ss64": {
        "base": PILOT / "ss64/active",
        "render_cond": PILOT / "ss64/active/renders_cond",
        "ss_latent": PILOT / "ss64/active/ss_latents/ss_enc_conv3d_16l8_fp16_64_view",
    },
    "shape512": {
        "base": PILOT / "shape512/active",
        "render_cond": PILOT / "shape512/active/renders_cond",
        "shape_latent": PILOT / "shape512/active/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view",
    },
    "shape1024": {
        "base": PILOT / "shape1024/active",
        "render_cond": PILOT / "shape1024/active/renders_cond",
        "shape_latent": PILOT / "shape1024/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
    },
    "pbr1024": {
        "base": PILOT / "pbr1024/active",
        "render_cond": PILOT / "pbr1024/active/renders_cond",
        "shape_latent": PILOT / "pbr1024/active/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view",
        "pbr_latent": PILOT / "pbr1024/active/pbr_latents/tex_enc_next_dc_f16c32_fp16_1024_view_fix",
    },
}


def require_pilot():
    if not PILOT.exists():
        if os.environ.get("PIXAL3D_REQUIRE_PILOT") == "1":
            pytest.fail(f"pilot root is missing: {PILOT}")
        pytest.skip(f"pilot root is missing: {PILOT}")


def make_dataset(stage):
    config = json.loads(CONFIGS[stage].read_text())
    roots = {"ABO": {key: str(value) for key, value in STAGE_ROOTS[stage].items()}}
    return getattr(datasets, config["dataset"]["name"])(
        json.dumps(roots), **config["dataset"]["args"]
    )


@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
@pytest.mark.parametrize("anchor", [0, 1])
def test_pilot_stage_loads_anchor_first(stage, anchor, monkeypatch):
    require_pilot()
    dataset = make_dataset(stage)
    assert len(dataset) == 64
    root, asset, _ = dataset.instances[0]
    monkeypatch.setattr(np.random, "randint", lambda low, high: anchor)
    pack = dataset.get_instance(root, asset)
    assert pack["view_idx"] == anchor
    assert pack["view_indices"][0].item() == anchor
    assert sorted(pack["view_indices"].tolist()) == list(range(8))
    assert pack["cond"].shape[0] == 8
    assert pack["transform_matrix"].shape == (8, 4, 4)
    assert torch.isfinite(pack["cond"]).all()
    assert torch.isfinite(pack["transform_matrix"]).all()


@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
@pytest.mark.parametrize("num_views", [2, 6])
def test_pilot_collation_forces_batchwide_endpoint(stage, num_views, monkeypatch):
    require_pilot()
    dataset = make_dataset(stage)
    first = dataset[0]
    second = dataset[1]
    monkeypatch.setattr(np.random, "randint", lambda low, high: num_views)
    batch = dataset.collate_fn([first, second])
    assert batch["cond"].shape[:2] == (2, num_views)
    assert batch["camera_angle_x"].shape == (2, num_views)
    assert batch["transform_matrix"].shape == (2, num_views, 4, 4)
