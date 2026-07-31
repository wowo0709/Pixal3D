import json
import os
from pathlib import Path

import pytest
import torch

from pixal3d.datasets.components import load_anchor_first_conditions
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
)
from tests.multiview.test_configs import CONFIGS
from tests.multiview.test_pilot_dataset import PILOT, STAGE_ROOTS


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
@pytest.mark.parametrize("num_views", [2, 6])
def test_real_conditioner_outputs_are_finite(stage, num_views):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if os.environ.get("PIXAL3D_RUN_GPU_SMOKE") != "1":
        pytest.skip("set PIXAL3D_RUN_GPU_SMOKE=1 after the GPU0 resource gate")
    config = json.loads(CONFIGS[stage].read_text())
    roots = STAGE_ROOTS[stage]
    asset = sorted(path.name for path in roots["render_cond"].iterdir() if path.is_dir())[0]
    conditions = load_anchor_first_conditions(
        roots["render_cond"] / asset,
        anchor_index=0,
        image_size=config["dataset"]["args"]["image_size"],
        other_view_indices=[1, 2, 3, 4, 5, 6, 7],
    )
    latent_key = "pbr_latent" if stage == "pbr1024" else (
        "ss_latent" if stage == "ss64" else "shape_latent"
    )
    scale = json.loads((roots[latent_key] / asset / "view00_scale.json").read_text())
    model_args = config["trainer"]["args"]["image_cond_model"]["args"]
    conditioner = DinoV3ProjFeatureExtractor(**model_args).cuda().eval()
    with torch.no_grad():
        global_feature, projected_feature = conditioner(
            conditions["cond"][:num_views][None].cuda(),
            camera_angle_x=conditions["camera_angle_x"][:num_views][None].cuda(),
            distance=conditions["camera_distance"][:num_views][None].cuda(),
            mesh_scale=torch.tensor([scale["total_scale"]], device="cuda"),
            transform_matrix=conditions["transform_matrix"][:num_views][None].cuda(),
        )
    assert torch.isfinite(global_feature).all()
    assert torch.isfinite(projected_feature).all()
    assert global_feature.shape[0] == projected_feature.shape[0] == 1


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
def test_real_k1_matches_existing_single_view_path(stage):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if os.environ.get("PIXAL3D_RUN_GPU_SMOKE") != "1":
        pytest.skip("set PIXAL3D_RUN_GPU_SMOKE=1 after the GPU0 resource gate")
    config = json.loads(CONFIGS[stage].read_text())
    roots = STAGE_ROOTS[stage]
    asset = sorted(path.name for path in roots["render_cond"].iterdir() if path.is_dir())[0]
    conditions = load_anchor_first_conditions(
        roots["render_cond"] / asset,
        anchor_index=0,
        image_size=config["dataset"]["args"]["image_size"],
        other_view_indices=[1, 2, 3, 4, 5, 6, 7],
    )
    latent_key = "pbr_latent" if stage == "pbr1024" else (
        "ss_latent" if stage == "ss64" else "shape_latent"
    )
    scale = json.loads((roots[latent_key] / asset / "view00_scale.json").read_text())
    conditioner = DinoV3ProjFeatureExtractor(
        **config["trainer"]["args"]["image_cond_model"]["args"]
    ).cuda().eval()
    image = conditions["cond"][0][None].cuda()
    angle = conditions["camera_angle_x"][0][None].cuda()
    distance = conditions["camera_distance"][0][None].cuda()
    mesh_scale = torch.tensor([scale["total_scale"]], device="cuda")
    transform = conditions["transform_matrix"][0][None, None].cuda()
    with torch.no_grad():
        legacy_global, legacy_proj = conditioner(
            image,
            camera_angle_x=angle,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        k1_global, k1_proj = conditioner(
            image[:, None],
            camera_angle_x=angle[:, None],
            distance=distance[:, None],
            mesh_scale=mesh_scale,
            transform_matrix=transform,
        )
    torch.testing.assert_close(k1_global.float(), legacy_global.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k1_proj.float(), legacy_proj.float(), rtol=1e-5, atol=1e-5)
