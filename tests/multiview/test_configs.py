import gc
import json
import os
from pathlib import Path

import pytest
import torch

from pixal3d import models


CONFIGS = {
    "ss64": Path("configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_multiview_ft64.json"),
    "shape512": Path("configs/gen/slat_flow_img2shape_dit_1_3B_256_bf16_proj_multiview_ft512.json"),
    "shape1024": Path("configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
    "pbr1024": Path("configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_proj_multiview_ft1024.json"),
}
CHECKPOINTS = {
    "ss64": "ss_flow_img_dit_1_3B_64_bf16.pt",
    "shape512": "slat_flow_img2shape_dit_1_3B_512_bf16.pt",
    "shape1024": "slat_flow_img2shape_dit_1_3B_1024_bf16.pt",
    "pbr1024": "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.pt",
}
OUTPUT_DIRS = {
    stage: f"/file3/youngwoo/pixal3d/ckpts/{stage}"
    for stage in CONFIGS
}
BATCH_POLICIES = {
    "ss64": (8, 4),
    "shape512": (8, 4),
    "shape1024": (2, 1),
    "pbr1024": (2, 1),
}


def test_four_configs_use_batchwide_k_and_matching_checkpoints():
    datasets = {
        "ss64": "MultiViewImageConditionedSparseStructureLatentView",
        "shape512": "MultiViewImageConditionedSLatShapeView",
        "shape1024": "MultiViewImageConditionedSLatShapeView",
        "pbr1024": "MultiViewImageConditionedSLatPbrView",
    }
    for stage, path in CONFIGS.items():
        config = json.loads(path.read_text())
        dataset_args = config["dataset"]["args"]
        trainer_args = config["trainer"]["args"]
        assert config["dataset"]["name"] == datasets[stage]
        assert dataset_args["condition_num_views"] == 8
        assert dataset_args["min_condition_views"] == 2
        assert dataset_args["max_condition_views"] == 6
        expected_batch, expected_split = BATCH_POLICIES[stage]
        assert trainer_args["batch_size_per_gpu"] == expected_batch
        assert trainer_args["batch_split"] == expected_split
        assert expected_batch * 6 == (
            48 if stage in {"ss64", "shape512"} else 12
        )
        assert trainer_args["i_sample"] == -1
        assert trainer_args["i_save"] == 2000
        assert trainer_args["max_checkpoints"] == 5
        assert trainer_args["max_steps"] == 20_000
        assert trainer_args["multiview_stage"] == stage
        assert trainer_args["image_cond_model"]["name"] == "DinoV3ProjFeatureExtractor"
        assert trainer_args["finetune_ckpt"] == {
            "denoiser": (
                "/file3/youngwoo/pixal3d/train/checkpoints/single_view/"
                + CHECKPOINTS[stage]
            )
        }
        assert config["models"]["denoiser"]["args"]["image_attn_mode"] == "proj"
        assert config["default_output_dir"] == OUTPUT_DIRS[stage]


def test_shape_512_is_an_independent_inference_checkpoint():
    config = json.loads(CONFIGS["shape512"].read_text())
    assert config["models"]["denoiser"]["args"]["resolution"] == 32
    assert config["dataset"]["args"]["resolution"] == 512
    assert config["trainer"]["args"]["finetune_ckpt"]["denoiser"].endswith(
        "slat_flow_img2shape_dit_1_3B_512_bf16.pt"
    )


@pytest.mark.integration
@pytest.mark.parametrize("stage", tuple(CONFIGS))
def test_released_checkpoint_strictly_matches_denoiser(stage):
    config = json.loads(CONFIGS[stage].read_text())
    checkpoint = Path(config["trainer"]["args"]["finetune_ckpt"]["denoiser"])
    if not checkpoint.exists():
        if os.environ.get("PIXAL3D_REQUIRE_CHECKPOINTS") == "1":
            pytest.fail(f"required checkpoint is missing: {checkpoint}")
        pytest.skip(f"checkpoint has not been materialized: {checkpoint}")
    model_config = config["models"]["denoiser"]
    denoiser = getattr(models, model_config["name"])(**model_config["args"])
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = denoiser.load_state_dict(state_dict, strict=False)
    assert set(incompatible.missing_keys) <= {"rope_phases"}
    assert incompatible.unexpected_keys == []
    del state_dict, denoiser
    gc.collect()
