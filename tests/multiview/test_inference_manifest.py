import json
import warnings

import numpy as np
import pytest
import torch
from PIL import Image
from pixal3d.pipelines.projection_aggregation import ProjectionAggregationConfig

import inference
from inference import (
    FLOW_MODEL_KEYS,
    build_parser,
    load_calibrated_manifest,
    load_flow_overrides,
)


def test_projection_aggregation_cli_is_opt_in():
    args = build_parser().parse_args(["--image", "input.png"])

    assert args.projection_aggregation is None
    assert args.projection_stages == "shape512,shape1024,pbr1024"
    assert args.projection_alpha == 0.5
    assert args.projection_temperature == 0.1


def test_projection_aggregation_cli_parses_consensus():
    args = build_parser().parse_args(
        [
            "--transforms",
            "transforms.json",
            "--projection_aggregation",
            "consensus",
            "--projection_stages",
            "shape512,pbr1024",
            "--projection_alpha",
            "0.25",
            "--projection_temperature",
            "0.2",
        ]
    )

    assert args.projection_aggregation == "consensus"
    assert args.projection_stages == "shape512,pbr1024"
    assert args.projection_alpha == 0.25
    assert args.projection_temperature == 0.2


def write_manifest(tmp_path, frames, **metadata):
    manifest = tmp_path / "transforms.json"
    manifest.write_text(json.dumps({"frames": frames, **metadata}))
    return manifest


def calibrated_frame(file_path, *, angle=0.7, transform=None):
    return {
        "file_path": file_path,
        "camera_angle_x": angle,
        "transform_matrix": (
            np.eye(4, dtype=np.float32) if transform is None else transform
        ).tolist(),
    }


def test_manifest_preserves_first_frame_anchor_order(tmp_path):
    frames = []
    for index in range(2):
        Image.new("RGBA", (4, 4), color=(index * 10, 0, 0, 255)).save(
            tmp_path / f"{index:03d}.png"
        )
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = index
        transform[2, 3] = 2.0
        frames.append(calibrated_frame(f"{index:03d}.png", transform=transform))

    images, cameras = load_calibrated_manifest(
        write_manifest(tmp_path, frames), mesh_scale=1.25
    )

    assert [image.getpixel((0, 0)) for image in images] == [
        (0, 0, 0, 255),
        (10, 0, 0, 255),
    ]
    assert cameras["camera_angle_x"] == [0.7, 0.7]
    assert cameras["distance"] == pytest.approx([2.0, np.sqrt(5.0)])
    assert np.asarray(cameras["transform_matrix"]).shape == (2, 4, 4)
    assert cameras["mesh_scale"] == 1.25


@pytest.mark.parametrize("frame_count", [0, 9])
def test_manifest_rejects_frame_counts_outside_one_to_eight(
    tmp_path, frame_count
):
    manifest = write_manifest(tmp_path, [{}] * frame_count)

    with pytest.raises(ValueError, match="between 1 and 8"):
        load_calibrated_manifest(manifest, mesh_scale=1.0)


def test_manifest_rejects_frame_path_escape(tmp_path):
    manifest = write_manifest(
        tmp_path,
        [calibrated_frame("../outside.png")],
    )

    with pytest.raises(ValueError, match="inside"):
        load_calibrated_manifest(manifest, mesh_scale=1.0)


def test_manifest_rejects_symlink_whose_target_escapes_directory(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    Image.new("RGBA", (4, 4)).save(outside)
    (tmp_path / "linked.png").symlink_to(outside)
    manifest = write_manifest(
        tmp_path,
        [calibrated_frame("linked.png")],
    )

    with pytest.raises(ValueError, match="inside"):
        load_calibrated_manifest(manifest, mesh_scale=1.0)


@pytest.mark.parametrize("angle", [float("nan"), float("inf")])
def test_manifest_rejects_nonfinite_camera_angle(tmp_path, angle):
    Image.new("RGBA", (4, 4)).save(tmp_path / "000.png")
    manifest = write_manifest(
        tmp_path,
        [calibrated_frame("000.png", angle=angle)],
    )

    with pytest.raises(ValueError, match="camera_angle_x must be finite"):
        load_calibrated_manifest(manifest, mesh_scale=1.0)


def test_manifest_rejects_camera_angle_outside_float32_range(tmp_path):
    Image.new("RGBA", (4, 4)).save(tmp_path / "000.png")
    manifest = write_manifest(
        tmp_path,
        [calibrated_frame("000.png", angle=1e39)],
    )

    with pytest.raises(ValueError, match="camera_angle_x must be finite"):
        load_calibrated_manifest(manifest, mesh_scale=1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_manifest_rejects_nonfinite_transform(tmp_path, value):
    Image.new("RGBA", (4, 4)).save(tmp_path / "000.png")
    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = value
    manifest = write_manifest(
        tmp_path,
        [calibrated_frame("000.png", transform=transform)],
    )

    with pytest.raises(ValueError, match=r"transform_matrix must be finite \[4, 4\]"):
        load_calibrated_manifest(manifest, mesh_scale=1.0)


def test_manifest_rejects_distance_outside_float32_range(tmp_path):
    Image.new("RGBA", (4, 4)).save(tmp_path / "000.png")
    transform = np.eye(4, dtype=np.float32)
    transform[:2, 3] = 3e38
    manifest = write_manifest(
        tmp_path,
        [calibrated_frame("000.png", transform=transform)],
    )

    with pytest.raises(ValueError, match="distance must be finite"):
        load_calibrated_manifest(manifest, mesh_scale=1.0)


def test_manifest_uses_top_level_camera_angle(tmp_path):
    Image.new("RGBA", (4, 4)).save(tmp_path / "000.png")
    frame = calibrated_frame("000.png")
    del frame["camera_angle_x"]

    _, cameras = load_calibrated_manifest(
        write_manifest(tmp_path, [frame], camera_angle_x=0.65),
        mesh_scale=1.0,
    )

    assert cameras["camera_angle_x"] == [0.65]


def test_manifest_missing_mesh_scale_warns_once_and_assumes_unit_scale(tmp_path):
    Image.new("RGBA", (4, 4)).save(tmp_path / "000.png")
    manifest = write_manifest(tmp_path, [calibrated_frame("000.png")])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, cameras = load_calibrated_manifest(manifest, mesh_scale=None)

    scale_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and "assuming canonical unit scale (1.0)" in str(warning.message)
    ]
    assert len(scale_warnings) == 1
    assert cameras["mesh_scale"] == 1.0


class RecordingPipeline:
    def __init__(self):
        self.models = {}
        self.pbr_attr_layout = None
        self.camera_params = None

    def preprocess_image(self, image):
        return image

    def run(self, image, *, camera_params, **kwargs):
        self.camera_params = camera_params
        self.run_kwargs = kwargs
        mesh = type(
            "Mesh",
            (),
            {
                "vertices": None,
                "faces": None,
                "attrs": None,
                "coords": None,
            },
        )()
        return [mesh], (None, None, 1)


class RecordingGlb:
    def apply_transform(self, transform):
        pass

    def export(self, output_path, *, extension_webp):
        pass


def test_run_inference_forwards_projection_aggregation(tmp_path, monkeypatch):
    pipeline = RecordingPipeline()
    monkeypatch.setattr(inference, "init_pipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(
        inference.o_voxel.postprocess,
        "to_glb",
        lambda **kwargs: RecordingGlb(),
    )
    image_path = tmp_path / "input.png"
    Image.new("RGBA", (4, 4)).save(image_path)

    inference.run_inference(
        image_path=str(image_path),
        output_path=str(tmp_path / "output.glb"),
        manual_fov=0.7,
        projection_aggregation={
            "shape512": ProjectionAggregationConfig(mode="consensus")
        },
    )

    assert (
        pipeline.run_kwargs["projection_aggregation"]["shape512"].mode
        == "consensus"
    )


def test_programmatic_calibrated_inference_omission_warns_once(
    tmp_path, monkeypatch
):
    Image.new("RGBA", (4, 4)).save(tmp_path / "000.png")
    manifest = write_manifest(tmp_path, [calibrated_frame("000.png")])
    pipeline = RecordingPipeline()
    monkeypatch.setattr(inference, "init_pipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(
        inference.o_voxel.postprocess,
        "to_glb",
        lambda **kwargs: RecordingGlb(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inference.run_inference(
            image_path=None,
            output_path=str(tmp_path / "output.glb"),
            transforms_path=str(manifest),
        )

    scale_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and "assuming canonical unit scale (1.0)" in str(warning.message)
    ]
    assert len(scale_warnings) == 1
    assert pipeline.camera_params["mesh_scale"] == 1.0


@pytest.mark.parametrize("mesh_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_manifest_rejects_invalid_explicit_mesh_scale(tmp_path, mesh_scale):
    Image.new("RGBA", (4, 4)).save(tmp_path / "000.png")
    manifest = write_manifest(tmp_path, [calibrated_frame("000.png")])

    with pytest.raises(ValueError, match="mesh_scale must be finite and positive"):
        load_calibrated_manifest(manifest, mesh_scale=mesh_scale)


def test_flow_overrides_load_the_four_exact_pipeline_models(tmp_path):
    expected_model_keys = (
        "sparse_structure_flow_model",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "tex_slat_flow_model_1024",
    )
    assert FLOW_MODEL_KEYS == expected_model_keys
    pipeline = type("Pipeline", (), {})()
    pipeline.models = {
        key: torch.nn.Linear(2, 2, bias=False) for key in expected_model_keys
    }
    overrides = {}
    for index, key in enumerate(expected_model_keys):
        checkpoint = tmp_path / f"{key}.pt"
        torch.save(
            {"weight": torch.full((2, 2), float(index + 1))},
            checkpoint,
        )
        overrides[key] = checkpoint

    load_flow_overrides(pipeline, overrides)

    for index, key in enumerate(expected_model_keys):
        assert torch.equal(
            pipeline.models[key].weight,
            torch.full((2, 2), float(index + 1)),
        )


def test_flow_overrides_reject_unknown_model_key(tmp_path):
    pipeline = type("Pipeline", (), {"models": {}})()

    with pytest.raises(ValueError, match="unknown flow checkpoint overrides"):
        load_flow_overrides(pipeline, {"not_a_pipeline_model": tmp_path / "x.pt"})


class ModelWithRopePhases(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.register_buffer("rope_phases", torch.ones(2))


def test_flow_overrides_allow_only_missing_rope_phases(tmp_path):
    checkpoint = tmp_path / "ss.pt"
    expected = torch.full((2, 2), 3.0)
    torch.save({"weight": expected}, checkpoint)
    pipeline = type("Pipeline", (), {})()
    pipeline.models = {"sparse_structure_flow_model": ModelWithRopePhases()}

    load_flow_overrides(
        pipeline,
        {"sparse_structure_flow_model": checkpoint},
    )

    assert torch.equal(
        pipeline.models["sparse_structure_flow_model"].weight,
        expected,
    )


@pytest.mark.parametrize(
    "state_dict, match",
    [
        ({}, "missing=\\['rope_phases', 'weight'\\]"),
        (
            {
                "weight": torch.zeros(2, 2),
                "rope_phases": torch.ones(2),
                "unexpected": torch.ones(1),
            },
            "unexpected=\\['unexpected'\\]",
        ),
    ],
)
def test_flow_overrides_reject_other_checkpoint_incompatibilities(
    tmp_path, state_dict, match
):
    checkpoint = tmp_path / "ss.pt"
    torch.save(state_dict, checkpoint)
    pipeline = type("Pipeline", (), {})()
    pipeline.models = {"sparse_structure_flow_model": ModelWithRopePhases()}

    with pytest.raises(RuntimeError, match=match):
        load_flow_overrides(
            pipeline,
            {"sparse_structure_flow_model": checkpoint},
        )


def test_cli_requires_exactly_one_image_source_and_observes_scale_omission():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--image", "image.png", "--transforms", "transforms.json"])

    single = parser.parse_args(["--image", "image.png"])
    calibrated = parser.parse_args(["--transforms", "transforms.json"])
    assert single.image == "image.png"
    assert single.transforms is None
    assert calibrated.transforms == "transforms.json"
    assert calibrated.image is None
    assert single.mesh_scale is None
    assert calibrated.mesh_scale is None
