import pytest
import torch
from PIL import Image

from pixal3d.pipelines.pixal3d_image_to_3d import normalize_calibrated_views


def image(color):
    return Image.new("RGB", (8, 8), color=color)


def test_uncalibrated_single_view_normalizes_to_legacy_k1():
    images, cameras = normalize_calibrated_views(
        image("red"),
        {"camera_angle_x": 0.7, "distance": 2.5, "mesh_scale": 1.0},
    )
    assert len(images) == 1
    assert cameras["camera_angle_x"].shape == (1, 1)
    assert cameras["distance"].shape == (1, 1)
    assert cameras["mesh_scale"].shape == (1,)
    assert cameras["transform_matrix"] is None


def test_calibrated_multiview_preserves_order_and_shapes():
    transforms = torch.eye(4).repeat(2, 1, 1)
    images, cameras = normalize_calibrated_views(
        [image("red"), image("blue")],
        {
            "camera_angle_x": [0.7, 0.8],
            "distance": [2.5, 2.7],
            "mesh_scale": 1.0,
            "transform_matrix": transforms,
        },
    )
    assert [view.getpixel((0, 0)) for view in images] == [
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert cameras["camera_angle_x"].shape == (1, 2)
    assert cameras["distance"].shape == (1, 2)
    assert cameras["transform_matrix"].shape == (1, 2, 4, 4)


@pytest.mark.parametrize("num_views", [0, 9])
def test_inference_rejects_view_counts_outside_one_to_eight(num_views):
    with pytest.raises(ValueError, match="between 1 and 8"):
        normalize_calibrated_views(
            [image("red")] * num_views,
            {
                "camera_angle_x": [0.7] * num_views,
                "distance": [2.5] * num_views,
            },
        )


def test_multiview_requires_calibrated_transforms():
    with pytest.raises(ValueError, match="transform_matrix"):
        normalize_calibrated_views(
            [image("red"), image("blue")],
            {"camera_angle_x": [0.7, 0.8], "distance": [2.5, 2.7]},
        )


class RecordingGrid(torch.nn.Module):
    def __init__(self, grid_resolution=2, image_resolution=8):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution


class RecordingConditioner(torch.nn.Module):
    def __init__(self, image_size=8, grid_resolution=2):
        super().__init__()
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.proj_grid = RecordingGrid(grid_resolution, image_size)
        self.calls = []

    def forward(self, image, **camera):
        transform_matrix = camera["transform_matrix"]
        transform_shape = (
            None if transform_matrix is None else transform_matrix.shape
        )
        self.calls.append((image.shape, transform_shape))
        batch = image.shape[0]
        return (
            torch.zeros(batch, 5, 4),
            torch.zeros(batch, self.grid_resolution ** 3, 4),
        )


def test_all_four_inference_conditioners_receive_the_same_k2_bundle():
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioners = [RecordingConditioner() for _ in range(4)]
    pipeline.image_cond_model_ss = conditioners[0]
    images = [image("red"), image("blue")]
    transforms = torch.eye(4).repeat(2, 1, 1)
    cameras = {
        "camera_angle_x": [[0.7, 0.8]],
        "distance": [[2.5, 2.7]],
        "mesh_scale": [1.0],
        "transform_matrix": transforms[None],
    }
    pipeline.get_proj_cond_ss(images, **cameras)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    for conditioner in conditioners[1:]:
        pipeline.get_proj_cond_shape(conditioner, images, coords, **cameras)
    assert [call[0][1] for model in conditioners for call in model.calls] == [
        2,
        2,
        2,
        2,
    ]
    assert [call[1][1] for model in conditioners for call in model.calls] == [
        2,
        2,
        2,
        2,
    ]


def test_shape_conditioner_preserves_positional_grid_resolution_override():
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner()
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)

    pipeline.get_proj_cond_shape(
        conditioner, [image("red")], coords, 0.7, 2.5, 1.0, 3
    )

    assert conditioner.calls == [(torch.Size([1, 3, 8, 8]), None)]
    assert conditioner.grid_resolution == 2
    assert conditioner.proj_grid.grid_resolution == 2
