import pytest
import torch
from PIL import Image

from pixal3d.pipelines.pixal3d_image_to_3d import (
    Pixal3DImageTo3DPipeline,
    normalize_calibrated_views,
)
from pixal3d.pipelines.projection_aggregation import (
    ProjectionAggregationConfig,
)
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid,
)


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


def test_missing_mesh_scale_warns_and_assumes_canonical_unit_scale():
    transforms = torch.eye(4).repeat(2, 1, 1)
    with pytest.warns(
        UserWarning,
        match="mesh_scale was not provided; assuming canonical unit scale \\(1.0\\)",
    ):
        _, cameras = normalize_calibrated_views(
            [image("red"), image("blue")],
            {
                "camera_angle_x": [0.7, 0.8],
                "distance": [2.5, 2.7],
                "transform_matrix": transforms,
            },
        )

    torch.testing.assert_close(cameras["mesh_scale"], torch.tensor([1.0]))


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


class RecordingConditioner(torch.nn.Module):
    def __init__(self, image_size=8, grid_resolution=2):
        super().__init__()
        self.image_size = image_size
        self.grid_resolution = grid_resolution
        self.proj_grid = ProjGrid(grid_resolution, image_size)
        self.calls = []

    @property
    def fixed_projection_transform(self):
        return self.proj_grid.front_view_transform_matrix

    def iter_view_features(self, image, **camera):
        num_views = image.shape[1] if image.ndim == 5 else 1
        for view_index in range(num_views):
            value = float(view_index + 1)
            yield (
                torch.full((1, 5, 4), value, device=image.device),
                torch.full(
                    (1, self.grid_resolution ** 3, 4),
                    value,
                    device=image.device,
                ),
            )

    def forward(self, image, **camera):
        self.calls.append(
            {
                "image": image.detach().clone(),
                "camera_angle_x": camera["camera_angle_x"].detach().clone(),
                "distance": camera["distance"].detach().clone(),
                "mesh_scale": camera["mesh_scale"].detach().clone(),
                "transform_matrix": (
                    None
                    if camera["transform_matrix"] is None
                    else camera["transform_matrix"].detach().clone()
                ),
            }
        )
        groups = list(self.iter_view_features(image, **camera))
        return tuple(
            torch.stack([group[index] for group in groups])
            .float()
            .mean(dim=0)
            for index in range(2)
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
    transforms[0, 0, 3] = 1.0
    transforms[1, 0, 3] = 2.0
    cameras = {
        "camera_angle_x": [[0.7, 0.8]],
        "distance": [[2.5, 2.7]],
        "mesh_scale": [0.6],
        "transform_matrix": transforms[None],
    }
    pipeline.get_proj_cond_ss(images, **cameras)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    for conditioner in conditioners[1:]:
        pipeline.get_proj_cond_shape(conditioner, images, coords, **cameras)

    red = torch.tensor([1.0, 0.0, 0.0])[:, None, None].expand(3, 8, 8)
    blue = torch.tensor([0.0, 0.0, 1.0])[:, None, None].expand(3, 8, 8)
    expected = {
        "image": torch.stack([red, blue]).unsqueeze(0),
        "camera_angle_x": torch.tensor([[0.7, 0.8]]),
        "distance": torch.tensor([[2.5, 2.7]]),
        "mesh_scale": torch.tensor([0.6]),
        "transform_matrix": transforms.unsqueeze(0),
    }
    for conditioner in conditioners:
        assert len(conditioner.calls) == 1
        call = conditioner.calls[0]
        for name, value in expected.items():
            torch.testing.assert_close(call[name], value, rtol=0, atol=0)


def test_shape_conditioner_preserves_positional_grid_resolution_override():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner()
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)

    pipeline.get_proj_cond_shape(
        conditioner, [image("red")], coords, 0.7, 2.5, 1.0, 3
    )

    assert len(conditioner.calls) == 1
    assert conditioner.calls[0]["image"].shape == torch.Size([1, 3, 8, 8])
    assert conditioner.calls[0]["transform_matrix"] is None
    assert conditioner.grid_resolution == 2
    assert conditioner.proj_grid.grid_resolution == 2


def test_sparse_first_explicit_mean_matches_default_dense_mean():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner(grid_resolution=2)
    images = [image("red"), image("blue")]
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32
    )
    transforms = torch.eye(4).repeat(1, 2, 1, 1)
    cameras = {
        "camera_angle_x": torch.tensor([[0.7, 0.8]]),
        "distance": torch.tensor([[2.5, 2.7]]),
        "mesh_scale": torch.tensor([1.0]),
        "transform_matrix": transforms,
    }

    default = pipeline.get_proj_cond_shape(
        conditioner, images, coords, **cameras
    )
    experimental = pipeline.get_proj_cond_shape(
        conditioner,
        images,
        coords,
        aggregation_config=ProjectionAggregationConfig(mode="mean"),
        **cameras,
    )

    torch.testing.assert_close(
        experimental["cond"]["global"], default["cond"]["global"]
    )
    torch.testing.assert_close(
        experimental["cond"]["proj"].feats,
        default["cond"]["proj"].feats,
    )
    assert experimental["cond"]["proj"].feats.shape == (2, 4)
    assert torch.equal(
        experimental["neg_cond"]["proj"].coords,
        coords,
    )
    assert torch.count_nonzero(
        experimental["neg_cond"]["proj"].feats
    ) == 0


def test_sparse_first_rejects_nonzero_sparse_batch_indices():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner()
    coords = torch.tensor([[1, 0, 0, 0]], dtype=torch.int32)
    with pytest.raises(ValueError, match="B=1"):
        pipeline.get_proj_cond_shape(
            conditioner,
            [image("red")],
            coords,
            0.7,
            2.5,
            1.0,
            aggregation_config=ProjectionAggregationConfig(mode="mean"),
        )


class FailingIteratorConditioner(RecordingConditioner):
    def iter_view_features(self, *args, **kwargs):
        raise RuntimeError("synthetic iterator failure")
        yield


def test_sparse_first_restores_grid_override_after_iterator_error():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = FailingIteratorConditioner()
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="synthetic iterator failure"):
        pipeline.get_proj_cond_shape(
            conditioner,
            [image("red")],
            coords,
            0.7,
            2.5,
            1.0,
            grid_resolution_override=3,
            aggregation_config=ProjectionAggregationConfig(mode="mean"),
        )
    assert conditioner.grid_resolution == 2
    assert conditioner.proj_grid.grid_resolution == 2


def test_oracle_mask_uses_same_active_projection_coordinates(monkeypatch):
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner(grid_resolution=2)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 4:, 4:] = 1.0
    diagnostics = {}
    transforms = torch.eye(4).repeat(1, 2, 1, 1)

    pipeline.get_proj_cond_shape(
        conditioner,
        [image("red"), image("blue")],
        coords,
        camera_angle_x=torch.tensor([[0.7, 0.7]]),
        distance=torch.tensor([[2.5, 2.5]]),
        mesh_scale=torch.tensor([1.0]),
        transform_matrix=transforms,
        aggregation_config=ProjectionAggregationConfig(
            mode="oracle", alpha=1.0
        ),
        oracle_masks=masks,
        diagnostics=diagnostics,
    )

    assert diagnostics["projected_corruption"].shape == (2, 1)
    assert diagnostics["pixel_xy"].shape == (2, 1, 2)
    assert diagnostics["depth"].shape == (2, 1)
    assert diagnostics["valid_mask"].shape == (2, 1)


def test_consensus_oracle_mask_is_diagnostic_only():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner(grid_resolution=2)
    images = [image("red"), image("blue")]
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    cameras = {
        "camera_angle_x": torch.tensor([[0.7, 0.7]]),
        "distance": torch.tensor([[2.5, 2.5]]),
        "mesh_scale": torch.tensor([1.0]),
        "transform_matrix": torch.eye(4).repeat(1, 2, 1, 1),
    }
    masks = torch.zeros(2, 1, 8, 8)
    masks[1, :, 4:, 4:] = 1.0
    config = ProjectionAggregationConfig(
        mode="consensus", alpha=1.0, temperature=0.1
    )
    without_mask = pipeline.get_proj_cond_shape(
        conditioner,
        images,
        coords,
        aggregation_config=config,
        oracle_masks=None,
        diagnostics={},
        **cameras,
    )
    diagnostics = {}
    with_mask = pipeline.get_proj_cond_shape(
        conditioner,
        images,
        coords,
        aggregation_config=config,
        oracle_masks=masks,
        diagnostics=diagnostics,
        **cameras,
    )
    torch.testing.assert_close(
        with_mask["cond"]["proj"].feats,
        without_mask["cond"]["proj"].feats,
    )
    assert diagnostics["projected_corruption"].shape == (
        len(images), coords.shape[0]
    )


def test_uncalibrated_single_view_projects_oracle_mask():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner(grid_resolution=2)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    diagnostics = {}

    pipeline.get_proj_cond_shape(
        conditioner,
        [image("red")],
        coords,
        camera_angle_x=torch.tensor([[0.7]]),
        distance=torch.tensor([[2.5]]),
        mesh_scale=torch.tensor([1.0]),
        transform_matrix=None,
        aggregation_config=ProjectionAggregationConfig(
            mode="oracle", alpha=1.0
        ),
        oracle_masks=torch.zeros(1, 1, 8, 8),
        diagnostics=diagnostics,
    )

    assert diagnostics["projected_corruption"].shape == (1, 1)
    assert diagnostics["pixel_xy"].shape == (1, 1, 2)


def test_oracle_mode_requires_oracle_masks():
    pipeline = Pixal3DImageTo3DPipeline()
    pipeline._device = "cpu"
    pipeline.low_vram = False
    conditioner = RecordingConditioner(grid_resolution=2)
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)

    with pytest.raises(ValueError, match="oracle mode requires oracle_masks"):
        pipeline.get_proj_cond_shape(
            conditioner,
            [image("red")],
            coords,
            aggregation_config=ProjectionAggregationConfig(mode="oracle"),
        )
