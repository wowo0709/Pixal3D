import pytest
import torch
import torch.nn as nn

from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
)


class ConditionerHarness(DinoV3ProjFeatureExtractor):
    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer("front", torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))

    @property
    def fixed_projection_transform(self):
        return self.front

    def _forward_single_view(
        self, image, camera_angle_x, distance, mesh_scale, transform_matrix
    ):
        global_feature = image.mean(dim=(-2, -1))[:, None, :]
        projected = transform_matrix[:, :3, :4].reshape(image.shape[0], 1, 12)
        return global_feature, projected


def cameras(k):
    transforms = torch.eye(4).repeat(1, k, 1, 1)
    transforms[0, :, 0, 3] = torch.arange(k, dtype=torch.float32)
    return {
        "camera_angle_x": torch.full((1, k), 0.7),
        "distance": torch.full((1, k), 2.5),
        "mesh_scale": torch.ones(1),
        "transform_matrix": transforms,
    }


def test_k1_matches_single_view_with_exact_fp32_gate():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    camera = cameras(1)
    actual = model(image[:, None], **camera)
    fixed = model.front[None].clone()
    fixed[:, 1, 3] = -2.5
    expected = model._forward_single_view(
        image, camera["camera_angle_x"][:, 0], camera["distance"][:, 0],
        camera["mesh_scale"], fixed,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_repeated_view_mean_matches_single_view():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    repeated = image[:, None].repeat(1, 4, 1, 1, 1)
    camera = cameras(4)
    camera["transform_matrix"][:] = torch.eye(4)
    actual = model(repeated, **camera)
    expected = model(image[:, None], **cameras(1))
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-5)


def test_non_anchor_permutation_preserves_arithmetic_mean():
    model = ConditionerHarness()
    image = torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    camera = cameras(4)
    first = model(image, **camera)
    order = torch.tensor([0, 3, 1, 2])
    second = model(
        image[:, order],
        camera_angle_x=camera["camera_angle_x"][:, order],
        distance=camera["distance"][:, order],
        mesh_scale=camera["mesh_scale"],
        transform_matrix=camera["transform_matrix"][:, order],
    )
    torch.testing.assert_close(first[0], second[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(first[1], second[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("key", ["camera_angle_x", "distance", "transform_matrix"])
def test_multiview_rejects_misaligned_camera_shapes(key):
    camera = cameras(2)
    camera[key] = camera[key][:, :1]
    with pytest.raises(ValueError, match=key):
        ConditionerHarness()(torch.zeros(1, 2, 3, 2, 2), **camera)


def test_multiview_conditioner_adds_no_trainable_parameter():
    assert [p for p in ConditionerHarness().parameters() if p.requires_grad] == []
