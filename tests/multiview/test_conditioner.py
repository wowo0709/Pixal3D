import pytest
import torch
import torch.nn as nn

from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    DinoV3ProjFeatureExtractor,
    ImageConditionedProjMixin,
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
        if transform_matrix is None:
            transform_matrix = self.front[None].expand(image.shape[0], -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance
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


def test_iter_view_features_matches_current_per_view_projection_order():
    model = ConditionerHarness()
    image = torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    camera = cameras(4)

    groups = list(model.iter_view_features(image, **camera))

    assert len(groups) == 4
    default_global, default_projected = model(image, **camera)
    expected_global = torch.stack([group[0] for group in groups]).mean(dim=0)
    expected_projected = torch.stack([group[1] for group in groups]).mean(dim=0)
    torch.testing.assert_close(default_global, expected_global)
    torch.testing.assert_close(default_projected, expected_projected)


def test_iter_view_features_supports_uncalibrated_k1_without_transform():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    groups = list(
        model.iter_view_features(
            image,
            camera_angle_x=torch.tensor([0.7]),
            distance=torch.tensor([2.5]),
            mesh_scale=torch.ones(1),
            transform_matrix=None,
        )
    )
    assert len(groups) == 1
    expected = model._forward_single_view(
        image, torch.tensor([0.7]), torch.tensor([2.5]), torch.ones(1), None
    )
    for actual, reference in zip(groups[0], expected):
        torch.testing.assert_close(actual, reference)


def test_k1_matches_single_view_with_exact_fp32_gate():
    model = ConditionerHarness()
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    camera = cameras(1)
    camera["transform_matrix"][0, 0] = torch.tensor([
        [0.8660254, -0.5, 0.0, 1.25],
        [0.5, 0.8660254, 0.0, -2.5],
        [0.0, 0.0, 1.0, 3.75],
        [0.0, 0.0, 0.0, 1.0],
    ])
    actual = model(image[:, None], **camera)
    fixed = model.front[None].clone()
    fixed[:, 1, 3] = -2.5
    expected = model._forward_single_view(
        image, camera["camera_angle_x"][:, 0], camera["distance"][:, 0],
        camera["mesh_scale"], fixed,
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


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


def test_multiview_delegates_a_lazy_view_stream_to_online_mean(monkeypatch):
    expected = (torch.tensor([17.0]), torch.tensor([23.0]))
    observed = {}

    def inspect_view_stream(groups):
        observed["is_one_shot_iterator"] = iter(groups) is groups
        observed["num_groups"] = sum(1 for _ in groups)
        return expected

    monkeypatch.setattr(
        image_conditioned_proj,
        "_online_mean_tensor_groups",
        inspect_view_stream,
    )
    actual = ConditionerHarness()(
        torch.zeros(1, 3, 3, 2, 2),
        **cameras(3),
    )

    assert actual is expected
    assert observed == {
        "is_one_shot_iterator": True,
        "num_groups": 3,
    }


@pytest.mark.parametrize("key", ["camera_angle_x", "distance", "transform_matrix"])
def test_multiview_rejects_misaligned_camera_shapes(key):
    camera = cameras(2)
    camera[key] = camera[key][:, :1]
    with pytest.raises(ValueError, match=key):
        ConditionerHarness()(torch.zeros(1, 2, 3, 2, 2), **camera)


def test_multiview_conditioner_adds_no_trainable_parameter():
    assert [p for p in ConditionerHarness().parameters() if p.requires_grad] == []


class _ConditioningSink:
    def get_cond(self, cond, **kwargs):
        return {"cond": cond, **kwargs}

    def get_inference_cond(self, cond, **kwargs):
        return {"cond": cond, **kwargs}


class ProjectionConditioningHarness(ImageConditionedProjMixin, _ConditioningSink):
    image_attn_mode = "proj"

    def __init__(self):
        pass

    def encode_image_proj(self, cond, **camera_info):
        self.received_camera_info = camera_info
        encoded = {"global": cond}
        negative = {"global": torch.zeros_like(cond)}
        return encoded, negative


@pytest.mark.parametrize("method_name", ["get_cond", "get_inference_cond"])
def test_projection_conditioning_consumes_view_indices_without_mutating_caller(
    method_name,
):
    harness = ProjectionConditioningHarness()
    view_indices = torch.tensor([[3, 1, 2]])
    camera_info = {
        "camera_angle_x": torch.tensor([[0.7, 0.8, 0.9]]),
        "distance": torch.tensor([[2.0, 2.1, 2.2]]),
        "mesh_scale": torch.ones(1),
        "transform_matrix": torch.eye(4).repeat(1, 3, 1, 1),
        "coords": None,
    }
    caller_data = {
        "cond": torch.ones(1, 4),
        "camera_info": camera_info,
        "view_indices": view_indices,
    }

    result = getattr(harness, method_name)(**caller_data)

    assert "view_indices" not in result
    assert harness.received_camera_info == camera_info
    assert caller_data["view_indices"] is view_indices
    torch.testing.assert_close(caller_data["view_indices"], torch.tensor([[3, 1, 2]]))
