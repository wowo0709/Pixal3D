import weakref

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
        self.multiview_aggregation = None
        self._last_multiview_aggregation_diagnostics = None
        self.single_view_calls = 0

    @property
    def fixed_projection_transform(self):
        return self.front

    def _forward_single_view(
        self, image, camera_angle_x, distance, mesh_scale, transform_matrix
    ):
        self.single_view_calls += 1
        global_feature = image.mean(dim=(-2, -1))[:, None, :]
        projected = transform_matrix[:, :3, :4].reshape(image.shape[0], 1, 12)
        return global_feature, projected


class ConsensusConditionerHarness(ConditionerHarness):
    def _forward_single_view(
        self, image, camera_angle_x, distance, mesh_scale, transform_matrix,
    ):
        self.single_view_calls += 1
        value = image[:, 0, 0, 0]
        global_feature = value[:, None, None]
        projected = value[:, None, None].expand(-1, 2, 4).clone()
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


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ({}, "mode"),
        ({"mode": "unsupported"}, "mode"),
        (
            {"mode": "consensus", "temperature": 0.2, "chunk_size": 1,
             "cache_device": "cpu"},
            "alpha",
        ),
        (
            {"mode": "consensus", "alpha": 1.0, "chunk_size": 1,
             "cache_device": "cpu"},
            "temperature",
        ),
        (
            {"mode": "consensus", "alpha": 1.0, "temperature": 0.2,
             "cache_device": "cpu"},
            "chunk_size",
        ),
        (
            {"mode": "consensus", "alpha": 1.0, "temperature": 0.2,
             "chunk_size": 1, "cache_device": "cuda"},
            "cache_device",
        ),
        (
            {"mode": "consensus", "alpha": 1.5, "temperature": 0.2,
             "chunk_size": 1, "cache_device": "cpu"},
            "alpha",
        ),
        (
            {"mode": "consensus", "alpha": 1.0, "temperature": 0.0,
             "chunk_size": 1, "cache_device": "cpu"},
            "temperature",
        ),
        (
            {"mode": "consensus", "alpha": 1.0, "temperature": 0.2,
             "chunk_size": 0, "cache_device": "cpu"},
            "chunk_size",
        ),
    ],
)
def test_aggregation_policy_rejects_invalid_mappings_before_view_extraction(
    policy, message,
):
    """Catches accepting an invalid policy after extracting expensive views."""
    model = ConditionerHarness()
    model.multiview_aggregation = policy

    with pytest.raises(ValueError, match=message):
        model(torch.zeros(1, 2, 3, 2, 2), **cameras(2))

    assert model.single_view_calls == 0


@pytest.mark.parametrize(
    ("policy", "num_views"),
    [
        (None, 3),
        ({"mode": "equal_mean"}, 3),
        (
            {"mode": "consensus", "alpha": 0.0, "temperature": 0.2,
             "chunk_size": 1, "cache_device": "cpu"},
            3,
        ),
        (
            {"mode": "consensus", "alpha": 1.0, "temperature": 0.2,
             "chunk_size": 1, "cache_device": "cpu"},
            1,
        ),
    ],
)
def test_aggregation_exact_bypass_preserves_lazy_stream_and_clears_diagnostics(
    monkeypatch, policy, num_views,
):
    """Catches bypasses that retain stale diagnostics or materialize the view stream."""
    expected = (torch.tensor([17.0]), torch.tensor([23.0]))
    observed = {}

    def inspect_view_stream(groups):
        observed["is_one_shot_iterator"] = iter(groups) is groups
        observed["num_groups"] = sum(1 for _ in groups)
        return expected

    monkeypatch.setattr(
        image_conditioned_proj, "_online_mean_tensor_groups", inspect_view_stream,
    )
    model = ConditionerHarness()
    model.multiview_aggregation = policy
    model._last_multiview_aggregation_diagnostics = object()

    actual = model(
        torch.zeros(1, num_views, 3, 2, 2),
        **cameras(num_views),
    )

    assert actual is expected
    assert observed == {
        "is_one_shot_iterator": True,
        "num_groups": num_views,
    }
    assert model.last_multiview_aggregation_diagnostics is None


def test_configured_consensus_favors_agreeing_projections_without_model_state():
    """Catches consensus configuration being ignored or persisting learned state."""
    model = ConsensusConditionerHarness()
    state_keys_before = set(model.state_dict())
    trainable_before = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.multiview_aggregation = {
        "mode": "consensus",
        "alpha": 1.0,
        "temperature": 0.2,
        "chunk_size": 1,
        "cache_device": "cpu",
    }
    image = torch.tensor([1.0, 1.0, 1.0, -1.0]).reshape(1, 4, 1, 1, 1)

    z_global, z_proj = model(image, **cameras(4))

    torch.testing.assert_close(z_global, torch.tensor([[[0.5]]]))
    assert torch.all(z_proj > 0.5)
    assert z_proj.shape == (1, 2, 4)
    assert z_proj.dtype is image.dtype
    assert z_proj.device == image.device
    diagnostics = model.last_multiview_aggregation_diagnostics
    assert diagnostics is not None
    assert diagnostics.scores.device.type == "cpu"
    assert diagnostics.weights.device.type == "cpu"
    torch.testing.assert_close(
        diagnostics.weights.sum(dim=1),
        torch.ones((1, 2), dtype=torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.all(diagnostics.weights[0, 3] < diagnostics.weights[0, 0])
    assert set(model.state_dict()) == state_keys_before
    assert [parameter for parameter in model.parameters() if parameter.requires_grad] == trainable_before


def test_configured_consensus_releases_original_projections_before_reuse(
    monkeypatch,
):
    """Catches retaining a full source projection after its CPU cache copy."""
    original_aggregate = image_conditioned_proj.aggregate_consensus_projection
    projection_refs = []
    released_before_extraction = []
    released_before_aggregation = []

    class ProjectionLifetimeHarness(ConsensusConditionerHarness):
        def _forward_single_view(
            self, image, camera_angle_x, distance, mesh_scale, transform_matrix,
        ):
            if projection_refs:
                released_before_extraction.append(
                    all(projection_ref() is None for projection_ref in projection_refs)
                )
            z_global, z_proj = super()._forward_single_view(
                image, camera_angle_x, distance, mesh_scale, transform_matrix,
            )
            projection_refs.append(weakref.ref(z_proj))
            return z_global, z_proj

    def inspect_projection_lifetime(*args, **kwargs):
        released_before_aggregation.append(
            all(projection_ref() is None for projection_ref in projection_refs)
        )
        return original_aggregate(*args, **kwargs)

    monkeypatch.setattr(
        image_conditioned_proj,
        "aggregate_consensus_projection",
        inspect_projection_lifetime,
    )
    model = ProjectionLifetimeHarness()
    model.multiview_aggregation = {
        "mode": "consensus",
        "alpha": 1.0,
        "temperature": 0.2,
        "chunk_size": 1,
        "cache_device": "cpu",
    }

    model(
        torch.tensor([1.0, 1.0, 1.0, -1.0]).reshape(1, 4, 1, 1, 1),
        **cameras(4),
    )

    assert released_before_extraction == [True, True, True]
    assert released_before_aggregation == [True]
    assert all(projection_ref() is None for projection_ref in projection_refs)


def test_aggregation_diagnostics_is_read_only():
    """Catches callers replacing the latest conditioner diagnostics."""
    model = ConsensusConditionerHarness()
    model.multiview_aggregation = {
        "mode": "consensus",
        "alpha": 1.0,
        "temperature": 0.2,
        "chunk_size": 1,
        "cache_device": "cpu",
    }

    model(torch.tensor([1.0, 1.0, 1.0, -1.0]).reshape(1, 4, 1, 1, 1), **cameras(4))
    diagnostics = model.last_multiview_aggregation_diagnostics

    assert diagnostics is not None
    with pytest.raises(AttributeError):
        model.last_multiview_aggregation_diagnostics = object()
    assert model.last_multiview_aggregation_diagnostics is diagnostics


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
