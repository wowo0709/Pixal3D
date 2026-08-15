import torch
import pytest

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    compute_multiview_projection_matrices,
)


def conditions():
    zero = torch.tensor([0.0, 0.0])
    return [
        {"cond": torch.tensor([3.0, 0.0]), "neg_cond": zero},
        {"cond": torch.tensor([0.0, 1.0]), "neg_cond": zero},
    ]


def parameters():
    return {
        "steps": 1,
        "rescale_t": 1.0,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "guidance_interval": [0.0, 1.0],
    }


def expected_velocity():
    weights = torch.softmax(torch.tensor([3.0, 1.0]) / 1.001, dim=0)
    return (
        weights[0] * torch.tensor([3.0, 0.0])
        + weights[1] * torch.tensor([0.0, 1.0])
    )


class DenseVelocityModel:
    def __call__(self, sample, timestep, cond, **kwargs):
        return cond.view(1, 2, 1, 1, 1).expand_as(sample)


class SparseVelocityModel:
    def __call__(self, sample, timestep, cond, **kwargs):
        return sample.replace(cond.view(1, 2).expand_as(sample.feats))


def sample(model, noise):
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    return Pixal3DImageTo3DPipeline._camera_aware_agw_sample(
        sampler,
        model,
        noise,
        conditions(),
        {},
        parameters(),
        verbose=False,
    )


def test_dense_tokens_use_guidance_magnitude_softmax():
    noise = torch.zeros(1, 2, 1, 1, 1)
    actual = sample(DenseVelocityModel(), noise)
    torch.testing.assert_close(actual, -expected_velocity().view_as(actual))


def test_sparse_tokens_use_guidance_magnitude_softmax():
    noise = SparseTensor(
        feats=torch.zeros(1, 2),
        coords=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
    )
    actual = sample(SparseVelocityModel(), noise)
    torch.testing.assert_close(actual.feats[0], -expected_velocity())


def test_view_camera_conditions_share_the_original_anchor_frame():
    pipeline = object.__new__(Pixal3DImageTo3DPipeline)
    images = [object(), object()]
    transforms = torch.eye(4).repeat(1, 2, 1, 1)
    transforms[0, 1, :3, :3] = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    cameras = {
        "camera_angle_x": torch.tensor([[0.1, 0.2]]),
        "distance": torch.tensor([[2.0, 3.0]]),
        "mesh_scale": torch.tensor([1.0]),
        "transform_matrix": transforms,
    }

    class Conditioner:
        fixed_projection_transform = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    conditioner = Conditioner()
    sliced = list(
        pipeline._per_view_camera_kwargs(images, cameras, conditioner)
    )
    expected, _ = compute_multiview_projection_matrices(
        transforms, cameras["distance"], conditioner.fixed_projection_transform
    )
    assert len(sliced) == 2
    assert sliced[0][0] == [images[0]]
    assert sliced[1][1]["camera_angle_x"].item() == pytest.approx(0.2)
    assert "transform_matrix" not in sliced[0][1]
    torch.testing.assert_close(
        sliced[0][1]["projection_matrix"], expected[:, 0]
    )
    torch.testing.assert_close(
        sliced[1][1]["projection_matrix"], expected[:, 1]
    )
    assert not torch.allclose(
        sliced[1][1]["projection_matrix"],
        sliced[0][1]["projection_matrix"],
    )


def test_single_view_delegates_to_ordinary_sampler():
    sentinel = object()

    class Result:
        samples = sentinel

    class Sampler:
        def sample(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return Result()

    sampler = Sampler()
    condition = conditions()[:1]
    actual = Pixal3DImageTo3DPipeline._camera_aware_agw_sample(
        sampler, object(), torch.zeros(1), condition,
        {"steps": 12}, {"guidance_strength": 7.5}, verbose=False,
    )
    assert actual is sentinel
    assert sampler.kwargs["steps"] == 12
    assert sampler.kwargs["guidance_strength"] == pytest.approx(7.5)
    assert sampler.kwargs["cond"] is condition[0]["cond"]
