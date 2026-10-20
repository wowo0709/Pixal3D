from unittest.mock import patch

import torch

from experiments.deformnext_retargeting.target_first_value_control import fixed_use_value_control
from pixal3d.modules.target_first_observed_query import ObservedQueryTargetFirstRayConditioner
from pixal3d.modules.target_first_observed_spatial import (
    ObservedQuerySpatialTargetFirstRayConditioner, native_spatial_ray_candidates,
)
from pixal3d.modules.ray_group_query import native_ray_membership, pool_ray_queries
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid


def fixture():
    config = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, hidden_dim=256, score_temperature=.5,
        projection_mode='native-bilinear', ray_chunk_size=4,
        task_attention=dict(decoder=True, score_prior=True, train_decoder=False,
            train_score=False, content_mode='residual-zero', initial_content_scale=16., chunk_size=4))
    torch.manual_seed(815)
    spatial = ObservedQuerySpatialTargetFirstRayConditioner(**config,
        ray_mode='native-spatial-attention', correspondence_mode='native-spatial')
    torch.manual_seed(815)
    semantic = ObservedQueryTargetFirstRayConditioner(**config, ray_mode='semantic-attention')
    data = dict(native_voxels=torch.randn(1, 5, 1024), target_omega_voxels=torch.randn(1, 5, 1024),
        target_native_voxels=torch.randn(1, 5, 1024), observed_query=torch.tensor([[True, False, True, False, False]]),
        ray_ids=torch.tensor([[[0, 0, 1, 1, -1], [2, 2, 1, 1, -1], [3, 3, 4, 4, -1]]]),
        source_keys=torch.randn(1, 3, 32, 1024), source_values=torch.randn(1, 3, 32, 1024),
        source_valid=torch.ones(1, 3, 32, dtype=torch.bool))
    candidates = dict(spatial_keys=torch.randn(1, 96, 3, 1024), spatial_values=torch.randn(1, 96, 3, 1024))
    return spatial, semantic, data, candidates


def activate(model):
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.normal_(std=.01)
        model.evidence_use.network[-1].weight.normal_(std=.03)
        model.task_attention.content_output.weight.normal_(std=.02)


def test_common_state_and_query_support_match_full_semantic_without_semantic_retrieval():
    spatial, semantic, data, candidates = fixture()
    assert spatial.state_dict().keys() == semantic.state_dict().keys()
    assert all(torch.equal(value, semantic.state_dict()[name]) for name, value in spatial.state_dict().items())
    with torch.no_grad():
        reference = semantic(**data)
        with patch('pixal3d.modules.target_first_observed_query.semantic_ray_proposals', side_effect=AssertionError('semantic retrieval used')):
            result = spatial(**data, **candidates)
    assert torch.equal(result.attention.query_valid, reference.attention.query_valid)
    assert result.attention.candidate_indices.shape[-1] == 3
    assert result.attention.candidate_indices.max() == 2
    assert not result.source_numerator.any() and not result.semantic_weight.any()
    assert spatial.retargeter.net[0].in_features == 2048


def test_all_source_tensors_receive_gradients_with_native_spatial_candidates():
    spatial, _, data, candidates = fixture()
    initial = spatial(**data, **candidates)
    (initial.source_numerator * torch.randn_like(initial.source_numerator)).sum().backward()
    assert spatial.fusion_gate.grad.abs() > 0
    assert spatial.retargeter.net[-1].weight.grad.abs().sum() > 0
    spatial.zero_grad(set_to_none=True)
    activate(spatial)
    spatial(**data, **candidates).source_numerator.square().mean().backward()
    for name, value in spatial.named_parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0, name
    assert all(value.grad is None for value in spatial.q2c.parameters())


def test_missing_observation_fallback_and_fixed_use_value_intervention():
    spatial, _, data, candidates = fixture()
    activate(spatial)
    spatial.eval()
    events = []
    with torch.no_grad():
        normal = spatial(**data, **candidates)
        with fixed_use_value_control(spatial, events):
            wrong = spatial(**data, **candidates)
            empty = spatial(**{**data, 'observed_query': torch.zeros_like(data['observed_query'])}, **candidates)
        absent = spatial(**{**data, 'source_valid': torch.zeros_like(data['source_valid'])}, **candidates)
        restored = spatial(**data, **candidates)
    for name in ['candidate_indices', 'matching_logits', 'candidate_weights', 'query_valid']:
        assert torch.equal(getattr(normal.attention, name), getattr(wrong.attention, name))
    assert torch.equal(normal.use_probability, wrong.use_probability)
    assert not torch.equal(normal.source_numerator, wrong.source_numerator)
    assert not empty.source_numerator.any() and not absent.source_numerator.any()
    assert not empty.use_probability.any() and not absent.use_probability.any()
    assert torch.equal(normal.source_numerator, restored.source_numerator)
    assert all(event['source_use_probability_exact_same_forward'] for event in events)


def test_empty_observed_query_keeps_zero_gradient_edges_for_every_trainable_tensor():
    spatial, _, data, candidates = fixture()
    activate(spatial)
    empty = spatial(**{**data, 'observed_query': torch.zeros_like(data['observed_query'])}, **candidates)
    empty.source_numerator.sum().backward()
    for value in spatial.parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all() and not value.grad.any()


def test_original_ProjGrid_constant_source_views_are_pooled_over_complete_rays():
    grid = ProjGrid(16, 512)
    matrices = torch.eye(4).repeat(1, 4, 1, 1)
    matrices[:, :, 2, 3] = 2
    camera = dict(transform_matrix=matrices, distance=torch.ones(1, 4) * 2,
        camera_angle_x=torch.ones(1, 4), mesh_scale=torch.ones(1))
    rays = native_ray_membership(grid, **camera, views=(1, 2, 3))
    maps = torch.arange(1, 4).float().reshape(1, 3, 1, 1).expand(1, 3, 1024, 1024)
    keys, values = native_spatial_ray_candidates(grid, maps * 10, maps, rays, camera)
    valid = pool_ray_queries(torch.ones(1, 4096, 1), rays, num_rays=1024).valid.flatten(1)
    for source in range(3):
        torch.testing.assert_close(values[:, :, source][valid],
            torch.full_like(values[:, :, source][valid], float(source + 1)), atol=2e-6, rtol=0)
        torch.testing.assert_close(keys[:, :, source][valid],
            torch.full_like(keys[:, :, source][valid], float(10 * (source + 1))), atol=2e-5, rtol=0)
    assert not values[~valid].any() and not keys[~valid].any()
