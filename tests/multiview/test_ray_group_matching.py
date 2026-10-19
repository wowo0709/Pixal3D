"""Shared ray queries, user-defined controls, native zero start and actual gradients."""
from contextlib import nullcontext

import pytest
import torch

from pixal3d.modules.ray_group_matching import RayGroupConditioner


def fixture(mode='semantic-attention', decoder=True):
    torch.manual_seed(813)
    cfg = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, hidden_dim=256, score_temperature=.5,
        projection_mode='native-bilinear', ray_mode=mode, ray_chunk_size=4)
    if mode.endswith('attention'):
        cfg['task_attention'] = dict(decoder=decoder, score_prior=True, train_decoder=False,
            train_score=False, content_mode='residual-zero', initial_content_scale=16., chunk_size=4)
    if mode == 'native-spatial-attention':
        cfg['correspondence_mode'] = 'native-spatial'
    if mode == 'semantic-top1':
        cfg['candidate_aggregation'] = 'top1'
    model = RayGroupConditioner(**cfg)
    ids = torch.tensor([[[0, 0, 1, 1, -1], [2, 2, 1, 1, -1], [3, 3, 4, 4, -1]]])
    data = dict(native_voxels=torch.randn(1, 5, 1024), target_omega_voxels=torch.randn(1, 5, 1024),
        target_native_voxels=torch.randn(1, 5, 1024), ray_ids=ids,
        source_keys=torch.randn(1, 3, 32, 1024), source_values=torch.randn(1, 3, 32, 1024),
        source_valid=torch.ones(1, 3, 32, dtype=torch.bool))
    if mode == 'native-spatial-attention':
        data.update(spatial_keys=torch.randn(1, 96, 3, 1024), spatial_values=torch.randn(1, 96, 3, 1024))
    return model, data


@pytest.mark.parametrize('amp', [False, True])
def test_native_zero_delta_then_gradients_through_ray_utility_and_retargeter(amp):
    model, data = fixture()
    frozen = {n: p.detach().clone() for n, p in model.q2c.named_parameters()}
    ctx = torch.autocast('cpu', dtype=torch.bfloat16) if amp else nullcontext()
    with ctx:
        output = model(**data)
        assert not output.map_delta.any()
        loss = (output.map_delta * torch.randn_like(output.map_delta)).sum()
    loss.backward()
    assert model.fusion_gate.grad.abs() > 0
    assert model.retargeter.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert model.task_attention.content_output.weight.grad.abs().sum() == 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.normal_(std=.001)
        model.task_attention.content_output.weight.normal_(std=.02)
    ctx = torch.autocast('cpu', dtype=torch.bfloat16) if amp else nullcontext()
    with ctx:
        trained = model(**data)
        trained.map_delta.square().mean().backward()
    for name in ['query', 'context', 'key', 'content_output']:
        grad = getattr(model.task_attention, name).weight.grad
        assert grad is not None and grad.isfinite().all() and grad.abs().sum() > 0
    for name, parameter in model.q2c.named_parameters():
        assert not parameter.requires_grad and parameter.grad is None
        assert torch.equal(parameter, frozen[name])
    assert model.retargeter.input_dim == 2048


def test_one_attention_result_per_camera_ray_is_invariant_to_member_order():
    model, data = fixture()
    with torch.no_grad():
        model.fusion_gate.fill_(.3)
        model.retargeter.net[-1].weight.normal_(std=.001)
        model.task_attention.content_output.weight.normal_(std=.02)
    permutation = torch.tensor([1, 0, 3, 2, 4])
    assert torch.equal(data['ray_ids'], data['ray_ids'][..., permutation])
    normal = model(**data)
    altered = model(**{**data, **{key: data[key][:, permutation] for key in
        ['native_voxels', 'target_omega_voxels', 'target_native_voxels']}})
    assert normal.map_delta.shape == (1, 3, 32, 1024)
    assert normal.attention.query_valid.sum() == 6  # camera/ray slots, not5 individual voxels
    torch.testing.assert_close(normal.map_delta, altered.map_delta, rtol=1e-5, atol=2e-6)
    assert torch.equal(normal.attention.candidate_indices, altered.attention.candidate_indices)
    torch.testing.assert_close(normal.attention.candidate_weights, altered.attention.candidate_weights)
    assert not normal.map_delta.flatten(1, 2)[~normal.attention.query_valid].any()


def test_user_no_score_means_exactly_one_matched_feature_without_utility_parameters():
    model, data = fixture('semantic-top1')
    result = model(**data).attention
    assert model.task_attention is None
    valid = result.query_valid
    weights = result.candidate_weights[valid]
    assert torch.equal(weights.max(-1).values, torch.ones(len(weights)))
    assert torch.equal((weights > 0).sum(-1), torch.ones(len(weights), dtype=torch.long))
    ids = result.candidate_indices[valid].gather(1, weights.argmax(-1, keepdim=True))[:, 0]
    torch.testing.assert_close(result.base_evidence[valid], data['source_values'][0].flatten(0, 1)[ids], rtol=0, atol=0)


def test_native_spatial_control_consumes_explicit_projected_values_and_physical_no_decoder():
    model, data = fixture('native-spatial-attention', decoder=False)
    assert len(model.q2c.decoder.blocks) == 0
    assert not any(name.startswith('q2c.decoder.') for name in model.state_dict())
    data['spatial_values'].fill_(17.)
    result = model(**data).attention
    assert result.candidate_weights.shape[-1] == 3
    torch.testing.assert_close(result.base_evidence[result.query_valid],
        torch.full_like(result.base_evidence[result.query_valid], 17.))
    with pytest.raises(ValueError, match='projected'):
        model(**{k: v for k, v in data.items() if k != 'spatial_values'})


def test_empty_rays_leave_zero_updates_and_finite_ddp_gradient_edges():
    model, data = fixture()
    data['ray_ids'].fill_(-1)
    result = model(**data)
    assert not result.attention.query_valid.any() and not result.map_delta.any()
    result.map_delta.sum().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters() if p.requires_grad)


def test_cannot_accidentally_reuse_target_patch_candidate_ids_for_ray_queries():
    model, data = fixture()
    with pytest.raises(TypeError, match='semantic_indices'):
        model(**data, semantic_indices=torch.zeros(1, 1024, 64, dtype=torch.long))
