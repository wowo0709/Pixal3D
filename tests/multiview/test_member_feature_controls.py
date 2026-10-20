import copy

import pytest
import torch

from pixal3d.modules.member_feature_controls import Top1FeatureReadout, Top1FeatureSourceRayConditioner, SpatialFeatureSourceRayConditioner
from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from experiments.deformnext_retargeting.member_feature_control_value import feature_value_control
from experiments.deformnext_retargeting.member_feature_control_recipe import NAME, SCHEMA, CONTROLS, validate_control_recipe
from test_member_posterior_source_ray import fixture, activate
from test_member_feature_runtime import recipe


def pair(kind):
    _, data = fixture('semantic-top1')
    kwargs = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024, query_dim=1024,
                  hidden_dim=256, topk=64, score_temperature=.5, projection_mode='native-bilinear',
                  ray_mode='semantic-top1', ray_chunk_size=4, candidate_aggregation='top1')
    torch.manual_seed(1317); full = MemberFeatureSourceRayConditioner(**kwargs)
    torch.manual_seed(1317)
    if kind == 'top1-feature':
        model = Top1FeatureSourceRayConditioner(**kwargs)
    else:
        model = SpatialFeatureSourceRayConditioner(correspondence_mode='native-spatial', **kwargs)
        data['spatial_keys'] = torch.randn(1, 3072, 3, 1024)
        data['spatial_values'] = torch.randn(1, 3072, 3, 1024)
    return full, model, data


@pytest.mark.parametrize('kind,count,head', [('top1-feature', 12, 2), ('spatial-feature-attention', 19, 9)])
def test_physical_control_tensors_common_weights_and_native_zero(kind, count, head):
    full, model, data = pair(kind)
    assert sum(p.requires_grad for p in model.parameters()) == count
    assert len(list(model.ray_readout.parameters())) == head
    assert all(torch.equal(v, full.state_dict()[k]) for k, v in model.state_dict().items())
    output = model(**data)
    assert not source_ray_delta(output.source_numerator, output.semantic_weight, data['source_values']).any()
    assert model.retargeter.net[0].in_features == 2048
    if kind == 'top1-feature':
        assert set(model.ray_readout.state_dict()) == {'value.weight', 'output.weight'}
        active = output.attention.query_valid
        assert ((model.last_feature_probabilities[1] > 0).sum(-1) == 1).all()
        assert output.attention.candidate_indices.shape[-1] == 64
    else:
        assert output.attention.candidate_indices.shape[-1] == 3
        expected = data['spatial_values'].gather(2, output.attention.matching_logits.argmax(-1)[..., None, None].expand(-1, -1, 1, 1024)).squeeze(2)
        assert torch.equal(output.attention.base_evidence[output.attention.query_valid], expected[output.attention.query_valid])


def test_physical_top1_matches_same_V_output_single_candidate_attention():
    torch.manual_seed(51); full = RayFeatureReadout(); torch.nn.init.normal_(full.output.weight, std=.01)
    head = Top1FeatureReadout(full)
    native, context = torch.randn(3, 1024), torch.randn(3, 1024)
    keys, values = torch.randn(3, 1, 1024), torch.randn(3, 1, 1024)
    valid, prior = torch.ones(3, 1, dtype=torch.bool), torch.zeros(3, 1)
    a, ap = full(native, context, keys, values, valid, prior)
    b, bp = head(native, context, keys, values, valid, prior)
    assert torch.equal(a, b) and torch.equal(ap, bp)
    # Additional unselected correspondences cannot contribute to the feature.
    values = torch.cat([values, torch.randn(3, 4, 1024)], 1)
    valid = torch.ones(3, 5, dtype=torch.bool); prior = torch.tensor([[1., 0., 0., 0., 0.]]).expand(3, -1)
    first, _ = head(native, context, torch.randn_like(values), values, valid, prior)
    values[:, 1:] *= 100
    second, p = head(native * 50, context * -20, torch.randn_like(values), values, valid, prior)
    assert torch.equal(first, second) and ((p > 0).sum(-1) == 1).all()


@pytest.mark.parametrize('kind', list(CONTROLS))
def test_all_source_gradients_and_frozen_matching(kind):
    _, model, data = pair(kind)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    target = torch.randn_like(data['source_values'])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); out = model(**data)
        delta = source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values'])
        (delta - target).square().mean().backward(); optimizer.step()
    for name, value in model.named_parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0, name
        else:
            assert value.grad is None


@pytest.mark.parametrize('kind', list(CONTROLS))
def test_absent_query_native_prior_and_connected_zero_gradients(kind):
    _, model, data = pair(kind); activate(model)
    data['observed_query'].zero_(); out = model(**data)
    delta = source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values'])
    assert not delta.any() and not out.use_probability.any()
    delta.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize('kind', list(CONTROLS))
def test_saved_value_control_keeps_actual_matching_Top1_and_use(kind):
    _, model, data = pair(kind); activate(model); model.eval()
    torch.nn.init.normal_(model.ray_readout.output.weight, std=.01); events = []
    with torch.no_grad():
        normal = model(**data)
        with feature_value_control(model, events): wrong = model(**data)
        restored = model(**data)
    assert events and events[0]['source_use_probability_exact_same_forward']
    assert torch.equal(normal.use_probability, wrong.use_probability)
    assert torch.equal(normal.attention.base_evidence, wrong.attention.base_evidence)
    assert torch.equal(normal.attention.matching_logits, wrong.attention.matching_logits)
    assert not torch.equal(normal.source_numerator, wrong.source_numerator)
    assert torch.equal(normal.source_numerator, restored.source_numerator)
    assert 'forward' not in model.__dict__ and not model.ray_readout._forward_hooks


def test_spatial_control_keeps_distinct_member_membership_not_semantic_patch_retrieval():
    _, model, data = pair('spatial-feature-attention'); model(**data)
    assert model._last_distinct_member_count[0, 0] == 2
    assert model._last_distinct_member_count[0, 1028] == 2
    before = model._last_distinct_member_count.clone()
    duplicate = {k: v for k, v in data.items()}
    for key in ['native_voxels', 'target_omega_voxels', 'target_native_voxels', 'target_patch_ids', 'observed_query']:
        duplicate[key] = torch.cat([data[key], data[key][:, :1]], dim=1)
    duplicate['ray_ids'] = torch.cat([data['ray_ids'], data['ray_ids'][:, :, :1]], dim=-1)
    model(**duplicate)
    assert torch.equal(before, model._last_distinct_member_count)


@pytest.mark.parametrize('kind', list(CONTROLS))
def test_control_recipe_preserves_all_native_science_fields(kind):
    cfg = recipe(); spec = CONTROLS[kind]
    cfg['trainer']['name'] = NAME; cfg['trainer']['args']['feature_control'] = kind
    cfg['trainer']['args']['deform_conditioner']['correspondence_mode'] = spec['correspondence']
    cfg['experiment'].update(conditioning_schema=SCHEMA, readout_arm=spec['arm'], feature_readout=spec['readout'], matching_proposals=spec['proposal'])
    old = copy.deepcopy(cfg)
    assert validate_control_recipe(cfg) == spec['arm'] and cfg == old
    cfg['trainer']['args']['batch_split'] = 2
    with pytest.raises(ValueError): validate_control_recipe(cfg)
