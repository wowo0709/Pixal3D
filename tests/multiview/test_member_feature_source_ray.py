import pytest
import torch

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner, _feature_chunk
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from test_member_posterior_source_ray import fixture, activate


def pair():
    baseline, data = fixture('semantic-top1')
    torch.manual_seed(1317)
    model = MemberFeatureSourceRayConditioner(arm='c1', implementation='r9-native-v5',
        key_dim=1024, value_dim=1024, query_dim=1024, hidden_dim=256, topk=64,
        score_temperature=.5, projection_mode='native-bilinear', ray_mode='semantic-top1',
        ray_chunk_size=4, candidate_aggregation='top1')
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in baseline.state_dict().items())
    return baseline, model, data


def delta(model, data, **kwargs):
    out = model(**data, **kwargs)
    return source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values']), out


def test_native_prior_and_warm_top1_R_function_exact_at_zero_feature_output():
    baseline, model, data = pair()
    cold, _ = delta(model, data)
    assert not cold.any()
    assert model.retargeter.net[0].in_features == 2048
    assert not any('deformation_projection' in key for key in model.state_dict())
    activate(baseline)
    merged = model.state_dict(); merged.update(baseline.state_dict()); model.load_state_dict(merged)
    with torch.no_grad():
        a, out_a = delta(baseline, data); b, out_b = delta(model, data)
    assert torch.equal(a, b)
    assert torch.equal(out_a.use_probability, out_b.use_probability)
    assert torch.equal(out_a.attention.retarget_residual, out_b.attention.retarget_residual)
    assert not model.last_direct_feature_correction.any()


def test_direct_output_receives_gradient_with_both_original_gates_zero():
    _, model, data = pair()
    change, _ = delta(model, data)
    (change*torch.randn_like(change)).sum().backward()
    assert not change.any()
    assert model.ray_readout.output.weight.grad.abs().sum() > 0
    assert model.fusion_gate.grad.abs() > 0
    assert model.retargeter.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.q2c.parameters())


def test_nine_historical_head_tensors_learn_without_semantic_gate_warmup():
    _, model, data = pair()
    optimizer = torch.optim.AdamW(model.ray_readout.parameters(), lr=1e-3)
    target = torch.randn(1, 3, 1024, 1024)
    assert len(list(model.ray_readout.parameters())) == 9
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        change, _ = delta(model, data)
        ((change-target).square().mean()).backward()
        optimizer.step()
    assert model.fusion_gate.detach() == 0
    assert not model.retargeter.net[-1].weight.detach().any()
    for name, parameter in model.ray_readout.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert not model.ray_readout._forward_hooks


@pytest.mark.parametrize('missing', ['query', 'source'])
def test_missing_inputs_keep_native_prior_and_zero_DDP_gradients(missing):
    _, model, data = pair(); activate(model)
    torch.nn.init.normal_(model.ray_readout.output.weight, std=.01)
    data['observed_query' if missing == 'query' else 'source_valid'].zero_()
    change, out = delta(model, data)
    assert not change.any() and not out.use_probability.any()
    change.sum().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and not parameter.grad.any(), name


def test_value_intervention_keeps_all_four_head_weights_top1_and_use_exact():
    _, model, data = pair(); activate(model)
    torch.nn.init.normal_(model.ray_readout.output.weight, std=.2)
    model.eval()
    with torch.no_grad():
        original, normal = delta(model, data)
        selected, probability = model.last_feature_probabilities
        altered, wrong = delta(model, data, control='wrong-feature-values-fixed-use')
        changed_selected, changed_probability = model.last_feature_probabilities
        assert torch.equal(selected, changed_selected)
        assert torch.equal(probability, changed_probability)
        assert torch.equal(normal.use_probability, wrong.use_probability)
        for field in ['candidate_indices', 'candidate_weights', 'matching_logits', 'base_evidence']:
            assert torch.equal(getattr(normal.attention, field), getattr(wrong.attention, field)), field
        assert not torch.equal(original, altered)
        assert model.last_direct_feature_correction.abs().max() <= model.g_max
        restored, _ = delta(model, data)
        assert torch.equal(original, restored)
    assert not model.ray_readout.value._forward_hooks
    with pytest.raises(ValueError, match='eval and no_grad'):
        model(**data, control='wrong-feature-values-fixed-use')


def test_chunk_calls_actual_historical_head_and_restores_value_hook_on_error():
    torch.manual_seed(9)
    head = RayFeatureReadout(); torch.nn.init.normal_(head.output.weight, std=.01)
    native, context = torch.randn(2, 1024), torch.randn(2, 1024)
    keys, values = torch.randn(1, 12, 1024), torch.randn(1, 12, 1024)
    batch, ids = torch.zeros(2, dtype=torch.long), torch.arange(8).reshape(2, 4)
    valid, prior = torch.ones(2, 4, dtype=torch.bool), torch.randn(2, 4)
    actual = _feature_chunk(head, native, context, values, keys, batch, ids, valid, prior, 'normal')
    expected = head(native, context, keys[batch[:, None], ids], values[batch[:, None], ids], valid, prior)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    head.eval()
    with torch.no_grad(), pytest.raises(ValueError, match='aligned RGB'):
        _feature_chunk(head, native[:, :1], context, values, keys, batch, ids, valid, prior,
                       'wrong-feature-values-fixed-use')
    assert not head.value._forward_hooks


def test_utility_load_requires_complete_state_and_untouched_output():
    _, model, _ = pair(); utility = model.ray_readout.score_selector.state_dict()
    model.load_utility(utility)
    with pytest.raises(ValueError, match='complete finite'):
        model.load_utility({key: value for key, value in utility.items() if key != 'query.weight'})
    torch.nn.init.normal_(model.ray_readout.output.weight, std=.01)
    with pytest.raises(ValueError, match='untouched'):
        model.load_utility(utility)
