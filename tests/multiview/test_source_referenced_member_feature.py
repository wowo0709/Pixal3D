from unittest.mock import patch

import pytest
import torch

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.modules.source_referenced_member_feature import SourceReferencedMemberFeatureConditioner
from test_member_posterior_source_ray import fixture, activate


def pair():
    _, data = fixture('semantic-top1')
    kwargs = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, hidden_dim=256, topk=64, score_temperature=.5,
        projection_mode='native-bilinear', ray_mode='semantic-top1', ray_chunk_size=4,
        candidate_aggregation='top1')
    torch.manual_seed(1317)
    old = MemberFeatureSourceRayConditioner(**kwargs)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(1317)
    new = SourceReferencedMemberFeatureConditioner(**kwargs)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    return old, new, data


def test_only_one_zero_parameter_added_and_cold_native_prior_exact():
    old, new, data = pair()
    assert set(new.state_dict()) - set(old.state_dict()) == {'retargeter.source_projection.weight'}
    assert sum(p.requires_grad for p in new.parameters()) == 20
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    out = new(**data)
    assert not source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values']).any()
    assert new.retargeter._source_context is None


def test_zero_branch_preserves_nontrivial_trained_parent_function_and_selection():
    old, new, data = pair()
    activate(old)
    torch.nn.init.normal_(old.ray_readout.output.weight, std=.03)
    state = new.state_dict(); state.update(old.state_dict()); new.load_state_dict(state)
    with torch.no_grad():
        a, b = old(**data), new(**data)
    assert torch.equal(a.source_numerator, b.source_numerator)
    assert torch.equal(a.semantic_weight, b.semantic_weight)
    assert torch.equal(a.use_probability, b.use_probability)
    assert torch.equal(a.attention.candidate_indices, b.attention.candidate_indices)
    assert torch.equal(a.attention.candidate_weights, b.attention.candidate_weights)


def test_uses_own_source_pixel_not_the_matching_candidate_and_backpropagates():
    old, new, data = pair(); activate(new)
    seen = []
    hook = new.retargeter.register_forward_pre_hook(
        lambda module, args: seen.append(module._source_context.detach().clone()))
    out = new(**data); hook.remove()
    expected = data['source_values'].reshape(-1, 1024)[new.last_source_reference_rows]
    assert torch.equal(seen[0], expected)
    # This includes source-view1 pixels even though its M0 foreground mask is
    # empty: their real native background feature must not become a zero input.
    assert (new.last_source_reference_rows >= 1024).any()
    assert new.retargeter._source_context is None
    (out.source_numerator * torch.randn_like(out.source_numerator)).sum().backward()
    assert new.retargeter.source_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in new.q2c.parameters())


def test_missing_target_observation_keeps_prior_and_connected_zero_gradients():
    _, new, data = pair(); data['observed_query'].zero_()
    out = new(**data)
    delta = source_ray_delta(out.source_numerator, out.semantic_weight, data['source_values'])
    assert not delta.any()
    delta.sum().backward()
    assert all(p.grad is not None and not p.grad.any() for p in new.parameters() if p.requires_grad)


def test_source_context_clears_on_parent_exception():
    _, new, data = pair()
    with patch.object(MemberFeatureSourceRayConditioner, 'forward', side_effect=RuntimeError('stop')):
        with pytest.raises(RuntimeError, match='stop'):
            new(**data)
    assert new.retargeter._source_context is None
    with pytest.raises(ValueError, match='actual own-ray'):
        new.retargeter(torch.zeros(1, 2048), .25)
