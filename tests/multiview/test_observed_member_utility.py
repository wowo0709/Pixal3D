from copy import deepcopy

import pytest
import torch

from experiments.deformnext_retargeting.observed_member_utility import observed_member_utility
from experiments.deformnext_retargeting.member_feature_query_support import utility_query_support
from pixal3d.modules import member_feature_source_ray as native
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from test_member_feature_source_ray import pair
from test_member_feature_query_support import prepared


def test_exact_same_query_design_as_frozen_single_argument_intervention():
    model, data = prepared(); original_forward = type(model).forward; original_pool = native.pool_ray_queries
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        with utility_query_support(model, [], mode='observed-query-only'):
            expected = model(**data)
        with observed_member_utility() as statistics:
            actual = model(**data)
    assert torch.equal(actual.source_numerator, expected.source_numerator)
    assert torch.equal(actual.semantic_weight, expected.semantic_weight)
    assert torch.equal(actual.use_probability, expected.use_probability)
    for field in ['candidate_indices', 'matching_logits', 'candidate_weights', 'base_evidence', 'retarget_residual']:
        assert torch.equal(getattr(actual.attention, field), getattr(expected.attention, field))
    assert statistics['forwards'] == statistics['utility_pool_calls'] == statistics['target_pool_calls'] == 1
    assert 0 < statistics['observed_utility_voxel_observations'] < statistics['utility_input_voxel_observations']
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert type(model).forward is original_forward and native.pool_ray_queries is original_pool


def test_preserves_cold_native_prior_and_warm_Top1_at_zero_feature_output():
    _, model, data = pair(); model.eval()
    with torch.no_grad():
        reference = model(**data)
        with observed_member_utility():actual = model(**data)
    assert torch.equal(actual.source_numerator, reference.source_numerator)
    assert not source_ray_delta(actual.source_numerator, actual.semantic_weight, data['source_values']).any()
    assert not model.last_direct_feature_correction.any()


def test_checkpoint_backward_keeps_changed_query_after_context_is_removed(monkeypatch):
    model, data = prepared(); model.train(); other = deepcopy(model)
    probe = torch.randn(1, 3, 1024, 1024)
    with observed_member_utility():actual = model(**data)
    # Backward occurs after the temporary pool/forward override was removed.
    (actual.source_numerator*probe).sum().backward()
    with monkeypatch.context() as patch:
        patch.setattr(native, 'checkpoint', lambda function, *args, **kwargs: function(*args))
        with observed_member_utility():expected = other(**data)
        (expected.source_numerator*probe).sum().backward()
    assert torch.equal(actual.source_numerator, expected.source_numerator)
    gradients = dict(other.named_parameters())
    for name, parameter in model.named_parameters():
        expected = gradients[name].grad
        assert (parameter.grad is None) == (expected is None), name
        if expected is not None:assert torch.equal(parameter.grad, expected), name
    assert model.ray_readout.output.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('missing', ['source_valid', 'observed_query'])
def test_missing_inputs_keep_original_native_fallback_and_DDP_gradient_edges(missing):
    model, data = prepared(); model.train(); data[missing].zero_()
    with observed_member_utility():out = model(**data)
    out.source_numerator.sum().backward()
    assert not out.source_numerator.any() and not out.semantic_weight.any()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:assert parameter.grad is not None and not parameter.grad.any(), name


def test_failure_restores_both_pool_and_forward(monkeypatch):
    model, data = prepared(); original_forward = type(model).forward
    def broken(*args, **kwargs):raise RuntimeError('original pool failed')
    monkeypatch.setattr(native, 'pool_ray_queries', broken)
    with pytest.raises(RuntimeError, match='original pool failed'):
        with observed_member_utility():model(**data)
    assert type(model).forward is original_forward and native.pool_ray_queries is broken


def test_runtime_keeps_native_cli_and_captures_only_the_declared_query_change(tmp_path, monkeypatch):
    import json
    import sys
    from experiments.deformnext_retargeting import observed_member_utility_runtime as entry
    from experiments.deformnext_retargeting import member_feature_runtime as original
    model, data = prepared(); before = type(model).forward
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(trainer=dict(name='MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer'))))
    remaining = ['--phase', 'native', '--config', str(config), '--output', str(tmp_path)]
    def run_native():
        assert sys.argv[1:] == remaining and type(model).forward is not before
        with torch.no_grad():model(**data)
    monkeypatch.setattr(original, 'main', run_native)
    argv = ['observed-runtime', '--utility-query-design', 'observed-input-target', *remaining]
    monkeypatch.setattr(sys, 'argv', argv)
    entry.main(); result = json.loads((tmp_path/'observed-member-utility-runtime.json').read_text())
    assert result['statistics']['forwards'] == 1 and not result['physical_module_or_parameter_layout_changed']
    assert result['native_identity_capacity_and_quality_require_their_separate_actual_reports']
    assert sys.argv is argv and type(model).forward is before
