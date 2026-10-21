from copy import deepcopy
import math

import pytest
import torch

from pixal3d.modules.bounded_member_feature_source_ray import (
    BoundedPriorRayFeatureReadout, BoundedPriorMemberFeatureSourceRayConditioner,
    bounded_probabilities, masked_probabilities,
)
from test_member_feature_source_ray import pair, activate, delta


def head_inputs():
    torch.manual_seed(13)
    valid = torch.tensor([[True]*8, [True]*5+[False]*3, [False]*8])
    prior = torch.randn(3, 8).masked_fill(~valid, -torch.inf)
    return torch.randn(3, 1024), torch.randn(3, 1024), torch.randn(3, 8, 1024), torch.randn(3, 8, 1024), valid, prior


def test_initial_matching_posterior_exact_after_pretrained_utility_capture():
    head = BoundedPriorRayFeatureReadout(); args = head_inputs()
    with torch.no_grad():head.score_selector.content_output.weight.normal_(std=.05)
    head.capture_initial_function()
    _, actual = head(*args)
    expected = masked_probabilities(args[-1][:, None].expand(-1, 4, -1), args[-2])
    assert torch.equal(actual, expected)
    assert not head(*args)[0].any()
    assert len(list(head.parameters())) == 9


def test_residual_odds_bounded_constant_shift_invariant_and_finite_empty_rows():
    torch.manual_seed(3); valid = torch.tensor([[True]*7, [False]*7]); prior = torch.randn(2, 7).masked_fill(~valid, -torch.inf)
    initial = torch.randn(2, 4, 7); current = (initial+torch.randn_like(initial)*100).requires_grad_()
    p = bounded_probabilities(current, initial, prior, valid, 1.)
    other = bounded_probabilities(current+19., initial, prior, valid, 1.)
    assert torch.allclose(p, other, atol=2e-7, rtol=2e-6)
    base = masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)
    relative = (p[0]/base[0]).log()
    assert float((relative.amax(-1)-relative.amin(-1)).max().detach()) <= 2.+2e-6
    assert torch.isfinite(p).all() and not p[1].any()
    (p*torch.randn_like(p)).sum().backward();assert torch.isfinite(current.grad).all() and current.grad.abs().sum()>0


def bounded_pair():
    _, original, data = pair(); torch.manual_seed(1317)
    bounded = BoundedPriorMemberFeatureSourceRayConditioner(residual_logit_bound=1.,
        arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024, query_dim=1024,
        hidden_dim=256, topk=64, score_temperature=.5, projection_mode='native-bilinear',
        ray_mode='semantic-top1', ray_chunk_size=4, candidate_aggregation='top1')
    return original, bounded, data


def test_all_original_parameters_and_cold_prior_preserved_then_all19_gradients():
    original, model, data = bounded_pair()
    assert set(dict(original.named_parameters())) == set(dict(model.named_parameters()))
    assert all(torch.equal(p, dict(model.named_parameters())[name]) for name, p in original.named_parameters())
    # Native runs load the trained seven-epoch utility, whose residual output
    # is active. An untrained zero utility needs a separate first output update.
    utility = deepcopy(original.ray_readout.score_selector.state_dict())
    utility['content_output.weight'].normal_(std=.02)
    original.load_utility(utility);model.load_utility(utility)
    before = {name: p.clone() for name, p in model.named_buffers()}
    change, _ = delta(model, data); assert not change.any()
    (change*torch.randn_like(change)).sum().backward()
    assert model.ray_readout.output.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True);activate(model)
    with torch.no_grad():model.ray_readout.output.weight.normal_(std=.02)
    change, _ = delta(model, data);(change*torch.randn_like(change)).sum().backward()
    trainable = [(name,p)for name,p in model.named_parameters()if p.requires_grad]
    assert len(trainable) == 19
    for name,p in trainable:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    assert all(torch.equal(value, dict(model.named_buffers())[name])for name,value in before.items())


def test_anchor_is_serialized_and_does_not_follow_trained_parameters():
    head = BoundedPriorRayFeatureReadout(); args = head_inputs(); before = {name:v.clone() for name,v in head.named_buffers()}
    with torch.no_grad():
        head.query.weight.add_(.01); head.score_selector.content_output.weight.normal_(std=.1);head.output.weight.normal_(std=.02)
    expected = head(*args)
    assert all(torch.equal(value, dict(head.named_buffers())[name])for name,value in before.items())
    restored = BoundedPriorRayFeatureReadout();restored.load_state_dict(deepcopy(head.state_dict()))
    actual = restored(*args);assert all(torch.equal(a,b)for a,b in zip(actual,expected))
    wrong = BoundedPriorRayFeatureReadout(residual_logit_bound=2.)
    with pytest.raises(RuntimeError, match='bounds differ'):wrong.load_state_dict(head.state_dict())


def test_valid_image_candidate_with_zero_prior_mass_has_finite_backward():
    head = BoundedPriorRayFeatureReadout(); args = list(head_inputs())
    args[-1][0, 0] = -torch.inf
    with torch.no_grad():head.output.weight.normal_(std=.01)
    out, probability = head(*args)
    assert not probability[0, :, 0].any() and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_bfloat16_autocast_starts_exactly_from_matching_posterior():
    head = BoundedPriorRayFeatureReadout(); args = head_inputs()
    with torch.no_grad():head.score_selector.content_output.weight.normal_(std=.02)
    head.capture_initial_function()
    with torch.autocast('cpu', dtype=torch.bfloat16):_, actual = head(*args)
    expected = masked_probabilities(args[-1][:, None].expand(-1, 4, -1), args[-2])
    assert torch.equal(actual, expected) and head.first_forward_prior_exact


def test_bounded_projected_value_control_keeps_actual_attention_and_original_carrier():
    from experiments.deformnext_retargeting.bounded_member_feature_value_control import feature_value_control
    _, model, data = bounded_pair();activate(model);model.eval()
    with torch.no_grad():
        model.ray_readout.output.weight.normal_(std=.02);model.ray_readout.query.weight.add_(.002)
        normal = model(**data);normal_weights = model.last_feature_probabilities[1].clone()
        events = []
        with feature_value_control(model, events):wrong = model(**data)
    assert torch.equal(model.last_feature_probabilities[1], normal_weights)
    assert torch.equal(normal.use_probability, wrong.use_probability)
    assert torch.equal(normal.attention.base_evidence, wrong.attention.base_evidence)
    assert not torch.equal(normal.source_numerator, wrong.source_numerator)
    assert events[0]['source_use_probability_exact_same_forward']
    assert all(row['bounded_functional_residual_and_frozen_reference_checked']for row in events[0]['chunks'])
    assert not model.ray_readout._forward_hooks and not model.ray_readout.value._forward_hooks


@pytest.mark.parametrize('fails', [False, True])
def test_runtime_scopes_constructor_and_bounded_value_verifier(tmp_path, monkeypatch, fails):
    import json
    import sys
    from experiments.deformnext_retargeting import bounded_member_feature_runtime as entry
    from experiments.deformnext_retargeting import member_feature_runtime as native_entry
    from experiments.deformnext_retargeting import member_feature_value_control as value_module
    from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer_module
    old_class, old_control = trainer_module.MemberFeatureSourceRayConditioner, value_module.feature_value_control
    _, _, data = bounded_pair()
    path = tmp_path/'config.json'
    path.write_text(json.dumps(dict(trainer=dict(name='MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer'),
        experiment=dict(attention_parameterization='bounded-matching-prior-functional-residual-v1',residual_logit_bound=1.))))
    remaining = ['--phase','native','--config',str(path),'--output',str(tmp_path)]
    def run_native():
        assert sys.argv[1:] == remaining
        assert trainer_module.MemberFeatureSourceRayConditioner is not old_class
        assert value_module.feature_value_control is not old_control
        if fails:raise RuntimeError('native failed')
        model = trainer_module.MemberFeatureSourceRayConditioner(arm='c1',implementation='r9-native-v5',
            key_dim=1024,value_dim=1024,query_dim=1024,hidden_dim=256,topk=64,score_temperature=.5,
            projection_mode='native-bilinear',ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1')
        model.load_utility(model.ray_readout.score_selector.state_dict());model.eval()
        with torch.no_grad():model(**data)
    monkeypatch.setattr(native_entry,'main',run_native)
    argv = ['bounded-runtime','--bounded-prior-logit-bound','1.0',*remaining];monkeypatch.setattr(sys,'argv',argv)
    if fails:
        with pytest.raises(RuntimeError,match='native failed'):entry.main()
    else:
        entry.main();result=json.loads((tmp_path/'bounded-member-feature-runtime.json').read_text())
        assert result['first_forward_prior_exact']==[True]
        assert result['original19_trainable_source_tensors_and_nine_feature_tensors']
    assert sys.argv is argv and trainer_module.MemberFeatureSourceRayConditioner is old_class
    assert value_module.feature_value_control is old_control
