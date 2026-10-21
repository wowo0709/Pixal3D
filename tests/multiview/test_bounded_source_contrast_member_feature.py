from copy import deepcopy

import pytest
import torch

from pixal3d.modules.bounded_source_contrast_member_feature import BoundedSourceContrastMemberFeatureConditioner
from pixal3d.modules.bounded_member_feature_source_ray import masked_probabilities
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from test_source_contrast_member_feature import fixture
from test_member_posterior_source_ray import activate


def pair():
    _,parent,data=fixture()
    torch.manual_seed(1317)
    model=BoundedSourceContrastMemberFeatureConditioner(residual_logit_bound=1.,
        arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,query_dim=1024,
        hidden_dim=256,topk=64,score_temperature=.5,projection_mode='native-bilinear',
        ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1')
    utility=deepcopy(parent.ray_readout.score_selector.state_dict())
    utility['content_output.weight'].normal_(std=.02)
    parent.load_utility(utility);model.load_utility(utility)
    return parent,model,data


def test_same_nineteen_initial_parameters_but_actual_four_heads_start_at_prior():
    parent,model,data=pair()
    p=dict(parent.named_parameters());q=dict(model.named_parameters())
    assert p.keys()==q.keys()and sum(v.requires_grad for v in q.values())==19
    assert all(torch.equal(v,q[k])for k,v in p.items())
    out=model(**data);selected,actual=model.last_feature_probabilities
    prior=out.attention.matching_logits.reshape(-1,64)[selected]
    expected=masked_probabilities(prior[:,None].expand(-1,4,-1),torch.ones_like(prior,dtype=torch.bool))
    assert torch.equal(actual,expected)
    assert not source_ray_delta(out.source_numerator,out.semantic_weight,data['source_values']).any()
    assert model.retargeter.net[0].in_features==2048
    assert not any('deformation_projection'in key for key in model.state_dict())
    assert model.retargeter._source_context is None


def test_real_source_reference_and_frozen_anchors_survive_nonzero_learning_paths():
    _,model,data=pair();before={k:v.clone()for k,v in model.named_buffers()}
    seen=[];hook=model.retargeter.register_forward_pre_hook(lambda m,a:seen.append(m._source_context.detach().clone()))
    activate(model)
    with torch.no_grad():model.ray_readout.output.weight.normal_(std=.02)
    out=model(**data);hook.remove()
    expected=data['source_values'].reshape(-1,1024)[model.last_source_reference_rows]
    assert torch.equal(seen[0],expected)
    (out.source_numerator*torch.randn_like(out.source_numerator)).sum().backward()
    for name,p in model.named_parameters():
        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all()and p.grad.abs().sum()>0,name
    assert all(p.grad is None for p in model.q2c.parameters())
    assert all(torch.equal(v,dict(model.named_buffers())[k])for k,v in before.items())
    assert model.retargeter._source_context is None


def test_strict_distinct_checkpoint_identity_and_exact_roundtrip():
    parent,model,data=pair()
    with pytest.raises(RuntimeError):model.load_state_dict(parent.state_dict(),strict=True)
    _,model,data=pair()
    with pytest.raises(RuntimeError):parent.load_state_dict(model.state_dict(),strict=True)
    activate(model)
    with torch.no_grad():model.ray_readout.output.weight.normal_(std=.01)
    state=deepcopy(model.state_dict());_,restored,_=pair();restored.load_state_dict(state,strict=True)
    with torch.no_grad():a,b=model(**data),restored(**data)
    assert torch.equal(a.source_numerator,b.source_numerator)
    assert torch.equal(a.semantic_weight,b.semantic_weight)
    assert 'bounded_source_contrast_schema'in state


def test_missing_support_keeps_native_with_all_nineteen_finite_zero_gradients():
    _,model,data=pair();data['observed_query'].zero_();out=model(**data)
    delta=source_ray_delta(out.source_numerator,out.semantic_weight,data['source_values'])
    assert not delta.any();delta.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()and not p.grad.any()for p in model.parameters()if p.requires_grad)
    assert model.retargeter._source_context is None


def test_bfloat16_source_context_and_head_keep_exact_cold_prior_and_native_delta():
    _,model,data=pair();model.eval()
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):out=model(**data)
    selected,actual=model.last_feature_probabilities
    prior=out.attention.matching_logits.reshape(-1,64)[selected]
    expected=masked_probabilities(prior[:,None].expand(-1,4,-1),torch.ones_like(prior,dtype=torch.bool))
    assert torch.equal(actual,expected)and model.ray_readout.first_forward_prior_exact
    assert not source_ray_delta(out.source_numerator,out.semantic_weight,data['source_values']).any()
    assert model.retargeter._source_context is None
