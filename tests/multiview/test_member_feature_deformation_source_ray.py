import pytest
import torch

from pixal3d.modules.member_feature_deformation_source_ray import (
    MemberFeatureDeformationSourceRayConditioner,selected_target_ray_context,geometry_feature_chunk)
from pixal3d.modules.ray_group_query import pool_ray_queries
from experiments.deformnext_retargeting.member_feature_value_control import feature_value_control
from test_member_feature_source_ray import pair,delta
from test_member_posterior_source_ray import activate


def matched():
    _,original,data=pair()
    settings=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.5,projection_mode='native-bilinear',
        ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1')
    torch.manual_seed(1317)
    model=MemberFeatureDeformationSourceRayConditioner(deformation_dim=2048,**settings)
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in original.state_dict().items())
    raw=dict(data,target_context=torch.cat((data['frame_target_keys'],torch.randn(1,1024,1024)),-1),
        source_context=torch.cat((data['source_keys'],torch.randn(1,3,1024,1024)),-1))
    return original,model,data,raw


def test_physical_channels_and_zero_projection_preserve_cold_and_activated_NoDQ_exact():
    original,model,data,raw=matched()
    assert not hasattr(original.retargeter,'deformation_projection')
    assert original.retargeter.input_dim==2048 and model.retargeter.input_dim==4096
    assert model.retargeter.deformation_projection.in_features==2048
    assert sum(p.requires_grad for p in model.parameters())==20
    with torch.no_grad():
        cold,_=delta(model,raw)
        assert not cold.any() and model.last_deformation_query.abs().sum()>0
        activate(original)
        torch.nn.init.normal_(original.ray_readout.output.weight,std=.01)
        state=model.state_dict();state.update(original.state_dict());model.load_state_dict(state)
        before,a=delta(original,data);after,b=delta(model,raw)
        assert torch.equal(before,after)
        assert torch.equal(a.use_probability,b.use_probability)
        assert all(torch.equal(x,y) for x,y in zip(original.last_feature_probabilities,model.last_feature_probabilities))
    assert not model.retargeter.deformation_projection.weight.any()


def test_new_RAW_geometry_and_all_twenty_tensors_reach_the_loss_after_R_activation():
    _,model,_,raw=matched();activate(model)
    torch.nn.init.normal_(model.ray_readout.output.weight,std=.01)
    torch.nn.init.normal_(model.ray_readout.score_selector.content_output.weight,std=.01)
    change,_=delta(model,raw)
    change.square().mean().backward()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
            assert parameter.grad.abs().sum()>0,name
    assert all(p.grad is None for p in model.q2c.parameters())


def test_GLOBAL_context_changes_geometry_only_matching_feature_weights_and_use_stay_exact():
    _,model,_,raw=matched();activate(model)
    torch.nn.init.normal_(model.ray_readout.output.weight,std=.01)
    torch.nn.init.normal_(model.retargeter.deformation_projection.weight,std=.002)
    changed=raw['target_context'].clone();changed[...,1024:]+=2
    with torch.no_grad():
        normal,a=delta(model,raw);query=model.last_deformation_query.clone()
        selected,weights=model.last_feature_probabilities
        altered,b=delta(model,dict(raw,target_context=changed))
    assert not torch.equal(normal,altered) and not torch.equal(query,model.last_deformation_query)
    assert torch.equal(a.use_probability,b.use_probability)
    assert all(torch.equal(x,y) for x,y in zip((selected,weights),model.last_feature_probabilities))
    for field in ['candidate_indices','matching_logits','candidate_weights','base_evidence']:
        assert torch.equal(getattr(a.attention,field),getattr(b.attention,field)),field


@pytest.mark.parametrize('missing',['observed_query','source_valid'])
def test_missing_evidence_keeps_native_prior_and_DQ_has_explicit_zero_DDP_edge(missing):
    _,model,_,raw=matched();activate(model);raw[missing].zero_()
    value,out=delta(model,raw)
    assert not value.any() and not out.use_probability.any() and model.last_deformation_query.shape==(0,2048)
    value.sum().backward()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:assert parameter.grad is not None and not parameter.grad.any(),name


def test_same_forward_V_only_intervention_keeps_the_RAW_deformation_query_exact():
    _,model,_,raw=matched();activate(model);model.eval()
    torch.nn.init.normal_(model.ray_readout.output.weight,std=.01)
    torch.nn.init.normal_(model.retargeter.deformation_projection.weight,std=.001)
    events=[]
    with torch.no_grad():
        normal,_=delta(model,raw);geometry=model.last_deformation_query.clone()
        with feature_value_control(model,events):wrong,_=delta(model,raw)
    assert not torch.equal(normal,wrong)
    assert torch.equal(geometry,model.last_deformation_query)
    assert events and events[0]['source_use_probability_exact_same_forward']
    assert all(x['all_four_head_weights_exact_same_forward'] for x in events[0]['chunks'])


def test_selected_target_context_matches_full_unmasked_native_ray_pool_for_multiple_assets():
    torch.manual_seed(9)
    raw=torch.randn(2,1024,2048);patch=torch.tensor([[2,2,7,8,9],[7,3,3,9,7]])
    rays=torch.tensor([[[0,0,1,2,2],[4,4,5,5,6],[-1,2,2,-1,2]],
                       [[3,3,2,0,1],[2,3,4,4,4],[-1,-1,8,8,9]]])
    selected=torch.tensor([3072+1024+4,0,1024+5,3072+2048+8,2])
    expected=pool_ray_queries(raw.gather(1,patch[...,None].expand(-1,-1,2048)),rays,num_rays=1024).values.reshape(-1,2048)[selected]
    actual=selected_target_ray_context(raw,patch,rays,selected)
    assert torch.equal(actual,expected)


def test_RAW_channels_are_required_and_must_correspond_to_the_same_FRAME():
    _,model,data,raw=matched()
    with pytest.raises(TypeError):model(**data)
    malformed=raw['source_context'].clone();malformed[...,:1024]+=1
    with pytest.raises(ValueError,match='original FRAME'):model(**dict(raw,source_context=malformed))
    malformed=raw['source_context'].clone();malformed[...,1024:]=torch.nan
    with pytest.raises(ValueError,match='must be finite'):model(**dict(raw,source_context=malformed))


def test_constructor_preserves_common_parameter_initialization_and_caller_rng():
    original,model,_,_=matched()
    settings=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,query_dim=1024,
        hidden_dim=256,topk=64,score_temperature=.5,projection_mode='native-bilinear',
        ray_mode='semantic-top1',ray_chunk_size=4,candidate_aggregation='top1')
    torch.manual_seed(1317);type(original)(**settings);before=torch.rand(4)
    torch.manual_seed(1317);type(model)(deformation_dim=2048,**settings);after=torch.rand(4)
    assert torch.equal(before,after)


def test_score_selected_RAW_geometry_carries_information_when_appearance_is_identical():
    from pixal3d.modules.ray_feature_attention import RayFeatureReadout
    torch.manual_seed(47);head=RayFeatureReadout()
    native,context=torch.randn(1,1024),torch.randn(1,1024)
    keys=torch.randn(1,2,1024);values=torch.randn(1,1,1024).expand(1,2,-1)
    raw=torch.cat((keys,torch.randn(1,2,1024)),-1);target=torch.randn(1,2048)
    batch=torch.zeros(1,dtype=torch.long);indices=torch.tensor([[0,1]]);valid=torch.ones(1,2,dtype=torch.bool)
    arguments=(head,native,context,values,keys,raw,target,batch,indices,valid)
    a,dq_a,weights_a=geometry_feature_chunk(*arguments,torch.tensor([[30.,-30.]]),'normal')
    b,dq_b,weights_b=geometry_feature_chunk(*arguments,torch.tensor([[-30.,30.]]),'normal')
    assert torch.equal(a,b) and not torch.equal(weights_a,weights_b)
    assert not torch.allclose(dq_a,dq_b)
