from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.task_attention_controls import task_value_context
from pixal3d.modules.task_matching_attention import TaskCandidateAttention


def test_value_intervention_preserves_keys_weights_logits_support_and_inputs():
    torch.manual_seed(17);layer=TaskCandidateAttention(width=8,hidden=4);m=SimpleNamespace(task_attention=layer)
    q=torch.randn(2,8);values=torch.randn(2,6,8);valid=torch.tensor([[True,False,True,True,False,False],[True]*6])
    prior=torch.randn(2,6);saved=values.clone();normal=layer(q,q,values,valid,prior);events=[]
    with task_value_context(m,'wrong-address',events):altered=layer(q,q,values,valid,prior)
    torch.testing.assert_close(normal[1],altered[1],rtol=0,atol=0)
    torch.testing.assert_close(normal[2],altered[2],rtol=0,atol=0)
    assert (normal[0]-altered[0]).abs().max()>0 and torch.equal(values,saved)
    assert events[0]['reassigned_valid_slots']==9
    # Uniform weights provide a multiset conservation check, independent of
    # the implementation's donor construction.
    layer.log_content_scale.data.fill_(-100);plain=layer(q,q,values,valid)
    with task_value_context(m,'wrong-address',[]):changed=layer(q,q,values,valid)
    torch.testing.assert_close(changed[0],plain[0],rtol=1e-6,atol=1e-6)


def test_hook_restores_on_exception_and_normal_is_exact():
    layer=TaskCandidateAttention(width=8,hidden=4);m=SimpleNamespace(task_attention=layer)
    q=torch.randn(1,8);v=torch.randn(1,4,8);mask=torch.ones(1,4,dtype=torch.bool)
    baseline=layer(q,q,v,mask)
    with pytest.raises(RuntimeError):
        with task_value_context(m,'wrong-address',[]):raise RuntimeError('interrupted diagnostic')
    with task_value_context(m,'normal',[]):actual=layer(q,q,v,mask)
    for x,y in zip(actual,baseline):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert not layer._forward_hooks


def test_control_does_not_silently_apply_to_frozen_or_native_V1():
    with pytest.raises(ValueError,match='requires a task'):
        with task_value_context(None,'wrong-address',[]):pass


def test_static_soft_value_control_preserves_each_retrieved_pool_weights_and_state():
    from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
    model=DeformationAwareConditioner(arm='c1',key_dim=1024,value_dim=4,query_dim=4,hidden_dim=8)
    model.fusion_gate.data.fill_(.2)
    values=torch.randn(1,3,32,4)
    data=dict(source_values=values,source_valid=torch.ones(1,3,32,dtype=torch.bool),
        semantic_indices=torch.arange(64).reshape(1,1,64).expand(1,3,64),
        matching_logits=torch.arange(64).float().reshape(1,1,64).expand(1,3,64)/100,
        target_valid=torch.tensor([[True,False,True]]),target_query=torch.randn(1,3,4))
    normal=model(**data);saved={k:v.clone() for k,v in model.state_dict().items()};events=[]
    with task_value_context(model,'wrong-address',events):changed=model(**data)
    for field in ['candidate_indices','matching_logits','candidate_weights','view_weights','query_valid','fusion_scale']:
        torch.testing.assert_close(getattr(normal,field),getattr(changed,field),rtol=0,atol=0)
    candidates=values.flatten(1,2)[0,:64].roll(32,0)
    expected=(normal.candidate_weights[0,:,:,None]*candidates[None]).sum(1)
    torch.testing.assert_close(changed.evidence[0],expected,rtol=0,atol=0)
    assert events[0]['reassigned_valid_slots']==128 and changed.evidence[0,1].eq(0).all()
    assert not torch.equal(changed.evidence,normal.evidence)
    assert '_attention' not in model.__dict__
    for key,val in saved.items():torch.testing.assert_close(model.state_dict()[key],val,rtol=0,atol=0)
    torch.testing.assert_close(model(**data).evidence,normal.evidence,rtol=0,atol=0)
    with pytest.raises(RuntimeError):
        with task_value_context(model,'wrong-address',[]):raise RuntimeError('interrupted')
    assert '_attention' not in model.__dict__


def test_value_reassignment_preserves_partial_pool_mean_and_empty_query():
    from experiments.deformnext_retargeting.task_attention_controls import _reassign
    values=torch.randn(3,6,4);valid=torch.tensor([[True,False,True,False,True,False],[False]*6,[False,True,False,False,False,False]])
    weights=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
    evidence=(weights[...,None]*values).sum(1);logits=torch.randn(3,6);events=[]
    changed,w,l=_reassign(values,valid,evidence,weights,logits,events)
    torch.testing.assert_close(changed,evidence,rtol=1e-6,atol=1e-6)
    assert w is weights and l is logits and events[0]['reassigned_valid_slots']==3


def diagnostic_records():
    rows=[{'sample_id':str(i),'source_cluster_id':str(i),'seed':i,'tp':2,'fp':1,'fn':2,
           'target_voxels_sha256':'a'*64,'target_voxel_shape':[64,64,64]} for i in range(128)]
    common={'status':'PASS','step':480,'weights':'raw','sampling_steps':12,'guidance':3.,
        'optimizer_updates_during_evaluation':0,'target_modality':'rgb','input_camera_ids':[8,9,10,11],
        'input_states':['M1','M0','M0','M0']}
    normal={**common,'task_value_control':'normal','per_asset':rows[:32]}
    wrong={**common,'task_value_control':'wrong-address','per_asset':rows[:32],
        'task_value_control_events':[{'matching_logits_and_weights_unchanged':True,'reassigned_valid_slots':64}]}
    return normal,wrong,{'per_asset':rows}


def test_diagnostic_requires_actual_fixed_final_normal_replay():
    from experiments.deformnext_retargeting.summarize_task_value_diagnostic import summarize
    a,b,c=diagnostic_records();result=summarize(a,b,c)
    assert result['normal_minus_wrong_address']['gain']==0
    c['per_asset'][0]={**c['per_asset'][0],'tp':3}
    with pytest.raises(ValueError,match='exactly replay'):summarize(a,b,c)


def test_diagnostic_rejects_keys_or_weights_changing_with_values():
    from experiments.deformnext_retargeting.summarize_task_value_diagnostic import summarize
    a,b,c=diagnostic_records();b['task_value_control_events'][0]['matching_logits_and_weights_unchanged']=False
    with pytest.raises(ValueError,match='preserve keys'):summarize(a,b,c)
