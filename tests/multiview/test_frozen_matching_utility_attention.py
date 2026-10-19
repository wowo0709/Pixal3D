"""Frozen semantic matching with a trainable, zero-start utility correction."""
import torch
import pytest

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.modules.task_matching_attention import TaskCandidateAttention
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin


def frozen():
    torch.manual_seed(87)
    return DeformationAwareConditioner(arm='c1',key_dim=1024,value_dim=1024,query_dim=1024,
        score_temperature=.05,task_attention={'decoder':True,'score_prior':True,'chunk_size':2,
            'train_decoder':False,'train_score':False,'content_mode':'residual-zero'})


def inputs():
    torch.manual_seed(31)
    return dict(target_keys=torch.randn(1,3,1024),source_keys=torch.randn(1,3,24,1024),
        source_values=torch.randn(1,3,24,1024),source_valid=torch.ones(1,3,24,dtype=torch.bool),
        target_query=torch.randn(1,3,1024),semantic_indices=torch.arange(64).reshape(1,1,64).expand(1,3,-1),
        matching_logits=torch.randn(1,3,64),candidate_valid=torch.ones(1,3,64,dtype=torch.bool),
        target_valid=torch.tensor([[True,True,False]]))


def test_zero_utility_preserves_known_matching_weights_and_excludes_invalid_candidates():
    torch.manual_seed(5);m=TaskCandidateAttention(width=8,hidden=4,content_mode='residual-zero')
    q=torch.randn(2,8);k=torch.randn(2,4,8);valid=torch.tensor([[True,True,False,True],[False]*4])
    prior=torch.tensor([[.2,.3,1.,.5],[1.,1.,1.,1.]]).log();prior[0,2]=100.
    expected=torch.tensor([[.2,.3,0.,.5],[0.,0.,0.,0.]])
    evidence,weights,_=m(q,q,k,valid,prior)
    torch.testing.assert_close(weights,expected,rtol=0,atol=1e-6)
    torch.testing.assert_close(evidence,(expected[...,None]*k).sum(1),rtol=0,atol=1e-6)
    k[~valid]=float('nan')
    torch.testing.assert_close(m(q,q,k,valid,prior)[0],evidence,rtol=0,atol=0)


def test_utility_output_can_learn_first_and_then_train_query_and_key_projections():
    torch.manual_seed(13);m=TaskCandidateAttention(width=8,hidden=4,content_mode='residual-zero')
    q=torch.randn(2,8);context=torch.randn(2,8);k=torch.randn(2,4,8)
    valid=torch.ones(2,4,dtype=torch.bool);prior=torch.randn(2,4);direction=torch.randn(2,8)
    opt=torch.optim.SGD(m.parameters(),lr=.001)
    (m(q,context,k,valid,prior)[0]*direction).sum().backward()
    assert m.content_output.weight.grad.abs().sum()>0
    assert m.query.weight.grad.count_nonzero()==0 and m.key.weight.grad.count_nonzero()==0
    opt.step();opt.zero_grad(set_to_none=True)
    (m(q,context,k,valid,prior)[0]*direction).sum().backward()
    for layer in [m.query,m.context,m.key]:
        assert layer.weight.grad.abs().sum()>0 and torch.isfinite(layer.weight.grad).all()
    assert m.log_content_scale.grad.abs()>0


def test_frozen_decoder_remains_present_and_utility_learning_preserves_all_matching_weights():
    m=frozen();p=inputs()
    assert len(m.q2c.decoder.blocks)==1 and not any(x.requires_grad for x in m.q2c.parameters())
    assert m.retargeter.input_dim==2048
    before={k:v.clone() for k,v in m.q2c.state_dict().items()}
    with torch.no_grad():m.task_attention.content_output.weight.normal_(std=.01)
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
    m(**p).evidence.square().mean().backward()
    assert all(x.grad is None for x in m.q2c.parameters())
    assert m.task_attention.content_output.weight.grad.abs().sum()>0
    assert m.task_attention.query.weight.grad.abs().sum()>0
    optimizer.step()
    assert all(torch.equal(value,m.q2c.state_dict()[key]) for key,value in before.items())


def test_frozen_zero_utility_keeps_exact_native_projection_and_first_retargeter_gradient():
    m=frozen();p=inputs();p['patch_grid']=torch.tensor([[[[0,1],[2,0]],[[1,2],[0,1]]]])
    bridge=object.__new__(ImageConditionedProjMixin);bridge.deform_conditioner=m;bridge.training_models={}
    native=torch.randn(1,8,1024);anchor=torch.randn_like(native)
    output=bridge._inject_deform_transport({'proj':native.clone()},p,None,anchor_condition={'proj':anchor})
    torch.testing.assert_close(output['proj'],native,rtol=0,atol=0)
    output['proj'].square().mean().backward()
    assert m.fusion_gate.grad.abs()>0 and m.retargeter.net[-1].weight.grad.abs().sum()>0
    assert all(x.grad is None for x in m.q2c.parameters())


def test_legacy_attention_checkpoint_keys_and_default_training_flags_are_unchanged():
    m=TaskCandidateAttention()
    assert set(m.state_dict())=={'query.weight','context.weight','key.weight','log_content_scale'}
    cond=DeformationAwareConditioner(arm='c1',key_dim=1024,value_dim=1024,query_dim=1024,
        task_attention={'decoder':True,'score_prior':True,'chunk_size':2})
    assert all(p.requires_grad for p in cond.q2c.parameters())
    assert not hasattr(cond.task_attention,'content_output')


@pytest.mark.parametrize('override',[
    {'decoder':False,'train_decoder':True}, {'score_prior':False,'train_score':True},
    {'train_decoder':0}, {'train_score':'false'}, {'content_mode':'unknown'},
])
def test_invalid_training_or_content_modes_are_rejected(override):
    spec={'decoder':True,'score_prior':True,'chunk_size':2,**override}
    with pytest.raises(ValueError):
        DeformationAwareConditioner(arm='c1',key_dim=1024,value_dim=1024,query_dim=1024,task_attention=spec)
