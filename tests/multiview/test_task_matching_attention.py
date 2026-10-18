"""Task-path contracts: independent decoder/score routes, source addresses and native start."""
import copy

import pytest
import torch

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.modules.task_matching_attention import TaskCandidateAttention, contextual_query
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin


def make(decoder=True, score=True, chunk=2):
    torch.manual_seed(17)
    m=DeformationAwareConditioner(arm='c1',key_dim=1024,value_dim=1024,query_dim=1024,
        task_attention={'decoder':decoder,'score_prior':score,'chunk_size':chunk},score_temperature=.05)
    with torch.no_grad():
        for block in m.q2c.decoder.blocks:
            block.attention_gate.fill_(.2);block.ffn_gate.fill_(.1)
    return m


def inputs():
    torch.manual_seed(41)
    return dict(target_keys=torch.randn(1,3,1024),source_keys=torch.randn(1,3,24,1024),
        source_values=torch.randn(1,3,24,1024),source_valid=torch.ones(1,3,24,dtype=torch.bool),
        target_query=torch.randn(1,3,1024),semantic_indices=torch.arange(64).reshape(1,1,64).expand(1,3,-1),
        matching_logits=torch.randn(1,3,64),candidate_valid=torch.ones(1,3,64,dtype=torch.bool),
        target_valid=torch.tensor([[True,True,False]]))


def test_context_mask_matches_authority_and_ignores_invalid_values():
    m=make();q=torch.randn(2,1024);k=torch.randn(2,64,1024);v=torch.ones(2,64,dtype=torch.bool)
    expected=m.q2c.decoder(q,k)[0]
    torch.testing.assert_close(contextual_query(m.q2c.decoder,q,k,v),expected,rtol=0,atol=0)
    v[:,17:]=False
    first=contextual_query(m.q2c.decoder,q,k,v)
    k[:,17:]=float('nan')
    torch.testing.assert_close(contextual_query(m.q2c.decoder,q,k,v),first,rtol=0,atol=0)


def test_score_prior_independent_of_context_and_invalid_fallback():
    torch.manual_seed(4);m=TaskCandidateAttention();q=torch.randn(2,1024);k=torch.randn(2,64,1024)
    valid=torch.ones(2,64,dtype=torch.bool);valid[1]=False
    prior=torch.zeros(2,64);prior[:,3]=20
    soft=m(q,q,k,valid);biased=m(q,q,k,valid,prior)
    assert biased[1][0,3]>.99 and (biased[0][0]-soft[0][0]).abs().max()>0
    assert soft[0][1].count_nonzero()==biased[0][1].count_nonzero()==0
    assert torch.isfinite(soft[0]).all() and torch.isfinite(biased[0]).all()
    shifted=m(q,q,k,valid,prior+100)
    torch.testing.assert_close(shifted[0],biased[0],atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('decoder,score',[(True,True),(False,True),(True,False)])
def test_physical_decoder_and_independent_gradients(decoder,score):
    m=make(decoder,score);p=inputs();out=m(**p)
    assert len(m.q2c.decoder.blocks)==int(decoder)
    assert all(x.requires_grad==score for x in m.q2c.fusion_head.parameters())
    assert out.query_valid.tolist()==[[True,True,False]]
    assert out.evidence[0,2].count_nonzero()==0
    # Warm the zero-output retargeter to check the actual trainable attention
    # routes. At native initialization its output layer correctly blocks them.
    out.evidence.square().mean().backward()
    for layer in [m.task_attention.query,m.task_attention.context,m.task_attention.key]:
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum()>0
    if decoder:
        assert m.q2c.decoder.blocks[0].attention_gate.grad.abs()>0
    if score:
        assert m.q2c.fusion_head.output.weight.grad.abs().sum()>0
    else:
        assert all(x.grad is None for x in m.q2c.fusion_head.parameters())


def test_chunking_preserves_evidence_and_gradients_and_uses_addresses():
    a=make(chunk=1);b=copy.deepcopy(a);b.task_attention_config['chunk_size']=4;p=inputs()
    oa=a(**p);ob=b(**p)
    torch.testing.assert_close(oa.evidence,ob.evidence,rtol=2e-5,atol=2e-6)
    oa.evidence.square().mean().backward();ob.evidence.square().mean().backward()
    for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        if x.grad is None:assert y.grad is None
        else:torch.testing.assert_close(x.grad,y.grad,rtol=2e-3,atol=2e-6)
    changed={**p,'semantic_indices':torch.roll(p['semantic_indices'],5,-1)}
    # Joint reordering of the same candidate tuples must be invariant.
    torch.testing.assert_close(a(**changed).evidence,oa.evidence,rtol=2e-5,atol=2e-6)
    changed={**p,'source_values':torch.roll(p['source_values'],3,2)}
    assert (a(**changed).evidence-oa.evidence).abs().max()>0


def test_exact_native_start_and_empty_rank_has_zero_grad_edges():
    m=make();bridge=object.__new__(ImageConditionedProjMixin);bridge.deform_conditioner=m;bridge.training_models={}
    p=inputs();p['patch_grid']=torch.tensor([[[[0,1],[2,0]],[[1,2],[0,1]]]])
    native=torch.randn(1,8,1024);anchor=torch.randn_like(native)
    out=bridge._inject_deform_transport({'proj':native.clone()},p,None,anchor_condition={'proj':anchor})
    torch.testing.assert_close(out['proj'],native,rtol=0,atol=0)
    out['proj'].square().mean().backward()
    assert m.fusion_gate.grad.abs()>0 and m.retargeter.net[-1].weight.grad.abs().sum()>0
    empty=make();p=inputs();p['target_valid'].zero_();empty(**p).evidence.sum().backward()
    assert all(x.grad is not None and x.grad.count_nonzero()==0 for x in empty.parameters() if x.requires_grad)


def test_no_silent_cached_only_or_invalid_task_contract():
    m=make();p=inputs();p.pop('target_keys')
    with pytest.raises(ValueError,match='explicit Omega'):m(**p)
    with pytest.raises(ValueError,match='SS C1'):
        DeformationAwareConditioner(arm='b3',key_dim=1024,value_dim=1024,query_dim=1024,
            task_attention={'decoder':True,'score_prior':True,'chunk_size':2})


def test_native_spatial_candidates_are_actual_camera_projection_not_semantic_or_GT():
    from types import SimpleNamespace
    from pixal3d.modules.task_matching_attention import inject_native_spatial_task
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid,compute_multiview_projection_matrices
    m=make(score=False);m.correspondence_mode='native-spatial'
    grid=ProjGrid(2,512);extractor=SimpleNamespace(proj_grid=grid,fixed_projection_transform=grid.front_view_transform_matrix)
    torch.manual_seed(81)
    transport={'source_values':torch.randn(1,3,1024,1024),'source_keys':torch.randn(1,3,1024,1024),
        'target_keys':torch.randn(1,1024,1024),'target_valid':torch.ones(1,1024,dtype=torch.bool),
        'source_valid':torch.ones(1,3,1024,dtype=torch.bool)}
    camera={'transform_matrix':torch.eye(4).reshape(1,1,4,4).repeat(1,4,1,1),
        'distance':torch.full((1,4),2.5),'camera_angle_x':torch.full((1,4),.7),'mesh_scale':torch.ones(1)}
    projection,_=compute_multiview_projection_matrices(camera['transform_matrix'],camera['distance'],extractor.fixed_projection_transform)
    def project(p,v):return grid(p.reshape(1,32,32,-1),camera['camera_angle_x'][:,v],camera['distance'][:,v],camera['mesh_scale'],projection[:,v])
    anchor=project(torch.randn(1,1024,1024),0)
    sources=torch.stack([project(transport['source_values'][:,i],i+1) for i in range(3)],2)
    native=(anchor+sources.sum(2))/4
    # With uniform learned view attention and no residual, the spatial route
    # recovers exactly the original four-view conditioning, even with gate ON.
    m.task_attention.log_content_scale.data.fill_(-100);m.fusion_gate.data.fill_(1.)
    result,trace=inject_native_spatial_task(m,m,extractor,{'proj':native.clone()},{'proj':anchor},transport,camera)
    torch.testing.assert_close(result['proj'],native,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(trace.base_evidence,sources.mean(2),rtol=1e-6,atol=1e-6)
    assert trace.candidate_weights.shape==(1,8,3)
    with pytest.raises(ValueError,match='original projected'):m(**inputs())


def test_decoder_removal_keeps_other_initial_weights_identical_and_score_off_keeps_context_active():
    full=make();no_decoder=make(decoder=False);no_prior=make(score=False)
    for name,p in full.state_dict().items():
        if not name.startswith('q2c.decoder.'):
            torch.testing.assert_close(p,no_decoder.state_dict()[name],rtol=0,atol=0)
        torch.testing.assert_close(p,no_prior.state_dict()[name],rtol=0,atol=0)
    p=inputs();baseline=no_prior(**p).evidence
    no_prior.q2c.decoder.blocks[0].attention_gate.data.zero_()
    assert (no_prior(**p).evidence-baseline).abs().max()>0


def test_native_spatial_larger_query_chunks_preserve_values_and_gradients():
    from pixal3d.modules.task_matching_attention import native_spatial_evidence
    a=make(chunk=1);b=copy.deepcopy(a);b.task_attention_config['chunk_size']=4
    torch.manual_seed(28);args=dict(target_keys=torch.randn(1,35,1024),target_query=torch.randn(1,35,1024),
        keys=torch.randn(1,35,3,1024),values=torch.randn(1,35,3,1024),target_valid=torch.ones(1,35,dtype=torch.bool))
    left=native_spatial_evidence(a,**args);right=native_spatial_evidence(b,**args)
    torch.testing.assert_close(left.evidence,right.evidence,rtol=3e-5,atol=3e-6)
    left.evidence.square().mean().backward();right.evidence.square().mean().backward()
    for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        if x.grad is None:assert y.grad is None
        else:torch.testing.assert_close(x.grad,y.grad,rtol=3e-3,atol=2e-6)


def test_task_layer_initialization_preserves_rank_CUDA_rng_and_parent_CPU_rng(monkeypatch):
    common=dict(arm='c1',key_dim=1024,value_dim=1024,query_dim=1024)
    torch.manual_seed(91);DeformationAwareConditioner(**common);expected=torch.get_rng_state()
    torch.manual_seed(91);calls=[]
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda seed:calls.append(seed))
    DeformationAwareConditioner(**common,task_attention={'decoder':True,'score_prior':True,'chunk_size':64})
    assert calls==[]
    assert torch.equal(torch.get_rng_state(),expected)
