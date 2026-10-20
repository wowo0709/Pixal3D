from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from pixal3d.modules.member_posterior_source_ray import MemberPosteriorSourceRayConditioner
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.modules.target_first_ray_v2 import TargetFirstRayEvidence
from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin,ProjGrid


def fixture(mode):
    torch.manual_seed(1317)
    config=dict(arm='c1',implementation='r9-native-v5',key_dim=1024,value_dim=1024,
        query_dim=1024,hidden_dim=256,topk=64,score_temperature=.5,
        projection_mode='native-bilinear',ray_mode=mode,ray_chunk_size=4,
        candidate_aggregation='top1' if mode=='semantic-top1' else 'weighted')
    if mode=='semantic-attention':
        config['task_attention']=dict(decoder=True,score_prior=True,train_decoder=False,train_score=False,
            content_mode='residual-zero',initial_content_scale=16.,chunk_size=4)
    model=MemberPosteriorSourceRayConditioner(**config)
    torch.manual_seed(7124)
    frame=torch.randn(1,1024,1024);native=torch.randn(1,1024,1024)
    patch_ids=torch.tensor([[7,7,9,11,11,12]])
    valid=torch.zeros(1,3,1024,dtype=torch.bool);valid.flatten()[:128]=True
    data=dict(native_voxels=torch.randn(1,6,1024),target_omega_voxels=frame[:,patch_ids[0]],
        target_native_voxels=native[:,patch_ids[0]],frame_target_keys=frame,target_patch_ids=patch_ids,
        ray_ids=torch.tensor([[[0,0,0,1,1,2],[4,4,4,-1,-1,5],[-1,-1,-1,-1,-1,-1]]]),
        observed_query=torch.tensor([[True,True,True,False,False,True]]),
        source_keys=torch.randn(1,3,1024,1024),source_values=torch.randn(1,3,1024,1024),source_valid=valid)
    return model,data


def activate(model,*,attention=True):
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.normal_(std=.01)
        model.evidence_use.network[-1].weight.normal_(std=.01)
        if attention and model.task_attention is not None:
            model.task_attention.content_output.weight.normal_(std=.01)


@pytest.mark.parametrize('mode',['semantic-top1','semantic-weighted','semantic-attention'])
def test_zero_correction_initial_gradients_and_all_activated_source_paths(mode):
    model,data=fixture(mode);out=model(**data)
    delta=source_ray_delta(out.source_numerator,out.semantic_weight,data['source_values'])
    assert not delta.any() and not out.semantic_weight.any()
    assert model.retargeter.net[0].in_features==2048
    (delta*torch.randn_like(delta)).sum().backward()
    assert model.fusion_gate.grad.abs()>0 and model.retargeter.net[-1].weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True);activate(model)
    out=model(**data);source_ray_delta(out.source_numerator,out.semantic_weight,data['source_values']).square().mean().backward()
    for name,value in model.named_parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0,name
    assert all(value.grad is None for value in model.q2c.parameters())


def test_matched_readouts_share_weights_and_cold_attention_equals_weighted():
    fixed,data=fixture('semantic-weighted');attention,_=fixture('semantic-attention');top1,_=fixture('semantic-top1')
    assert all(torch.equal(value,attention.state_dict()[key]) for key,value in fixed.state_dict().items())
    assert all(torch.equal(value,top1.state_dict()[key]) for key,value in fixed.state_dict().items())
    with torch.no_grad():
        for model in [fixed,attention,top1]:
            model.fusion_gate.fill_(.2);model.retargeter.net[-1].weight.fill_(.001)
        a,b,c=[model(**data) for model in [fixed,attention,top1]]
    for field in ['candidate_indices','matching_logits','candidate_weights','base_evidence','retarget_residual']:
        assert torch.equal(getattr(a.attention,field),getattr(b.attention,field)),field
    assert torch.equal(a.source_numerator,b.source_numerator) and torch.equal(a.semantic_weight,b.semantic_weight)
    assert torch.equal(a.attention.candidate_indices,c.attention.candidate_indices)
    active=c.attention.query_valid
    assert ((c.attention.candidate_weights[active]>0).sum(-1)==1).all()
    index=c.attention.candidate_indices.gather(-1,c.attention.matching_logits.argmax(-1,keepdim=True))
    expected=data['source_values'].flatten(1,2)[0,index[0,:,0]]
    assert torch.equal(c.attention.base_evidence[active],expected[active[0]])
    assert (a.attention.candidate_weights[active]>0).sum(-1).min()>1


@pytest.mark.parametrize('mode',['semantic-top1','semantic-weighted','semantic-attention'])
def test_empty_observations_abstain_without_zero_feature_observations_and_keep_DDP_edges(mode):
    model,data=fixture(mode);activate(model)
    data['observed_query'].zero_();out=model(**data)
    delta=source_ray_delta(out.source_numerator,out.semantic_weight,data['source_values'])
    assert not delta.any() and not out.use_probability.any() and not out.attention.query_valid.any()
    delta.sum().backward()
    for name,value in model.named_parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all() and not value.grad.any(),name


def test_trainer_passes_input_patch_addresses_and_preserves_native_MV4_carrier():
    grid=ProjGrid(16,512);transforms=torch.eye(4).repeat(1,4,1,1);transforms[:,:,2,3]=2
    camera=dict(transform_matrix=transforms,distance=torch.full((1,4),2.),camera_angle_x=torch.ones(1,4),mesh_scale=torch.ones(1))
    native=dict(proj=torch.randn(1,4096,1024),**{'global':torch.randn(1,1,1024)})
    negative={key:torch.randn_like(value) for key,value in native.items()};target={key:torch.randn_like(value) for key,value in native.items()}
    seen=[]
    class Conditioner:
        active=False
        def __call__(self,**kwargs):
            seen.append(kwargs);mass=torch.full((1,3,1024,1),.125 if self.active else 0.)
            numerator=mass*(kwargs['source_values']+1)+(.02 if self.active else 0.)
            return TargetFirstRayEvidence(numerator,mass,SimpleNamespace(),torch.zeros(1,3072))
    trainer=object.__new__(Trainer);trainer.image_attn_mode='proj';trainer.deform_conditioner=Conditioner()
    trainer.image_cond_model=SimpleNamespace(proj_grid=grid,fixed_projection_transform=grid.front_view_transform_matrix)
    trainer.encode_image_proj=lambda images,**kw:(target,negative)
    images=torch.rand(1,4,3,8,8)
    transport=dict(target_keys=torch.randn(1,1024,1024),source_keys=torch.randn(1,3,1024,1024),
        source_values=torch.randn(1,3,1024,1024),source_valid=torch.ones(1,3,1024,dtype=torch.bool),
        target_query=torch.randn(1,1024,1024),target_valid=torch.ones(1,1024,dtype=torch.bool),
        patch_grid=(torch.arange(4096)%1024).reshape(1,16,16,16))
    with patch.object(ImageConditionedProjMixin,'_encode_deform_views',return_value=(native,negative)) as base:
        untouched,neg=trainer._encode_deform_views(images,camera,None)
        cold,_=trainer._encode_deform_views(images,camera,transport)
        trainer.deform_conditioner.active=True;active,_=trainer._encode_deform_views(images,camera,transport)
    assert untouched is native and neg is negative and base.call_count==3
    assert all(torch.equal(cold[key],value) for key,value in native.items())
    assert seen[0]['frame_target_keys'] is transport['target_keys']
    assert torch.equal(seen[0]['target_patch_ids'],transport['patch_grid'].flatten(1))
    assert seen[0]['source_values'] is transport['source_values']
    assert not {'deformation','semantic_indices','matching_logits'}&seen[0].keys()
    torch.testing.assert_close(active['proj'],native['proj']+(.125+.02)*.75,atol=1e-6,rtol=0)
    assert active['global'] is native['global']


@pytest.mark.parametrize('mode',['semantic-top1','semantic-weighted','semantic-attention'])
def test_saved_wrong_values_preserve_same_forward_matching_and_source_use(mode):
    from experiments.deformnext_retargeting.target_first_value_control import fixed_use_value_control
    model,data=fixture(mode);activate(model);model.eval();events=[]
    with torch.no_grad():
        normal=model(**data)
        with fixed_use_value_control(model,events):
            wrong=model(**data)
        restored=model(**data)
    assert events and all(row['source_use_probability_exact_same_forward'] for row in events)
    assert torch.equal(normal.use_probability,wrong.use_probability)
    for field in ['candidate_indices','matching_logits','candidate_weights']:
        assert torch.equal(getattr(normal.attention,field),getattr(wrong.attention,field))
    assert not torch.equal(normal.source_numerator,wrong.source_numerator)
    assert torch.equal(normal.source_numerator,restored.source_numerator)
