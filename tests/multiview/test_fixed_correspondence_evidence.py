"""Physical constructors, candidate arithmetic and unchanged delivery; no CPU NN forward."""
import ast
import inspect
import textwrap

import pytest
import torch

from experiments.deformnext_retargeting.fixed_correspondence_native_recipe import (
    VARIANTS, construct_conditioner,
)
from pixal3d.modules import fixed_correspondence_evidence as module
from pixal3d.modules.consistent_ray_evidence import ConsistentRayEvidenceConditioner
from test_consistent_ray_evidence import config
from test_shared_evidence_ablation import matcher


def make(arm):
    kwargs=config('E_FIXED_RAW')[0]['trainer']['args']['deform_conditioner']
    kwargs['freeze_q2c']=False
    return construct_conditioner(arm,kwargs)


@pytest.mark.parametrize('arm',VARIANTS)
def test_same_cold_core10_and_physical_matcher_removal(matcher,arm):
    before=torch.random.get_rng_state().clone();m=make(arm)
    assert torch.equal(before,torch.random.get_rng_state())
    assert m._actual_initial_core10_equal_Fixed and m._actual_initial_core10_equal_Top1
    assert sum(p.requires_grad for p in m.parameters())==10
    assert m.ray_readout is None and m.retargeter.input_dim==2048
    assert (m.q2c is None)==(arm!='M_FIXED_D1_COS')
    if m.q2c is not None:
        assert m.q2c.fusion_head is None and not any(p.requires_grad for p in m.q2c.parameters())
    m.train()
    assert m.q2c is None or not m.q2c.training


@pytest.mark.parametrize('a',VARIANTS)
@pytest.mark.parametrize('b',VARIANTS)
def test_another_correspondence_checkpoint_cannot_be_silently_loaded(matcher,a,b):
    own,other=make(a),make(b)
    if a==b: own.load_state_dict(other.state_dict(),strict=True)
    else:
        with pytest.raises(RuntimeError):own.load_state_dict(other.state_dict(),strict=True)


def test_semantic_gather_preserves_batch_and_original_view_addresses():
    flat=torch.arange(2*3072).reshape(2,3072,1).float().expand(-1,-1,1024)
    selected=torch.tensor([5,3072+6]);ids=torch.tensor([[0,1025,3071],[7,2048,3070]])
    out=module.candidate_feature_rows(flat,selected,ids)
    assert torch.equal(out[:,:,0],torch.tensor([[0,1025,3071],[3079,5120,6142.]]))


def test_spatial_values_are_real_projected_features_not_flat_image_patches():
    flat=torch.zeros(1,3072,1024)
    spatial=torch.tensor([3.,7.,11.]).reshape(1,1,3,1).expand(1,3072,3,1024)
    selected=torch.tensor([0,1400]);ids=torch.arange(3).expand(2,3)
    out=module.candidate_feature_rows(flat,selected,ids,spatial_values=spatial)
    assert torch.equal(out[:,:,0],torch.tensor([[3.,7.,11.],[3.,7.,11.]]))
    with pytest.raises(ValueError):
        module.candidate_feature_rows(flat,selected,torch.tensor([[0,1024,2048]]).expand(2,3),spatial_values=spatial)


def test_spatial_and_semantic_view_probability_conventions_are_distinct():
    weights=torch.tensor([[[.2,.3,.5]]]);spatial=torch.arange(3).reshape(1,1,3)
    semantic=torch.tensor([[[0,1025,1026]]])
    assert torch.equal(module.candidate_view_weights(spatial,weights,spatial=True),weights)
    assert torch.equal(module.candidate_view_weights(semantic,weights,spatial=False),torch.tensor([[[.2,.8,0.]]]))
    with pytest.raises(ValueError):module.candidate_view_weights(semantic,weights,spatial=True)


@pytest.mark.parametrize('arm',VARIANTS)
def test_each_matcher_routes_to_its_existing_posterior_without_other_inputs(matcher,monkeypatch,arm):
    m=make(arm);seen=[];sentinel=object()
    def call(*args,**kw):seen.append((args,kw));return sentinel
    for name in ['raw_member_ray_posteriors','d1_cosine_member_ray_posteriors','raw_spatial_member_posteriors']:
        monkeypatch.setattr(module,name,call if name=={
            'M_FIXED_RAW':'raw_member_ray_posteriors','M_FIXED_D1_COS':'d1_cosine_member_ray_posteriors',
            'M_FIXED_SPATIAL':'raw_spatial_member_posteriors'}[arm] else lambda *a,**k:pytest.fail('wrong matcher'))
    kwargs=dict(target_keys='target',source_keys='source',source_valid='alpha',ray_ids='ray',
        patch_ids='patch',observed_query='observed',temperature=.05,chunk_size=64)
    assert m._posterior(spatial_keys='projected' if arm=='M_FIXED_SPATIAL' else None,**kwargs) is sentinel
    args,kw=seen[0]
    assert ('source_keys' in kw)==(arm!='M_FIXED_SPATIAL')
    assert ('spatial_keys' in kw)==(arm=='M_FIXED_SPATIAL')
    assert bool(args)==(arm=='M_FIXED_D1_COS')


def test_retargeting_use_fusion_fallback_and_return_match_parent_ast():
    def tail(function):
        tree=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        i=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='residual')
        result=[]
        for node in tree.body[i:]:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='view_weights':
                continue
            result.append(ast.dump(node,include_attributes=False))
        return result
    assert tail(module.FixedCorrespondenceEvidenceConditioner.forward)==tail(ConsistentRayEvidenceConditioner.forward)
