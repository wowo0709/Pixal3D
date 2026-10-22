from dataclasses import replace

import pytest
import torch

from experiments.deformnext_retargeting.pbr_source_transfer_controls import mask_source_transfer
from pixal3d.modules.deform_conditioner import DeformationCondition
from pixal3d.modules.target_first_ray_v2 import TargetFirstRayEvidence


def fixture(gate=-.2):
    e=torch.linspace(-1,1,2048).expand(1,3072,-1).clone()
    r=torch.full_like(e,.13)
    u=torch.linspace(0,1,3072)[None]
    g=torch.tensor(gate)
    m=u[...,None]*g
    attention=DeformationCondition(e+r,None,None,None,None,r,
        fusion_scale=g,base_evidence=e)
    return TargetFirstRayEvidence((m*e+u[...,None]*r).reshape(1,3,1024,2048),
        m.reshape(1,3,1024,1),attention,u)


def correction(output,source):
    return output.source_numerator-output.semantic_weight*source


@torch.no_grad()
@pytest.mark.parametrize('gate',[-.2,.3])
def test_R_off_retains_the_signed_direct_Top1_minus_source_term(gate):
    o=fixture(gate);s=torch.zeros_like(o.source_numerator)
    c=mask_source_transfer(o,'R-off')
    assert torch.equal(c.semantic_weight,o.semantic_weight)
    expected=(o.use_probability[...,None]*gate*o.attention.base_evidence).reshape_as(s)
    assert torch.equal(correction(c,s),expected)


@torch.no_grad()
def test_direct_shift_off_must_also_remove_the_subtracted_native_source_mass():
    o=fixture();s=torch.full_like(o.source_numerator,7.)
    c=mask_source_transfer(o,'direct-top1-shift-off')
    expected=(o.use_probability[...,None]*o.attention.retarget_residual).reshape_as(s)
    assert torch.equal(correction(c,s),expected)
    assert not torch.count_nonzero(c.semantic_weight)


@torch.no_grad()
def test_equal_spatial_evidence_cancels_only_direct_delivery_not_R():
    o=fixture();s=o.attention.base_evidence.reshape_as(o.source_numerator)
    c=mask_source_transfer(o,'R-off')
    assert not torch.count_nonzero(correction(c,s))
    c=mask_source_transfer(o,'direct-top1-shift-off')
    assert torch.count_nonzero(correction(c,s))>0


@torch.no_grad()
def test_both_off_restores_native_conditioning_and_preserves_metadata():
    o=fixture();before=o.source_numerator.clone();c=mask_source_transfer(o,'both-source-contributions-off')
    assert c.attention is o.attention and c.use_probability is o.use_probability
    assert not torch.count_nonzero(correction(c,torch.ones_like(c.source_numerator)))
    assert torch.equal(o.source_numerator,before)


@torch.no_grad()
def test_normal_is_the_same_output_object():
    o=fixture();assert mask_source_transfer(o,'normal') is o


def test_training_gradient_context_is_rejected():
    with pytest.raises(ValueError):mask_source_transfer(fixture(),'R-off')


@torch.no_grad()
@pytest.mark.parametrize('mutation',['numerator','mass','nonfinite','shape','unknown_control'])
def test_inconsistent_source_decomposition_cannot_be_silently_masked(mutation):
    o=fixture();control='R-off'
    if mutation=='numerator':o=replace(o,source_numerator=o.source_numerator+1)
    elif mutation=='mass':o=replace(o,semantic_weight=o.semantic_weight+1)
    elif mutation=='nonfinite':o=replace(o,source_numerator=o.source_numerator*float('nan'))
    elif mutation=='shape':o=replace(o,source_numerator=o.source_numerator[:,:,:1])
    else:control='uniform-NoScore'
    with pytest.raises(ValueError):mask_source_transfer(o,control)
