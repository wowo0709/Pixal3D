from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.deformnext_retargeting.ray_patch_score_pool import score_pool_ray_patches


class IdentityDecoder(nn.Module):
    def forward(self,query,keys):return query,keys


class CosineScore(nn.Module):
    def forward(self,query,keys,pair,cosine):return cosine


def inputs():
    target=torch.zeros(1024,1024);target[0,0]=1;target[1,1]=1
    source=torch.zeros(3,1024,1024);source[0,0,0]=1;source[0,1,1]=1
    return dict(target_keys=target,source_keys=source,source_valid=torch.ones(3,1024,dtype=torch.bool),
        candidates=torch.arange(64)[None].repeat(3,1),
        ray_ids=torch.tensor([[0,0,0],[-1,-1,-1],[-1,-1,-1]]),
        patch_ids=torch.tensor([0,0,1]),rays_per_view=1)


def test_each_ray_has_one_multimodal_distribution_without_voxel_repetition_bias():
    matcher=SimpleNamespace(decoder=IdentityDecoder(),fusion_head=CosineScore())
    weights,counts=score_pool_ray_patches(matcher,**inputs())
    assert counts.tolist()==[2,0,0]
    assert weights.shape==(3,64)
    assert torch.allclose(weights[0,:2],torch.tensor([.5,.5]),atol=1e-5,rtol=0)
    assert weights[1:].count_nonzero()==0
    assert torch.allclose(weights[0].sum(),torch.tensor(1.))


def test_readout_is_frozen_and_chunk_size_does_not_change_result():
    matcher=SimpleNamespace(decoder=IdentityDecoder(),fusion_head=CosineScore())
    values=inputs();values['target_keys'].requires_grad_(True)
    left,_=score_pool_ray_patches(matcher,**values,chunk_size=1)
    right,_=score_pool_ray_patches(matcher,**values,chunk_size=64)
    assert not left.requires_grad
    assert torch.equal(left,right)


def test_invalid_source_support_cannot_be_used_as_a_zero_feature():
    values=inputs();values['source_valid'][0,0]=False
    with pytest.raises(ValueError,match='alpha-valid'):
        score_pool_ray_patches(SimpleNamespace(decoder=IdentityDecoder(),fusion_head=CosineScore()),**values)


def test_disabled_native_matching_has_zero_readout_without_fabricating_candidates():
    values=inputs();values['ray_ids'].fill_(-1);values['source_valid'].zero_()
    weights,counts=score_pool_ray_patches(SimpleNamespace(decoder=IdentityDecoder(),fusion_head=CosineScore()),**values)
    assert weights.count_nonzero()==0
    assert counts.count_nonzero()==0
