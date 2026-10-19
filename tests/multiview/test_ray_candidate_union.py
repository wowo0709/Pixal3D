import pytest
import torch

from experiments.deformnext_retargeting.ray_candidate_union import merge_ray_patch_proposals


def data():
    proposals=torch.zeros(1024,64,dtype=torch.long)
    proposals[0]=torch.arange(64);proposals[1]=torch.arange(64,128)
    return [torch.tensor([[0,0,0],[-1,-1,-1],[-1,-1,-1]]),torch.tensor([0,0,1]),
        proposals,torch.ones(3072,dtype=torch.bool)]


def test_ray_budget_is_shared_across_distinct_target_patches_not_depth_repetitions():
    result,active=merge_ray_patch_proposals(*data(),rays_per_view=1)
    assert active.tolist()==[True,False,False]
    assert result[0,:6].tolist()==[0,64,1,65,2,66]
    assert len(set(result[0].tolist()))==64
    assert int((result[0]<64).sum())==32


def test_duplicate_proposals_are_not_counted_as_additional_correspondence():
    values=data();values[2][1]=values[2][0]
    result,_=merge_ray_patch_proposals(*values,rays_per_view=1)
    assert result[0].tolist()==list(range(64))


def test_invalid_alpha_or_insufficient_distinct_support_is_rejected():
    values=data();values[3][127]=False
    with pytest.raises(ValueError,match='alpha-invalid'):
        merge_ray_patch_proposals(*values,rays_per_view=1)
    values=data();values[2].zero_()
    with pytest.raises(ValueError,match='64 distinct'):
        merge_ray_patch_proposals(*values,rays_per_view=1)
