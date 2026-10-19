import torch
import pytest

from experiments.deformnext_retargeting.ray_correspondence_audit import audit_ray_correspondences


def inputs():
    # One ray touches two different target patches; voxel repetition must not
    # make the first patch more important. Scalar Top1 misses both truths.
    rays=torch.tensor([[0,0,0],[-1,-1,-1],[-1,-1,-1]])
    candidates=torch.arange(64)[None].repeat(3,1)
    prior=torch.zeros(3,64);prior[:,50]=1
    learned=torch.zeros(3,64);learned[:,5]=.5;learned[:,10]=.5
    positives=torch.zeros(1024,3,dtype=torch.long);positives[0,0]=5;positives[1,0]=10
    visible=torch.zeros(1024,3,dtype=torch.bool);visible[:2,0]=True
    return dict(ray_ids=rays,patch_ids=torch.tensor([0,0,1]),candidates=candidates,
        prior_weights=prior,learned_weights=learned,source_valid=torch.ones(3072,dtype=torch.bool),
        positive_indices=positives,positive_visible=visible,target_foreground=torch.ones(1024,dtype=torch.bool),rays_per_view=1)


def test_multi_target_rays_require_coverage_of_each_distinct_patch():
    r=audit_ray_correspondences(**inputs())
    assert r['distinct_ray_target_patch_pairs']==2
    assert r['observable_ray_target_patch_pairs']==2
    assert r['coverage']['exact']['prior']['1']==0
    assert r['coverage']['exact']['prior']['64']==2
    assert r['scalar_top1_miss_top64_hit']['exact']==2
    assert r['ray_union_probability_mass']['exact']['learned']==1
    assert r['ray_union_probability_mass']['exact']['prior']==0
    assert r['per_target_patch_probability_mass']['exact']['learned']==1


def test_unobservable_labels_are_excluded_instead_of_counted_as_search_failures():
    data=inputs();data['positive_visible'][1]=False
    r=audit_ray_correspondences(**data)
    assert r['observable_ray_target_patch_pairs']==1
    assert r['unobservable_pairs_excluded']==1


def test_invalid_source_alpha_is_not_a_retrievable_positive():
    data=inputs();data['source_valid'][10]=False;data['learned_weights'].zero_();data['learned_weights'][:,5]=1
    r=audit_ray_correspondences(**data)
    assert r['observable_ray_target_patch_pairs']==1
    assert r['visible_GT_pairs_excluded_by_source_patch_alpha']==1


def test_invalid_probabilities_or_wrong_camera_addresses_are_rejected():
    data=inputs();data['learned_weights'][0,5]=2
    with pytest.raises(ValueError,match='normalized'):
        audit_ray_correspondences(**data)
    data=inputs();data['positive_indices'][0,0]=3072
    with pytest.raises(ValueError,match='address'):
        audit_ray_correspondences(**data)
