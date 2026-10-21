import numpy as np
import pytest

from experiments.deformnext_retargeting.member_correspondence_diagnostic import (
    distinct_members,topology_sets,summarize_probabilities,
)


def fixture():
    rays=np.full((3,3),-1,np.int64);rays[0]=[5,5,5]
    patches=np.array([7,7,8]);observed=np.ones(3,bool)
    positive=np.tile(np.array([31,1024+31,2048+31]),(1024,1))
    visible=np.zeros((1024,3),bool);visible[7,0]=True;visible[8,1]=True
    fg=np.zeros(1024,bool);fg[7:9]=True
    ids=np.zeros((3072,64),np.int64);valid=np.zeros_like(ids,bool)
    ids[5,:5]=[31,30,32,63,1024+31];valid[5,:5]=True
    source=np.ones((3,1024),bool)
    return rays,patches,observed,positive,visible,fg,ids,valid,source


def test_distinct_patch_votes_once_and_gt_sets_keep_camera_identity():
    args=fixture();ray,patch=distinct_members(*args[:3])
    assert ray.tolist()==[5,5] and patch.tolist()==[7,8]
    sets,relations=topology_sets(*args)
    assert len(relations['relation_ray'])==2
    assert sets['all_source_views']['GT_set_size'][5]==2
    assert sets['same_source_camera']['GT_set_size'][5]==1
    assert sets['all_source_views']['exact'][5,:5].tolist()==[True,False,False,False,True]
    assert sets['same_source_camera']['near1'][5,:5].tolist()==[True,True,False,True,False]


def test_ineligible_or_unsupported_gt_remains_in_denominator():
    args=list(fixture());args[-1][0,31]=False;args[-2][5,0]=False
    sets,_=topology_sets(*args)
    assert sets['same_source_camera']['observable'][5]
    assert not sets['same_source_camera']['eligible_GT_exists'][5]
    assert not sets['same_source_camera']['exact'][5].any()
    assert sets['same_source_camera']['near1'][5].any()
    args[-2][:]=False
    sets,_=topology_sets(*args)
    probabilities=np.zeros((3072,64));heads=np.zeros((3072,4,64))
    report=summarize_probabilities(sets,probabilities,heads,args[-2])
    assert report['same_source_camera']['observable_rays']==1
    assert report['same_source_camera']['native_supported_rays']==0
    assert report['same_source_camera']['exact']['prior_top1_hit_rays']==0


def test_probability_mass_and_top1_compare_actual_same_candidates():
    args=fixture();sets,_=topology_sets(*args)
    prior=np.zeros((3072,64));prior[5,:2]=[.2,.8]
    heads=np.repeat(prior[:,None],4,axis=1);heads[5,:,:2]=[.7,.3]
    r=summarize_probabilities(sets,prior,heads,args[-2])['same_source_camera']['exact']
    assert r['proposal_hit_rays']==1 and r['prior_top1_hit_rays']==0
    assert r['mean_head_top1_hit_rays']==1 and r['per_head_top1_hits']==4
    assert r['prior_GT_mass_sum']==pytest.approx(.2)
    assert r['mean_head_GT_mass_sum']==pytest.approx(.7)


def test_wrong_gt_source_view_is_rejected_and_background_is_unobservable():
    args=list(fixture());args[3]=args[3].copy();args[3][7,0]=1024+31
    with pytest.raises(ValueError,match='topology'):topology_sets(*args)
    args=list(fixture());args[5][:]=False
    sets,_=topology_sets(*args)
    assert not sets['all_source_views']['observable'].any()
