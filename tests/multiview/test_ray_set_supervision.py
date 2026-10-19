import torch

from experiments.deformnext_retargeting.ray_set_supervision import multi_positive_ray_loss,ray_positive_sets


def test_loss_requires_complementary_target_parts_rather_than_any_one_correct_candidate():
    labels=torch.tensor([[True,False],[False,True]])
    queries=torch.tensor([0,0])
    balanced,_=multi_positive_ray_loss(torch.tensor([[.5,.5]]),queries,labels)
    collapsed,_=multi_positive_ray_loss(torch.tensor([[.99,.01]]),queries,labels)
    assert collapsed>balanced
    assert abs(float(balanced)-.69314718056)<1e-6


def test_proposal_misses_remain_reported_and_do_not_create_visibility_targets():
    probability=torch.tensor([[.7,.3]],requires_grad=True)
    loss,stats=multi_positive_ray_loss(probability,torch.tensor([0]),torch.zeros(1,2,dtype=torch.bool))
    loss.backward()
    assert stats==dict(observable_relations=1,covered_relations=0,proposal_misses=1,supervised_queries=0)
    assert torch.equal(probability.grad,torch.zeros_like(probability))


def test_each_ray_is_balanced_despite_a_different_number_of_observed_target_patches():
    probabilities=torch.tensor([[.5,.5],[.9,.1]])
    labels=torch.tensor([[True,False],[True,False],[True,False]])
    loss,_=multi_positive_ray_loss(probabilities,torch.tensor([0,0,1]),labels)
    expected=(-torch.log(torch.tensor(.5))-torch.log(torch.tensor(.9)))/2
    assert torch.allclose(loss,expected)


def test_teacher_deduplicates_voxel_repetitions_and_retains_per_target_patch_sets():
    rays=torch.tensor([[0,0,0],[-1,-1,-1],[-1,-1,-1]])
    positive=torch.zeros(1024,3,dtype=torch.long);positive[0,0]=5;positive[1,0]=10
    visible=torch.zeros(1024,3,dtype=torch.bool);visible[:2,0]=True
    result=ray_positive_sets(ray_ids=rays,patch_ids=torch.tensor([0,0,1]),
        candidates=torch.arange(64)[None].repeat(3,1),source_valid=torch.ones(3072,dtype=torch.bool),
        positive_indices=positive,positive_visible=visible,target_foreground=torch.ones(1024,dtype=torch.bool),rays_per_view=1)
    assert result['ray_indices'].tolist()==[0,0]
    assert result['target_patch_indices'].tolist()==[0,1]
    assert result['positive_exact'].nonzero().tolist()==[[0,5],[1,10]]
    assert result['positive_near1'].any(-1).all()
