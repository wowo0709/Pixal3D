import torch

from experiments.deformnext_retargeting.ray_set_head_training import ray_set_logit_loss


def test_hard_positive_keeps_finite_loss_and_nonzero_gradient():
    logits=torch.tensor([[-80.,80.]],requires_grad=True)
    loss,stats=ray_set_logit_loss(logits,torch.tensor([0]),torch.tensor([[True,False]]))
    loss.backward()
    assert torch.isfinite(loss) and abs(float(loss)-160)<1e-5
    assert torch.equal(logits.grad,torch.tensor([[-1.,1.]]))
    assert stats['covered_relations']==1


def test_complementary_sets_prefer_balanced_mass_over_one_correct_part():
    relations=torch.tensor([0,0]);labels=torch.tensor([[True,False],[False,True]])
    balanced,_=ray_set_logit_loss(torch.tensor([[0.,0.]]),relations,labels)
    collapsed,_=ray_set_logit_loss(torch.tensor([[4.,-4.]]),relations,labels)
    assert collapsed>balanced


def test_missing_proposal_is_not_relabelled_as_invisible():
    logits=torch.tensor([[1.,2.]],requires_grad=True)
    loss,stats=ray_set_logit_loss(logits,torch.tensor([0]),torch.tensor([[False,False]]))
    loss.backward()
    assert loss==0 and stats['proposal_misses']==1 and stats['supervised_queries']==0
    assert logits.grad.count_nonzero()==0
