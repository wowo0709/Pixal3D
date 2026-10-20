"""Controls must isolate payload selection and the location of weighted pooling."""
import torch

from experiments.deformnext_retargeting.consistent_ray_controls import (
    mean_input_correction, top1_probability)


def test_top1_payload_is_valid_one_hot_and_distinguishes_native_and_attention_scores():
    probability = torch.tensor([[[.1,.8,.1],[.1,.8,.1]], [[0.,0.,0.],[0.,0.,0.]]])
    prior = torch.tensor([[3.,2.,10.],[0.,0.,0.]])
    valid = torch.tensor([[True,True,False],[False,False,False]])
    native = top1_probability(probability, prior, valid, native=True)
    learned = top1_probability(probability, prior, valid, native=False)
    assert torch.equal(native[0], torch.tensor([[1.,0.,0.],[1.,0.,0.]]))
    assert torch.equal(learned[0], torch.tensor([[0.,1.,0.],[0.,1.,0.]]))
    assert native[1].count_nonzero() == learned[1].count_nonzero() == 0


def test_mean_control_pools_both_appearance_and_deformation_before_the_operator():
    class Nonlinear:
        def correction(self, appearance, deformation, g_max):
            return g_max*(appearance[:, :1]+deformation).square()

    model = Nonlinear()
    target, raw_target = torch.tensor([[9.]]), torch.tensor([[10.]])
    values, raw = torch.tensor([[[1.],[3.],[999.]]]), torch.tensor([[[2.],[6.],[999.]]])
    ids, valid = torch.tensor([[0,1,2]]), torch.tensor([[True,True,False]])
    p, batch = torch.tensor([[.25,.75,0.]]), torch.tensor([0])
    result = mean_input_correction(model,target,raw_target,values,raw,batch,ids,valid,p,.5,2)
    assert torch.equal(result, torch.tensor([[.5*(2.5+10.-5.)**2]]))
    wrong_top1 = .5*(1.+10.-5.)**2
    paired = .5*(.25*(1.+10.-2.)**2+.75*(3.+10.-6.)**2)
    assert result.item() != wrong_top1 and result.item() != paired
