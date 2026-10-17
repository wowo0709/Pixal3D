import torch

from experiments.deformnext_retargeting.audit_native_transport_projection import normalized_evidence
from experiments.deformnext_retargeting.probe_source_target_balance import blend_projected


def test_partial_support_does_not_dilute_transport_value_and_empty_support_is_finite():
    # A sample halfway between a valid constant feature and an invalid slot.
    def project(x):
        return (x[:, :1] + x[:, 1:2]) / 2
    values=torch.tensor([[[2.,4.],[100.,200.]],[[5.,7.],[8.,9.]]],requires_grad=True)
    valid=torch.tensor([[True,False],[False,False]])
    result,coverage=normalized_evidence(project,values,valid)
    torch.testing.assert_close(result[0,0],torch.tensor([2.,4.]))
    torch.testing.assert_close(coverage[:,0,0],torch.tensor([.5,0.]))
    assert torch.equal(result[1],torch.zeros_like(result[1]))
    result.sum().backward()
    assert torch.equal(values.grad[:,1],torch.zeros_like(values.grad[:,1]))
    assert torch.isfinite(values.grad).all()


def test_role_probe_keeps_source_views_and_zero_strength_is_native():
    anchor=torch.tensor([2.]);sources=torch.tensor([3.,5.,7.])
    native=(anchor+sources.sum())/4
    assert torch.equal(blend_projected(native,anchor,0),native)
    expected=.625*anchor+.125*sources.sum()
    torch.testing.assert_close(blend_projected(native,anchor,.5),expected)
