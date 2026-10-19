"""A zero feature residual must learn without first opening a scalar gate."""
import torch

from pixal3d.modules.ray_feature_attention import RayFeatureReadout


def inputs(width):
    return (torch.randn(2, 1024), torch.randn(2, 1024), torch.randn(2, 64, 1024),
            torch.randn(2, 64, width), torch.ones(2, 64, dtype=torch.bool), torch.randn(2, 64))


def test_zero_output_and_first_step_feature_gradient():
    torch.manual_seed(101)
    head = RayFeatureReadout()
    change, probability = head(*inputs(1024))
    assert torch.equal(change, torch.zeros_like(change))
    assert probability.shape == (2, 4, 64)
    assert torch.allclose(probability.sum(-1), torch.ones(2, 4))
    (change * torch.randn_like(change)).sum().backward()
    assert head.output.weight.grad.abs().sum() > 0
    assert head.query.weight.grad.abs().sum() == 0
    assert not any('ray_mix' in name for name,_ in head.named_parameters())


def test_pbr_width_and_finite_all_masked_output():
    head = RayFeatureReadout(2048)
    native, context, keys, values, valid, prior = inputs(2048)
    valid.zero_();values.fill_(float('nan'));keys.fill_(float('nan'))
    change, probability = head(native, context, keys, values, valid, prior)
    assert torch.equal(change, torch.zeros_like(change))
    assert torch.equal(probability, torch.zeros_like(probability))
