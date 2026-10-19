"""Geometry-only correspondence preserves native probabilities and appearance."""
import copy

import pytest
import torch

from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from pixal3d.modules.ray_geometry_retargeting import RayGeometryReadout
from pixal3d.modules.ray_deformation_query import geometry_readout


@pytest.mark.parametrize('width', [1024, 2048])
def test_geometry_has_no_appearance_projections_and_matches_anchored_native_probability(width):
    torch.manual_seed(179)
    original = RayFeatureReadout(width)
    original.query.weight.data.zero_()
    head = RayGeometryReadout(copy.deepcopy(original))
    assert len(list(head.parameters())) == 7
    assert not hasattr(head, 'value') and not hasattr(head, 'output')
    native, context = torch.randn(3, 1024), torch.randn(3, 1024)
    keys, values = torch.randn(3, 5, 1024), torch.randn(3, 5, width)
    valid = torch.tensor([[True]*5, [True, False, True, True, False], [False]*5])
    prior = torch.randn(3, 5)
    old_delta, old_probability = original(native, context, keys, values, valid, prior)
    delta, probability = head(native, context, keys, values, valid, prior)
    assert torch.equal(delta, old_delta)
    assert torch.equal(probability, old_probability)
    assert torch.count_nonzero(probability[-1]) == 0
    assert probability[1, :, ~valid[1]].count_nonzero() == 0
    assert not torch.equal(probability[0], torch.full_like(probability[0], .2))


def test_attention_combines_real_geometry_and_learns_without_changing_appearance():
    torch.manual_seed(181)
    head = RayGeometryReadout(RayFeatureReadout())
    native, context = torch.randn(2, 1024), torch.randn(2, 1024)
    keys, values = torch.randn(1, 3, 1024), torch.randn(1, 3, 1024)
    raw = torch.cat([keys, torch.randn(1, 3, 1024)], -1)
    target = torch.randn(2, 2048)
    batch, indices = torch.zeros(2, dtype=torch.long), torch.tensor([[0, 1], [1, 2]])
    valid, prior = torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 2)
    args = (head, native, context, values, keys, raw, target, batch, indices, valid, prior)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    gradients = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        appearance, deformation, probability = geometry_readout(*args)
        assert torch.count_nonzero(appearance) == 0
        expected = target - (probability.mean(1)[..., None] * raw[batch[:, None], indices]).sum(1)
        assert torch.equal(deformation, expected)
        deformation.square().mean().backward()
        gradients.append({name: float(p.grad.abs().sum()) for name, p in head.named_parameters()})
        optimizer.step()
    assert gradients[0]['query.weight'] > 0 and gradients[0]['key.weight'] == 0
    assert all(value > 0 for value in gradients[-1].values())
    assert torch.count_nonzero(raw) > 0 and torch.count_nonzero(target) > 0
