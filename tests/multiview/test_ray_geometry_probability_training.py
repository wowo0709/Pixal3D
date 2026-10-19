"""Removing appearance VO preserves matching math after Q/K have learned."""
import copy

import pytest
import torch

from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from pixal3d.modules.ray_geometry_retargeting import RayGeometryReadout


@pytest.mark.parametrize('width', [1024, 2048])
def test_learned_probability_and_matching_gradients_equal_native_with_nonzero_projections(width):
    torch.manual_seed(191)
    native_head = RayFeatureReadout(width)
    torch.nn.init.normal_(native_head.score_selector.content_output.weight, std=.02)
    torch.nn.init.normal_(native_head.output.weight, std=.02)
    geometry = RayGeometryReadout(copy.deepcopy(native_head))
    matching = {name: value for name, value in native_head.state_dict().items()
                if name not in {'value.weight', 'output.weight'}}
    geometry.load_state_dict(matching, strict=True)
    assert torch.count_nonzero(geometry.query.weight) > 0
    args = (torch.randn(3, 1024), torch.randn(3, 1024),
            torch.randn(3, 7, 1024), torch.randn(3, 7, width),
            torch.tensor([[True]*7, [True, False, True, False, True, False, True], [False]*7]),
            torch.randn(3, 7))
    feature, native_probability = native_head(*args)
    absent_feature, geometry_probability = geometry(*args)
    assert torch.count_nonzero(feature) > 0 and torch.count_nonzero(absent_feature) == 0
    assert torch.equal(native_probability, geometry_probability)
    reward = torch.randn_like(native_probability)
    (native_probability*reward).sum().backward()
    (geometry_probability*reward).sum().backward()
    native_parameters = dict(native_head.named_parameters())
    for name, value in geometry.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.equal(value.grad, native_parameters[name].grad)
