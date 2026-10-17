"""Check original learned content behavior and the physical query ablation."""
import pytest
import torch

from experiments.deformnext_retargeting.geometry_content_adapter import GeometryContentRetargetAdapter
from experiments.deformnext_retargeting.historical_reproduction import GeometryFactorialVoxelAdapter


def test_converted_content_matches_nonzero_historical_r4_function_and_input_gradients():
    torch.manual_seed(8)
    original = GeometryFactorialVoxelAdapter(channels=4, hidden=8, geometry_hidden=3)
    with torch.no_grad():
        original.residual[-1].weight.normal_(0, .1)
        original.residual[-1].bias.normal_(0, .1)
    converted = GeometryContentRetargetAdapter.from_historical_content(original)
    target = torch.randn(1, 4096, 4, requires_grad=True)
    source = torch.randn_like(target, requires_grad=True)
    geometry = torch.randn(1, 4096, 9, requires_grad=True)
    expected = original(target, source, geometry, mode='r4_content')[0]
    actual = converted(target, source, geometry)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    gradient = torch.randn_like(expected)
    previous = torch.autograd.grad((expected * gradient).sum(), (target, source, geometry))
    current = torch.autograd.grad((actual * gradient).sum(), (target, source, geometry))
    for a, b in zip(previous, current):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)


def test_noquery_has_no_encoder_or_extra_channels_and_rejects_zero_padding():
    adapter = GeometryContentRetargetAdapter(with_query=False, channels=4, hidden=8, geometry_hidden=3)
    assert adapter.content_encoder is None
    assert not any('content_encoder' in key for key in adapter.state_dict())
    assert adapter.residual[0].normalized_shape == (3 * 4 + 3,)
    assert adapter.residual[1].in_features == 3 * 4 + 3
    target, source = torch.randn(1, 4096, 4), torch.randn(1, 4096, 4)
    actual, trace = adapter(target, source)
    assert torch.equal(actual, source)
    assert trace['geometry_channels'] == trace['content_channels'] == 0
    with pytest.raises(ValueError, match='do not pass zero channels'):
        adapter(target, source, torch.zeros(1, 4096, 9))


@pytest.mark.parametrize('query', [False, True])
def test_both_additive_arms_have_exact_native_start_and_connected_learning(query):
    adapter = GeometryContentRetargetAdapter(with_query=query, channels=4, hidden=8, geometry_hidden=3)
    target, source = torch.randn(1, 4096, 4), torch.randn(1, 4096, 4)
    geometry = torch.randn(1, 4096, 9) if query else None
    actual, _ = adapter(target, source, geometry)
    assert torch.equal(actual, source)
    (actual - target).square().mean().backward()
    assert adapter.residual[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(adapter.residual[-1].weight.grad).all()
    if query:
        assert adapter.residual[1].in_features == 3 * 4 + 3 + 3
