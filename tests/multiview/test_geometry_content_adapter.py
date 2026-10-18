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


def test_historical_network_preserves_six_full_optimizer_steps_of_original_r4():
    torch.manual_seed(43)
    original=GeometryFactorialVoxelAdapter(channels=4,hidden=8,geometry_hidden=3)
    restored=GeometryContentRetargetAdapter.from_historical_content(original)
    target,source,geometry=torch.randn(1,4096,4),torch.randn(1,4096,4),torch.randn(1,4096,9)
    desired=torch.randn_like(target)
    left=torch.optim.AdamW(original.parameters(),lr=1e-4,weight_decay=.01)
    right=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=.01)
    for update in range(6):
        for model,optimizer in [(original,left),(restored,right)]:
            optimizer.zero_grad(set_to_none=True)
            output,trace=(model(target,source,geometry,mode='r4_content') if model is original else model(target,source,geometry))
            loss=(output-desired).square().mean()+.001*trace['correction'].square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        expected={key.replace('gate_encoder.','constant_gate_network.'):value for key,value in original.state_dict().items()}
        actual=restored.state_dict()
        assert expected.keys()==actual.keys()
        for key,value in expected.items():torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
    assert restored.constant_gate_network[-1].bias.grad.abs().sum()>0


def test_reference_initialization_matches_common_weights_and_rng_with_physical_noquery():
    torch.manual_seed(61)
    original=GeometryFactorialVoxelAdapter(channels=4,hidden=8,geometry_hidden=3)
    expected_rng=torch.get_rng_state()
    arms=[]
    for query in [False,True]:
        torch.manual_seed(61)
        model=GeometryContentRetargetAdapter.initialized_from_reference(with_query=query,channels=4,hidden=8,geometry_hidden=3)
        assert torch.equal(torch.get_rng_state(),expected_rng)
        assert (model.content_encoder is None)==(not query)
        for key,value in original.gate_encoder.state_dict().items():
            torch.testing.assert_close(model.constant_gate_network.state_dict()[key],value,rtol=0,atol=0)
        arms.append(model)
    torch.testing.assert_close(arms[0].residual[1].weight,arms[1].residual[1].weight[:,:15],rtol=0,atol=0)
    assert arms[0].residual[1].in_features==15 and arms[1].residual[1].in_features==18
    with pytest.raises(ValueError,match='do not pass zero channels'):
        arms[0](torch.randn(1,4096,4),torch.randn(1,4096,4),torch.zeros(1,4096,9))


def test_scalar_conversion_is_inference_equivalent_but_not_optimizer_equivalent():
    torch.manual_seed(8)
    original=GeometryFactorialVoxelAdapter(channels=4,hidden=8,geometry_hidden=3)
    with torch.no_grad():original.residual[-1].weight.normal_(0,.1)
    scalar=GeometryContentRetargetAdapter.from_historical_content(original,gate_parameterization='scalar')
    target,source,geometry=torch.randn(1,4096,4),torch.randn(1,4096,4),torch.randn(1,4096,9)
    torch.testing.assert_close(original(target,source,geometry,mode='r4_content')[0],scalar(target,source,geometry)[0],rtol=1e-6,atol=1e-6)
    for model in [original,scalar]:
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
        output=(model(target,source,geometry,mode='r4_content') if model is original else model(target,source,geometry))[0]
        output.square().mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    assert not torch.allclose(original(target,source,geometry,mode='r4_content')[0],scalar(target,source,geometry)[0],rtol=0,atol=1e-7)
