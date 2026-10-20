import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from pixal3d.modules.checkpointed_ffn_chunks import install_checkpointed_ffn_chunks
from pixal3d.modules.transformer.blocks import FeedForwardNet


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([FeedForwardNet(8, 5.3334) for _ in range(2)])

    def forward(self, x):
        for block in self.blocks:
            def body(value, selected=block):
                return value + selected(torch.sin(value))
            x = checkpoint(body, x, use_reentrant=False)
        return x


@pytest.mark.parametrize('frozen', [False, True])
def test_native_outer_checkpoint_outputs_and_backward_match(frozen):
    torch.manual_seed(724)
    native = Model().double()
    native.requires_grad_(not frozen)
    chunked = copy.deepcopy(native)
    keys = tuple(chunked.state_dict())
    identities = {name: id(value) for name, value in chunked.named_parameters()}
    info = install_checkpointed_ffn_chunks(chunked, chunk_assets=2, expected_layers=2)
    assert tuple(chunked.state_dict()) == keys
    assert {name: id(value) for name, value in chunked.named_parameters()} == identities
    assert info['global_batch_or_accumulation_changed'] is False
    a = torch.randn(7, 11, 8, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    actual, expected = chunked(b), native(a)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(b.grad, a.grad, atol=1e-12, rtol=1e-12)
    for left, right in zip(native.parameters(), chunked.parameters()):
        if frozen:
            assert left.grad is None and right.grad is None
        else:
            torch.testing.assert_close(right.grad, left.grad, atol=1e-12, rtol=1e-12)


def test_forward_and_backward_never_expand_full_batch_in_chunked_ffn():
    model = Model()
    install_checkpointed_ffn_chunks(model, chunk_assets=2, expected_layers=2)
    batches = []
    handles = [block.mlp[0].register_forward_pre_hook(lambda _, inputs: batches.append(inputs[0].shape[0]))
               for block in model.blocks]
    try:
        model(torch.randn(7, 11, 8, requires_grad=True)).square().mean().backward()
        assert batches and max(batches) <= 2
        batches.clear()
        with torch.no_grad():
            model(torch.randn(7, 11, 8))
        assert batches == [7, 7]
    finally:
        for handle in handles:
            handle.remove()


def test_eval_and_saved_weights_are_native_exact():
    torch.manual_seed(254)
    native = Model().eval()
    chunked = copy.deepcopy(native)
    before = {name: value.clone() for name, value in chunked.state_dict().items()}
    install_checkpointed_ffn_chunks(chunked, chunk_assets=2, expected_layers=2)
    with torch.no_grad():
        x = torch.randn(7, 11, 8)
        assert torch.equal(native(x), chunked(x))
    assert all(torch.equal(value, before[name]) for name, value in chunked.state_dict().items())


def test_repeated_install_or_unverified_layers_are_rejected():
    model = Model()
    with pytest.raises(ValueError, match='layer count'):
        install_checkpointed_ffn_chunks(model, chunk_assets=2)
    install_checkpointed_ffn_chunks(model, chunk_assets=2, expected_layers=2)
    with pytest.raises(ValueError, match='already installed'):
        install_checkpointed_ffn_chunks(model, chunk_assets=2, expected_layers=2)
