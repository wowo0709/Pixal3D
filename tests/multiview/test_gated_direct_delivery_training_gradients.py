"""Pure tensor gradient contracts; no neural model construction on CPU."""
import torch

from experiments.deformnext_retargeting.direct_delivery_controls import controlled_residual, nearest_dense_consumer


def test_existing_gate_keeps_attention_learning_through_R_and_direct_delivery():
    F = torch.tensor([[[.03, -.02]]], requires_grad=True)
    gate = torch.tensor(.025, requires_grad=True)
    R = 3*F  # Analytic stand-in for the original R(Top1+F,target) path.
    C = .25*torch.tanh(F/.25)
    residual = controlled_residual(R+C, R, C, gate, 'direct-C-fusion-gated')
    residual.sum().backward()
    torch.testing.assert_close(F.grad, 3 + gate.detach()*(1-torch.tanh(F.detach()/.25).square()))
    torch.testing.assert_close(gate.grad, C.detach().sum())


def test_zero_output_has_exact_original_consumer_identity_and_nonzero_learning_signal():
    F = torch.zeros(1, 2, 3, requires_grad=True)
    gate = torch.tensor(.02, requires_grad=True)
    R = torch.full_like(F, .01) + 2*F
    C = .25*torch.tanh(F/.25)
    original = R+C
    gated = controlled_residual(original, R, C, gate, 'direct-C-fusion-gated')
    carrier = torch.ones_like(F)
    native = torch.zeros(1, 1, 3)
    args = (torch.ones(1, 2, dtype=torch.bool), gate, torch.zeros(1, 1, 1, 1, dtype=torch.long))
    expected = nearest_dense_consumer(native, native, carrier, original, *args)
    actual = nearest_dense_consumer(native, native, carrier, gated, *args)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert torch.all(F.grad[:, 0] > 0)
    assert torch.isfinite(gate.grad) and gate.grad != 0
