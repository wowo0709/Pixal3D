"""Consumer and gradient controls; no CPU neural forwards or CUDA allocation."""
import ast
from pathlib import Path

import torch

from experiments.deformnext_retargeting.direct_delivery_controls import controlled_residual, nearest_dense_consumer


def test_removing_direct_c_keeps_attention_gradient_through_retargeter():
    feature = torch.tensor([[[.03, -.02]]], requires_grad=True)
    gate = torch.tensor(.025, requires_grad=True)
    retargeted = 3 * feature
    correction = .25 * torch.tanh(feature / .25)
    delivered = controlled_residual(retargeted + correction, retargeted, correction, gate, 'direct-C-zero')
    assert delivered is retargeted
    delivered.sum().backward()
    torch.testing.assert_close(feature.grad, torch.full_like(feature, 3.))
    assert gate.grad is None  # The separate semantic carrier still uses this gate.


def test_initial_zero_c_preserves_consumer_and_attention_still_learns_through_r():
    feature = torch.zeros(1, 2, 3, requires_grad=True)
    gate = torch.tensor(.02, requires_grad=True)
    retargeted = torch.full_like(feature, .01) + 2 * feature
    correction = .25 * torch.tanh(feature / .25)
    original = retargeted + correction
    delivered = controlled_residual(original, retargeted, correction, gate, 'direct-C-zero')
    native = torch.zeros(1, 1, 3)
    carrier = torch.ones_like(feature)
    args = (torch.ones(1, 2, dtype=torch.bool), gate, torch.zeros(1, 1, 1, 1, dtype=torch.long))
    expected = nearest_dense_consumer(native, native, carrier, original, *args)
    actual = nearest_dense_consumer(native, native, carrier, delivered, *args)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert torch.all(feature.grad[:, 0] > 0)
    assert torch.isfinite(gate.grad) and gate.grad != 0


def test_nonzero_c_consumer_keeps_semantic_gate_and_r_exactly():
    native = torch.zeros(1, 1, 2)
    carrier = torch.ones(1, 2, 2)
    retargeted = torch.full_like(carrier, .01)
    correction = torch.full_like(carrier, .2)
    gate = torch.tensor(.025)
    payload = controlled_residual(retargeted + correction, retargeted, correction, gate, 'direct-C-zero')
    actual = nearest_dense_consumer(native, native, carrier, payload,
        torch.ones(1, 2, dtype=torch.bool), gate, torch.zeros(1, 1, 1, 1, dtype=torch.long))
    torch.testing.assert_close(actual, gate * .75 + .75 * retargeted[:, :1])


def test_runtime_uses_zero_direct_delivery_without_replacing_original_module():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / 'experiments/deformnext_retargeting/r_only_direct_delivery_runtime.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'controlled_residual']
    assert len(calls) == 1 and isinstance(calls[0].args[-1], ast.Constant)
    assert calls[0].args[-1].value == 'direct-C-zero'
