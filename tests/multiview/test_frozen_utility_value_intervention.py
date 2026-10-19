"""The value intervention must preserve the new utility model's learned scores."""
from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.task_attention_controls import task_value_context
from pixal3d.modules.task_matching_attention import TaskCandidateAttention


@pytest.mark.parametrize('learned_output', [False, True])
def test_value_only_control_keeps_utility_logits_weights_inputs_and_restores_hook(learned_output):
    attention = TaskCandidateAttention(width=4, hidden=3, content_mode='residual-zero').eval()
    if learned_output:
        with torch.no_grad():
            attention.content_output.weight.fill_(.1)
    target = torch.tensor([[1., 0., 1., 0.]])
    values = torch.eye(4)[None]
    original_values = values.clone()
    valid = torch.ones(1, 4, dtype=torch.bool)
    prior = torch.tensor([[12., -12., 0., 1.]])
    args = (target, target, values, valid, prior)
    with torch.no_grad():
        normal = attention(*args)
        events = []
        with task_value_context(SimpleNamespace(task_attention=attention), 'wrong-address', events):
            altered = attention(*args)
        restored = attention(*args)
    assert torch.equal(normal[1], altered[1])
    assert torch.equal(normal[2], altered[2])
    assert torch.equal(values, original_values)
    assert not torch.equal(normal[0], altered[0])
    assert all(torch.equal(a, b) for a, b in zip(normal, restored))
    assert events[0]['reassigned_valid_slots'] == 4
    assert events[0]['matching_logits_and_weights_unchanged']
    assert events[0]['evidence_change_RMS'] > 0
