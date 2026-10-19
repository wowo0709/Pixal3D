"""Changing residual-logit strength must retain native initialization and masking."""
import math

import pytest
import torch

from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.modules.task_matching_attention import TaskCandidateAttention


def attention(scale):
    torch.manual_seed(119)
    return TaskCandidateAttention(width=8, hidden=4, content_mode='residual-zero', initial_content_scale=scale)


def test_scale16_preserves_prior_output_and_multiplies_first_output_gradient_by_four():
    low, high = attention(4.), attention(16.)
    assert set(low.state_dict()) == set(high.state_dict())
    assert all(torch.equal(low.state_dict()[k], high.state_dict()[k])
               for k in low.state_dict() if k != 'log_content_scale')
    torch.manual_seed(27)
    target, context, candidates = torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 5, 8)
    valid = torch.tensor([[True, True, False, True, True], [True] * 5, [False] * 5])
    prior, direction = torch.randn(3, 5), torch.randn(3, 8)
    candidates[~valid] = float('nan')
    lower = low(target, context, candidates, valid, prior)
    higher = high(target, context, candidates, valid, prior)
    for left, right in zip(lower, higher):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for module, result in [(low, lower), (high, higher)]:
        (result[0] * direction).sum().backward()
        assert module.content_output.weight.grad.abs().sum() > 0
        assert module.query.weight.grad.count_nonzero() == 0
        assert torch.isfinite(result[0]).all() and result[0][-1].count_nonzero() == 0
    torch.testing.assert_close(high.content_output.weight.grad, 4 * low.content_output.weight.grad,
                               rtol=1e-6, atol=1e-6)


def test_conditioner_receives_scale16_without_new_parameter_or_deformation_channels():
    cond = DeformationAwareConditioner(arm='c1', key_dim=1024, value_dim=1024, query_dim=1024,
        task_attention={'decoder': True, 'score_prior': True, 'chunk_size': 64,
            'train_decoder': False, 'train_score': False, 'content_mode': 'residual-zero',
            'initial_content_scale': 16.})
    assert cond.task_attention.log_content_scale.item() == pytest.approx(math.log(16))
    assert len(list(cond.task_attention.parameters())) == 5
    assert cond.retargeter.input_dim == 2048
    assert not any(p.requires_grad for p in cond.q2c.parameters())


@pytest.mark.parametrize('scale', [0, -1, 21., float('nan'), float('inf'), True, '16'])
def test_invalid_initial_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        attention(scale)


def test_legacy_default_scale_and_checkpoint_keys_remain_unchanged():
    module = TaskCandidateAttention(width=8, hidden=4)
    assert module.log_content_scale.item() == pytest.approx(math.log(4))
    assert set(module.state_dict()) == {'query.weight', 'context.weight', 'key.weight', 'log_content_scale'}
