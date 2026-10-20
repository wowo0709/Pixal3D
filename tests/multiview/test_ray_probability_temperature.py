"""Pooled temperature changes confidence, preserving native T1 and ranking."""
import pytest
import torch

from pixal3d.modules.ray_feature_attention import RayFeatureReadout
from pixal3d.modules.ray_probability_temperature import (
    matching_logits, native_probability, pooled_temperature)


@pytest.mark.parametrize('width', [1024, 2048])
def test_native_probability_and_matching_gradients_are_exact_at_temperature_one(width):
    torch.manual_seed(211)
    head = RayFeatureReadout(width)
    torch.nn.init.normal_(head.score_selector.content_output.weight, std=.02)
    args = (torch.randn(3, 1024), torch.randn(3, 1024), torch.randn(3, 5, 1024),
            torch.randn(3, 5, width),
            torch.tensor([[True]*5, [True, False, True, True, False], [False]*5]), torch.randn(3, 5))
    valid = args[4]
    _, expected = head(*args)
    logits = matching_logits(head, *args)
    assert torch.equal(native_probability(logits, valid), expected)
    actual = pooled_temperature(logits, valid, 1.)
    assert torch.equal(actual, expected.mean(1))
    names, params = zip(*[(n, p) for n, p in head.named_parameters()
                         if n not in {'value.weight', 'output.weight'}])
    reward = torch.randn_like(actual)
    left = torch.autograd.grad((expected.mean(1)*reward).sum(), params, retain_graph=True)
    right = torch.autograd.grad((actual*reward).sum(), params)
    assert len(names) == 7
    assert all(torch.equal(a, b) for a, b in zip(left, right))


def test_pooled_calibration_keeps_ranking_and_handles_underflow_without_fake_observations():
    logits = torch.tensor([[[2., 1., -120., 99.], [1., 3., -121., 99.]],
                           [[-torch.inf]*4, [-torch.inf]*4]])
    valid = torch.tensor([[True, True, True, False], [False]*4])
    baseline = pooled_temperature(logits, valid, 1.)
    assert baseline[0, 2] == 0
    for temperature in [.125, .25, .5, .75, 1.5, 2., 4.]:
        result = pooled_temperature(logits, valid, temperature)
        assert torch.isfinite(result).all() and result[~valid].count_nonzero() == 0
        assert result[0].argmax() == baseline[0].argmax()
        assert torch.allclose(result[0].sum(), torch.tensor(1.))
    assert pooled_temperature(logits, valid, 4.)[0, 2] > 0
    for bad in [True, 0., -1., float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            pooled_temperature(logits, valid, bad)
