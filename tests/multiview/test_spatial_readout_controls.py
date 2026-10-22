import pytest
import torch

from experiments.deformnext_retargeting.spatial_readout_controls import (
    combine_projected_values, select_probabilities,
)
from experiments.deformnext_retargeting.evaluate_spatial_readout_control import tensor_digest


def inputs():
    baseline = torch.tensor([.1, .3, .6]).reshape(1, 1, 3).expand(2, 4, 3).clone()
    prior = torch.tensor([1., 2., 7.]).log().expand(2, 3).clone()
    valid = torch.ones(2, 3, dtype=torch.bool)
    projected = torch.tensor([1., 10., 100.]).reshape(1, 1, 3, 1).expand(2, 4, 3, 64).clone()
    return baseline, prior, valid, projected


def test_normal_preserves_actual_probability_object():
    p, prior, valid, _ = inputs()
    assert select_probabilities(p, prior, valid, 'normal') is p


def test_matching_prior_retains_nonuniform_score():
    p, prior, valid, _ = inputs()
    actual = select_probabilities(p, prior, valid, 'matching-prior-only')
    torch.testing.assert_close(actual, torch.tensor([.1, .2, .7]).expand(2, 4, 3))
    assert not torch.equal(actual, torch.full_like(actual, 1/3))


def test_top1_selects_matching_correspondence_not_learned_argmax_or_uniform():
    p, prior, valid, _ = inputs()
    prior[:, 1] = 9.
    actual = select_probabilities(p, prior, valid, 'top1-feature-only')
    assert torch.equal(actual, torch.tensor([0., 1., 0.]).expand_as(p))


def test_top1_ignores_invalid_high_score_and_ties_choose_first():
    p, prior, valid, _ = inputs()
    valid[:, 2] = False; p[:, :, 2] = 0; p[:, :, :2] = .5; prior[:, :2] = 1.; prior[:, 2] = 100.
    assert torch.equal(select_probabilities(p, prior, valid, 'top1-feature-only'),
                       torch.tensor([1., 0., 0.]).expand_as(p))


def test_wrong_projected_values_change_mapping_without_changing_probabilities():
    p, _, _, projected = inputs(); saved = p.clone()
    torch.testing.assert_close(combine_projected_values(projected, p, 'normal'), torch.full((2, 256), 63.1))
    torch.testing.assert_close(combine_projected_values(projected, p, 'wrong-projected-V'), torch.full((2, 256), 16.3))
    assert torch.equal(p, saved)


@pytest.mark.parametrize('mutation', ['wide', 'negative', 'nonfinite', 'unnormalized', 'masked_mass', 'empty'])
def test_invalid_actual_native3_probabilities_are_rejected(mutation):
    p, prior, valid, _ = inputs()
    if mutation == 'wide': p = torch.ones(2, 4, 64)/64
    elif mutation == 'negative': p[0, 0, 0] = -.1
    elif mutation == 'nonfinite': prior[0, 0] = torch.nan
    elif mutation == 'unnormalized': p *= .5
    elif mutation == 'masked_mass': valid[:, 2] = False
    else: valid[:] = False
    with pytest.raises(ValueError): select_probabilities(p, prior, valid, 'normal')


def test_wrong_value_width_is_rejected():
    p, _, _, projected = inputs()
    with pytest.raises(ValueError): combine_projected_values(projected[..., :32], p, 'normal')


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_actual_condition_hash_supports_native_scalar_and_bfloat16_bits(dtype):
    value=torch.tensor(1.,dtype=dtype)
    assert tensor_digest(value)==tensor_digest(value.clone())
    assert tensor_digest(value)!=tensor_digest(value+1)
