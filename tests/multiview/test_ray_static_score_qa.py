import pytest
import torch

from experiments.deformnext_retargeting.verify_ray_static_conditioning_contract import (
    compare_static_trace, score_temperature,
)


def test_scaled_static_logit_compares_in_the_original_head_units():
    left = torch.tensor([1., float('-inf')])
    right = torch.tensor([1.000025, float('-inf')])
    result = compare_static_trace(left, right, name='matching_logits', temperature=.05)
    assert result['max_abs_error'] > 1e-5
    assert result['head_score_max_abs_error'] < 1e-5
    assert result['masked_negative_infinity_values'] == 1


def test_material_head_score_difference_is_still_rejected():
    with pytest.raises(AssertionError):
        compare_static_trace(torch.tensor([1.]), torch.tensor([1.001]),
            name='matching_logits', temperature=.05)


def test_invalid_mask_difference_is_still_rejected():
    with pytest.raises(ValueError, match='masks differ'):
        compare_static_trace(torch.tensor([1., float('-inf')]), torch.tensor([1., 0.]),
            name='matching_logits', temperature=.05)


def test_weight_and_projection_comparisons_keep_the_original_tolerance():
    with pytest.raises(AssertionError):
        compare_static_trace(torch.tensor([.2]), torch.tensor([.200025]),
            name='candidate_weights', temperature=.05)


def test_learned_utility_logits_cannot_use_the_static_score_adapter():
    cfg = {'trainer': {'name': 'RayGroupImageConditionedProjFlowMatchingCFGTrainer',
        'args': {'deform_conditioner': {'ray_mode': 'semantic-weighted',
            'task_attention': None, 'score_temperature': .05}}}}
    assert score_temperature(cfg) == .05
    cfg['trainer']['args']['deform_conditioner']['task_attention'] = {'score_prior': True}
    with pytest.raises(ValueError, match='static-ray'):
        score_temperature(cfg)
