from copy import deepcopy

import pytest
import torch

from experiments.deformnext_retargeting.r_mediated_interventions import (
    CONTROLS, SavedRMediatedIntervention, controlled_readout,
)
from test_r_mediated_source_contrast import model_pair
from test_member_posterior_source_ray import activate


def trained_like():
    _, model, data = model_pair('raw-topk-plus-feature')
    activate(model)
    with torch.no_grad():
        model.ray_readout.output.weight.normal_(std=.02)
        model.ray_readout.query.weight.add_(.1*torch.randn_like(model.ray_readout.query.weight))
    model.eval()
    return model, data


@pytest.mark.parametrize('control', CONTROLS)
def test_scoped_controls_preserve_matching_support_carrier_use_rng_and_weights(control):
    model, data = trained_like()
    before = deepcopy(model.state_dict())
    rng = torch.random.get_rng_state()
    intervention = SavedRMediatedIntervention(model, control)
    try:
        with torch.no_grad():
            result = model(**data)
        assert len(intervention.events) == 1
        event = intervention.events[0]
        assert all(event['fixed'].values())
        assert torch.equal(rng, torch.random.get_rng_state())
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
        if control in {'normal', 'wrong-retrieved-values-fixed-use', 'no-transformed-value'}:
            assert event['attention_probabilities_exact_to_normal']
        if control == 'normal':
            assert event['R_output_difference_L1'] == 0
        else:
            assert event['R_output_difference_L1'] > 0
        if control == 'no-transformed-value':
            assert event['R_input_appearance_exact_to_normal']
        if control == 'top1-feature-only':
            selected = model.last_source_reference_rows
            prior = result.attention.matching_logits.reshape(-1, 64)[selected]
            p = model.last_feature_probabilities[1]
            assert torch.equal(p.argmax(-1), prior.argmax(-1)[:, None].expand(-1, 4))
            assert (p.sum(-1) == 1).all()
    finally:
        intervention.close()
    assert not intervention.handles
    with torch.no_grad():
        restored = model(**data)
    if control == 'normal':
        assert torch.equal(restored.source_numerator, result.source_numerator)


def test_diagnostic_cannot_be_used_for_training_or_with_gradients():
    model, data = trained_like()
    intervention = SavedRMediatedIntervention(model, 'wrong-retrieved-values-fixed-use')
    try:
        with pytest.raises(ValueError, match='eval/no_grad'):
            model(**data)
        model.train()
        with torch.no_grad(), pytest.raises(ValueError, match='eval/no_grad'):
            model(**data)
    finally:
        intervention.close()


def test_changed_parameters_between_readonly_forwards_are_rejected():
    model, data = trained_like()
    intervention = SavedRMediatedIntervention(model, 'normal')
    try:
        with torch.no_grad():
            model(**data)
            model.fusion_gate.add_(.1)
            with pytest.raises(ValueError, match='parameters changed'):
                model(**data)
    finally:
        intervention.close()


def test_top1_control_masks_invalid_candidates_and_empty_prior_support():
    model, data = trained_like()
    head = model.ray_readout
    native, context = torch.randn(2, 1024), torch.randn(2, 1024)
    keys, values = torch.randn(2, 3, 1024), torch.randn(2, 3, 1024)
    valid = torch.tensor([[True, False, True], [True, True, True]])
    prior = torch.tensor([[1., 99., .5], [-torch.inf]*3])
    args = (native, context, keys, values, valid, prior)
    with torch.no_grad():
        original = head(*args)
        change, p = controlled_readout(head, args, original, 'top1-feature-only')
    assert torch.isfinite(change).all()
    assert (p[0, :, 0] == 1).all() and (p[0, :, 1:] == 0).all()
    assert not p[1].any()
