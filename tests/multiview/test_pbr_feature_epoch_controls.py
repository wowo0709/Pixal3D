"""Pure tensor intervention contracts; no CPU neural forwards."""
import pytest
import torch

from experiments.deformnext_retargeting.pbr_feature_epoch_controls import feature_output_control


@torch.no_grad()
def test_feature_zero_preserves_actual_attention_probabilities_and_original_feature():
    feature = torch.tensor([[.03, -.02]])
    original = feature.clone()
    weights = torch.tensor([[[.8, .2], [.7, .3], [.6, .4], [.5, .5]]])
    changed, actual_weights = feature_output_control(feature, weights, 'feature-output-zero')
    assert not torch.count_nonzero(changed)
    assert actual_weights is weights and torch.equal(feature, original)


@torch.no_grad()
@pytest.mark.parametrize('control', ['normal', 'wrong-projected-V-fixed-use', 'R-off', 'both-source-contributions-off'])
def test_other_controls_do_not_silently_zero_feature(control):
    feature, weights = torch.ones(2, 3), torch.ones(2, 4, 5) / 5
    actual, probability = feature_output_control(feature, weights, control)
    assert actual is feature and probability is weights


def test_saved_weight_control_rejects_gradients():
    with pytest.raises(ValueError):
        feature_output_control(torch.ones(2, 3), torch.ones(2, 4, 5), 'feature-output-zero')
