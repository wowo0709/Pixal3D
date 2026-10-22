from copy import deepcopy

import pytest
import torch

from experiments.deformnext_retargeting.r_native_source_weight_audit import audit_source_weights


def states():
    names = ['ray_readout.' + str(i) for i in range(9)]
    initial = {n: torch.ones(2) for n in names}
    initial.update({'ray_readout.initial_' + str(i): torch.ones(2) for i in range(7)})
    initial.update({'q2c.decoder.weight': torch.ones(2), 'q2c.score.weight': torch.ones(2),
        'ray_readout.residual_logit_bound': torch.tensor(1.), 'retargeter.net.2.weight': torch.zeros(2),
        'fusion_gate': torch.tensor(0.)})
    return initial, deepcopy(initial), names


def test_actual_weight_movement_does_not_claim_attention_or_generation_effect():
    initial, current, names = states()
    current[names[0]][0] += .01
    current['retargeter.net.2.weight'][0] = .02
    result = audit_source_weights(initial, current, names)
    assert result['learned_attention_tensors_changed'] == 1
    assert result['learned_head_tensors'][names[0]]['delta_L2'] > 0
    assert result['attention_distributions_not_evaluated']
    assert result['quality_gain_proven'] is False


@pytest.mark.parametrize('name', ['q2c.decoder.weight', 'q2c.score.weight', 'ray_readout.initial_3'])
def test_frozen_decoder_score_and_reference_drift_is_rejected(name):
    initial, current, names = states()
    current[name][0] += .01
    with pytest.raises(ValueError, match='frozen matching'): audit_source_weights(initial, current, names)


def test_matching_checkpoint_dtype_cannot_change_under_an_exact_frozen_claim():
    initial, current, names = states()
    current['q2c.decoder.weight'] = current['q2c.decoder.weight'].bfloat16()
    with pytest.raises(ValueError, match='frozen matching'): audit_source_weights(initial, current, names)
