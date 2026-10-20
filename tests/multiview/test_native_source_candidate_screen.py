"""Selection must not promote noise repeats or unmatched control gains."""
import copy

import pytest

from experiments.deformnext_retargeting.native_source_candidate_screen import screen


def model(tp, fp):
    return [dict(status='PASS', stage='ss64', step=480, optimizer_updates_during_evaluation=0,
                 sampling_steps=12, guidance=3.0,
                 per_asset=[dict(sample_id=str(i), source_cluster_id='cluster'+str(i // 2),
                                 seed=42 + noise * 100 + i, tp=tp, fp=fp, fn=100-tp)
                            for i in range(6)]) for noise in range(3)]


def data():
    return {'v1': model(30, 30), 'top1': model(35, 25), 'spatial': model(60, 5),
            'candidate': model(50, 10)}


POLICY = {'candidate': dict(required_controls=['v1', 'top1'], diagnostic_controls=['spatial'])}


def test_matched_positive_candidate_is_only_a_confirmation_candidate():
    result = screen(data(), POLICY, draws=60)
    assert result['selected_candidate'] == 'candidate'
    assert not result['model_adopted'] and not result['full_goal_complete']
    assert not result['independent_training_confirmation']
    assert result['generation_noise_repeats'] == 3
    diagnostic = result['candidate_evidence']['candidate']['comparisons']['spatial']
    assert not diagnostic['positive_source_cluster_CI']
    assert not diagnostic['required_for_eligibility']
    assert not result['isolated_matching_decoder_effect_proven']


def test_gain_over_v1_without_gain_over_matched_top1_is_not_eligible():
    values = data();values['top1'] = model(55, 5)
    result = screen(values, POLICY, draws=60)
    assert result['selected_candidate'] is None
    assert result['status'] == 'NO_PROMISING_NATIVE_SOURCE_COMBINATION_YET'


def test_pooled_positive_iou_does_not_hide_one_negative_noise():
    values = data();values['candidate'][2] = model(32, 28)[2]
    result = screen(values, POLICY, draws=60)
    assert result['selected_candidate'] is None
    assert not result['candidate_evidence']['candidate']['comparisons']['top1']['positive_at_all_three_noises']


def test_additive_module_must_beat_its_declared_predecessor():
    values = data();values['nodq'] = model(60, 5)
    policy = copy.deepcopy(POLICY);policy['candidate']['required_controls'].append('nodq')
    assert screen(values, policy, draws=60)['selected_candidate'] is None


@pytest.mark.parametrize('damage', ['two_noises', 'wrong_step', 'training_in_eval', 'unpaired_seed'])
def test_incomplete_or_unpaired_saved_results_rejected(damage):
    values = data()
    if damage == 'two_noises':
        values['candidate'].pop()
    elif damage == 'wrong_step':
        values['candidate'][0]['step'] = 384
    elif damage == 'training_in_eval':
        values['candidate'][0]['optimizer_updates_during_evaluation'] = 1
    else:
        values['candidate'][1]['per_asset'][0]['seed'] += 1
    with pytest.raises(ValueError):
        screen(values, POLICY, draws=60)


@pytest.mark.parametrize('controls', [[], ['top1'], ['v1', 'missing'], ['v1', 'candidate'], ['v1', 'v1']])
def test_policy_controls_cannot_be_silently_omitted(controls):
    with pytest.raises(ValueError):
        screen(data(), {'candidate': dict(required_controls=controls)}, draws=60)
