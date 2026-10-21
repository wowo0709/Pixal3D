import pytest
import torch

from experiments.deformnext_retargeting.member_feature_readout_control import (
    MODES, feature_readout_control, probabilities,
)
from test_member_feature_query_support import prepared


def clean(model):
    for layer in model.modules():
        assert not layer._forward_hooks and not layer._forward_pre_hooks


@pytest.mark.parametrize('mode', MODES)
def test_frozen_causal_modes_keep_original_context_and_restore_weights(mode):
    model, data = prepared(); events = []
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        baseline = model(**data)
        with feature_readout_control(model, events, mode=mode):
            actual = model(**data)
        restored = model(**data)
    assert torch.equal(baseline.source_numerator, restored.source_numerator)
    assert torch.equal(baseline.use_probability, actual.use_probability)
    assert torch.equal(baseline.semantic_weight, actual.semantic_weight)
    for field in ['candidate_indices', 'candidate_weights', 'matching_logits', 'base_evidence', 'query_valid']:
        assert torch.equal(getattr(baseline.attention, field), getattr(actual.attention, field))
    if mode == 'normal':
        assert torch.equal(baseline.source_numerator, actual.source_numerator)
    else:
        assert not torch.equal(baseline.source_numerator, actual.source_numerator)
    assert len(events) == 1 and events[0]['queries'] == int(actual.attention.query_valid.sum())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert events[0]['actual_matching_decoder_unchanged']
    assert not events[0]['retrained_NoScore_NoCorr_or_decoder_ablation']
    if mode == 'matching-prior-only':
        assert all(abs(row['KL_to_matching_prior_mean']) < 1e-7 for row in events[0]['chunks'])
    if mode == 'zero-feature-correction':
        assert all(row['actual_feature_correction_RMS'] == 0 for row in events[0]['chunks'])
    clean(model)


def test_cross_mode_fixed_context_identity_and_actual_probability_change():
    model, data = prepared(); results = {}
    with torch.no_grad():
        for mode in MODES:
            events = []
            with feature_readout_control(model, events, mode=mode):
                model(**data)
            results[mode] = events[0]
    expected = results['normal']['fixed_context_SHA256']
    assert all(row['fixed_context_SHA256'] == expected for row in results.values())
    assert all(row['probability_change_RMS'] > 0 for row in results['matching-prior-only']['chunks'])
    assert all(row['probability_change_RMS'] > 0 for row in results['without-readout-logprior']['chunks'])
    clean(model)


@pytest.mark.parametrize('missing', ['source_valid', 'observed_query'])
def test_empty_source_or_query_still_native_prior(missing):
    model, data = prepared(); data[missing].zero_(); events = []
    with torch.no_grad(), feature_readout_control(model, events, mode='matching-prior-only'):
        result = model(**data)
    assert not result.source_numerator.any() and not result.semantic_weight.any()
    assert events[0]['queries'] == 0 and not events[0]['chunks']
    clean(model)


@pytest.mark.parametrize('condition', ['training', 'gradient', 'combined-control'])
def test_reject_unsafe_or_combined_probe_and_remove_hooks(condition):
    model, data = prepared()
    if condition == 'training': model.train()
    if condition == 'combined-control': data['control'] = 'wrong-feature-values-fixed-use'
    context = torch.enable_grad if condition == 'gradient' else torch.no_grad
    with context(), pytest.raises(ValueError, match='eval/no_grad'):
        with feature_readout_control(model, [], mode='matching-prior-only'):
            model(**data)
    clean(model)


def test_malformed_same_forward_fails_closed_and_restores():
    model, data = prepared()
    def corrupt(head, args, result):
        return result[0], result[1].roll(1, -1)
    hook = model.ray_readout.register_forward_hook(corrupt)
    try:
        with torch.no_grad(), pytest.raises(ValueError, match='probability differs'):
            with feature_readout_control(model, [], mode='normal'):
                model(**data)
    finally:
        hook.remove()
    clean(model)


def test_all_invalid_softmax_is_finite_zero():
    logits = torch.full((2, 4, 5), -torch.inf)
    actual = probabilities(logits, torch.zeros(2, 5, dtype=torch.bool))
    assert torch.isfinite(actual).all() and not actual.any()


@pytest.mark.parametrize('step', [480, 192])
def test_native_entry_preserves_arguments_and_restores_trainer_after_rejection(tmp_path, monkeypatch, step):
    import json
    import sys
    from types import SimpleNamespace
    from experiments.deformnext_retargeting import evaluate_member_feature_readout_control as entry
    from experiments.deformnext_retargeting import evaluate_member_feature_source_ray as native
    from pixal3d.trainers.flow_matching.member_feature_source_ray import MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer as trainer_class
    model, data = prepared(); trainer = object.__new__(trainer_class)
    trainer.deform_conditioner = model; trainer.optimizer = SimpleNamespace(state={})
    def encode(self):
        return self.deform_conditioner(**data)
    monkeypatch.setattr(trainer_class, '_encode_deform_views', encode)
    remaining = ['--expected-assets', '32', '--step', str(step), '--output', str(tmp_path)]
    def run_native():
        assert sys.argv[1:] == remaining
        with torch.no_grad():trainer._encode_deform_views()
        (tmp_path/'evaluation.json').write_text(json.dumps(dict(step=step, optimizer_updates_during_evaluation=0)))
    monkeypatch.setattr(native, 'main', run_native)
    argv = ['readout-probe', '--feature-readout-control', 'matching-prior-only', *remaining]
    monkeypatch.setattr(sys, 'argv', argv)
    if step == 480:
        entry.main(); result = json.loads((tmp_path/'evaluation.json').read_text())
        assert result['diagnostic_only'] and result['feature_readout_control']['actual_optimizer_states'] == 0
        assert result['feature_readout_control']['events'][0]['queries'] > 0
    else:
        with pytest.raises(ValueError, match='final480'):entry.main()
    assert sys.argv is argv and trainer_class._encode_deform_views is encode
    clean(model)
