import pytest
import torch

from experiments.deformnext_retargeting.member_feature_query_support import utility_query_support
from test_member_feature_source_ray import pair, activate


def prepared():
    _, model, data = pair()
    activate(model)
    torch.nn.init.normal_(model.ray_readout.output.weight, std=.05)
    data['observed_query'][0, 0] = False
    model.eval()
    return model, data


def check_clean(model):
    for layer in [model, model.ray_readout, model.retargeter, model.evidence_use]:
        assert not layer._forward_hooks and not layer._forward_pre_hooks


def test_normal_probe_exact_and_observed_probe_changes_only_feature_branch():
    model, data = prepared()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    normal_events, observed_events = [], []
    with torch.no_grad():
        reference = model(**data)
        with utility_query_support(model, normal_events, mode='normal'):
            normal = model(**data)
        with utility_query_support(model, observed_events, mode='observed-query-only'):
            observed = model(**data)
        restored = model(**data)
    assert torch.equal(reference.source_numerator, normal.source_numerator)
    assert torch.equal(reference.source_numerator, restored.source_numerator)
    assert not torch.equal(normal.source_numerator, observed.source_numerator)
    assert torch.equal(normal.use_probability, observed.use_probability)
    assert torch.equal(normal.semantic_weight, observed.semantic_weight)
    for field in ['candidate_indices', 'candidate_weights', 'matching_logits', 'base_evidence', 'query_valid']:
        assert torch.equal(getattr(normal.attention, field), getattr(observed.attention, field))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    event = observed_events[0]
    assert event['utility_query_change_RMS'] > 0 and event['excluded_input_alpha_fraction'] > 0
    assert event['original_R_and_source_use_target_context_exact']
    assert event['queries'] == event['actual_head_rows'] == int(normal.attention.query_valid.sum())
    assert len(normal_events) == len(observed_events) == 1
    assert normal_events[0]['fixed_context_SHA256'] == event['fixed_context_SHA256']
    check_clean(model)


@pytest.mark.parametrize('missing', ['observed_query', 'source_valid'])
def test_missing_support_retains_native_fallback_without_new_query_or_write_mask(missing):
    model, data = prepared(); data[missing].zero_(); events = []
    with torch.no_grad(), utility_query_support(model, events, mode='observed-query-only'):
        result = model(**data)
    assert not result.source_numerator.any() and not result.semantic_weight.any()
    assert events[0]['queries'] == 0 and events[0]['utility_query_change_RMS'] is None
    check_clean(model)


@pytest.mark.parametrize('condition', ['training', 'gradient', 'combined-control'])
def test_rejects_training_gradients_and_combined_controls_and_removes_hooks(condition):
    model, data = prepared()
    if condition == 'training':
        model.train()
    if condition == 'combined-control':
        data['control'] = 'wrong-feature-values-fixed-use'
    context = torch.enable_grad if condition == 'gradient' else torch.no_grad
    with context(), pytest.raises(ValueError, match='eval/no_grad'):
        with utility_query_support(model, [], mode='observed-query-only'):
            model(**data)
    check_clean(model)


def test_fails_closed_on_changed_head_row_order_and_restores_all_hooks():
    model, data = prepared()

    def corrupt(head, args):
        return (args[0]+1, *args[1:])

    handle = model.ray_readout.register_forward_pre_hook(corrupt)
    try:
        with torch.no_grad(), pytest.raises(ValueError, match='row order changed'):
            with utility_query_support(model, [], mode='observed-query-only'):
                model(**data)
        assert list(model.ray_readout._forward_pre_hooks.values()) == [corrupt]
    finally:
        handle.remove()
    check_clean(model)


@pytest.mark.parametrize('step', [480, 192])
def test_entry_keeps_native_arguments_and_restores_trainer_on_completion_or_rejection(tmp_path, monkeypatch, step):
    import json
    import sys
    from types import SimpleNamespace
    from experiments.deformnext_retargeting import evaluate_member_feature_query_support as entry
    from experiments.deformnext_retargeting import evaluate_member_feature_source_ray as native
    from pixal3d.trainers.flow_matching.member_feature_source_ray import MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer as trainer_class

    model, data = prepared()
    trainer = object.__new__(trainer_class)
    trainer.deform_conditioner = model
    trainer.optimizer = SimpleNamespace(state={})

    def encode(self):
        return self.deform_conditioner(**data)

    monkeypatch.setattr(trainer_class, '_encode_deform_views', encode)
    remaining = ['--expected-assets', '32', '--stage', 'ss64', '--step', str(step), '--output', str(tmp_path)]

    def run_native():
        assert sys.argv[1:] == remaining
        with torch.no_grad():
            trainer._encode_deform_views()
        (tmp_path/'evaluation.json').write_text(json.dumps(dict(
            step=step, optimizer_updates_during_evaluation=0, diagnostic_only=False)))

    monkeypatch.setattr(native, 'main', run_native)
    argv = ['query-probe', '--utility-query-control', 'observed-query-only', *remaining]
    monkeypatch.setattr(sys, 'argv', argv)
    if step == 480:
        entry.main()
        result = json.loads((tmp_path/'evaluation.json').read_text())
        assert result['diagnostic_only']
        assert result['utility_query_support']['training_updates'] == 0
        assert result['utility_query_support']['events'][0]['queries'] > 0
    else:
        with pytest.raises(ValueError, match='final480'):
            entry.main()
    assert sys.argv is argv and trainer_class._encode_deform_views is encode
    check_clean(model)
