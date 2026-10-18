"""Prevent inference-only feature statistics from being logged for other models."""
import copy
import math

import pytest

from experiments.deformnext_retargeting.semantic_topk_matching_drift import (
    CONTRASTS, DISTRIBUTIONS, FIELDS, PREFIX, collect_matching_drift, validate_reports,
)


def reports():
    rows = [{'sample_id': str(i), 'source_cluster_id': str(i % 93)} for i in range(128)]
    qa = {'selected_head': {'sha256': 'a'*64}, 'trained_conditioner': {'sha256': 'b'*64},
          'source_recipe': {'sha256': 'c'*64}}
    distribution = {'mean_effective_candidates': 4., 'mean_max_weight': .5, 'mean_entropy_nats': 1.}
    contrast = dict(zip(FIELDS, [.4, .1, .2, .1, .99]))
    base = {'status': 'PASS_FIXED128_SAVED_MATCHING_PRIOR_DIAGNOSIS',
            'cases': 128, 'source_clusters': 93, 'active_queries': 9883,
            'device': 'CPU_FP32', 'GPU_used': False, 'optimizer_updates': 0,
            'inputs': copy.deepcopy(rows), 'weights': {'initializer': qa['selected_head'],
                'trained_conditioner': qa['trained_conditioner']}, 'recipe': qa['source_recipe'],
            'original_saved_QA': {'sha256': 'd'*64}, 'panel': {'sha256': 'e'*64}, 'runtime_source': [],
            'per_case_context_and_score_changes': [
                {'sample_id': str(i), 'active_queries': 77 if i < 127 else 104} for i in range(128)],
            'attention_distributions': {k: copy.deepcopy(distribution) for k in DISTRIBUTIONS[:5]},
            'contrasts': {k: copy.deepcopy(contrast) for k in CONTRASTS[:5]}}
    hybrid = copy.deepcopy(base)
    hybrid.update(status='PASS_FIXED128_SAVED_MATCHING_PRIOR_HYBRID_DIAGNOSIS',
                  full128_original_diagnostic_replay={'status': 'PASS', 'max_absolute_statistic_difference': 0.})
    hybrid['attention_distributions'].update({k: copy.deepcopy(distribution) for k in DISTRIBUTIONS[5:]})
    hybrid['contrasts'].update({k: copy.deepcopy(contrast) for k in CONTRASTS[5:]})
    return base, hybrid, rows, qa


def test_later_diagnostic_is_optional_but_a_partial_pair_is_not_silently_skipped(tmp_path):
    assert collect_matching_drift(tmp_path, [], {}) is None
    p = tmp_path/'matching-prior-drift-panel128-v1/summary.json'
    p.parent.mkdir();p.write_text('{}')
    with pytest.raises(FileNotFoundError):
        collect_matching_drift(tmp_path, [], {})


def test_component_swap_statistics_stay_in_diagnostic_namespace():
    result = validate_reports(*reports())
    assert len(result['metrics']) == 66
    assert all(key.startswith(PREFIX+'/') for key in result['metrics'])
    assert len(result['tables'][PREFIX+'/attention_distributions']['data']) == 7
    assert len(result['tables'][PREFIX+'/component_swaps']['data']) == 9
    assert 'not accuracy' in result['scope'] and 'not independently trained' in result['scope']


@pytest.mark.parametrize('field,value', [('optimizer_updates', 1), ('GPU_used', True), ('device', 'GPU_BF16')])
def test_training_or_other_precision_cannot_be_reported_as_this_cpu_inference_probe(field, value):
    base, hybrid, rows, qa = reports();hybrid[field] = value
    with pytest.raises(ValueError, match='scope'):
        validate_reports(base, hybrid, rows, qa)


def test_wrong_model_cannot_supply_a_diagnostic_for_the_completed_full_model():
    base, hybrid, rows, qa = reports();qa['trained_conditioner'] = {'sha256': 'f'*64}
    with pytest.raises(ValueError, match='saved Full'):
        validate_reports(base, hybrid, rows, qa)


def test_changed_source_order_is_detected_even_with_the_same_case_count():
    base, hybrid, rows, qa = reports();hybrid['inputs'].reverse()
    with pytest.raises(ValueError, match='case/source order'):
        validate_reports(base, hybrid, rows, qa)


def test_missing_queries_cannot_be_hidden_in_panel_level_statistics():
    base, hybrid, rows, qa = reports();hybrid['per_case_context_and_score_changes'][0]['active_queries'] -= 1
    with pytest.raises(ValueError, match='query accounting'):
        validate_reports(base, hybrid, rows, qa)


def test_hybrid_replay_claim_cannot_hide_a_changed_original_result():
    base, hybrid, rows, qa = reports();hybrid['attention_distributions']['trained_prior_T05']['mean_max_weight'] += .01
    with pytest.raises(ValueError, match='original attention'):
        validate_reports(base, hybrid, rows, qa)


def test_nonfinite_new_component_statistics_do_not_enter_wandb_payload():
    base, hybrid, rows, qa = reports()
    hybrid['contrasts'][CONTRASTS[-1]]['candidate_argmax_changed_fraction'] = math.nan
    with pytest.raises(ValueError, match='component-swap'):
        validate_reports(base, hybrid, rows, qa)


def test_a_missing_component_swap_is_incomplete_even_when_status_says_pass():
    base, hybrid, rows, qa = reports();del hybrid['contrasts'][CONTRASTS[-1]]
    with pytest.raises(ValueError, match='incomplete'):
        validate_reports(base, hybrid, rows, qa)
