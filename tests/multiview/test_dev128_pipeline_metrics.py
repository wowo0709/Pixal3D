"""Keep camera roles and paired sources intact when extending the panel."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.cascade_view_roles import IMAGE_METRICS
from experiments.deformnext_retargeting.dev128_pipeline_contract import ARMS
from experiments.deformnext_retargeting.dev128_pipeline_metrics import summarize


def population():
    ids = list(map(str, range(128)))
    clusters = {sid:str(i % 93) for i, sid in enumerate(ids)}
    rows = [dict(sample_id=sid, source_cluster_id=clusters[sid], CD=.01, FScore=.8,
        views={cid:{m:.2 for m in IMAGE_METRICS} for cid in (8,9,10)}) for sid in ids]
    return {arm:deepcopy(rows) for arm in ARMS}, ids, clusters


def test_source_and_target_effects_remain_separate_with_full_paired_population():
    records, ids, clusters = population()
    for row in records['E_LEARNED_RAW']:
        row['views'][8]['LPIPS'] += .03
        row['views'][9]['LPIPS'] -= .01
        row['views'][10]['LPIPS'] -= .01
    result = summarize(records, sample_ids=ids, clusters=clusters, draws=30)
    key = 'E_LEARNED_RAW_minus_C_DECODER'
    assert result['assets'] == 128 and result['source_clusters'] == 93
    assert result['roles']['target8']['contrasts'][key]['deformation_weighted_delta']['LPIPS'] == pytest.approx(.03)
    assert result['roles']['source9_10']['contrasts'][key]['deformation_weighted_delta']['LPIPS'] == pytest.approx(-.01)
    assert not result['candidate_selected'] and not result['full_goal_complete']


@pytest.mark.parametrize('damage', ['missing-suffix', 'reordered', 'source-cluster', 'wrong-model'])
def test_suffix_coverage_or_pairing_errors_cannot_be_hidden_by_good_prefix(damage):
    records, ids, clusters = population()
    if damage == 'missing-suffix':
        records['E_FIXED_RAW'].pop()
    elif damage == 'reordered':
        records['E_FIXED_RAW'][100:102] = records['E_FIXED_RAW'][100:102][::-1]
    elif damage == 'source-cluster':
        records['E_FIXED_RAW'][100]['source_cluster_id'] = 'different'
    else:
        records['FULL'] = records.pop('E_LEARNED_RAW')
    with pytest.raises(ValueError):
        summarize(records, sample_ids=ids, clusters=clusters, draws=30)
