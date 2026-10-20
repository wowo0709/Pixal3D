"""Paired camera-role statistics must retain clustering and fixed references."""
import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.deformnext_retargeting.cascade_view_roles import (
    ClosedInputs, IMAGE_METRICS, existing_statistic_difference, summarize_view_roles,
)


def records():
    baseline = []
    for index, cluster in enumerate(('a', 'a', 'a', 'b')):
        views = {cid: {metric: float(index) for metric in IMAGE_METRICS} for cid in (8, 9, 10)}
        baseline.append(dict(sample_id=str(index), source_cluster_id=cluster,
                             views=views, CD=0.01 * index, FScore=0.8))
    candidate = copy.deepcopy(baseline)
    for row in candidate:
        for cid, view in row['views'].items():
            for metric in view:
                view[metric] += 3 if cid == 8 else 1
    return {'baseline': baseline, 'candidate': candidate}


def test_role_effect_and_paired_interaction_not_separate_significance():
    result = summarize_view_roles(records(), [('candidate', 'baseline')], draws=300)
    for role, delta in [('target8', 3), ('source9_10', 1), ('all8_9_10', 5 / 3)]:
        c = result['roles'][role]['contrasts']['candidate_minus_baseline']
        assert c['deformation_weighted_delta']['PSNR'] == pytest.approx(delta)
        assert c['deformation_weighted_cluster_CI95']['PSNR'] == pytest.approx([delta, delta])
    interaction = result['interaction_target_minus_source_effect']['candidate_minus_baseline']
    assert interaction['deformation_weighted_delta']['PSNR'] == 2
    assert interaction['deformation_weighted_cluster_CI95']['PSNR'] == [2, 2]
    assert 'CD' not in result['roles']['target8']['means']['candidate']['deformation_weighted']
    assert result['roles']['all8_9_10']['contrasts']['candidate_minus_baseline']['deformation_weighted_delta']['CD'] == 0


def test_deformation_and_source_weightings_keep_unequal_cluster_sizes():
    result = summarize_view_roles(records(), [], draws=300)
    means = result['roles']['target8']['means']['baseline']
    assert result['source_clusters'] == 2
    assert means['deformation_weighted']['PSNR'] == 1.5
    assert means['source_weighted']['PSNR'] == 2.0


def test_bootstrap_reproducible_and_original_allview_intervals_preserved():
    data = records()
    for i, row in enumerate(data['candidate']):
        row['views'][8]['PSNR'] += i * i
    pairs = [('candidate', 'baseline')]
    first = summarize_view_roles(data, pairs, draws=300)
    assert first == summarize_view_roles(data, pairs, draws=300)
    existing = first['roles']['all8_9_10']
    assert existing_statistic_difference(first, existing) == 0
    damaged = copy.deepcopy(existing)
    damaged['contrasts']['candidate_minus_baseline']['deformation_weighted_cluster_CI95']['PSNR'][0] += .001
    with pytest.raises(ValueError, match='statistics changed'):
        existing_statistic_difference(first, damaged)


@pytest.mark.parametrize('damage', ['duplicate_asset', 'asset_order', 'cluster', 'camera', 'nonfinite'])
def test_unpaired_or_invalid_results_rejected(damage):
    data = records()
    if damage == 'duplicate_asset':
        data['baseline'][1]['sample_id'] = data['baseline'][0]['sample_id']
    elif damage == 'asset_order':
        data['candidate'].reverse()
    elif damage == 'cluster':
        data['candidate'][0]['source_cluster_id'] = 'other'
    elif damage == 'camera':
        del data['candidate'][0]['views'][10]
    else:
        data['candidate'][0]['views'][8]['PSNR'] = np.nan
    with pytest.raises(ValueError):
        summarize_view_roles(data, [('candidate', 'baseline')], draws=30)


@pytest.mark.parametrize('pairs', [[('candidate', 'absent')], [('candidate', 'candidate')],
                                  [('candidate', 'baseline'), ('candidate', 'baseline')]])
def test_invalid_contrasts_rejected(pairs):
    with pytest.raises(ValueError):
        summarize_view_roles(records(), pairs, draws=30)


def test_receipt_checks_real_bytes_size_and_canonical_verification(tmp_path):
    path = tmp_path / 'input.json'
    path.write_bytes(b'{}')
    receipt = tmp_path / 'receipt.json'
    record = dict(status='PASS', local_equals_actual_file4_SHA256=True,
                  files=[dict(relative_path='input.json', bytes=2,
                              sha256=hashlib.sha256(b'{}').hexdigest())])
    receipt.write_text(json.dumps(record))
    checker = ClosedInputs(tmp_path)
    assert checker.check(path, receipt) == record['files'][0]['sha256']
    path.write_bytes(b'[]')
    with pytest.raises(ValueError, match='identity mismatch'):
        checker.check(path, receipt)
    path.write_bytes(b'{}')
    record['local_equals_actual_file4_SHA256'] = False
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Unverified canonical'):
        ClosedInputs(tmp_path).check(path, receipt)
