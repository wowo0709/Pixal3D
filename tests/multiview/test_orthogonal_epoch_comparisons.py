"""Guard trend reports against comparing different GT, noise or panel orders."""
import copy

import pytest

from experiments.deformnext_retargeting.compare_orthogonal_utility_epochs import paired_records, summarize


def records():
    return [dict(sample_id=f'case-{i}', source_cluster_id=f'source-{i // 2}', seed=100 + i,
                 tp=8, fp=2, fn=2, target_voxels_sha256=f'GT-{i}', target_voxel_shape=[64] * 3)
            for i in range(32)]


def test_paired_gain_uses_voxel_counts_and_keeps_source_clusters():
    baseline = records()
    candidate = copy.deepcopy(baseline)
    for row in candidate:
        row.update(tp=9, fn=1)
    result = paired_records(baseline, candidate, [r['sample_id'] for r in baseline])
    assert result['gain'] == pytest.approx(1 / 12)
    assert result['source_clusters'] == 16
    assert result['paired_target_hashes_verified'] == 32
    assert result['training_seed_replication'] is False


@pytest.mark.parametrize('field,value', [('target_voxels_sha256', 'other-GT'),
                                        ('seed', 999), ('source_cluster_id', 'other-source')])
def test_same_voxel_counts_cannot_hide_different_identity(field, value):
    baseline = records()
    candidate = copy.deepcopy(baseline)
    candidate[3][field] = value
    with pytest.raises(ValueError):
        paired_records(baseline, candidate, [r['sample_id'] for r in baseline])


def test_missing_hash_or_reordered_panel_cannot_be_reported():
    baseline = records()
    ids = [r['sample_id'] for r in baseline]
    with pytest.raises(ValueError, match='order'):
        paired_records(baseline, list(reversed(baseline)), ids)
    candidate = copy.deepcopy(baseline)
    del candidate[0]['target_voxels_sha256']
    with pytest.raises(ValueError, match='all32'):
        paired_records(baseline, candidate, ids)


def test_unknown_candidate_or_epoch_rejected_before_reading_data(tmp_path):
    for epoch, name in [(0, 'voxel-query-c1'), (6, 'utility-softprior-c1'), (1, 'unregistered')]:
        with pytest.raises(ValueError, match='declared candidate'):
            summarize(tmp_path, epoch, name)
