from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.r_native_paired_epoch import count_metrics, summarize_native_epoch


def records():
    result = {}
    for name, tp in [('vanilla', 10), ('R_top1_only', 11), ('R_feature', 12), ('R_raw_topk', 13)]:
        counts = [tp, 20-tp, 20-tp]
        rows = [dict(sample_id=str(i), source_cluster_id=str(i//2), seed=i,
            target_voxels_sha256=str(i), target_voxel_shape=[64]*3, target_representation='fixedGT',
            tp=counts[0], fp=counts[1], fn=counts[2], **count_metrics(*counts)) for i in range(32)]
        result[name] = dict(step=192, weights='raw', quick_panel_ids=[str(i) for i in range(32)],
            quick_per_asset=rows, quick_metrics=count_metrics(*(32*n for n in counts)))
    return result


def test_shared_epoch_and_counts_define_both_aggregations_and_cluster_contrasts():
    r = summarize_native_epoch(records(), step=192, draws=100)
    assert r['step'] == 192 and r['source_clusters'] == 16
    delta = 12/28-11/29
    assert r['exploratory_unadjusted_contrasts']['R_feature_minus_R_top1_only']['mean_asset_delta']['voxel_iou'] == pytest.approx(delta)
    assert r['quality_gain_proven'] is False and r['full_goal_complete'] is False


@pytest.mark.parametrize('field', ['seed', 'target_voxels_sha256', 'target_voxel_shape', 'target_representation'])
def test_same_means_cannot_hide_different_noise_or_GT(field):
    value = records(); value['R_feature']['quick_per_asset'][0][field] = 'different'
    with pytest.raises(ValueError, match='must pair'):
        summarize_native_epoch(value, step=192, draws=10)


def test_mismatched_epoch_or_stale_aggregate_metrics_are_rejected():
    value = records(); value['R_feature']['step'] = 96
    with pytest.raises(ValueError): summarize_native_epoch(value, step=192, draws=10)
    value = records(); value['R_feature']['quick_metrics']['voxel_iou'] += .01
    with pytest.raises(ValueError, match='occupancy counts'):
        summarize_native_epoch(value, step=192, draws=10)
    value = records(); value['R_feature']['quick_per_asset'][0]['tp'] += 1
    with pytest.raises(ValueError, match='occupancy counts'):
        summarize_native_epoch(value, step=192, draws=10)
