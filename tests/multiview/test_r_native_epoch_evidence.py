from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.r_native_epoch_evidence import summarize_epoch96


def records():
    out = {}
    for name, offset in [('vanilla', 0.), ('R_top1_only', .005), ('R_feature', .01), ('R_raw_topk', .02)]:
        out[name] = dict(step=96, weights='raw', quick_panel_ids=[str(i) for i in range(32)], quick_metrics={},
            quick_per_asset=[dict(sample_id=str(i), source_cluster_id=str(i // 2), seed=i,
                target_voxels_sha256=str(i), voxel_iou=.4 + offset,
                voxel_precision=.5, voxel_recall=.6, voxel_f1=.55) for i in range(32)])
    return out


def test_first_epoch_is_bounded_paired_evidence_without_final_or_replay_claim():
    result = summarize_epoch96(records(), draws=100)
    assert result['assets'] == 32 and result['source_clusters'] == 16
    assert len(result['exploratory_unadjusted_contrasts']) == 6
    assert result['exploratory_unadjusted_contrasts']['R_feature_minus_R_top1_only']['mean_asset_delta']['voxel_iou'] == pytest.approx(.005)
    assert result['independent_checkpoint_GPU_replay_not_proven_by_this_summary']
    assert result['quality_gain_proven'] is False


def test_matching_means_do_not_hide_wrong_GT_or_noise():
    for field in ['seed', 'target_voxels_sha256']:
        value = deepcopy(records())
        value['R_feature']['quick_per_asset'][0][field] = 'wrong'
        with pytest.raises(ValueError): summarize_epoch96(value, draws=10)
