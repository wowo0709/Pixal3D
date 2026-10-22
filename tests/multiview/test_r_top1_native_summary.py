import copy

import pytest

from experiments.deformnext_retargeting.r_native_paired_summary import summarize_native_three
from experiments.deformnext_retargeting.r_top1_native_summary import summarize_native_four
from test_r_native_paired_summary import results


def four():
    value = results()
    value['R_top1_only'] = copy.deepcopy(value['R_feature'])
    for name, evaluations in value.items():
        for evaluation in evaluations:
            evaluation['native_variant'] = name
            evaluation['actual_source_tensors'] = 0 if name == 'vanilla' else 10 if name == 'R_top1_only' else 19
            if name == 'R_top1_only':
                for row in evaluation['per_asset']: row['voxel_iou'] -= .005
    return value


def test_four_way_control_preserves_all_original_three_way_paired_statistics():
    value = four()
    original = summarize_native_three({k: v for k, v in value.items() if k != 'R_top1_only'}, draws=100)
    new = summarize_native_four(value, draws=100)
    assert len(new['comparisons']) == 6
    assert new['physical_Top1_is_not_uniform_attention']
    for key, contrast in original['comparisons'].items():
        for field in ['mean_asset_delta', 'source_cluster_CI95', 'mean_source_delta', 'equal_source_cluster_CI95']:
            assert new['comparisons'][key][field] == contrast[field]
    contrast = new['comparisons']['R_feature_minus_R_top1_only']
    assert contrast['mean_asset_delta']['voxel_iou'] == pytest.approx(.005)
    assert new['independent_training_seeds'] == 1 and new['automatic_model_adoption'] is False


def test_declared_Top1_model_cannot_contain_the_old_learned_head():
    value = four()
    value['R_top1_only'][0]['actual_source_tensors'] = 19
    with pytest.raises(ValueError): summarize_native_four(value, draws=10)
