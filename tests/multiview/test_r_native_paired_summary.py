import copy

import pytest

from experiments.deformnext_retargeting.r_native_paired_summary import summarize_native_three


def results():
    out = {}
    for name, offset in [('vanilla', 0.), ('R_feature', .01), ('R_raw_topk', .02)]:
        out[name] = [dict(stage='ss64', step=480, native_variant=name, evaluation_mode='final128',
            optimizer_updates_during_evaluation=0, quick_metrics={}, per_asset=[dict(
                sample_id=str(i), source_cluster_id=str(i // 2), seed=noise * 1000 + i,
                target_voxels_sha256='GT' + str(i), voxel_iou=.4 + offset,
                voxel_precision=.5, voxel_recall=.6, voxel_f1=.55) for i in range(128)]) for noise in range(3)]
    return out


def test_noise_is_averaged_within_asset_before_source_cluster_bootstrap():
    value = summarize_native_three(results(), draws=100)
    assert value['assets'] == 128 and value['source_clusters'] == 64
    assert value['independent_training_seeds'] == 1
    contrast = value['comparisons']['R_raw_topk_minus_R_feature']
    assert contrast['mean_asset_delta']['voxel_iou'] == pytest.approx(.01)
    assert contrast['primary_IoU_three_contrast_family_CI95'] == pytest.approx([.01, .01])
    assert value['automatic_model_adoption'] is False


@pytest.mark.parametrize('damage', ['seed', 'GT', 'duplicate-noise', 'source', 'wrong-step'])
def test_apparent_gains_do_not_override_pairing_failures(damage):
    value = copy.deepcopy(results())
    if damage == 'duplicate-noise': value['vanilla'][1] = copy.deepcopy(value['vanilla'][0])
    elif damage == 'wrong-step': value['R_raw_topk'][0]['step'] = 96
    else:
        row = value['R_raw_topk'][0]['per_asset'][0]
        row[{'seed': 'seed', 'GT': 'target_voxels_sha256', 'source': 'source_cluster_id'}[damage]] = 'wrong'
    with pytest.raises(ValueError): summarize_native_three(value, draws=10)
