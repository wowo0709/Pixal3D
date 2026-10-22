import pytest

from experiments.deformnext_retargeting.native_pbr_epoch_comparison import compare_epoch
from test_native_pbr_stage_summary import aggregate, snapshot


def paired(step=380, effect=-.5):
    ref=snapshot('vanilla')['validation_rows']
    src=snapshot('top1-only',effect)['validation_rows']
    return src[:1+step//380],ref


def test_intermediate_negative_effect_is_preserved_without_claiming_completion():
    src,ref=paired()
    r=compare_epoch(src,ref,mode='top1-only',reference_mode='vanilla',step=380)
    assert r['same_epoch_delta']['mean_asset_metrics']['render_psnr']==pytest.approx(-.1)
    assert r['full_stage_completion_proven'] is False
    assert r['full_goal_complete'] is False


def test_same_epoch_strong_top1_control_is_supported():
    ref=snapshot('top1-only')['validation_rows']
    src=snapshot('raw-topk-plus-feature')['validation_rows'][:3]
    r=compare_epoch(src,ref,mode='raw-topk-plus-feature',reference_mode='top1-only',step=760)
    assert r['reference_mode']=='top1-only' and r['step']==760


@pytest.mark.parametrize('field',['seed','actual_RGB_input_SHA256','actual_GT_PBR_latent_SHA256',
    'actual_GT_Shape_condition_SHA256','actual_camera_SHA256','sample_id'])
def test_changed_actual_inputs_cannot_be_paired_by_epoch_number_alone(field):
    src,ref=paired();src[-1]['quick_per_asset'][0][field]='changed'
    with pytest.raises(ValueError):compare_epoch(src,ref,mode='top1-only',reference_mode='vanilla',step=380)


def test_cold_prediction_mismatch_is_rejected():
    src,ref=paired();src[0]['quick_per_asset'][0]['actual_prediction_PBR_latent_SHA256']='changed'
    with pytest.raises(ValueError):compare_epoch(src,ref,mode='top1-only',reference_mode='vanilla',step=380)


def test_incomplete_baseline_or_missing_source_epoch_cannot_be_invented():
    src,ref=paired()
    with pytest.raises(ValueError):compare_epoch(src,ref[:-1],mode='top1-only',reference_mode='vanilla',step=380)
    with pytest.raises(ValueError):compare_epoch(src,ref,mode='top1-only',reference_mode='vanilla',step=760)


def test_mean_asset_decibels_cannot_be_substituted_for_pooled_PSNR():
    src,ref=paired()
    for row in src+ref:
        for i,a in enumerate(row['quick_per_asset']):a['metrics']['render_psnr']=10.+i*10
        row['quick_metrics']=aggregate([a['metrics'] for a in row['quick_per_asset']])
    src[-1]['quick_metrics']['render_psnr']=25.
    with pytest.raises(ValueError):compare_epoch(src,ref,mode='top1-only',reference_mode='vanilla',step=380)


@pytest.mark.parametrize('flag,value',[('full_cascade',True),('source_preservation_metrics',True),
    ('fixed_GT_target_shape',False),('optimizer_updates_during_evaluation',1)])
def test_wrong_scope_or_evaluation_updates_are_rejected(flag,value):
    src,ref=paired();src[-1][flag]=value
    with pytest.raises(ValueError):compare_epoch(src,ref,mode='top1-only',reference_mode='vanilla',step=380)


def test_later_published_epochs_do_not_change_an_earlier_requested_comparison():
    src,ref=paired(1520);a=compare_epoch(src,ref,mode='top1-only',reference_mode='vanilla',step=380)
    b=compare_epoch(src[:2],ref,mode='top1-only',reference_mode='vanilla',step=380)
    assert a['same_epoch_delta']==b['same_epoch_delta']
    assert a['reported_source_epoch_steps']==[0,380]
