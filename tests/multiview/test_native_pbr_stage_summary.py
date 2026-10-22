import copy
import hashlib
import math

import pytest

from experiments.deformnext_retargeting.native_pbr_stage_summary import (
    MODES, STEPS, aggregate, compare_snapshots, validate_completion, validate_curve,
)


def metrics(psnr=20., lpips=.1):
    return dict(foreground_fraction=.25, foreground_psnr=psnr,
        foreground_base_color_psnr=psnr,crop_ssim=.9,crop_lpips=lpips,
        render_psnr=psnr,render_ssim=.9,render_lpips=lpips,
        base_color_psnr=psnr,base_color_ssim=.9,gt_shape_conditioned_alpha_iou=1.)


def completion(mode):
    source=0 if mode=='vanilla' else 10 if mode=='top1-only' else 19
    return dict(active=dict(PID=123,birth=100.,mode=mode,physical_GPUs=['0','1','2','3']),
        terminal=dict(PID=123,birth=100.,returncode=0,physical_GPUs=['0','1','2','3']),
        wrapper=dict(status='PASS_ACTUAL_FRESH_NATIVE_PBR1900_SCIENCE',mode=mode,
            source_multiview_step=16850,fresh_optimizer=True,world_size=4,global_batch=24,
            microbatch=6,accumulation=1,actual_optimizer_updates=1900,
            actual_five_epoch_NN_source_Adam_canonical_archives_verified=True),
        ranks=[dict(status='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE',
            mode=mode,rank=rank,training_seed=101,world_size=4,global_batch=24,microbatch=6,
            accumulation=1,source_multiview_step=16850,fresh_optimizer=True,
            actual_optimizer_updates=1900,actual_optimizer_states=360+source,
            actual_training_assets=9139,actual_dataloader_batches=380,source_tensors=source,
            initial4_identity_before_first_update=True,frozen_base_matcher_and_anchors_exact=True,
            post_training_DDP_parameters_exact=True,stage_training_complete=True)for rank in range(4)],
        progress=dict(stage='completed',current=1900))


def snapshot(mode, effect=0.):
    rows=[]
    for step in STEPS:
        m=metrics(psnr=20+step/1900*(1+effect),lpips=.1-step/1900*effect*.01)
        per=[dict(sample_id=str(i),source_cluster_id=str(i),conditioning_input_sha256=str(i),
            seed=i,actual_noise_SHA256=str(i),actual_RGB_input_SHA256=str(i),
            actual_GT_PBR_latent_SHA256=str(i),actual_GT_Shape_condition_SHA256=str(i),
            actual_camera_SHA256={'camera':str(i)},metrics=copy.deepcopy(m),
            actual_prediction_PBR_latent_SHA256=hashlib.sha256(f'{i}:{step}:{mode if step else "base"}'.encode()).hexdigest())for i in range(4)]
        rows.append(dict(step=step,weights='raw',fixed_GT_target_shape=True,full_cascade=False,
            source_preservation_metrics=False,optimizer_updates_during_evaluation=0,
            quick_panel_ids=list(map(str,range(4))),quick_per_asset=per,quick_metrics=copy.deepcopy(m)))
    return dict(status='PASS_ACTUAL_NATIVE_PBR1900_FIXED4_STAGE_SNAPSHOT',mode=mode,
        validation_rows=rows,completion_proof=completion(mode),
        training_trace_digests={str(i):str(i)*64 for i in range(4)},
        GT_PNG_pixel_SHA256={str(step):{str(i):str(i)*64 for i in range(4)}for step in STEPS})


def test_pooled_psnr_is_not_the_mean_of_asset_decibels():
    result=aggregate([metrics(p)for p in [10.,20.,30.,40.]])
    assert result['render_psnr']==pytest.approx(-10*math.log10((.1+.01+.001+.0001)/4))
    assert result['render_psnr']!=pytest.approx(25.)


def test_completed_stage_needs_terminal_proof_without_a_live_training_PID():
    validate_completion('vanilla',**completion('vanilla'))


@pytest.mark.parametrize('field,value', [('actual_optimizer_updates',4),('actual_optimizer_states',1),
    ('frozen_base_matcher_and_anchors_exact',False),('post_training_DDP_parameters_exact',False),
    ('source_tensors',19),('actual_training_assets',9149)])
def test_capacity_or_wrong_stage_data_cannot_replace_full_science(field,value):
    proof=completion('vanilla');proof['ranks'][0][field]=value
    with pytest.raises(ValueError):validate_completion('vanilla',**proof)


@pytest.mark.parametrize('field,value', [('PID',999),('birth',99.),('returncode',1)])
def test_terminal_must_belong_to_actual_owned_child(field,value):
    proof=completion('vanilla');proof['terminal'][field]=value
    with pytest.raises(ValueError):validate_completion('vanilla',**proof)


@pytest.mark.parametrize('field,value', [('full_cascade',True),('fixed_GT_target_shape',False),
    ('source_preservation_metrics',True),('optimizer_updates_during_evaluation',1),('weights','ema')])
def test_scope_or_validation_update_cannot_be_mislabelled(field,value):
    rows=snapshot('vanilla')['validation_rows'];rows[-1][field]=value
    with pytest.raises(ValueError):validate_curve(rows)


def test_missing_final_epoch_and_wrong_aggregation_are_rejected():
    rows=snapshot('vanilla')['validation_rows']
    with pytest.raises(ValueError):validate_curve(rows[:-1])
    rows[-1]['quick_per_asset'][0]['metrics']['render_psnr']=0
    with pytest.raises(ValueError):validate_curve(rows)


@pytest.mark.parametrize('field', ['seed','actual_RGB_input_SHA256','actual_GT_Shape_condition_SHA256','actual_camera_SHA256'])
def test_same_asset_names_do_not_hide_changed_inputs(field):
    rows=snapshot('vanilla')['validation_rows'];rows[-1]['quick_per_asset'][0][field]='changed'
    with pytest.raises(ValueError):validate_curve(rows)


def test_negative_source_result_is_preserved_without_automatic_adoption():
    result=compare_snapshots({'vanilla':snapshot('vanilla'),'top1-only':snapshot('top1-only',-.5)})
    assert result['contrasts']['top1-only_minus_vanilla']['per_epoch'][1900]['mean_asset_metrics']['render_psnr']==pytest.approx(-.5)
    assert result['quality_gain_proven'] is False and result['full_goal_complete'] is False


@pytest.mark.parametrize('mutation', ['trace','GT_pixels','initial_prediction'])
def test_matched_comparison_requires_actual_training_and_image_identity(mutation):
    records={'vanilla':snapshot('vanilla'),'top1-only':snapshot('top1-only')}
    if mutation=='trace':records['top1-only']['training_trace_digests']['1']='f'*64
    elif mutation=='GT_pixels':records['top1-only']['GT_PNG_pixel_SHA256']['1900']['0']='f'*64
    else:records['top1-only']['validation_rows'][0]['quick_per_asset'][0]['actual_prediction_PBR_latent_SHA256']='changed'
    with pytest.raises(ValueError):compare_snapshots(records)


def test_all_four_models_use_the_strong_top1_control_and_six_contrasts():
    result=compare_snapshots({m:snapshot(m)for m in MODES})
    assert len(result['contrasts'])==6
    assert 'raw-topk-plus-feature_minus_top1-only' in result['contrasts']
    assert result['source_preservation'] is False and result['full_cascade'] is False
