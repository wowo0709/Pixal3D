import hashlib

import pytest

from experiments.deformnext_retargeting.pbr_source_transfer_summary import KEYS,SOURCE_FIELDS,summarize,validate_normal_pair
from test_native_pbr_stage_summary import aggregate,metrics


def digest(value):return hashlib.sha256(str(value).encode()).hexdigest()


def record(key):
    effect=-.5 if key=='R-off' else 0.
    control='normal' if key=='normal-repeat' else key
    rows=[]
    for i in range(4):
        rows.append(dict(sample_id=str(i),source_cluster_id=str(i),conditioning_input_sha256=digest(i),
            seed=i,actual_noise_SHA256=digest(i),actual_RGB_input_SHA256=digest(i),
            actual_GT_PBR_latent_SHA256=digest(i),actual_GT_Shape_condition_SHA256=digest(i),
            actual_camera_SHA256={'camera':digest(i)},actual_prediction_PBR_latent_SHA256=digest((i,control)),
            metrics=metrics(20.+effect)))
    return dict(status='PASS_ACTUAL_NATIVE_PBR1900_SOURCE_TRANSFER_READOUT_FIXED4',diagnostic_key=key,
        control=control,physical_GPU='1' if key=='normal-repeat' else '0',model_mode='top1-only',actual_trained_model_updates=1900,optimizer_updates_in_diagnostic=0,
        unused_eval_optimizer_states=0,actual_deterministic_algorithms=True,all_weights_unchanged=True,
        frozen_original_base_and_matcher_exact=True,fixed_GT_target_shape=True,full_cascade=False,source_preservation=False,
        actual_model_files={'denoiser':{'path':'own1900','SHA256':digest('weight')},'deform_conditioner':{'path':'source1900','SHA256':digest('source')}},
        actual_model_state_SHA256={'denoiser':digest('weight'),'deform_conditioner':digest('source')},
        actual_frozen_encoder_SHA256=digest('encoder'),per_asset=rows,quick_metrics=aggregate([r['metrics'] for r in rows]),
        module_events=[{'fixed_source_fields':{k:digest((i,k)) for k in SOURCE_FIELDS}} for i in range(4)],
        consumer_events=[dict(native_global_object_preserved=True,native_global_SHA256=digest(i),
            native_proj_SHA256=digest(i),actual_delta_RMS=0. if control=='both-source-contributions-off' else .1) for i in range(4)],
        condition_events=[{'negative':digest(i)} for i in range(4)],
        panel_pixel_SHA256=[{'GT':digest(i),'prediction':digest((i,control))} for i in range(4)],
        actual_original_epoch_prediction_exact=[False]*4)


def records():return {k:record(k) for k in KEYS}


def test_normal_parity_gate_needs_no_fabricated_control_results():
    r=validate_normal_pair({k:record(k) for k in KEYS[:2]})
    assert r['no_control_quality_results_claimed'] is True


@pytest.mark.parametrize('mutation',['same_GPU','changed_consumer'])
def test_normal_gate_rejects_wrong_device_or_unstable_actual_consumer(mutation):
    r={k:record(k) for k in KEYS[:2]}
    if mutation=='same_GPU':r['normal-repeat']['physical_GPU']='0'
    else:r['normal-repeat']['consumer_events'][0]['actual_delta_RMS']=.2
    with pytest.raises(ValueError):validate_normal_pair(r)


def test_negative_control_effect_is_retained_without_quality_adoption():
    r=summarize(records())
    assert r['contrasts']['R-off_minus_normal']['metric_deltas']['pooled_metrics']['render_psnr']==pytest.approx(-.5)
    assert r['original_epoch_metrics_not_used_for_control_contrasts'] is True
    assert r['quality_gain_proven'] is False and r['full_goal_complete'] is False


@pytest.mark.parametrize('mutation',['missing_repeat','repeat_prediction','repeat_metrics','GT_pixels',
    'noise','source_scores','negative_condition','model_weights','both_off_nonzero','step','optimizer'])
def test_unpaired_or_incomplete_readout_is_rejected(mutation):
    r=records();x=r['R-off']
    if mutation=='missing_repeat':del r['normal-repeat']
    elif mutation=='repeat_prediction':r['normal-repeat']['per_asset'][0]['actual_prediction_PBR_latent_SHA256']='changed'
    elif mutation=='repeat_metrics':r['normal-repeat']['per_asset'][0]['metrics']['render_psnr']=0.
    elif mutation=='GT_pixels':x['panel_pixel_SHA256'][0]['GT']='changed'
    elif mutation=='noise':x['per_asset'][0]['actual_noise_SHA256']='changed'
    elif mutation=='source_scores':x['module_events'][0]['fixed_source_fields']['matching_logits']=digest('changed')
    elif mutation=='negative_condition':x['condition_events'][0]['negative']='changed'
    elif mutation=='model_weights':x['actual_model_state_SHA256']['denoiser']=digest('changed')
    elif mutation=='both_off_nonzero':r['both-source-contributions-off']['consumer_events'][0]['actual_delta_RMS']=.1
    elif mutation=='step':x['actual_trained_model_updates']=380
    else:x['optimizer_updates_in_diagnostic']=1
    with pytest.raises(ValueError):summarize(r)
