import copy
import hashlib
import pytest

from experiments.deformnext_retargeting.spatial_readout_summary import CONTROLS,FIXED_FIELDS,summarize


def digest(value):return hashlib.sha256(str(value).encode()).hexdigest()


def records():
    result={}
    for control in CONTROLS:
        rows=[];artifacts=[]
        for i in range(32):
            tp=10+(control=='normal');fp=3;fn=20-tp
            rows.append(dict(sample_id=str(i),source_cluster_id=str(i),seed=i,target_voxels_sha256=digest(i),
                target_voxel_shape=[64,64,64],target_representation='synthetic test only',tp=tp,fp=fp,fn=fn,
                voxel_iou=tp/(tp+fp+fn),voxel_precision=tp/(tp+fp),voxel_recall=tp/(tp+fn),voxel_f1=2*tp/(2*tp+fp+fn)))
            artifacts.append(dict(sample_id=str(i),seed=i,target_voxels_SHA256=digest(i),occupancy_counts=dict(tp=tp,fp=fp,fn=fn)))
        event=dict(fixed_source_fields={k:digest(k) for k in FIXED_FIELDS},
            original_candidates_Top1_R_target_and_source_use_exact=True,same_forward_weights_unchanged=True)
        diag=dict(status='PASS_ACTUAL_SAVED_SPATIAL480_READOUT_CONTROL32',control=control,
            actual_original32_prediction_latents_and_voxels_exact=control=='normal',
            actual_trained_updates=480,optimizer_updates=0,actual_unused_optimizer_states=0,
            actual_model_states_unchanged=True,reference_archive_SHA256=digest('ref'),
            actual_loaded_model_state_SHA256={k:digest(k) for k in ['denoiser','deform_conditioner']},
            actual_source_checkpoint_SHA256={k:digest(k) for k in ['NN480','source480']},
            events=[copy.deepcopy(event) for _ in range(4)],actual_SS_artifacts=artifacts,
            consumer_events=[dict(target_native_global_object_preserved=True,native_projected_SHA256=digest('native')) for _ in range(4)],
            encoder_events=[dict(RGB=digest(i),negative=digest('negative')) for i in range(4)],
            sampler_events=[dict(noise=digest(i)) for i in range(4)])
        result[control]=dict(status='PASS',stage='ss64',step=480,weights='raw',per_asset=rows,
            sampling_steps=12,guidance=3.,generation_batch_size=8,input_states=['M1','M0','M0','M0'],
            input_camera_ids=[8,9,10,11],deterministic_evaluation=dict(actual_deterministic_algorithms=True),
            optimizer_updates_during_evaluation=0,spatial_readout_diagnostic=diag,
            native_prior_source_ray=dict(feature_control='spatial-feature-attention',actual_feature_head_tensors=9,
                original_correspondence_candidate_count=3,physical_deformation_query_channels=0))
    return result


def test_saved_control_reliance_retains_negative_results_without_adoption():
    value=summarize(records())
    assert value['model_adopted'] is False and value['primary_IoU_family_comparisons']==4
    assert value['contrasts']['top1-feature-only_minus_normal']['mean_asset_delta']['voxel_iou']<0


@pytest.mark.parametrize('mutation',['normal_not_exact','noise','matching_prior','native','model'])
def test_wrong_actual_reference_or_fixed_inputs_are_rejected(mutation):
    r=records();d=r['top1-feature-only']['spatial_readout_diagnostic']
    if mutation=='normal_not_exact':r['normal']['spatial_readout_diagnostic']['actual_original32_prediction_latents_and_voxels_exact']=False
    elif mutation=='noise':d['sampler_events'][0]['noise']=digest('changed')
    elif mutation=='matching_prior':d['events'][0]['fixed_source_fields']['matching_prior']=digest('changed')
    elif mutation=='native':d['consumer_events'][0]['native_projected_SHA256']=digest('changed')
    else:d['actual_source_checkpoint_SHA256']['NN480']=digest('changed')
    with pytest.raises(ValueError):summarize(r)
