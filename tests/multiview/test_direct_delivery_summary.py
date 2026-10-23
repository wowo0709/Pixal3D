import copy

import pytest

from experiments.deformnext_retargeting.direct_delivery_summary import normal_parity, summarize


def record(control='normal', offset=0.):
    rows = [dict(sample_id=str(i), source_cluster_id=str(i//2), seed=42+i,
        target_voxels_sha256='gt'+str(i), target_voxel_shape=[64]*3, target_representation='SS64',
        tp=10, fp=3, fn=2, voxel_iou=.5+offset, voxel_precision=.7+offset,
        voxel_recall=.8+offset, voxel_f1=.75+offset) for i in range(32)]
    events = [dict(fixed_fields={'F': 'same', 'R': 'same', 'probability': 'same'},
        actual_CUDA_autocast_enabled=False, actual_CUDA_autocast_dtype='torch.bfloat16',
        original_consumer_reconstruction_exact=True, original_R_plus_C_exact=True,
        native_global_object_preserved=True, original_module_output_unchanged=True,
        actual_fusion_gate=.03, actual_C_coefficient=1. if control == 'normal' else (
            0. if control == 'direct-C-zero' else .03),
        delivered_residual_SHA256='same' if control == 'normal' else control,
        delivered_projected_SHA256='same' if control == 'normal' else control) for _ in range(4)]
    artifacts = [dict(sample_id=r['sample_id'], seed=r['seed'], target_voxels_SHA256=r['target_voxels_sha256'],
        target_latent_SHA256='target-latent'+r['sample_id'], prediction_voxels_SHA256='voxel'+r['sample_id'],
        prediction_latent_SHA256='pred-latent'+r['sample_id'],
        occupancy_counts={k:r[k] for k in ['tp', 'fp', 'fn']}) for r in rows]
    return dict(status='PASS', stage='ss64', step=288, weights='raw', optimizer_updates_during_evaluation=0,
        sampling_steps=12, guidance=3., generation_batch_size=8, input_states=['M1', 'M0', 'M0', 'M0'],
        input_camera_ids=[8, 9, 10, 11], deterministic_evaluation={'actual_deterministic_algorithms':True},
        direct_no_query=dict(variant='direct-physical-NoDQ', physical_deformation_channels=0,
            retargeter_input_channels=2048, deformation_projection_present=False),
        per_asset=rows, direct_delivery_diagnostic=dict(status='PASS_ACTUAL_FROZEN_NODQ288_DIRECT_C_CONTROL32',
            control=control, actual_trained_step=288, optimizer_updates=0, actual_unused_optimizer_states=0,
            actual_model_states_unchanged=True, original_matching_F_attention_and_R_computation_preserved=True,
            direct_C_coefficient_only_intervened=True, actual_loaded_model_state_SHA256={
                'denoiser':'a'*64, 'deform_conditioner':'b'*64}, actual_checkpoint_SHA256={'NN':'a'*64, 'source':'b'*64},
            events=events, encoder_events=[{'RGB':'same'}]*4, sampler_events=[{'noise':'same'}]*4,
            actual_numeric_runtime={'deterministic_algorithms':True, 'TF32':False},
            actual_SS_artifacts=artifacts, actual_original_quick288_counts_and_metrics_exact=False))


def test_parity_does_not_fabricate_historical_prediction_identity():
    actual = normal_parity(record(), record())
    assert actual['historical_quick_prediction_hashes_unavailable'] is True
    assert actual['original_quick288_counts_and_metrics_exact'] is False


@pytest.mark.parametrize('mutation', ['prediction', 'R', 'noise', 'optimizer', 'coefficient', 'GT'])
def test_actual_parity_rejects_uncontrolled_changes(mutation):
    normal = record(); other = copy.deepcopy(normal); d = other['direct_delivery_diagnostic']
    if mutation == 'prediction': d['actual_SS_artifacts'][0]['prediction_latent_SHA256'] = 'changed'
    elif mutation == 'R': d['events'][0]['fixed_fields']['R'] = 'changed'
    elif mutation == 'noise': d['sampler_events'][0]['noise'] = 'changed'
    elif mutation == 'optimizer': d['optimizer_updates'] = 1
    elif mutation == 'coefficient': d['events'][0]['actual_C_coefficient'] = .5
    else: other['per_asset'][0]['source_cluster_id'] = 'different'
    with pytest.raises(ValueError): normal_parity(normal, other)


def test_negative_diagnostic_is_valid_and_never_adopts_a_model():
    result = summarize({'normal':record(), 'normal-repeat':record(),
        'direct-C-zero':record('direct-C-zero', -.02),
        'direct-C-fusion-gated':record('direct-C-fusion-gated', -.01)})
    assert result['source_clusters'] == 16
    assert result['contrasts']['direct-C-zero_minus_normal']['primary_IoU_family2_CI95'][1] < 0
    assert result['model_adopted'] is False and result['quality_gain_proven'] is False


def test_controls_cannot_change_matching_or_R_even_if_metrics_improve():
    records = {'normal':record(), 'normal-repeat':record(),
        'direct-C-zero':record('direct-C-zero', .02),
        'direct-C-fusion-gated':record('direct-C-fusion-gated', .01)}
    records['direct-C-zero']['direct_delivery_diagnostic']['events'][0]['fixed_fields']['F'] = 'changed'
    with pytest.raises(ValueError, match='matching/F'): summarize(records)
