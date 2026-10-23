"""Actual normal parity and paired direct-C diagnostics; never select Full."""
import math

import numpy as np

from .direct_delivery_controls import CONTROLS


METRICS = ('voxel_iou', 'voxel_precision', 'voxel_recall', 'voxel_f1')
IDENTITY = ('sample_id', 'source_cluster_id', 'seed', 'target_voxels_sha256',
            'target_voxel_shape', 'target_representation')
ARTIFACT_IDENTITY = ('sample_id', 'seed', 'target_voxels_SHA256', 'target_latent_SHA256')
PREDICTION = ('prediction_voxels_SHA256', 'prediction_latent_SHA256', 'occupancy_counts')


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(record):
    d = record['direct_delivery_diagnostic']
    require(record['status'] == 'PASS' and record['stage'] == 'ss64' and record['step'] == 288
        and record['weights'] == 'raw' and record['optimizer_updates_during_evaluation'] == 0
        and record['sampling_steps'] == 12 and record['guidance'] == 3.
        and record['generation_batch_size'] == 8 and record['input_states'] == ['M1', 'M0', 'M0', 'M0']
        and record['input_camera_ids'] == [8, 9, 10, 11]
        and record['deterministic_evaluation']['actual_deterministic_algorithms'] is True,
        'actual deterministic RGB NoDQ288/batch8 protocol required')
    require(d['status'] == 'PASS_ACTUAL_FROZEN_NODQ288_DIRECT_C_CONTROL32' and d['control'] in CONTROLS
        and d['actual_trained_step'] == 288 and d['optimizer_updates'] == 0
        and d['actual_unused_optimizer_states'] == 0 and d['actual_model_states_unchanged'] is True
        and d['original_matching_F_attention_and_R_computation_preserved'] is True
        and d['direct_C_coefficient_only_intervened'] is True, 'actual frozen direct-C diagnostic required')
    require(set(d['actual_loaded_model_state_SHA256']) == {'denoiser', 'deform_conditioner'}
        and len(d['actual_checkpoint_SHA256']) == 2, 'both actual loaded model/checkpoint identities required')
    source = record['direct_no_query']
    require(source['variant'] == 'direct-physical-NoDQ' and source['physical_deformation_channels'] == 0
        and source['retargeter_input_channels'] == 2048 and source['deformation_projection_present'] is False,
        'original physically absent deformation-query structure required')
    rows, artifacts = record['per_asset'], d['actual_SS_artifacts']
    require(len(rows) == len(artifacts) == len({r['sample_id'] for r in rows}) == 32,
            'complete unique32 archived predictions required')
    require(all(len(d[k]) == 4 for k in ['events', 'encoder_events', 'sampler_events']),
            'four actual batch8 consumer/input/noise events required')
    for row, artifact in zip(rows, artifacts):
        require(row['sample_id'] == artifact['sample_id'] and row['seed'] == artifact['seed']
            and row['target_voxels_sha256'] == artifact['target_voxels_SHA256']
            and all(row[k] == artifact['occupancy_counts'][k] for k in ['tp', 'fp', 'fn'])
            and all(math.isfinite(row[k]) for k in METRICS), 'actual archived occupancy/metric identity required')
    for event in d['events']:
        require(all(event[k] is True for k in ['original_consumer_reconstruction_exact',
            'original_R_plus_C_exact', 'native_global_object_preserved', 'original_module_output_unchanged']),
            'actual consumer reconstruction and unchanged original module output required')
        coefficient = 1. if d['control'] == 'normal' else (
            0. if d['control'] == 'direct-C-zero' else event['actual_fusion_gate'])
        require(event['actual_C_coefficient'] == coefficient, 'wrong direct-C delivery coefficient')
    return d


def validate_pair(normal, actual, *, exact_predictions=False):
    nd, d = validate(normal), validate(actual)
    for key in ['actual_loaded_model_state_SHA256', 'actual_checkpoint_SHA256', 'actual_numeric_runtime',
                'encoder_events', 'sampler_events']:
        require(d[key] == nd[key], 'actual model/RGB/global/noise protocol changed: '+key)
    for left, right in zip(d['events'], nd['events']):
        require(left['fixed_fields'] == right['fixed_fields'],
                'matching/F/probabilities/R input/output/carrier/native/gate/valid changed')
        require(all(left[k] == right[k] for k in ['actual_CUDA_autocast_enabled', 'actual_CUDA_autocast_dtype']),
                'actual consumer CUDA autocast contract changed')
    for row, base, artifact, old in zip(actual['per_asset'], normal['per_asset'],
                                      d['actual_SS_artifacts'], nd['actual_SS_artifacts']):
        require(all(row[k] == base[k] for k in IDENTITY)
            and all(artifact[k] == old[k] for k in ARTIFACT_IDENTITY), 'paired ordered asset/GT/seed changed')
        if exact_predictions:
            require(row == base and all(artifact[k] == old[k] for k in PREDICTION),
                    'independent normal prediction latent/voxel/count/metric mismatch')
    if exact_predictions:
        require(d['control'] == nd['control'] == 'normal' and d['events'] == nd['events'],
                'both independent normal forwards and consumers must be exact')


def normal_parity(normal, repeat):
    validate_pair(normal, repeat, exact_predictions=True)
    return dict(status='PASS_ACTUAL_NODQ288_TWO_INDEPENDENT_NORMALS', assets=32,
        actual_checkpoint_SHA256=normal['direct_delivery_diagnostic']['actual_checkpoint_SHA256'],
        actual_inputs_noise_GT_predictions_latents_voxels_metrics_and_consumers_exact=True,
        original_quick288_counts_and_metrics_exact=normal['direct_delivery_diagnostic'][
            'actual_original_quick288_counts_and_metrics_exact'],
        historical_quick_prediction_hashes_unavailable=True, model_adopted=False, full_goal_complete=False)


def summarize(records):
    require(tuple(records) == ('normal', 'normal-repeat', *CONTROLS[1:]),
            'two normal repeats and both declared direct-C interventions required')
    parity = normal_parity(records['normal'], records['normal-repeat'])
    normal = records['normal']
    arrays, metrics = {}, {}
    for name, record in records.items():
        validate_pair(normal, record, exact_predictions=name in {'normal', 'normal-repeat'})
        values = np.array([[r[k] for k in METRICS] for r in record['per_asset']], dtype=np.float64)
        arrays[name] = values
        counts = {k: sum(r[k] for r in record['per_asset']) for k in ['tp', 'fp', 'fn']}
        metrics[name] = dict(mean_asset=dict(zip(METRICS, values.mean(0).tolist())),
                             pooled_IoU=counts['tp']/sum(counts.values()), counts=counts)
    clusters = {}
    for i, row in enumerate(normal['per_asset']):
        clusters.setdefault(row['source_cluster_id'], []).append(i)
    groups = [clusters[key] for key in sorted(clusters)]
    sizes = np.array([len(group) for group in groups])
    draws = np.random.default_rng(20261023).integers(0, len(groups), (4000, len(groups)))
    contrasts = {}
    for name in CONTROLS[1:]:
        delta = arrays[name] - arrays['normal']
        group_sums = np.stack([delta[group].sum(0) for group in groups])
        boot = group_sums[draws].sum(1)/sizes[draws].sum(1)[:, None]
        contrasts[name+'_minus_normal'] = dict(
            mean_asset_delta=dict(zip(METRICS, delta.mean(0).tolist())),
            exploratory_cluster_CI95={key: np.quantile(boot[:, i], [.025, .975]).tolist()
                                       for i, key in enumerate(METRICS)},
            primary_IoU_family2_CI95=np.quantile(boot[:, 0], [.0125, .9875]).tolist())
    return dict(status='PASS_ACTUAL_NODQ288_DIRECT_DELIVERY32_ANALYSIS', assets=32,
        source_clusters=len(groups), primary_metric='mean-asset voxel IoU',
        normal_parity=parity, metrics=metrics, contrasts=contrasts,
        bootstrap_draws=4000, bootstrap_seed=20261023, primary_IoU_family_comparisons=2,
        secondary_metrics_exploratory_unadjusted=True, independent_training_seeds=1,
        generation_noise_repeats=1, saved_weight_delivery_control_not_retrained_module_effect=True,
        R_still_conditioned_on_original_attention_F=True, not_NoScore_NoCorr_or_R_off=True,
        historical_quick_prediction_hashes_unavailable=True, model_adopted=False,
        quality_gain_proven=False, full_goal_complete=False)
