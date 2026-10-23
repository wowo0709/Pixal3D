"""Exploratory shared-evidence follow-up with two predeclared module contrasts."""
import numpy as np


def summarize_consistent_evidence(results, training_seeds, *, draws=4000, seed=42022):
    names = ('vanilla', 'R_top1_only', 'E_FIXED_RAW', 'E_LEARNED_RAW')
    metrics = ('voxel_iou', 'voxel_precision', 'voxel_recall', 'voxel_f1')
    if set(results) != set(names) or any(len(results[n]) != 3 for n in names):
        raise ValueError('four actual shared-evidence follow-up models and three paired noises required')
    if training_seeds != {n: 101 for n in names}:
        raise ValueError('actual closed configuration TRAIN seed101 for all four models required')
    first = results['vanilla'][0]['per_asset']
    ids = [r['sample_id'] for r in first]
    clusters = [r['source_cluster_id'] for r in first]
    if len(ids) != 128 or len(set(ids)) != 128 or any(not c for c in clusters):
        raise ValueError('unique128 assets with actual source cluster IDs required')
    matrices = {}
    for name in names:
        matrices[name] = []
        for noise, result in enumerate(results[name]):
            rows = result['per_asset']
            reference = results['vanilla'][noise]['per_asset']
            if (result['stage'] != 'ss64' or result['step'] != 480
                    or result['native_variant'] != name or result['evaluation_mode'] != 'final128'
                    or result['optimizer_updates_during_evaluation'] != 0
                    or result['actual_source_tensors'] != dict(vanilla=0,R_top1_only=10,E_FIXED_RAW=10,E_LEARNED_RAW=17)[name]
                    or [r['sample_id'] for r in rows] != ids
                    or [r['source_cluster_id'] for r in rows] != clusters
                    or any((a['seed'], a['target_voxels_sha256']) !=
                           (b['seed'], b['target_voxels_sha256']) for a, b in zip(rows, reference))):
                raise ValueError('paired actual native checkpoint/asset/seed/GT identity differs')
            same = ['manifest_sha256','target_modality','input_states','input_camera_ids',
                    'generation_batch_size','sampling_steps','guidance','deterministic_algorithms']
            expected = dict(target_modality='rgb', input_states=['M1','M0','M0','M0'],
                input_camera_ids=[8,9,10,11],generation_batch_size=8,sampling_steps=12,
                guidance=3.,deterministic_algorithms=True)
            if (any(k not in result for k in same)
                    or any(result[k] != results['vanilla'][noise][k] for k in same)
                    or any(result[k] != v for k,v in expected.items())):
                raise ValueError('same actual data, RGB roles, cameras and sampler required')
            if name.startswith('E_') and (result.get('conditioning_schema') != 'consistent-raw-ray-evidence-native-v1'
                    or result.get('followup_study_schema') != 'consistent-raw-ray-evidence-followup-v1'
                    or result.get('shared_carrier_use_retarget_evidence') is not True
                    or result.get('learned_projection_tensors') != 0
                    or result.get('learned_attention_head_tensors') != (7 if name == 'E_LEARNED_RAW' else 0)
                    or result.get('training_seed') != 101):
                raise ValueError('actual new shared-evidence architecture and training101 required')
            matrices[name].append([[r[m] for m in metrics] for r in rows])
        matrices[name] = np.asarray(matrices[name], dtype=np.float64)
    if any(len({results['vanilla'][j]['per_asset'][i]['seed'] for j in range(3)}) != 3 for i in range(128)):
        raise ValueError('three distinct generation seeds per asset required')
    if not all(np.isfinite(m).all() for m in matrices.values()):
        raise ValueError('nonfinite native metrics')
    # Average each asset across noise before resampling its source cluster.
    assets = {n: m.mean(0) for n, m in matrices.items()}
    groups = [np.flatnonzero(np.asarray(clusters) == c) for c in sorted(set(clusters))]
    resamples = np.random.default_rng(seed).integers(0, len(groups), (draws, len(groups)))
    sizes = np.asarray([len(g) for g in groups])
    comparisons = {}
    contrasts=[('E_FIXED_RAW','R_top1_only',2),('E_LEARNED_RAW','E_FIXED_RAW',2),
               ('E_LEARNED_RAW','vanilla',2),('E_LEARNED_RAW','R_top1_only',2)]
    for a,b,family in contrasts:
        delta = assets[a] - assets[b]
        sums = np.stack([delta[g].sum(0) for g in groups])
        boot = sums[resamples].sum(1) / sizes[resamples].sum(1)[:, None]
        source_boot = (sums / sizes[:, None])[resamples].mean(1)
        comparisons[a + '_minus_' + b] = dict(
            mean_asset_delta=dict(zip(metrics, delta.mean(0).tolist())),
            source_cluster_CI95={m: np.quantile(boot[:, k], [.025, .975]).tolist() for k, m in enumerate(metrics)},
            mean_source_delta=dict(zip(metrics, (sums / sizes[:, None]).mean(0).tolist())),
            equal_source_cluster_CI95={m: np.quantile(source_boot[:, k], [.025, .975]).tolist() for k, m in enumerate(metrics)},
            prespecified_IoU_family_size=family,
            primary_IoU_family_CI95=np.quantile(boot[:,0],[.025/family,1-.025/family]).tolist())
    return dict(status='PASS_PAIRED_CONSISTENT_EVIDENCE_EXPLORATORY_DEV128_THREE_NOISE_ANALYSIS',
        assets=128, source_clusters=len(groups), generation_noises=3,
        independent_training_seeds=1, noise_is_not_independent_training_seed=True,
        primary_metric='mean asset voxel IoU, averaged over three paired noises',
        per_model_mean_asset={n: dict(zip(metrics, m.mean((0, 1)).tolist())) for n, m in matrices.items()},
        per_model_native_global_occupancy_metrics={n: [r['quick_metrics'] for r in results[n]] for n in names},
        comparisons=comparisons, held_out_test508_evaluated=False,
        physical_Top1_is_not_uniform_attention=True,
        independent_decoder_or_native_spatial_effect_not_identified=True,
        shared_raw_evidence_with_identity_V=True,
        only_learned_probabilities_differ_between_followups=True,
        followup_is_exploratory_after_previous_module_results=True,
        original_twelve_arm_family_and_results_unchanged=True,
        prespecified_module_contrasts=2,prespecified_pipeline_contrasts=2,
        independent_training_seed_confirmation=False,full_cascade_gain_not_proven=True,
        automatic_model_adoption=False,full_goal_complete=False)
