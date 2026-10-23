"""Paired native-task evidence including the physical Top1-only control."""
import numpy as np


def summarize_projected_modules(results, *, draws=4000, seed=42022):
    names = ('vanilla', 'R_top1_only', 'R_raw_topk', 'C_TOP1_V', 'C_FIXED_TOPK')
    metrics = ('voxel_iou', 'voxel_precision', 'voxel_recall', 'voxel_f1')
    if set(results) != set(names) or any(len(results[n]) != 3 for n in names):
        raise ValueError('five actual common-pipeline models and three paired noises required')
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
                    or result['actual_source_tensors'] != (0 if name == 'vanilla' else 10 if name == 'R_top1_only' else 12 if name.startswith('C_') else 19)
                    or [r['sample_id'] for r in rows] != ids
                    or [r['source_cluster_id'] for r in rows] != clusters
                    or any((a['seed'], a['target_voxels_sha256']) !=
                           (b['seed'], b['target_voxels_sha256']) for a, b in zip(rows, reference))):
                raise ValueError('paired actual native checkpoint/asset/seed/GT identity differs')
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
    contrasts=[('R_raw_topk','vanilla',2),('R_raw_topk','R_top1_only',2),
               ('C_TOP1_V','R_top1_only',10),('C_FIXED_TOPK','C_TOP1_V',10),
               ('R_raw_topk','C_FIXED_TOPK',10)]
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
    return dict(status='PASS_PAIRED_PROJECTED_MODULES_FIVE_MODEL_DEV128_FAMILY10_V2',
        assets=128, source_clusters=len(groups), generation_noises=3,
        independent_training_seeds=1, noise_is_not_independent_training_seed=True,
        primary_metric='mean asset voxel IoU, averaged over three paired noises',
        per_model_mean_asset={n: dict(zip(metrics, m.mean((0, 1)).tolist())) for n, m in matrices.items()},
        per_model_native_global_occupancy_metrics={n: [r['quick_metrics'] for r in results[n]] for n in names},
        comparisons=comparisons, held_out_test508_evaluated=False,
        physical_Top1_is_not_uniform_attention=True,
        independent_decoder_or_native_spatial_effect_not_identified=True,
        value_TopK_attention_separated_by_shared_V_output=True,
        planned_ten_module_family_not_reduced_to_only_observed_three=True,
        decoder_and_scalar_score_separate_in_protocol_v2=True,
        independent_training_seed_confirmation=False,full_cascade_gain_not_proven=True,
        automatic_model_adoption=False,full_goal_complete=False)
