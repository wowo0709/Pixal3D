"""Paired native-task evidence including the physical Top1-only control."""
import numpy as np


def summarize_native_four(results, *, draws=4000, seed=42022):
    names = ('vanilla', 'R_top1_only', 'R_feature', 'R_raw_topk')
    metrics = ('voxel_iou', 'voxel_precision', 'voxel_recall', 'voxel_f1')
    if set(results) != set(names) or any(len(results[n]) != 3 for n in names):
        raise ValueError('four actual native-task models and three paired noises required')
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
                    or result['actual_source_tensors'] != (0 if name == 'vanilla' else 10 if name == 'R_top1_only' else 19)
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
    for a, b in [('R_top1_only', 'vanilla'), ('R_feature', 'R_top1_only'), ('R_raw_topk', 'R_feature'),
                 ('R_feature', 'vanilla'), ('R_raw_topk', 'vanilla'), ('R_raw_topk', 'R_top1_only')]:
        delta = assets[a] - assets[b]
        sums = np.stack([delta[g].sum(0) for g in groups])
        boot = sums[resamples].sum(1) / sizes[resamples].sum(1)[:, None]
        source_boot = (sums / sizes[:, None])[resamples].mean(1)
        comparisons[a + '_minus_' + b] = dict(
            mean_asset_delta=dict(zip(metrics, delta.mean(0).tolist())),
            source_cluster_CI95={m: np.quantile(boot[:, k], [.025, .975]).tolist() for k, m in enumerate(metrics)},
            mean_source_delta=dict(zip(metrics, (sums / sizes[:, None]).mean(0).tolist())),
            equal_source_cluster_CI95={m: np.quantile(source_boot[:, k], [.025, .975]).tolist() for k, m in enumerate(metrics)},
            primary_IoU_six_contrast_family_CI95=np.quantile(boot[:, 0], [.05 / 12, 1 - .05 / 12]).tolist())
    return dict(status='PASS_PAIRED_NATIVE_FOUR_MODEL_DEV128_THREE_NOISE_ANALYSIS',
        assets=128, source_clusters=len(groups), generation_noises=3,
        independent_training_seeds=1, noise_is_not_independent_training_seed=True,
        primary_metric='mean asset voxel IoU, averaged over three paired noises',
        per_model_mean_asset={n: dict(zip(metrics, m.mean((0, 1)).tolist())) for n, m in matrices.items()},
        per_model_native_global_occupancy_metrics={n: [r['quick_metrics'] for r in results[n]] for n in names},
        comparisons=comparisons, held_out_test508_evaluated=False,
        physical_Top1_is_not_uniform_attention=True,
        independent_decoder_or_native_spatial_effect_not_identified=True,
        automatic_model_adoption=False, full_goal_complete=False)
