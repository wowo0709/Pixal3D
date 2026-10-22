"""Bounded paired evidence from actual in-process epoch96 validation rows."""
import numpy as np


def summarize_epoch96(records, *, draws=4000, seed=42096):
    allowed = {'vanilla', 'R_top1_only', 'R_feature', 'R_raw_topk'}
    if 'vanilla' not in records or not 2 <= len(records) <= 4 or not set(records) <= allowed:
        raise ValueError('native vanilla and declared paired source models required')
    reference = records['vanilla']
    first = reference['quick_per_asset']
    ids = [r['sample_id'] for r in first]
    clusters = [r['source_cluster_id'] for r in first]
    if len(ids) != 32 or len(set(ids)) != 32 or any(not c for c in clusters):
        raise ValueError('unique DEV32 with actual source cluster identities required')
    matrices = {}
    metrics = ('voxel_iou', 'voxel_precision', 'voxel_recall', 'voxel_f1')
    for name, record in records.items():
        rows = record['quick_per_asset']
        if (record['step'] != 96 or record['weights'] != 'raw' or record['quick_panel_ids'] != ids
                or [(r['sample_id'], r['source_cluster_id'], r['seed'], r['target_voxels_sha256']) for r in rows] !=
                    [(r['sample_id'], r['source_cluster_id'], r['seed'], r['target_voxels_sha256']) for r in first]):
            raise ValueError('actual same-epoch asset, source, seed and GT records must pair')
        matrices[name] = np.asarray([[r[m] for m in metrics] for r in rows], dtype=np.float64)
        if not np.isfinite(matrices[name]).all():
            raise ValueError('nonfinite native validation metric')
    groups = [np.flatnonzero(np.asarray(clusters) == c) for c in sorted(set(clusters))]
    sizes = np.asarray([len(g) for g in groups])
    resamples = np.random.default_rng(seed).integers(0, len(groups), (draws, len(groups)))
    pairs = [(n, 'vanilla') for n in records if n != 'vanilla']
    pairs += [(a, b) for a, b in [('R_feature', 'R_top1_only'), ('R_raw_topk', 'R_feature'),
                                ('R_raw_topk', 'R_top1_only')] if a in records and b in records]
    contrasts = {}
    for a, b in pairs:
        delta = matrices[a] - matrices[b]
        sums = np.stack([delta[g].sum(0) for g in groups])
        boot = sums[resamples].sum(1) / sizes[resamples].sum(1)[:, None]
        contrasts[a + '_minus_' + b] = dict(
            mean_asset_delta=dict(zip(metrics, delta.mean(0).tolist())),
            source_cluster_CI95={m: np.quantile(boot[:, k], [.025, .975]).tolist() for k, m in enumerate(metrics)})
    return dict(status='PASS_ACTUAL_IN_PROCESS_NATIVE_EPOCH96_PAIRED_DEV32_EVIDENCE',
        assets=32, source_clusters=len(groups), step=96, generation_noises=1, training_seed=101,
        native_global_occupancy_metrics={n: r['quick_metrics'] for n, r in records.items()},
        mean_asset_metrics={n: dict(zip(metrics, m.mean(0).tolist())) for n, m in matrices.items()},
        exploratory_unadjusted_contrasts=contrasts, independent_checkpoint_GPU_replay_not_proven_by_this_summary=True,
        final128_or_test508_not_evaluated=True, quality_gain_proven=False,
        automatic_training_restart_or_model_adoption=False, full_goal_complete=False)
